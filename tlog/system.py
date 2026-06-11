"""Background system metrics sampler (GPU/CPU/RAM), zero dependencies.

Runs in a daemon thread; samples nvidia-smi and /proc at a fixed interval and
appends to system.jsonl. Any probe that fails once (e.g. no nvidia-smi on a
mac, no /proc) is disabled for the rest of the run.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable

from .writer import JsonlWriter

_NVSMI_FIELDS = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def _sample_gpus() -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={_NVSMI_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if out.returncode != 0:
        raise OSError("nvidia-smi failed")
    metrics: dict = {}
    for i, line in enumerate(out.stdout.strip().splitlines()):
        vals = [v.strip() for v in line.split(",")]

        def num(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None  # e.g. "[N/A]"

        util, mem_used, mem_total, temp, power = (num(v) for v in vals[:5])
        if util is not None:
            metrics[f"gpu/{i}/util_pct"] = util
        if mem_used is not None:
            metrics[f"gpu/{i}/mem_gb"] = round(mem_used / 1024, 2)
        if mem_total is not None:
            metrics[f"gpu/{i}/mem_total_gb"] = round(mem_total / 1024, 2)
        if temp is not None:
            metrics[f"gpu/{i}/temp_c"] = temp
        if power is not None:
            metrics[f"gpu/{i}/power_w"] = power
    return metrics


class _CpuProbe:
    """CPU utilization from /proc/stat deltas (Linux only)."""

    def __init__(self):
        self._last: tuple[int, int] | None = None

    def sample(self) -> dict:
        with open("/proc/stat") as f:
            fields = f.readline().split()[1:]
        vals = [int(v) for v in fields]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        metrics: dict = {}
        if self._last is not None:
            dt_total = total - self._last[0]
            dt_idle = idle - self._last[1]
            if dt_total > 0:
                metrics["cpu/util_pct"] = round(100.0 * (1 - dt_idle / dt_total), 1)
        self._last = (total, idle)
        return metrics


def _sample_mem() -> dict:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0])  # kB
    total = info.get("MemTotal")
    avail = info.get("MemAvailable")
    metrics: dict = {}
    if total:
        metrics["ram/total_gb"] = round(total / 1024 / 1024, 2)
        if avail is not None:
            metrics["ram/used_gb"] = round((total - avail) / 1024 / 1024, 2)
    return metrics


def _sample_loadavg() -> dict:
    return {"cpu/loadavg_1m": round(os.getloadavg()[0], 2)}


class SystemSampler:
    def __init__(self, writer: JsonlWriter, interval: float = 10.0,
                 get_step: Callable[[], int | None] = lambda: None):
        self._writer = writer
        self._interval = interval
        self._get_step = get_step
        self._stop = threading.Event()
        cpu = _CpuProbe()
        self._probes: list[Callable[[], dict]] = [
            _sample_gpus,
            cpu.sample,
            _sample_mem,
            _sample_loadavg,
        ]
        self._thread = threading.Thread(
            target=self._loop, name="tlog-system-sampler", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            record: dict = {"_ts": time.time()}
            step = self._get_step()
            if step is not None:
                record["_step"] = step
            for probe in list(self._probes):
                try:
                    record.update(probe())
                except Exception:
                    self._probes.remove(probe)  # probe unsupported here; drop it
            if len(record) > (2 if "_step" in record else 1):
                self._writer.write(record)

    def stop(self) -> None:
        self._stop.set()
