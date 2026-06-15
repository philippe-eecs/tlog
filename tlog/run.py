"""Run lifecycle: directory creation, resume, logging, finish."""

from __future__ import annotations

import atexit
import datetime
import os
import random
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import meta as _meta
from .console import ConsoleCapture
from .media import save_image
from .system import SystemSampler
from .writer import JsonlWriter, atomic_write_json

_ADJECTIVES = (
    "amber brisk calm dapper eager fabled gentle hazy icy jolly keen lucid "
    "mellow noble opal proud quiet rapid sleek tidy vivid wry zesty bold"
).split()
_NOUNS = (
    "falcon birch comet dune ember fjord glacier harbor iris juniper kestrel "
    "lagoon meadow nebula otter pine quartz reef summit tundra vale willow "
    "yarrow zephyr"
).split()

HEARTBEAT_INTERVAL = 15.0


def generate_name() -> str:
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", s).strip("-") or "run"


def _find_resumable(
    project_dir: Path,
    run_id: str | None,
    job_id: str | None,
    name: str | None = None,
) -> Path | None:
    """Locate an existing run dir to re-attach to. Priority: explicit id, then
    matching SLURM job id, then the most recent run with the same name — so a
    rerun of `(project, name)` continues that run instead of forking a new one."""
    if not project_dir.is_dir():
        return None
    import json

    job_matches, name_matches = [], []
    for d in project_dir.iterdir():
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        if run_id and d.name.endswith(f"__{run_id}"):
            return d
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            m = {}
        created = m.get("created_at", 0)
        if job_id and m.get("env", {}).get("slurm", {}).get("SLURM_JOB_ID") == job_id:
            job_matches.append((created, d))
        if name and d.name.split("__", 1)[0] == name:
            name_matches.append((created, d))
    if job_matches:
        return max(job_matches)[1]
    if name_matches:
        return max(name_matches)[1]
    return None


class Run:
    """A live training run writing to its own directory. Create via tlog.init()."""

    def __init__(
        self,
        project: str = "default",
        name: str | None = None,
        config: dict | None = None,
        dir: str | Path | None = None,
        id: str | None = None,
        resume: str = "auto",
        group: str | None = None,
        new: bool = False,
        reset: bool = False,
        capture_console: bool = True,
        system_metrics: bool = True,
        system_interval: float = 10.0,
    ):
        root = Path(dir or os.environ.get("TLOG_DIR", "./runs")).expanduser()
        project = _sanitize(project)
        project_dir = root / project
        want_name = _sanitize(name) if name else None

        slurm_job = os.environ.get("SLURM_JOB_ID")
        # `new=True` forces a fresh parallel run; otherwise we re-attach by
        # explicit id, SLURM job, or — by default — a matching name.
        existing = None
        if not new and resume != "never":
            existing = _find_resumable(project_dir, id, slurm_job, want_name)
        if resume == "must" and existing is None:
            raise RuntimeError(
                f"resume='must' but no existing run found (id={id!r}, name={name!r}, "
                f"SLURM_JOB_ID={slurm_job!r}) under {project_dir}"
            )

        self.group = group
        self.resumed = existing is not None
        if existing is not None:
            self.dir = existing
            self.id = existing.name.rsplit("__", 1)[-1]
            self.name = existing.name.split("__", 1)[0]
            if reset:
                self._wipe_data()
        else:
            self.id = id or uuid.uuid4().hex[:6]
            self.name = want_name or generate_name()
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.dir = project_dir / f"{self.name}__{stamp}__{self.id}"
            self.dir.mkdir(parents=True, exist_ok=True)

        self.project = project
        self._lock = threading.Lock()
        self._finished = False
        self._last_step: int | None = None

        self._metrics = JsonlWriter(self.dir / "metrics.jsonl")
        self._media_index: JsonlWriter | None = None

        self._init_meta()
        if config is not None:
            atomic_write_json(self.dir / "config.json", dict(config))

        self._console = ConsoleCapture(self.dir / "console.log") if capture_console else None

        self._system: SystemSampler | None = None
        if system_metrics:
            self._system = SystemSampler(
                JsonlWriter(self.dir / "system.jsonl"),
                interval=system_interval,
                get_step=lambda: self._last_step,
            )
            self._system.start()

        self._stop_heartbeat = threading.Event()
        self._heartbeat_path = self.dir / "heartbeat"
        self._heartbeat_path.touch()
        threading.Thread(
            target=self._heartbeat_loop, name="tlog-heartbeat", daemon=True
        ).start()

        threading.Thread(
            target=self._capture_slow_meta, name="tlog-meta", daemon=True
        ).start()

        atexit.register(self.finish)

    # -- metadata -----------------------------------------------------------

    def _wipe_data(self) -> None:
        """reset=True: clear prior metrics/media so a same-name run starts clean."""
        import shutil

        for name in ("metrics.jsonl", "system.jsonl", "console.log"):
            (self.dir / name).unlink(missing_ok=True)
        media = self.dir / "media"
        if media.is_dir():
            shutil.rmtree(media, ignore_errors=True)

    def _init_meta(self) -> None:
        now = time.time()
        meta_path = self.dir / "meta.json"
        if self.resumed:
            import json

            try:
                m = json.loads(meta_path.read_text())
            except (OSError, ValueError):
                m = {}
            m.setdefault("restarts", []).append(now)
            m["state"] = "running"
            m["env"] = _meta.capture_fast()
            if self.group is not None:
                m["group"] = self.group
            else:
                self.group = m.get("group")
            self._meta = m
        else:
            self._meta = {
                "id": self.id,
                "name": self.name,
                "project": self.project,
                "group": self.group,
                "created_at": now,
                "created_at_iso": datetime.datetime.now().isoformat(timespec="seconds"),
                "state": "running",
                "restarts": [],
                "env": _meta.capture_fast(),
            }
        atomic_write_json(meta_path, self._meta)

    def _capture_slow_meta(self) -> None:
        try:
            extra = _meta.capture_slow(self.dir, os.getcwd())
        except Exception:
            return
        with self._lock:
            if self._finished:
                return
            self._meta["env"].update(extra)
            atomic_write_json(self.dir / "meta.json", self._meta)

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(HEARTBEAT_INTERVAL):
            try:
                os.utime(self._heartbeat_path)
            except OSError:
                pass

    # -- logging ------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalars at a step (wandb.log-shaped)."""
        if self._finished:
            return
        if step is None:
            step = 0 if self._last_step is None else self._last_step + 1
        self._last_step = int(step)
        record: dict[str, Any] = {"_step": self._last_step, "_ts": time.time()}
        record.update(metrics)
        self._metrics.write(record)

    def log_images(
        self,
        key: str,
        images: Any,
        step: int | None = None,
        caption: str | None = None,
    ) -> None:
        """Log one image or a list of images (PIL / torch / numpy) under `key`."""
        if self._finished:
            return
        if step is None:
            step = self._last_step if self._last_step is not None else 0
        step = int(step)
        if not isinstance(images, (list, tuple)):
            images = [images]

        subdir = _sanitize(key)
        files = []
        for i, img in enumerate(images):
            rel = f"{subdir}/step{step:08d}_{i}.png"
            save_image(img, self.dir / "media" / rel)
            files.append(rel)

        with self._lock:
            if self._media_index is None:
                self._media_index = JsonlWriter(self.dir / "media" / "index.jsonl")
        record: dict[str, Any] = {"_step": step, "_ts": time.time(), "key": key, "files": files}
        if caption:
            record["caption"] = caption
        self._media_index.write(record)

    # -- lifecycle ----------------------------------------------------------

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        atexit.unregister(self.finish)
        self._stop_heartbeat.set()
        if self._system is not None:
            self._system.stop()
        if self._console is not None:
            self._console.stop()
        self._metrics.close()
        if self._media_index is not None:
            self._media_index.close()
        self._meta["state"] = "finished"
        self._meta["finished_at"] = time.time()
        atomic_write_json(self.dir / "meta.json", self._meta)

    @property
    def url(self) -> str:
        return str(self.dir)

    def __repr__(self) -> str:
        return f"Run({self.project}/{self.name} id={self.id} dir={self.dir})"


class NoopRun:
    """Returned by init() on non-zero ranks: absorbs all calls silently."""

    resumed = False
    id = name = project = ""
    group = None
    dir = None

    def log(self, *a, **kw) -> None:
        pass

    def log_images(self, *a, **kw) -> None:
        pass

    def finish(self) -> None:
        pass

    def __repr__(self) -> str:
        return "NoopRun(rank != 0)"
