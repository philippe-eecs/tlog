"""Read side shared by every viewer (TUI, HTML export, web server).

Handles run discovery, incremental tailing of append-only JSONL files,
keep-last-per-(key, step) dedup (which makes preemption/checkpoint-rewind
resumes render correctly), bucket downsampling that preserves spikes, and
EMA smoothing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

HEARTBEAT_STALE_AFTER = 60.0  # seconds without heartbeat -> presumed dead


# -- run discovery -----------------------------------------------------------


@dataclass
class RunInfo:
    path: Path
    meta: dict
    config: dict

    @property
    def id(self) -> str:
        return self.meta.get("id", self.path.name.rsplit("__", 1)[-1])

    @property
    def name(self) -> str:
        return self.meta.get("name", self.path.name.split("__", 1)[0])

    @property
    def project(self) -> str:
        return self.meta.get("project", self.path.parent.name)

    @property
    def created_at(self) -> float:
        return self.meta.get("created_at", 0.0)

    @property
    def status(self) -> str:
        """'running' | 'finished' | 'dead' (no clean finish, heartbeat stale)."""
        if self.meta.get("state") == "finished":
            return "finished"
        hb = self.path / "heartbeat"
        try:
            if time.time() - hb.stat().st_mtime < HEARTBEAT_STALE_AFTER:
                return "running"
        except OSError:
            pass
        return "dead"

    @property
    def label(self) -> str:
        return f"{self.name}__{self.id}"

    @property
    def group(self) -> str | None:
        return self.meta.get("group")


def _is_run_dir(p: Path) -> bool:
    return (p / "meta.json").is_file()


def _load_run(p: Path) -> RunInfo | None:
    try:
        meta = json.loads((p / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    config = {}
    try:
        config = json.loads((p / "config.json").read_text())
    except (OSError, ValueError):
        pass
    return RunInfo(path=p, meta=meta, config=config)


def find_runs(root: str | Path) -> list[RunInfo]:
    """Discover runs under `root`, which may be a run dir, a project dir, or a
    root containing project dirs. Newest first."""
    root = Path(root).expanduser()
    runs: list[RunInfo] = []
    if _is_run_dir(root):
        info = _load_run(root)
        return [info] if info else []
    if not root.is_dir():
        return []
    for child in sorted(root.iterdir()):
        if _is_run_dir(child):
            if info := _load_run(child):
                runs.append(info)
        elif child.is_dir() and not child.name.startswith("."):
            for grandchild in sorted(child.iterdir()):
                if _is_run_dir(grandchild):
                    if info := _load_run(grandchild):
                        runs.append(info)
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs


def resolve_run(spec: str, root: str | Path = ".") -> RunInfo | None:
    """Resolve a CLI run spec: a path, a run id, or a (partial) run name.
    Falls back to searching under `root` (and ./runs)."""
    p = Path(spec).expanduser()
    if _is_run_dir(p):
        return _load_run(p)
    candidates = []
    for base in (root, Path(root) / "runs", "runs"):
        candidates = find_runs(base)
        if candidates:
            break
    for r in candidates:
        if r.id == spec or r.name == spec or r.path.name == spec:
            return r
    for r in candidates:
        if spec in r.name or spec in r.path.name:
            return r
    return None


def latest_run(root: str | Path) -> RunInfo | None:
    """Most recently *active* run (running first, then newest)."""
    runs = find_runs(root)
    if not runs:
        return None
    running = [r for r in runs if r.status == "running"]
    return running[0] if running else runs[0]


# -- groups + saved sets ------------------------------------------------------


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", name).strip("-") or "set"


def runs_in_group(root: str | Path, group: str) -> list[RunInfo]:
    """Every run tagged `group=` at init() with this name (newest first)."""
    return [r for r in find_runs(root) if r.group == group]


def _sets_dir(root: str | Path) -> Path:
    return Path(root).expanduser() / ".tlog" / "sets"


def set_path(root: str | Path, name: str) -> Path:
    return _sets_dir(root) / f"{_safe(name)}.json"


def list_sets(root: str | Path) -> list[dict]:
    out = []
    d = _sets_dir(root)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({"name": p.stem, "runs": data.get("runs", []),
                    "note": data.get("note", "")})
    return out


def load_set(root: str | Path, name: str) -> list[RunInfo]:
    p = set_path(root, name)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    runs = []
    for ref in data.get("runs", []):
        info = _load_run(Path(ref).expanduser())
        if info:
            runs.append(info)
    return runs


def save_set(
    root: str | Path, name: str, runs: list[RunInfo], note: str = ""
) -> Path:
    p = set_path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"name": name, "note": note, "runs": [str(r.path) for r in runs]}, indent=2
    ), encoding="utf-8")
    return p


def link_runs(
    root: str | Path, name: str, specs: list[str], note: str = ""
) -> tuple[int, int]:
    """Create or append to a saved set. Returns (added, total)."""
    existing = load_set(root, name)
    have = {str(r.path) for r in existing}
    added = 0
    for spec in specs:
        for info in resolve_runs(spec, root):
            if str(info.path) not in have:
                existing.append(info)
                have.add(str(info.path))
                added += 1
    save_set(root, name, existing, note)
    return added, len(existing)


def resolve_runs(spec: str, root: str | Path = ".") -> list[RunInfo]:
    """Resolve a token to one or more runs: a project dir, a saved set, a
    group tag, or a single run (path / id / name). Empty list if nothing
    matches. This is what `tlog <token>` and multi-run viewers use."""
    for base in (Path(spec).expanduser(), Path(root) / spec):
        if base.is_dir() and not _is_run_dir(base):
            rs = find_runs(base)
            if rs:
                return rs
    if set_path(root, spec).is_file():
        runs = load_set(root, spec)
        if runs:
            return runs
    grouped = runs_in_group(root, spec)
    if grouped:
        return grouped
    info = resolve_run(spec, root)
    return [info] if info else []


# -- incremental metrics reading ----------------------------------------------


class Series:
    """A single metric's history with keep-last-per-step dedup."""

    __slots__ = ("_by_step", "_sorted", "_dirty")

    def __init__(self):
        self._by_step: dict[int, float] = {}
        self._sorted: tuple[list[int], list[float]] = ([], [])
        self._dirty = False

    def add(self, step: int, value: float) -> None:
        self._by_step[step] = value
        self._dirty = True

    def points(self) -> tuple[list[int], list[float]]:
        """(steps, values) sorted by step, deduped keep-last."""
        if self._dirty:
            items = sorted(self._by_step.items())
            self._sorted = ([s for s, _ in items], [v for _, v in items])
            self._dirty = False
        return self._sorted

    @property
    def last(self) -> tuple[int, float] | None:
        steps, values = self.points()
        return (steps[-1], values[-1]) if steps else None

    def __len__(self) -> int:
        return len(self._by_step)


class JsonlTail:
    """Incrementally read complete lines appended to a JSONL file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset = 0

    def read_new(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:  # file replaced/truncated: re-read
            self._offset = 0
        if size == self._offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        # only consume up to the last complete line
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self._offset += end + 1
        records = []
        for line in chunk[: end + 1].splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        return records


class MetricsReader:
    """Live view over a run's metrics.jsonl (and optionally system.jsonl)."""

    def __init__(self, run: RunInfo, include_system: bool = False):
        self.run = run
        self._tails = [JsonlTail(run.path / "metrics.jsonl")]
        if include_system:
            self._tails.append(JsonlTail(run.path / "system.jsonl"))
        self.series: dict[str, Series] = {}
        self.last_step: int | None = None
        self.last_ts: float | None = None

    def refresh(self) -> bool:
        """Ingest newly appended records. Returns True if anything changed."""
        changed = False
        for tail in self._tails:
            for rec in tail.read_new():
                step = rec.get("_step")
                ts = rec.get("_ts")
                if step is None:
                    continue
                for key, value in rec.items():
                    if key.startswith("_") or not isinstance(value, (int, float)):
                        continue
                    self.series.setdefault(key, Series()).add(int(step), float(value))
                    changed = True
                if self.last_step is None or step >= self.last_step:
                    self.last_step = int(step)
                    if ts is not None:
                        self.last_ts = float(ts)
        return changed

    def keys(self) -> list[str]:
        return sorted(self.series)


def read_media_index(run: RunInfo) -> list[dict]:
    """All media records for a run: [{_step, key, files, caption?}, ...]."""
    return JsonlTail(run.path / "media" / "index.jsonl").read_new()


def last_record(path: Path, chunk: int = 65536, predicate=None) -> dict | None:
    """Cheaply read the last complete JSON record of a JSONL file (optionally
    the last one matching `predicate`, searching within the final `chunk` bytes)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - chunk))
            lines = f.read().splitlines()
    except OSError:
        return None
    fallback = None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if predicate is None or predicate(rec):
            return rec
        if fallback is None:
            fallback = rec
    return fallback


# -- transforms ----------------------------------------------------------------


def downsample(
    steps: list[int], values: list[float], max_points: int
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Bucket (steps, values) down to <= max_points buckets.

    Returns (step, mean, min, max) per bucket so spikes survive downsampling
    (mean is plotted; min/max can be drawn as a band or extent).
    """
    n = len(steps)
    if n <= max_points:
        return steps, values, values, values
    lo, hi = steps[0], steps[-1]
    span = max(hi - lo, 1)
    out_s: list[int] = []
    out_mean: list[float] = []
    out_min: list[float] = []
    out_max: list[float] = []
    i = 0
    for b in range(max_points):
        b_hi = lo + span * (b + 1) / max_points
        j = i
        acc = 0.0
        vmin = float("inf")
        vmax = float("-inf")
        while j < n and (steps[j] <= b_hi or b == max_points - 1):
            v = values[j]
            acc += v
            vmin = min(vmin, v)
            vmax = max(vmax, v)
            j += 1
        if j > i:
            out_s.append(steps[(i + j - 1) // 2])
            out_mean.append(acc / (j - i))
            out_min.append(vmin)
            out_max.append(vmax)
            i = j
        if i >= n:
            break
    return out_s, out_mean, out_min, out_max


def ema(values: list[float], weight: float) -> list[float]:
    """wandb-style debiased exponential moving average; weight in [0, 1)."""
    if weight <= 0 or len(values) < 2:
        return list(values)
    smoothed = []
    last = 0.0
    debias = 0.0
    for v in values:
        last = last * weight + (1 - weight) * v
        debias = debias * weight + (1 - weight)
        smoothed.append(last / debias)
    return smoothed


def group_keys(keys: list[str]) -> dict[str, list[str]]:
    """Group metric keys by namespace prefix ('loss/total' -> group 'loss')."""
    groups: dict[str, list[str]] = {}
    for k in keys:
        prefix = k.split("/", 1)[0] if "/" in k else "metrics"
        groups.setdefault(prefix, []).append(k)
    return groups


# -- console log ---------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def read_console(run: RunInfo, max_lines: int = 200) -> list[str]:
    """Tail of console.log with ANSI stripped and \\r-overwrites resolved
    (tqdm progress bars collapse to their final state)."""
    path = run.path / "console.log"
    try:
        # newline="" so \r survives for overwrite-resolution below
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            raw = f.read()
    except OSError:
        return []
    lines = []
    for line in raw.split("\n"):
        line = line.rstrip("\r")  # CRLF endings
        line = _ANSI_RE.sub("", line.rsplit("\r", 1)[-1])
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines[-max_lines:]
