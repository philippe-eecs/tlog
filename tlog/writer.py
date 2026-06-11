"""Append-only JSONL writing and atomic JSON file updates.

The write path must never corrupt history: metrics files are append-only
(one JSON object per line) and small state files (meta.json, config.json)
are replaced atomically via write-to-temp + rename.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    """Best-effort serialization for values coming out of training code
    (numpy scalars, torch tensors, Paths, ...) without importing those libs."""
    if isinstance(obj, Path):
        return str(obj)
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return repr(obj)


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=_json_default)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON to `path` atomically (temp file + rename)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class JsonlWriter:
    """Thread-safe append-only JSONL writer.

    Lines are written whole and flushed immediately so a crash can lose at
    most the line being written, never corrupt earlier history. An optional
    fsync interval bounds data loss on hard node failures without paying
    fsync cost on every training-loop log call.
    """

    def __init__(self, path: Path, fsync_interval: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._fsync_interval = fsync_interval
        self._last_fsync = time.monotonic()

    def write(self, obj: dict) -> None:
        line = dumps(obj) + "\n"
        with self._lock:
            if self._f.closed:
                return
            self._f.write(line)
            self._f.flush()
            now = time.monotonic()
            if now - self._last_fsync >= self._fsync_interval:
                try:
                    os.fsync(self._f.fileno())
                except OSError:
                    pass
                self._last_fsync = now

    def close(self) -> None:
        with self._lock:
            if self._f.closed:
                return
            self._f.flush()
            try:
                os.fsync(self._f.fileno())
            except OSError:
                pass
            self._f.close()
