"""Tee stdout/stderr of the training process into the run directory."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TextIO


class _Tee:
    """File-like wrapper that mirrors writes to the original stream and a log
    file. Exposes enough of the TextIO surface for print/tqdm/logging."""

    def __init__(self, stream: TextIO, logfile: TextIO, lock: threading.Lock):
        self._stream = stream
        self._logfile = logfile
        self._lock = lock

    def write(self, data: str) -> int:
        n = self._stream.write(data)
        with self._lock:
            if not self._logfile.closed:
                try:
                    self._logfile.write(data)
                except (OSError, ValueError):
                    pass
        return n

    def flush(self) -> None:
        self._stream.flush()
        with self._lock:
            if not self._logfile.closed:
                try:
                    self._logfile.flush()
                except (OSError, ValueError):
                    pass

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._stream, name)


class ConsoleCapture:
    def __init__(self, path: Path):
        # line-buffered so `tlog tail`/viewers see output promptly
        self._logfile = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        self._lock = threading.Lock()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._logfile, self._lock)
        sys.stderr = _Tee(self._orig_stderr, self._logfile, self._lock)

    def stop(self) -> None:
        if isinstance(sys.stdout, _Tee):
            sys.stdout = self._orig_stdout
        if isinstance(sys.stderr, _Tee):
            sys.stderr = self._orig_stderr
        with self._lock:
            if not self._logfile.closed:
                self._logfile.flush()
                self._logfile.close()
