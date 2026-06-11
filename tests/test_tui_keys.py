"""Regression test: arrow keys must page the TUI (escape sequences arrive as
multi-byte bursts; a buffered stdin reader once swallowed them — see tui.py
_read_key). Drives a real `tlog watch` through a pty."""

import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")

HIGHLIGHT = b"\x1b[7m %s \x1b[27m"


def make_run(tmp_path):
    from tlog.run import Run

    run = Run(project="p", name="r", dir=tmp_path,
              capture_console=False, system_metrics=False)
    for s in range(0, 100, 10):
        run.log({"loss/total": 1.0 / (s + 1), "eval/fid": 50.0 - s}, step=s)
    run.finish()
    return run.dir


def drain(master, seconds):
    out = b""
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                out += os.read(master, 65536)
            except OSError:
                break
    return out


def test_arrow_keys_page_the_tui(tmp_path):
    run_dir = make_run(tmp_path)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    p = subprocess.Popen(
        [sys.executable, "-m", "tlog.cli", "watch", str(run_dir)],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True,
    )
    os.close(slave)
    try:
        first = drain(master, 1.5)
        assert HIGHLIGHT % b"loss" in first

        os.write(master, b"\x1b[C")  # right arrow, CSI form
        assert HIGHLIGHT % b"eval" in drain(master, 1.0)

        os.write(master, b"\x1bOD")  # left arrow, SS3 (application cursor) form
        assert HIGHLIGHT % b"loss" in drain(master, 1.0)

        os.write(master, b"q")
        drain(master, 1.0)
        assert p.wait(timeout=5) == 0
    finally:
        if p.poll() is None:
            p.kill()
        os.close(master)
