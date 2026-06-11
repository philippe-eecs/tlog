"""Run metadata capture: who/what/where launched this training job.

Fast capture (microseconds, env vars and sys state only) happens inline in
init(). Slow capture (git subprocess, nvidia-smi, scontrol) runs in a
background thread so the training loop never waits on it.
"""

from __future__ import annotations

import getpass
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

_SLURM_VARS = [
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_RESTART_COUNT",
    "SLURM_SUBMIT_DIR",
]


def _run(cmd: list[str], timeout: float = 10.0, cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        if out.returncode == 0:
            return out.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def slurm_env() -> dict:
    return {k: os.environ[k] for k in _SLURM_VARS if k in os.environ}


def framework_versions() -> dict:
    """Versions of relevant libs *already imported* by the training script.
    Never imports anything heavy itself."""
    versions = {}
    for name in ("torch", "jax", "numpy", "transformers"):
        mod = sys.modules.get(name)
        if mod is not None:
            v = getattr(mod, "__version__", None)
            if v:
                versions[name] = str(v)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                versions["cuda"] = torch.version.cuda
        except Exception:
            pass
    return versions


def capture_fast() -> dict:
    return {
        "argv": sys.argv,
        "entrypoint": os.path.abspath(sys.argv[0]) if sys.argv else None,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1") or 1),
        "slurm": slurm_env(),
        "frameworks": framework_versions(),
    }


def capture_git(cwd: str) -> tuple[dict, str | None]:
    """Returns (git metadata dict, diff text or None). Empty dict outside a repo."""
    head = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    if head is None:
        return {}, None
    info: dict = {"commit": head.strip()}
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if branch:
        info["branch"] = branch.strip()
    remote = _run(["git", "remote", "get-url", "origin"], cwd=cwd)
    if remote:
        info["remote"] = remote.strip()
    status = _run(["git", "status", "--porcelain"], cwd=cwd)
    info["dirty"] = bool(status and status.strip())
    diff = None
    if info["dirty"]:
        diff = _run(["git", "diff", "HEAD"], timeout=30.0, cwd=cwd)
    return info, diff


def capture_gpus() -> list[dict]:
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    gpus = []
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0]:
                gpus.append({"name": parts[0], "memory": parts[1] if len(parts) > 1 else None})
    return gpus


def capture_sbatch_script(job_id: str) -> str | None:
    """Fetch the batch script that launched this SLURM job."""
    return _run(["scontrol", "write", "batch_script", job_id, "-"], timeout=15.0)


def capture_slow(run_dir: Path, cwd: str) -> dict:
    """Subprocess-based captures. Writes launch.sh / diff.patch side files into
    `run_dir` and returns a dict to merge into meta.json."""
    extra: dict = {}

    git, diff = capture_git(cwd)
    if git:
        extra["git"] = git
    if diff:
        try:
            (run_dir / "diff.patch").write_text(diff, encoding="utf-8")
        except OSError:
            pass

    gpus = capture_gpus()
    if gpus:
        extra["gpus"] = gpus

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        script = capture_sbatch_script(job_id)
        if script:
            try:
                (run_dir / "launch.sh").write_text(script, encoding="utf-8")
            except OSError:
                pass

    extra["meta_completed_at"] = time.time()
    return extra
