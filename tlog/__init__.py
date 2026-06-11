"""tlog — lightweight, local-first experiment logger for neural net training.

Drop-in wandb-shaped API:

    import tlog

    run = tlog.init(project="vitok", name="vae-L", config=vars(args))
    tlog.log({"loss/total": 0.41, "training/lr": 3e-4}, step=step)
    tlog.log_images("eval/recon", [orig, recon], step=step)
    tlog.finish()

View runs with `tlog watch` (terminal), `tlog serve` (browser via port
forward), or `tlog export -o report.html` (single shareable file).
"""

from __future__ import annotations

import os
from typing import Any

from .run import NoopRun, Run

__version__ = "0.1.0"
__all__ = ["init", "log", "log_images", "finish", "run", "Run", "NoopRun"]

run: Run | NoopRun | None = None  # the active run, set by init()


def init(
    project: str = "default",
    name: str | None = None,
    config: dict | None = None,
    dir: str | None = None,
    id: str | None = None,
    resume: str = "auto",
    capture_console: bool = True,
    system_metrics: bool = True,
    rank_zero_only: bool = True,
) -> Run | NoopRun:
    """Start (or resume) a run. On non-zero ranks (per the RANK env var set by
    torchrun/SLURM) returns a no-op run unless rank_zero_only=False.

    resume: "auto"  — resume iff an explicit `id` is given or this process is a
                      SLURM requeue (SLURM_RESTART_COUNT > 0) of a job that
                      already created a run; otherwise start fresh.
            "must"  — resume an existing run or raise.
            "never" — always start a fresh run.
    """
    global run
    if rank_zero_only and int(os.environ.get("RANK", "0") or 0) != 0:
        run = NoopRun()
        return run
    if run is not None and not isinstance(run, NoopRun):
        run.finish()
    run = Run(
        project=project,
        name=name,
        config=config,
        dir=dir,
        id=id,
        resume=resume,
        capture_console=capture_console,
        system_metrics=system_metrics,
    )
    print(f"tlog: logging to {run.dir}" + (" (resumed)" if run.resumed else ""))
    return run


def _require_run() -> Run | NoopRun:
    if run is None:
        raise RuntimeError("tlog.init() must be called before logging")
    return run


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dict of scalar metrics at a training step."""
    _require_run().log(metrics, step=step)


def log_images(key: str, images: Any, step: int | None = None, caption: str | None = None) -> None:
    """Log one image or a list of images (PIL / torch tensor / numpy array)."""
    _require_run().log_images(key, images, step=step, caption=caption)


def finish() -> None:
    """Mark the active run finished and flush all files."""
    global run
    if run is not None:
        run.finish()
