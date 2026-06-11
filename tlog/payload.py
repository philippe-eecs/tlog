"""Build the JSON payloads the web frontend consumes (export + serve)."""

from __future__ import annotations

from pathlib import Path

from .store import MetricsReader, RunInfo, downsample, last_record, read_media_index

MAX_POINTS = 1500


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def run_summary(info: RunInfo) -> dict:
    env = info.meta.get("env", {})
    git = env.get("git", {})
    last = last_record(info.path / "metrics.jsonl") or {}
    return {
        "id": info.id,
        "name": info.name,
        "project": info.project,
        "status": info.status,
        "step": last.get("_step"),
        "config": info.config,
        "summary": {
            "created": info.meta.get("created_at_iso", ""),
            "hostname": env.get("hostname", ""),
            "slurm_job": env.get("slurm", {}).get("SLURM_JOB_ID", ""),
            "git_commit": git.get("commit", ""),
            "git_dirty": git.get("dirty", False),
            "restarts": len(info.meta.get("restarts", [])),
            "dir": str(info.path),
        },
        "metrics_mtime": max(
            _mtime(info.path / "metrics.jsonl"), _mtime(info.path / "system.jsonl")
        ),
        "media_mtime": _mtime(info.path / "media" / "index.jsonl"),
    }


def run_metrics(info: RunInfo, max_points: int = MAX_POINTS) -> dict:
    reader = MetricsReader(info, include_system=True)
    reader.refresh()
    return metrics_from_reader(reader, max_points)


def metrics_from_reader(reader: MetricsReader, max_points: int = MAX_POINTS) -> dict:
    out = {}
    for key, series in reader.series.items():
        steps, values = series.points()
        s, mean, _, _ = downsample(steps, values, max_points)
        out[key] = {"steps": s, "values": [round(v, 8) for v in mean]}
    return out


def run_media(info: RunInfo) -> list[dict]:
    records = []
    for rec in read_media_index(info):
        records.append(
            {
                "step": rec.get("_step", 0),
                "key": rec.get("key", "media"),
                "files": rec.get("files", []),
                "caption": rec.get("caption"),
            }
        )
    return records
