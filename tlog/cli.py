"""`tlog` command line: watch (default), ls, tail, export, serve, rm."""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

from . import store

DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
_STATUS_COLOR = {"running": "\x1b[38;5;114m", "finished": "\x1b[38;5;75m", "dead": "\x1b[38;5;203m"}


def default_root() -> str:
    return os.environ.get("TLOG_DIR", "./runs")


def _resolve_or_die(spec: str | None, root: str) -> store.RunInfo:
    if spec is None:
        info = store.latest_run(root)
        if info is None:
            sys.exit(f"tlog: no runs found under {root!r} (set --dir or TLOG_DIR)")
        return info
    info = store.resolve_run(spec, root)
    if info is None:
        sys.exit(f"tlog: no run matching {spec!r} under {root!r}")
    return info


def cmd_ls(args: argparse.Namespace) -> None:
    runs = store.find_runs(args.root or default_root())
    if not runs:
        print(f"no runs under {args.root or default_root()!r}")
        return
    color = sys.stdout.isatty()
    rows = [("", "PROJECT/NAME", "ID", "STEP", "LAST LOSS", "STARTED", "SLURM", "STATUS")]
    for r in runs:
        last = store.last_record(r.path / "metrics.jsonl") or {}
        step = last.get("_step")
        loss_rec = store.last_record(
            r.path / "metrics.jsonl",
            predicate=lambda rec: any(k.startswith("loss") for k in rec),
        ) or {}
        loss = next(
            (v for k, v in loss_rec.items() if k.startswith("loss")),
            next((v for k, v in last.items() if not k.startswith("_")), None),
        )
        started = datetime.datetime.fromtimestamp(r.created_at).strftime("%m-%d %H:%M")
        slurm = r.meta.get("env", {}).get("slurm", {}).get("SLURM_JOB_ID", "")
        status = r.status
        dot = "●"
        if color:
            dot = _STATUS_COLOR.get(status, "") + "●" + RESET
        rows.append(
            (
                dot,
                f"{r.project}/{r.name}",
                r.id,
                f"{step:,}" if step is not None else "-",
                f"{loss:.4g}" if isinstance(loss, (int, float)) else "-",
                started,
                slurm,
                status,
            )
        )
    plain = [tuple(c if i or not color else "●" for i, c in enumerate(row)) for row in rows]
    widths = [max(len(str(r[i])) for r in plain) for i in range(len(rows[0]))]
    for row, p in zip(rows, plain):
        line = "  ".join(
            str(c) + " " * (widths[i] - len(str(p[i]))) for i, c in enumerate(row)
        )
        print(line.rstrip())


def cmd_watch(args: argparse.Namespace) -> None:
    from .tui import watch

    root = args.dir or default_root()
    info = _resolve_or_die(args.run, root)
    watch(info, interval=args.interval, ncols=args.cols)


def cmd_tail(args: argparse.Namespace) -> None:
    info = _resolve_or_die(args.run, args.dir or default_root())
    for line in store.read_console(info, max_lines=args.lines):
        print(line)


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export_html

    root = args.dir or default_root()
    runs = [_resolve_or_die(spec, root) for spec in args.runs] or None
    if runs is None:
        runs = store.find_runs(root)
        if not runs:
            sys.exit(f"tlog: no runs under {root!r}")
    out = export_html(runs, Path(args.output), max_image_px=args.max_image_px)
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:,.0f} KB, {len(runs)} run{'s' * (len(runs) != 1)})")


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(args.root or default_root(), host=args.host, port=args.port)


def cmd_rm(args: argparse.Namespace) -> None:
    info = _resolve_or_die(args.run, args.dir or default_root())
    if not args.yes:
        answer = input(f"delete {info.path}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    shutil.rmtree(info.path)
    print(f"deleted {info.path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tlog",
        description="lightweight local experiment logger — view training runs in "
        "the terminal, a browser, or a self-contained HTML file",
    )
    sub = parser.add_subparsers(dest="command")

    p_watch = sub.add_parser("watch", help="live terminal dashboard (default command)")
    p_watch.add_argument("run", nargs="?", help="run dir, id, or name (default: latest run)")
    p_watch.add_argument("--dir", help="runs root (default: $TLOG_DIR or ./runs)")
    p_watch.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    p_watch.add_argument(
        "--cols", type=int, default=None,
        help="chart columns (default: auto from pane width; keys 1-9/0 at runtime)",
    )
    p_watch.set_defaults(func=cmd_watch)

    p_ls = sub.add_parser("ls", help="list runs")
    p_ls.add_argument("root", nargs="?", help="runs root (default: $TLOG_DIR or ./runs)")
    p_ls.set_defaults(func=cmd_ls)

    p_tail = sub.add_parser("tail", help="show a run's captured console log")
    p_tail.add_argument("run", nargs="?", help="run dir, id, or name (default: latest)")
    p_tail.add_argument("-n", "--lines", type=int, default=50)
    p_tail.add_argument("--dir", help="runs root")
    p_tail.set_defaults(func=cmd_tail)

    p_export = sub.add_parser("export", help="write a self-contained HTML report")
    p_export.add_argument("runs", nargs="*", help="runs to include (default: all)")
    p_export.add_argument("-o", "--output", default="tlog_report.html")
    p_export.add_argument("--dir", help="runs root")
    p_export.add_argument(
        "--max-image-px", type=int, default=512,
        help="downscale embedded images to this max side (0 = keep original)",
    )
    p_export.set_defaults(func=cmd_export)

    p_serve = sub.add_parser("serve", help="live web dashboard (port-forward friendly)")
    p_serve.add_argument("root", nargs="?", help="runs root (default: $TLOG_DIR or ./runs)")
    p_serve.add_argument("-p", "--port", type=int, default=8585)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.set_defaults(func=cmd_serve)

    p_rm = sub.add_parser("rm", help="delete a run directory")
    p_rm.add_argument("run", help="run dir, id, or name")
    p_rm.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    p_rm.add_argument("--dir", help="runs root")
    p_rm.set_defaults(func=cmd_rm)

    args = parser.parse_args(argv)
    if args.command is None:  # bare `tlog` -> watch latest
        args = parser.parse_args(["watch"] + (argv or sys.argv[1:]))
    args.func(args)


if __name__ == "__main__":
    main()
