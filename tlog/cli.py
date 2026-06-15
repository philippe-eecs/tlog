"""`tlog` command line.

The headline is smart dispatch: `tlog <thing>` figures out what <thing> is —
a run, a project, a group, a saved set, or a file (image / .md / .jsonl /
video) — and does the obvious thing. Bare `tlog` watches the latest run.

Subcommands (rarely needed thanks to dispatch): watch · show · review ·
comment · comments · resolve · link · sets · ls · serve · report · rm.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

from . import comments, render, store, termimg

DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
_STATUS_COLOR = {"running": "\x1b[38;5;114m", "finished": "\x1b[38;5;75m", "dead": "\x1b[38;5;203m"}

_STATIC_EXTS = render.IMAGE_EXTS | render.VIDEO_EXTS | {".md", ".markdown"}


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


def _gather_runs(specs: list[str], root: str) -> list[store.RunInfo]:
    """Resolve specs (run / project / group / set) into a de-duped run list."""
    infos: list[store.RunInfo] = []
    for spec in specs:
        infos.extend(store.resolve_runs(spec, root))
    seen: set = set()
    return [r for r in infos if not (r.path in seen or seen.add(r.path))]


def _tmux_hint(images: str) -> None:
    """One-line nudge if tmux is swallowing graphics we could otherwise use."""
    if images != "auto" or not termimg.in_tmux():
        return
    if termimg.host_backend() in ("halfblock", "off"):
        return
    if not termimg.passthrough_enabled():
        print(
            "tlog: tmux is hiding image escapes (using half-block). For full-res, run "
            "`tmux set -g allow-passthrough on` and `export TLOG_TERM=ghostty`.",
            file=sys.stderr,
        )


# -- watch (default) -----------------------------------------------------------


def cmd_watch(args: argparse.Namespace) -> None:
    from .tui import watch

    root = args.dir or default_root()
    show_targets, run_specs = [], []
    for spec in args.runs:
        p = Path(spec).expanduser()
        suf = p.suffix.lower()
        if p.is_file() and suf in _STATIC_EXTS:
            show_targets.append(spec)  # `tlog plot.png` / `tlog notes.md`
        elif p.is_file() and suf == ".jsonl":
            run_specs.append(str(p.parent))  # `tlog run/metrics.jsonl`
        else:
            run_specs.append(spec)

    if show_targets and not run_specs:
        _tmux_hint(args.images)
        backend = termimg.detect_backend(args.images)
        for t in show_targets:
            render.show(t, root=root, backend=backend)
        return

    infos = _gather_runs(run_specs, root) if run_specs else []
    if not infos:
        infos = [_resolve_or_die(None, root)]
    _tmux_hint(args.images)
    watch(infos, interval=args.interval, ncols=args.cols, images=args.images, root=root)


# -- show / review -------------------------------------------------------------


def cmd_show(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    if args.console:
        info = _resolve_or_die(args.targets[0] if args.targets else None, root)
        for line in store.read_console(info, max_lines=args.lines):
            print(line)
        return
    if not args.targets:
        sys.exit("tlog: show needs a target (image / .md / .jsonl / run / dir)")
    _tmux_hint(args.images)
    backend = termimg.detect_backend(args.images)
    for t in args.targets:
        if not render.show(t, root=root, backend=backend):
            print(f"tlog: nothing to show for {t!r}", file=sys.stderr)


def cmd_review(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    doc = Path(args.doc).expanduser()
    if not doc.is_file():
        sys.exit(f"tlog: no such doc {str(doc)!r}")
    _tmux_hint(args.images)
    backend = termimg.detect_backend(args.images)
    runs = _gather_runs(args.runs, root) if args.runs else store.find_runs(root)
    render.emit_markdown(doc, runs, root, backend=backend)
    if args.no_comment:
        return
    print(f"\n{DIM}opening your editor to leave comments…{RESET}", file=sys.stderr)
    n = comments.comment_on_doc(root, doc)
    where = comments.comments_path(root)
    print(f"tlog: saved {n} comment(s) to {where}. "
          f"Agent reads them with: tlog comments --doc {doc.name} --json")


# -- comments ------------------------------------------------------------------


def cmd_comment(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    target = comments.parse_target(args.target, root)
    text = args.message
    if not text:
        raw = comments.edit_text(
            f"\n\n# Comment on {args.target}\n# Lines starting with '#' are ignored.\n"
        )
        text = "\n".join(
            l for l in raw.splitlines() if not l.lstrip().startswith("#")
        ).strip()
    if not text:
        print("tlog: empty comment, aborted")
        return
    cid = comments.add(root, target, text, author=args.author)
    c = comments.Comment(cid, 0, args.author, "open", text, target)
    print(f"tlog: added comment {cid} on {c.target_str()}")


def cmd_comments(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    cs = comments.load(root)
    status = None if args.status == "all" else args.status
    cs = comments.select(cs, doc=args.doc, run=args.run, status=status)
    if args.json:
        print(comments.to_json(cs))
    else:
        print(comments.format_list(cs, color=sys.stdout.isatty()))


def cmd_resolve(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    ok = comments.resolve(root, args.id, note=args.note or "", by=args.author)
    print(f"tlog: resolved {args.id}" if ok else f"tlog: no open comment {args.id!r}")


# -- sets / link ---------------------------------------------------------------


def cmd_link(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    added, total = store.link_runs(root, args.name, args.runs, note=args.note or "")
    print(f"tlog: set {args.name!r} now has {total} run(s) (+{added}); "
          f"view with: tlog {args.name}")


def cmd_sets(args: argparse.Namespace) -> None:
    root = args.dir or default_root()
    sets = store.list_sets(root)
    if not sets:
        print(f"no saved sets under {root!r} (create with: tlog link <name> <runs…>)")
        return
    for s in sets:
        note = f"  {DIM}{s['note']}{RESET}" if s["note"] else ""
        print(f"{BOLD}{s['name']}{RESET}  {len(s['runs'])} run(s){note}")


# -- ls ------------------------------------------------------------------------


def cmd_ls(args: argparse.Namespace) -> None:
    runs = store.find_runs(args.root or default_root())
    if not runs:
        print(f"no runs under {args.root or default_root()!r}")
        return
    color = sys.stdout.isatty()
    rows = [("", "PROJECT/NAME", "GROUP", "ID", "STEP", "LAST LOSS", "STARTED", "STATUS")]
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
        status = r.status
        dot = "●"
        if color:
            dot = _STATUS_COLOR.get(status, "") + "●" + RESET
        rows.append(
            (
                dot,
                f"{r.project}/{r.name}",
                r.group or "-",
                r.id,
                f"{step:,}" if step is not None else "-",
                f"{loss:.4g}" if isinstance(loss, (int, float)) else "-",
                started,
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


# -- serve / report / rm -------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(args.root or default_root(), host=args.host, port=args.port)


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export_html

    root = args.dir or default_root()
    infos = _gather_runs(args.runs, root) if args.runs else store.find_runs(root)
    if not infos:
        sys.exit(f"tlog: no runs under {root!r}")
    out = export_html(
        infos, Path(args.output or "tlog_report.html"), max_image_px=args.max_image_px
    )
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    print(f"wrote {out} ({out.stat().st_size / 1024:,.0f} KB, {len(infos)} run"
          f"{'s' * (len(infos) != 1)})")


def cmd_report(args: argparse.Namespace) -> None:
    from .report import report_html

    if not Path(args.spec).is_file():
        sys.exit(f"tlog: no spec file {args.spec!r} — `report` needs a markdown file "
                 f"with ```tlog blocks. For a plain HTML dump use: tlog export {args.spec}")
    root = args.dir or default_root()
    infos = _gather_runs(args.runs, root) if args.runs else store.find_runs(root)
    if not infos:
        sys.exit(f"tlog: no runs under {root!r}")
    out = report_html(
        Path(args.spec), infos, root,
        output=Path(args.output) if args.output else None,
        max_image_px=args.max_image_px, open_browser=args.open,
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:,.0f} KB, {len(infos)} run"
          f"{'s' * (len(infos) != 1)})")


def cmd_rm(args: argparse.Namespace) -> None:
    info = _resolve_or_die(args.run, args.dir or default_root())
    if not args.yes:
        answer = input(f"delete {info.path}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    shutil.rmtree(info.path)
    print(f"deleted {info.path}")


# -- parser --------------------------------------------------------------------

_IMG_HELP = ("image renderer: auto (kitty/Ghostty/iTerm2, incl. through tmux "
             "with allow-passthrough), halfblock, kitty, iterm2, off")
_SUBCOMMANDS = {
    "watch", "show", "review", "comment", "comments", "resolve",
    "link", "sets", "ls", "serve", "export", "report", "rm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tlog",
        description="local experiment logger + terminal review buddy. "
        "`tlog <run|project|group|set|image|.md|.jsonl>` just works; "
        "bare `tlog` watches the latest run.",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("watch", help="live terminal dashboard (default command)")
    p.add_argument("runs", nargs="*", help="runs / project / group / set, or a "
                   "static file to show; default: latest run")
    p.add_argument("--dir", help="runs root (default: $TLOG_DIR or ./runs)")
    p.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    p.add_argument("--cols", type=int, default=None, help="chart columns (default auto)")
    p.add_argument("--images", choices=list(termimg.BACKENDS), default="auto", help=_IMG_HELP)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("show", help="render image / markdown / run / dir to the terminal")
    p.add_argument("targets", nargs="*", help="paths or run specs to render")
    p.add_argument("--dir", help="runs root")
    p.add_argument("--images", choices=list(termimg.BACKENDS), default="auto", help=_IMG_HELP)
    p.add_argument("--console", action="store_true", help="print a run's console log")
    p.add_argument("-n", "--lines", type=int, default=200, help="--console: tail length")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("review", help="render a markdown doc, then open $EDITOR to comment")
    p.add_argument("doc", help="markdown file (may embed ```tlog chart/table/images)")
    p.add_argument("runs", nargs="*", help="default run set for blocks (default: all)")
    p.add_argument("--dir", help="runs root")
    p.add_argument("--images", choices=list(termimg.BACKENDS), default="auto", help=_IMG_HELP)
    p.add_argument("--no-comment", action="store_true", help="just render, don't open editor")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("comment", help="add a comment (target = doc[#anchor] | run[@step][:key] | file[:line])")
    p.add_argument("target")
    p.add_argument("-m", "--message", help="comment text (omit to open $EDITOR)")
    p.add_argument("--author", default="human")
    p.add_argument("--dir", help="runs root")
    p.set_defaults(func=cmd_comment)

    p = sub.add_parser("comments", help="list comments (agents: add --json)")
    p.add_argument("--doc", help="filter to a doc")
    p.add_argument("--run", help="filter to a run")
    p.add_argument("--status", choices=["open", "resolved", "all"], default="open")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--dir", help="runs root")
    p.set_defaults(func=cmd_comments)

    p = sub.add_parser("resolve", help="mark a comment resolved")
    p.add_argument("id")
    p.add_argument("-m", "--note", help="resolution note")
    p.add_argument("--author", default="claude")
    p.add_argument("--dir", help="runs root")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("link", help="create/append a saved set of runs to compare")
    p.add_argument("name")
    p.add_argument("runs", nargs="+", help="runs / projects / groups to add")
    p.add_argument("--note", help="set description")
    p.add_argument("--dir", help="runs root")
    p.set_defaults(func=cmd_link)

    p = sub.add_parser("sets", help="list saved sets")
    p.add_argument("--dir", help="runs root")
    p.set_defaults(func=cmd_sets)

    p = sub.add_parser("ls", help="list runs")
    p.add_argument("root", nargs="?", help="runs root (default: $TLOG_DIR or ./runs)")
    p.set_defaults(func=cmd_ls)

    p = sub.add_parser("serve", help="live web dashboard (port-forward friendly)")
    p.add_argument("root", nargs="?", help="runs root")
    p.add_argument("-p", "--port", type=int, default=8585)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("export", help="self-contained interactive HTML of the runs")
    p.add_argument("runs", nargs="*", help="runs / project / group / set (default: all)")
    p.add_argument("-o", "--output", help="output file (default: tlog_report.html)")
    p.add_argument("--dir", help="runs root")
    p.add_argument("--open", action="store_true", help="open in browser")
    p.add_argument("--max-image-px", type=int, default=512,
                   help="downscale embedded images (0 = keep original)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("report", help="render a markdown spec with ```tlog blocks into HTML")
    p.add_argument("spec", help="markdown spec with ```tlog chart/table/images blocks")
    p.add_argument("runs", nargs="*", help="runs / project / group / set (default: all)")
    p.add_argument("-o", "--output", help="output file (default: <spec>.html)")
    p.add_argument("--dir", help="runs root")
    p.add_argument("--open", action="store_true", help="open in browser")
    p.add_argument("--max-image-px", type=int, default=512,
                   help="downscale embedded images (0 = keep original)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("rm", help="delete a run directory")
    p.add_argument("run", help="run dir, id, or name")
    p.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    p.add_argument("--dir", help="runs root")
    p.set_defaults(func=cmd_rm)

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # smart default: anything that isn't a subcommand is a target for `watch`,
    # which itself routes static files (image/.md/.jsonl) to `show`.
    if not argv:
        argv = ["watch"]
    elif argv[0] not in _SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["watch"] + argv
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
