"""Render rich content to the *scrollback* (not the alt-screen) so an agent can
`tlog show` something and have it persist in the session for the human to read.

Covers images (full-res via kitty/iTerm2, half-block fallback), markdown with
inline `tlog chart/table/images` blocks (same spec as `tlog report`, terminal
target), plain run summaries, and — via `video.py` — short clips.

Everything writes to a stream (default stdout). Image escapes and half-block
lines are both plain text, so output is fully capturable in tests.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import termimg
from .chart import (
    BOLD,
    DIM,
    RESET,
    ChartSpec,
    fg,
    fmt_step,
    pad,
    render_chart,
    run_color,
    visible_len,
)
from .chart import PALETTE as _PALETTE
from .chart import SERIES_COLORS as _SERIES_COLORS
from .store import RunInfo, find_runs, read_media_index

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


def term_width() -> int:
    return shutil.get_terminal_size((100, 40)).columns


# -- images --------------------------------------------------------------------


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def emit_image(
    path: Path,
    out=sys.stdout,
    backend: str | None = None,
    width_cells: int | None = None,
) -> None:
    """Draw one image inline at the cursor, flowing in the scrollback."""
    path = Path(path)
    backend = backend or termimg.detect_backend()
    width_cells = width_cells or min(term_width() - 2, 100)
    if backend == "off":
        out.write(f"{DIM}[image: {path.name}]{RESET}\n")
        return
    try:
        data = path.read_bytes()
    except OSError as e:
        out.write(f"{DIM}(can't read {path.name}: {e}){RESET}\n")
        return

    if backend == "kitty":
        try:
            w, h = _image_size(path)
            rows = max(1, round(width_cells * (h / w) * 0.5))
        except Exception:
            rows = max(1, width_cells // 2)
        img_id = termimg.kitty_id_for(path)
        # reserve `rows` lines (scrolls if needed), step back up, draw, step below
        out.write("\n" * rows + f"\x1b[{rows}A")
        out.write(termimg.render_kitty(data, img_id, width_cells, rows))
        out.write(f"\x1b[{rows}B\r")
    elif backend == "iterm2":
        out.write(termimg.render_iterm2(data, width_cells) + "\n")
    else:  # halfblock
        for line in termimg.render_halfblock(data, width_cells):
            out.write(line + "\n")


def emit_images_dir(path: Path, out=sys.stdout, backend: str | None = None) -> None:
    path = Path(path)
    imgs = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        out.write(f"{DIM}(no images in {path}){RESET}\n")
        return
    for p in imgs:
        out.write(f"\n{BOLD}{p.name}{RESET}\n")
        emit_image(p, out=out, backend=backend)


# -- markdown (terminal ANSI subset) -------------------------------------------

import re

_INLINE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"\x1b[38;5;180m{m.group(1)}{RESET}"),
    (re.compile(r"\*\*([^*]+)\*\*"), lambda m: f"{BOLD}{m.group(1)}\x1b[22m"),
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), lambda m: f"\x1b[3m{m.group(1)}\x1b[23m"),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"),
     lambda m: f"\x1b[4m{m.group(1)}\x1b[24m {DIM}({m.group(2)}){RESET}"),
]


def _inline(text: str) -> str:
    for pat, repl in _INLINE:
        text = pat.sub(repl, text)
    return text


def md_ansi(text: str, width: int | None = None) -> list[str]:
    """Render a markdown subset (headings, lists, bold/italic/code/links, code
    fences, rules) to ANSI lines. Soft-wrapping is left to the terminal."""
    width = width or term_width()
    out: list[str] = []
    para: list[str] = []
    in_code = False

    def flush():
        if para:
            out.append(_inline(" ".join(para)))
            para.clear()

    for line in text.splitlines():
        if line.startswith("```"):
            flush()
            out.append("")
            in_code = not in_code
            continue
        if in_code:
            out.append(f"{DIM}  │ {line}{RESET}")
            continue
        s = line.strip()
        h = re.match(r"^(#{1,4})\s+(.*)", s)
        if h:
            flush()
            out.append("")
            n, txt = len(h.group(1)), _inline(h.group(2))
            if n == 1:
                out.append(f"{BOLD}\x1b[4m{txt}{RESET}")
            elif n == 2:
                out.append(f"{BOLD}{fg(75)}{txt}{RESET}")
            else:
                out.append(f"{BOLD}{txt}{RESET}")
        elif re.match(r"^(-{3,}|\*{3,})$", s):
            flush()
            out.append(f"{DIM}{'─' * min(width, 60)}{RESET}")
        elif re.match(r"^[-*]\s+", s):
            flush()
            out.append(f"  {fg(75)}•{RESET} " + _inline(re.sub(r"^[-*]\s+", "", s)))
        elif re.match(r"^\d+\.\s+", s):
            flush()
            out.append("  " + _inline(s))
        elif not s:
            flush()
            out.append("")
        else:
            para.append(s)
    flush()
    return out


# -- tlog blocks to the terminal -----------------------------------------------


def ansi_table(columns: list[str], rows: list[tuple]) -> list[str]:
    """rows: (color_idx, name, status, step, [cell strings]). The first column
    is the colored run dot, kept aligned via visible-length-aware padding."""
    header = ["", "run", "status", "step"] + columns
    disp = [header]
    for ci, name, status, step, cells in rows:
        dot = fg(_PALETTE[run_color(ci)]) + "●" + RESET
        disp.append([dot, name, status,
                     fmt_step(step) if isinstance(step, int) else "—", *cells])
    widths = [max(visible_len(str(r[i])) for r in disp) for i in range(len(header))]
    out = []
    for ri, row in enumerate(disp):
        cells_s = "  ".join(
            (DIM + pad(str(c), widths[i]) + RESET) if ri == 0 else pad(str(c), widths[i])
            for i, c in enumerate(row)
        )
        out.append("  " + cells_s)
    return out


def emit_markdown(
    path: Path,
    runs: list[RunInfo],
    root: str,
    out=sys.stdout,
    backend: str | None = None,
    width: int | None = None,
    max_image_px: int = 0,
) -> None:
    """Render a markdown spec (prose + ```tlog blocks) to the terminal."""
    from .report import _Ctx, chart_series, images_data, parse_spec, table_data, _as_bool

    path = Path(path)
    backend = backend or termimg.detect_backend()
    width = width or term_width()
    ctx = _Ctx(runs, root)
    text = path.read_text(encoding="utf-8")

    def line(s=""):
        out.write(s + "\n")

    for el in parse_spec(text):
        if el[0] == "md":
            for ln in md_ansi(el[1], width):
                line(ln)
            continue
        _, btype, params = el
        if btype == "chart":
            got = chart_series(ctx, params)
            if isinstance(got, str):
                line(f"{DIM}⚠ {got}{RESET}")
                continue
            key, series = got
            spec = ChartSpec(key, [
                (name, steps, values, _SERIES_COLORS[ci % len(_SERIES_COLORS)])
                for name, ci, steps, values in series
            ])
            line()
            for ln in render_chart(
                spec, min(width, 100), 14,
                float(params.get("smooth", 0) or 0), _as_bool(params.get("logy", "")),
            ):
                line(ln)
            line()
        elif btype == "table":
            got = table_data(ctx, params)
            if isinstance(got, str):
                line(f"{DIM}⚠ {got}{RESET}")
                continue
            cols, rows = got
            line()
            for ln in ansi_table(cols, rows):
                line(ln)
            line()
        elif btype == "images":
            got = images_data(ctx, params)
            if isinstance(got, str):
                line(f"{DIM}⚠ {got}{RESET}")
                continue
            key, img_runs, wanted, per_run = got
            for step in wanted:
                for info, by_step in zip(img_runs, per_run):
                    rec = by_step.get(step)
                    if not rec:
                        continue
                    cap = rec.get("caption", "")
                    label = f"{fg(_PALETTE[run_color(ctx.color(info))])}●{RESET} "
                    line(f"\n{label}{BOLD}{info.name}{RESET} {DIM}{key} @ step {step:,}"
                         f"{('  · ' + cap) if cap else ''}{RESET}")
                    for rel in rec.get("files", []):
                        emit_image(info.path / "media" / rel, out=out, backend=backend)
        else:
            line(f"{DIM}⚠ unknown tlog block: {btype or '(none)'}{RESET}")


# -- run summary ----------------------------------------------------------------


def emit_run(
    info: RunInfo,
    out=sys.stdout,
    backend: str | None = None,
    width: int | None = None,
) -> None:
    """A compact summary of a single run: header, latest charts, recent images."""
    from .store import MetricsReader, group_keys

    backend = backend or termimg.detect_backend()
    width = width or term_width()
    reader = MetricsReader(info, include_system=True)
    reader.refresh()

    dot = {"running": fg(114), "finished": fg(75), "dead": fg(203)}.get(info.status, "")
    out.write(
        f"{dot}●{RESET} {BOLD}{info.project}/{info.name}{RESET} {DIM}({info.id})"
        f" · step {fmt_step(reader.last_step)} · {info.status}{RESET}\n"
    )
    groups = group_keys(reader.keys())
    ncols = max(1, min(width // 45, 2))
    chart_w = width // ncols - 1
    keys = [k for g in groups.values() for k in g]
    blocks = []
    for k in keys:
        sr = reader.series.get(k)
        if not sr or not len(sr):
            continue
        steps, values = sr.points()
        spec = ChartSpec(k, [(info.name, steps, values, _SERIES_COLORS[0])])
        blocks.append(render_chart(spec, chart_w, 12, 0.0, False))
    for r0 in range(0, len(blocks), ncols):
        row = blocks[r0 : r0 + ncols]
        for li in range(max(len(b) for b in row)):
            out.write(" ".join(pad(b[li] if li < len(b) else "", chart_w) for b in row) + "\n")
        out.write("\n")

    media = read_media_index(info)
    if media:
        last = media[-1]
        out.write(f"{DIM}latest media: {last.get('key', 'media')} @ step "
                  f"{last.get('_step', 0):,}{RESET}\n")
        for rel in last.get("files", [])[:4]:
            emit_image(info.path / "media" / rel, out=out, backend=backend)


# -- top-level dispatch ---------------------------------------------------------


def show(target: str, root: str = "runs", out=sys.stdout, backend: str | None = None) -> bool:
    """`tlog show <target>` — render a path by type. Returns True if handled."""
    from .store import resolve_run

    p = Path(target).expanduser()
    if p.is_file():
        suf = p.suffix.lower()
        if suf in VIDEO_EXTS and suf != ".gif":
            from . import video

            video.play(p, out=out, backend=backend)
        elif suf in IMAGE_EXTS:
            emit_image(p, out=out, backend=backend)
        elif suf in (".md", ".markdown"):
            runs = find_runs(root)
            emit_markdown(p, runs, root, out=out, backend=backend)
        elif suf == ".jsonl":
            info = resolve_run(str(p.parent), root)
            if info:
                emit_run(info, out=out, backend=backend)
            else:
                out.write(f"{DIM}(not a run dir: {p.parent}){RESET}\n")
        else:
            out.write(p.read_text(encoding="utf-8", errors="replace"))
        return True
    if p.is_dir():
        info = resolve_run(str(p), root)
        if info:
            emit_run(info, out=out, backend=backend)
        else:
            emit_images_dir(p, out=out, backend=backend)
        return True
    info = resolve_run(target, root)
    if info:
        emit_run(info, out=out, backend=backend)
        return True
    return False
