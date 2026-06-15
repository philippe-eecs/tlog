"""Live terminal dashboard (`tlog watch`) — pure ANSI, made for a tmux pane.

Braille-canvas line charts laid out in a grid, one page per metric namespace
(loss, eval, timing, gpu, ...) plus media and console pages. Watching several
runs overlays them wandb-style with one color per run. Repaints every couple
of seconds from the incremental store tail.
"""

from __future__ import annotations

import math
import os
import select
import shutil
import sys
import time

from . import termimg
from .chart import (
    BOLD,
    ChartSpec,
    DIM,
    RESET,
    render_chart,
)
from .chart import PALETTE as _PALETTE
from .chart import SERIES_COLORS as _SERIES_COLORS
from .chart import SMOOTH_LEVELS as _SMOOTH_LEVELS
from .chart import fg as _fg
from .chart import fmt_step
from .chart import pad as _pad
from .store import (
    JsonlTail,
    MetricsReader,
    RunInfo,
    group_keys,
    read_console,
)

_GROUP_ORDER = ["loss", "eval", "training", "timing", "memory", "gpu", "cpu", "ram"]


def fmt_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


class WatchApp:
    def __init__(
        self,
        runs: RunInfo | list[RunInfo],
        interval: float = 2.0,
        ncols: int | None = None,
        images: str = "auto",
        root: str | None = None,
    ):
        self.runs = [runs] if isinstance(runs, RunInfo) else list(runs)
        self.readers = [MetricsReader(r, include_system=True) for r in self.runs]
        self._media_tails = [
            JsonlTail(r.path / "media" / "index.jsonl") for r in self.runs
        ]
        self.media: list[list[dict]] = [[] for _ in self.runs]
        self.interval = interval
        self.page = 0
        self.smooth_idx = 0
        self.logscale = False
        self.ncols = ncols  # None = auto by pane width; set via --cols or keys 1-9
        self.scroll = 0  # chart rows / media steps scrolled, console history chunks
        self.focus = 0  # which run the console page (and `r` key) targets
        self.media_key_idx = 0
        self.backend = termimg.detect_backend(images)
        self._kitty_sent: set = set()
        self._last_frame: str | None = None
        # root for the comment store; derive from a run dir if not given
        self.root = root or (
            str(self.runs[0].path.parent.parent) if self.runs else "."
        )
        self.hidden: set[int] = set()  # run indices toggled off with `v`
        self._comment_counts: dict[str, int] = {}
        self._old_attrs = None

    @property
    def multi(self) -> bool:
        return len(self.runs) > 1

    def run_color(self, i: int) -> int:
        return _SERIES_COLORS[i % len(_SERIES_COLORS)]

    def _refresh(self) -> None:
        for reader in self.readers:
            reader.refresh()
        for i, tail in enumerate(self._media_tails):
            self.media[i].extend(tail.read_new())
        self.runs = [_reload_info(r) for r in self.runs]
        self._refresh_comments()

    def _refresh_comments(self) -> None:
        """Tally open comments per run so the legend/header can show a badge."""
        try:
            from . import comments

            counts: dict[str, int] = {}
            for c in comments.load(self.root):
                if c.status == "open" and c.target.get("type") == "run":
                    run = c.target.get("run", "")
                    counts[run] = counts.get(run, 0) + 1
            self._comment_counts = counts
        except Exception:
            pass

    def _comment_badge(self, run: RunInfo) -> str:
        n = self._comment_counts.get(f"{run.project}/{run.name}", 0)
        return f" {_fg(209)}✎{n}{RESET}" if n else ""

    # -- pages ---------------------------------------------------------------

    def _pages(self) -> list[str]:
        groups: dict[str, list[str]] = {}
        for reader in self.readers:
            for g, ks in group_keys(reader.keys()).items():
                groups.setdefault(g, [])
                for k in ks:
                    if k not in groups[g]:
                        groups[g].append(k)
        self._groups = groups
        ordered = [g for g in _GROUP_ORDER if g in groups]
        ordered += sorted(g for g in groups if g not in _GROUP_ORDER)
        if self.backend != "off" and any(self.media):
            ordered.append("media")
        return ordered + ["console"]

    # -- frame ---------------------------------------------------------------

    def render(self) -> str:
        cols, rows = shutil.get_terminal_size()
        pages = self._pages()
        self.page %= max(len(pages), 1)
        current = pages[self.page] if pages else "console"

        header = self._header()
        tabs = []
        for i, g in enumerate(pages):
            if i == self.page:
                tabs.append(f"\x1b[7m {g} \x1b[27m")
            else:
                tabs.append(f"{DIM} {g} {RESET}")
        tabline = " " + "".join(tabs)

        top = [header, tabline]
        if self.multi:
            top.append(self._legend())
        top.append("")

        smooth = _SMOOTH_LEVELS[self.smooth_idx]
        extras = " · m media-key" if current == "media" else ""
        extras += " · r/v run focus/hide" if self.multi else ""
        footer = (
            f" {DIM}←/→ pages · ↑/↓ scroll · 1-9 cols ({self.ncols or 'auto'})"
            f"{extras} · s smooth ({smooth:g}) · l log "
            f"({'on' if self.logscale else 'off'}) · c comment · q quit{RESET}"
        )

        body_rows = rows - len(top) - 1
        placements: list[tuple[int, int, str]] = []
        if current == "console":
            body = self._render_console(cols, body_rows)
        elif current == "media":
            body, placements = self._render_media(cols, body_rows)
        else:
            body = self._render_charts(current, cols, body_rows)
        body += [""] * (body_rows - len(body))

        lines = top + body[:body_rows] + [footer]
        frame = "\x1b[H" + "\x1b[K\n".join(lines) + "\x1b[J"

        # kitty: clear placements every frame, even when this page has none,
        # so images don't linger after leaving the media page
        if placements or self.backend == "kitty":
            base = len(top)  # body starts at 1-based row base+1
            seqs = []
            if self.backend == "kitty":
                seqs.append(termimg.kitty_delete_all())
            for line_idx, col, seq in placements:
                if line_idx < body_rows:  # never draw over the footer
                    seqs.append(f"\x1b[{base + 1 + line_idx};{col}H{seq}")
            frame += "".join(seqs)
        return frame

    def _header(self) -> str:
        dots = {"running": _fg(114), "finished": _fg(75), "dead": _fg(203)}
        if not self.multi:
            info, reader = self.runs[0], self.readers[0]
            status = info.status
            age = ""
            if reader.last_ts:
                age = f" · updated {fmt_ago(time.time() - reader.last_ts)} ago"
            slurm = info.meta.get("env", {}).get("slurm", {}).get("SLURM_JOB_ID")
            slurm_s = f" · slurm {slurm}" if slurm else ""
            return (
                f" {dots[status]}●{RESET} {BOLD}{info.project}/{info.name}{RESET}"
                f" {DIM}({info.id}){RESET}{self._comment_badge(info)}"
                f" · step {BOLD}{fmt_step(reader.last_step)}{RESET}"
                f" · {status}{slurm_s}{age}"
            )
        projects = {r.project for r in self.runs}
        title = projects.pop() if len(projects) == 1 else "compare"
        last_ts = max((r.last_ts or 0) for r in self.readers)
        age = f" · updated {fmt_ago(time.time() - last_ts)} ago" if last_ts else ""
        return f" {BOLD}{title}{RESET} · {len(self.runs)} runs{age}"

    def _legend(self) -> str:
        parts = []
        for i, (run, reader) in enumerate(zip(self.runs, self.readers)):
            hidden = i in self.hidden
            dot = (DIM if hidden else _fg(_PALETTE[self.run_color(i)])) + "●" + RESET
            name = f"{DIM}{run.name}{RESET}" if hidden else run.name
            if i == self.focus:
                name = f"\x1b[4m{name}\x1b[24m"  # underline = focused (r key)
            parts.append(
                f"{dot} {name}{self._comment_badge(run)} {DIM}{fmt_step(reader.last_step)}"
                f" {run.status[:3]}{RESET}"
            )
        return " " + "   ".join(parts)

    # -- console page ----------------------------------------------------------

    def _render_console(self, cols: int, rows: int) -> list[str]:
        lines = read_console(self.runs[self.focus], max_lines=2000)
        max_offset = max(0, len(lines) - rows)
        self.scroll = min(self.scroll, (max_offset + 4) // 5)
        offset = min(self.scroll * 5, max_offset)
        end = len(lines) - offset
        body = rows - 1 if offset else rows
        out = [" " + l[: cols - 2] for l in lines[max(0, end - body) : end]]
        if offset:
            out.append(f" {DIM}── ↓ {offset} newer lines ──{RESET}")
        return out

    # -- charts page -------------------------------------------------------------

    def _series_for(self, key: str) -> list[tuple[str, list[int], list[float], int]]:
        series = []
        for i, (run, reader) in enumerate(zip(self.runs, self.readers)):
            if i in self.hidden:  # toggled off with `v`
                continue
            s = reader.series.get(key)
            if s is None or not len(s):
                continue
            steps, values = s.points()
            color = self.run_color(i) if self.multi else None
            series.append((run.name, steps, values, color))
        return series

    def _render_charts(self, group: str, cols: int, rows: int) -> list[str]:
        keys = getattr(self, "_groups", {}).get(group, [])
        if not keys:
            return [f" {DIM}no metrics yet — waiting for {group}/* ...{RESET}"]

        # columns: manual override (clamped so charts stay legible) or auto
        max_cols = max(1, cols // 24)
        if self.ncols:
            ncols = min(self.ncols, max_cols, len(keys))
        else:
            ncols = max(1, min(cols // 45, len(keys)))
        chart_w = cols // ncols - 1

        total_rows = math.ceil(len(keys) / ncols)
        chart_h = max(6, rows // total_rows - 1)
        needs_scroll = total_rows * (chart_h + 1) > rows
        avail = rows - 1 if needs_scroll else rows
        visible_rows = max(1, avail // (chart_h + 1))
        max_scroll = max(0, total_rows - visible_rows)
        self.scroll = min(self.scroll, max_scroll)
        start = self.scroll * ncols
        shown = keys[start : start + visible_rows * ncols]

        smooth = _SMOOTH_LEVELS[self.smooth_idx]
        blocks: list[list[str]] = []
        for ki, key in enumerate(shown):
            series = self._series_for(key)
            if not self.multi:
                # color by absolute position so colors stay stable while scrolling
                color = _SERIES_COLORS[(start + ki) % len(_SERIES_COLORS)]
                series = [(lbl, st, vals, color) for lbl, st, vals, _ in series]
            chart = ChartSpec(key, series)
            blocks.append(render_chart(chart, chart_w, chart_h, smooth, self.logscale))

        out: list[str] = []
        for r0 in range(0, len(blocks), ncols):
            row_blocks = blocks[r0 : r0 + ncols]
            for li in range(chart_h):
                line = " ".join(
                    _pad(b[li] if li < len(b) else "", chart_w) for b in row_blocks
                )
                out.append(line)
            out.append("")
        if needs_scroll:
            arrows = ("↑ " if self.scroll > 0 else "") + (
                "↓ " if self.scroll < max_scroll else ""
            )
            out = out[:avail]
            out.append(
                f" {DIM}charts {start + 1}–{start + len(shown)} of {len(keys)}"
                f" · {arrows}scroll{RESET}"
            )
        return out

    # -- media page ----------------------------------------------------------------

    def _media_keys(self) -> list[str]:
        keys: set[str] = set()
        for recs in self.media:
            keys.update(r.get("key", "media") for r in recs)
        return sorted(keys)

    def _render_media(
        self, cols: int, rows: int
    ) -> tuple[list[str], list[tuple[int, int, str]]]:
        keys = self._media_keys()
        if not keys:
            return (
                [f" {DIM}no media yet — log with tlog.log_images(key, imgs, step=){RESET}"],
                [],
            )
        self.media_key_idx %= len(keys)
        mkey = keys[self.media_key_idx]
        gutter = 11
        n = len(self.runs)
        thumb_w = max(16, min(48, (cols - gutter) // n - 2))

        # latest record per (run, step) for the selected key
        per_run: list[dict[int, dict]] = []
        steps: set[int] = set()
        for recs in self.media:
            by_step: dict[int, dict] = {}
            for rec in recs:
                if rec.get("key", "media") == mkey:
                    by_step[int(rec.get("_step", 0))] = rec
            per_run.append(by_step)
            steps.update(by_step)
        step_list = sorted(steps, reverse=True)

        out = [
            f" {BOLD}{mkey}{RESET}  {DIM}({self.media_key_idx + 1}/{len(keys)} keys"
            f" · m to cycle · ↑/↓ steps){RESET}"
        ]
        if self.multi:
            hdr = " " * gutter
            for i, run in enumerate(self.runs):
                hdr += _pad(
                    _fg(_PALETTE[self.run_color(i)]) + run.name + RESET, thumb_w + 1
                )
            out.append(hdr)

        placements: list[tuple[int, int, str]] = []
        self.scroll = min(self.scroll, max(0, len(step_list) - 1))
        shown_any = False
        for step in step_list[self.scroll :]:
            cells: list[list[str]] = []
            for ri, by_step in enumerate(per_run):
                rec = by_step.get(step)
                cells.append(self._thumb(rec, ri, thumb_w) if rec else [])
            height = max((len(c) for c in cells), default=0)
            if height == 0:
                continue
            if shown_any and len(out) + height + 1 > rows:
                remaining = len(step_list) - self.scroll
                out.append(f" {DIM}↓ more steps — scroll{RESET}")
                break
            row_base = len(out)
            for li in range(height):
                prefix = (
                    f" {DIM}{fmt_step(step):>8}{RESET}  " if li == 0 else " " * gutter
                )
                out.append(prefix + " ".join(_pad(c[li] if li < len(c) else "", thumb_w) for c in cells))
            # protocol backends: place real images over the reserved blank cells
            if self.backend in ("kitty", "iterm2"):
                for ri, by_step in enumerate(per_run):
                    rec = by_step.get(step)
                    if not rec or not rec.get("files"):
                        continue
                    col = gutter + ri * (thumb_w + 1) + 1
                    seq = self._protocol_image(ri, rec["files"][0], thumb_w)
                    if seq:
                        placements.append((row_base, col, seq))
            out.append("")
            shown_any = True
        return out, placements

    def _thumb(self, rec: dict, run_idx: int, thumb_w: int) -> list[str]:
        run = self.runs[run_idx]
        files = rec.get("files", [])[:3]
        if not files:
            return []
        if self.backend in ("kitty", "iterm2"):
            # reserve a blank box; the real image is overlaid by placement
            box_h = max(4, thumb_w // 2)
            lines = [""] * box_h
        else:
            k = len(files)
            each_w = max(12, (thumb_w - (k - 1)) // k)
            blocks = [
                termimg.cached_halfblock(run.path / "media" / f, each_w)
                for f in files
            ]
            height = max(len(b) for b in blocks)
            lines = [
                " ".join(_pad(b[li] if li < len(b) else "", each_w) for b in blocks)
                for li in range(height)
            ]
        caption = rec.get("caption")
        if caption:
            lines.append(f"{DIM}{caption[: thumb_w]}{RESET}")
        return lines

    def _protocol_image(self, run_idx: int, rel: str, thumb_w: int) -> str | None:
        path = self.runs[run_idx].path / "media" / rel
        try:
            data = path.read_bytes()
        except OSError:
            return None
        rows = max(4, thumb_w // 2)
        if self.backend == "kitty":
            img_id = termimg.kitty_id_for(path)
            key = (str(path), img_id)
            if key in self._kitty_sent:
                return termimg.render_kitty(None, img_id, thumb_w, rows)
            self._kitty_sent.add(key)
            return termimg.render_kitty(data, img_id, thumb_w, rows)
        return termimg.render_iterm2(data, thumb_w)

    # -- comments --------------------------------------------------------------

    def _add_comment(self, current: str) -> None:
        """Drop out of the alt-screen, open $EDITOR, append a comment tagged to
        the focused run and current page, then restore the dashboard."""
        from . import comments

        info = self.runs[self.focus]
        target = {"type": "run", "run": f"{info.project}/{info.name}"}
        if current == "media":
            keys = self._media_keys()
            if keys:
                target["key"] = keys[self.media_key_idx % len(keys)]
        elif current not in ("console", "media"):
            target["key"] = current  # the metric-group page

        # leave alt-screen + restore cooked mode for the editor
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        if self._old_attrs is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_attrs)
        try:
            raw = comments.edit_text(
                f"\n# Comment on {info.project}/{info.name}"
                f"{(' · ' + target['key']) if target.get('key') else ''}\n"
                "# Lines starting with '#' are ignored.\n"
            )
            text = "\n".join(
                l for l in raw.splitlines() if not l.lstrip().startswith("#")
            ).strip()
            if text:
                comments.add(self.root, target, text, author="human")
        except Exception:
            pass
        finally:  # re-enter alt-screen + cbreak
            if self._old_attrs is not None:
                import tty

                tty.setcbreak(sys.stdin.fileno())
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            sys.stdout.flush()
            self._last_frame = None  # force a full repaint
            self._refresh_comments()

    # -- input / main loop -----------------------------------------------------

    def _read_key(self, timeout: float) -> str | None:
        # Read raw bytes from the fd: buffered sys.stdin.read() would swallow
        # the tail of escape sequences past the select() check.
        if not sys.stdin.isatty():
            time.sleep(timeout)
            return None
        fd = sys.stdin.fileno()
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return None
        data = os.read(fd, 1)
        if data == b"\x1b":
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                data += os.read(fd, 8)
        return data.decode("utf-8", "replace")

    def run(self) -> None:
        is_tty = sys.stdin.isatty()
        old_attrs = None
        if is_tty:
            import termios
            import tty

            old_attrs = termios.tcgetattr(sys.stdin)
            self._old_attrs = old_attrs
            tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write("\x1b[?1049h\x1b[?25l")  # alt screen, hide cursor
        try:
            last_refresh = 0.0
            while True:
                now = time.monotonic()
                if now - last_refresh >= self.interval or last_refresh == 0:
                    self._refresh()
                    last_refresh = now
                frame = self.render()
                if frame != self._last_frame:  # skip identical repaints (SSH-friendly)
                    sys.stdout.write(frame)
                    sys.stdout.flush()
                    self._last_frame = frame
                key = self._read_key(0.25)
                pages = self._pages()
                current = pages[self.page % len(pages)] if pages else "console"
                on_console = current == "console"
                if key in ("q", "Q", "\x03"):
                    break
                elif key in ("\x1b[C", "\x1bOC", "n", "\t"):  # right arrow (CSI/SS3)
                    self.page += 1
                    self.scroll = 0
                elif key in ("\x1b[D", "\x1bOD", "p"):  # left arrow
                    self.page -= 1
                    self.scroll = 0
                elif key in ("\x1b[B", "\x1bOB", "j"):  # down arrow
                    # charts/media: move down; console: back toward newest
                    self.scroll = self.scroll - 1 if on_console else self.scroll + 1
                    self.scroll = max(0, self.scroll)
                elif key in ("\x1b[A", "\x1bOA", "k"):  # up arrow
                    self.scroll = self.scroll + 1 if on_console else self.scroll - 1
                    self.scroll = max(0, self.scroll)
                elif key == "s":
                    self.smooth_idx = (self.smooth_idx + 1) % len(_SMOOTH_LEVELS)
                elif key == "l":
                    self.logscale = not self.logscale
                elif key == "m" and current == "media":
                    self.media_key_idx += 1
                    self.scroll = 0
                elif key == "r" and self.multi:
                    self.focus = (self.focus + 1) % len(self.runs)
                elif key == "v" and self.multi:  # toggle focused run on/off
                    self.hidden ^= {self.focus}
                elif key == "c":  # leave a comment on the focused run/view
                    self._add_comment(current)
                elif key and key.isdigit():  # 1-9 force columns, 0 = auto
                    self.ncols = int(key) or None
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            if old_attrs is not None:
                import termios

                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)


def _reload_info(info: RunInfo) -> RunInfo:
    from .store import _load_run

    return _load_run(info.path) or info


def watch(
    runs: RunInfo | list[RunInfo],
    interval: float = 2.0,
    ncols: int | None = None,
    images: str = "auto",
    root: str | None = None,
) -> None:
    WatchApp(runs, interval=interval, ncols=ncols, images=images, root=root).run()
