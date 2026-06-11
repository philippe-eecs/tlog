"""Live terminal dashboard (`tlog watch`) — pure ANSI, made for a tmux pane.

Braille-canvas line charts laid out in a grid, one page per metric namespace
(loss, eval, timing, gpu, ...) plus a console page. Repaints every couple of
seconds from the incremental store tail.
"""

from __future__ import annotations

import math
import os
import select
import shutil
import sys
import time

from .store import MetricsReader, RunInfo, downsample, ema, group_keys, read_console

# braille dot bits indexed [y % 4][x % 2]
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

# palette indices -> ANSI 256 color codes
_BAND = 1
_PALETTE = {1: 238, 2: 75, 3: 209, 4: 114, 5: 176, 6: 221, 7: 80, 8: 168}
_SERIES_COLORS = [2, 3, 4, 5, 6, 7, 8]

_SMOOTH_LEVELS = [0.0, 0.6, 0.9, 0.99]

_GROUP_ORDER = ["loss", "eval", "training", "timing", "memory", "gpu", "cpu", "ram"]

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"


def _fg(code: int) -> str:
    return f"\x1b[38;5;{code}m"


def fmt_num(v: float) -> str:
    if v != v:  # nan
        return "nan"
    a = abs(v)
    if a >= 1e6 or (a < 1e-3 and a > 0):
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:,.1f}"
    return f"{v:.4g}"


def fmt_step(s: int | None) -> str:
    if s is None:
        return "-"
    if s >= 1_000_000:
        return f"{s / 1e6:.2f}M"
    if s >= 10_000:
        return f"{s / 1e3:.1f}k"
    return str(s)


def fmt_ago(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


class Canvas:
    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.w, self.h = cols * 2, rows * 4
        self._bits = bytearray(cols * rows)
        self._color = bytearray(cols * rows)

    def set(self, x: int, y: int, color: int) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y // 4) * self.cols + (x // 2)
            self._bits[i] |= _DOTS[y % 4][x % 2]
            if color != _BAND or self._color[i] == 0:
                self._color[i] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.set(x0, y0, color)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def vline(self, x: int, y0: int, y1: int, color: int) -> None:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            self.set(x, y, color)

    def render(self) -> list[str]:
        out = []
        for r in range(self.rows):
            parts = []
            cur = -1
            for c in range(self.cols):
                i = r * self.cols + c
                bits = self._bits[i]
                if not bits:
                    parts.append(" ")
                    continue
                color = self._color[i]
                if color != cur:
                    parts.append(_fg(_PALETTE.get(color, 250)))
                    cur = color
                parts.append(chr(0x2800 + bits))
            parts.append(RESET)
            out.append("".join(parts))
        return out


class ChartSpec:
    """One metric chart: possibly several runs' series of the same key."""

    def __init__(self, key: str, series: list[tuple[str, list[int], list[float], int]]):
        self.key = key
        self.series = series  # (run_label, steps, values, color)


def render_chart(
    chart: ChartSpec,
    width: int,
    height: int,
    smooth: float,
    logscale: bool,
) -> list[str]:
    """Render a chart into `height` lines of `width` visible chars."""
    gutter = 9
    plot_cols = max(8, width - gutter - 1)
    plot_rows = max(2, height - 2)

    prepared = []  # (steps, means, mins, maxs, color)
    vmin, vmax = math.inf, -math.inf
    last_val = None
    for _, steps, values, color in chart.series:
        if logscale:
            pts = [(s, v) for s, v in zip(steps, values) if v > 0]
            steps = [s for s, _ in pts]
            values = [math.log10(v) for _, v in pts]
        if not steps:
            continue
        if last_val is None and chart.series:
            last_val = values[-1]
        s, mean, lo, hi = downsample(steps, values, plot_cols * 2)
        mean = ema(mean, smooth)
        prepared.append((s, mean, lo, hi, color))
        vmin = min(vmin, min(lo))
        vmax = max(vmax, max(hi))

    title_val = ""
    if chart.series and chart.series[0][2]:
        raw_last = chart.series[0][2][-1]
        title_val = fmt_num(raw_last)
    title = f" {BOLD}{chart.key}{RESET}"
    pad = width - len(chart.key) - len(title_val) - 3
    title += " " * max(1, pad) + _fg(_PALETTE[chart.series[0][3]]) + title_val + RESET if chart.series else ""

    if not prepared or not (vmax > -math.inf):
        return [title] + [" " * width] * (height - 1)

    if vmax == vmin:
        vmax += 1e-12 + abs(vmax) * 1e-6
        vmin -= 1e-12 + abs(vmin) * 1e-6

    canvas = Canvas(plot_cols, plot_rows)
    px_h = canvas.h - 1
    px_w = canvas.w - 1
    all_min = min(p[0][0] for p in prepared)
    all_max = max(p[0][-1] for p in prepared)
    span = max(all_max - all_min, 1)

    def sx(step: int) -> int:
        return round((step - all_min) / span * px_w)

    def sy(v: float) -> int:
        return px_h - round((v - vmin) / (vmax - vmin) * px_h)

    single = len(prepared) == 1
    for steps, mean, lo, hi, color in prepared:
        if single and len(steps) < len(chart.series[0][1]):
            for s, l, h in zip(steps, lo, hi):  # min/max band under the line
                canvas.vline(sx(s), sy(h), sy(l), _BAND)
        prev = None
        for s, v in zip(steps, mean):
            pt = (sx(s), sy(v))
            if prev is not None:
                canvas.line(prev[0], prev[1], pt[0], pt[1], color)
            else:
                canvas.set(pt[0], pt[1], color)
            prev = pt

    def ylab(v: float) -> str:
        return fmt_num(10**v if logscale else v)

    lines = [title]
    rendered = canvas.render()
    for r, row in enumerate(rendered):
        if r == 0:
            lab = ylab(vmax)
        elif r == plot_rows - 1:
            lab = ylab(vmin)
        else:
            lab = ""
        lines.append(f"{DIM}{lab:>{gutter - 2}} {'┤' if lab else '│'}{RESET}{row}")
    x_left = fmt_step(int(all_min))
    x_right = fmt_step(int(all_max))
    inner = plot_cols - len(x_left) - len(x_right)
    lines.append(
        " " * (gutter - 1) + DIM + x_left + " " * max(1, inner) + x_right + RESET
    )
    return lines[:height]


def _visible_len(s: str) -> int:
    import re

    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _visible_len(s))


class WatchApp:
    def __init__(self, run: RunInfo, interval: float = 2.0, ncols: int | None = None):
        self.info = run
        self.reader = MetricsReader(run, include_system=True)
        self.interval = interval
        self.page = 0
        self.smooth_idx = 0
        self.logscale = False
        self.ncols = ncols  # None = auto by pane width; set via --cols or keys 1-9
        self.scroll = 0  # chart rows scrolled (charts) / 5-line chunks back (console)

    # -- pages ---------------------------------------------------------------

    def _pages(self) -> list[str]:
        groups = group_keys(self.reader.keys())
        ordered = [g for g in _GROUP_ORDER if g in groups]
        ordered += sorted(g for g in groups if g not in _GROUP_ORDER)
        return ordered + ["console"]

    # -- frame ---------------------------------------------------------------

    def render(self) -> str:
        cols, rows = shutil.get_terminal_size()
        pages = self._pages()
        self.page %= max(len(pages), 1)
        current = pages[self.page] if pages else "console"

        status = self.info.status
        dot = {"running": _fg(114) + "●", "finished": _fg(75) + "●", "dead": _fg(203) + "●"}[status]
        age = ""
        if self.reader.last_ts:
            age = f" · updated {fmt_ago(time.time() - self.reader.last_ts)} ago"
        slurm = self.info.meta.get("env", {}).get("slurm", {}).get("SLURM_JOB_ID")
        slurm_s = f" · slurm {slurm}" if slurm else ""
        header = (
            f" {dot}{RESET} {BOLD}{self.info.project}/{self.info.name}{RESET}"
            f" {DIM}({self.info.id}){RESET}"
            f" · step {BOLD}{fmt_step(self.reader.last_step)}{RESET}"
            f" · {status}{slurm_s}{age}"
        )

        tabs = []
        for i, g in enumerate(pages):
            if i == self.page:
                tabs.append(f"\x1b[7m {g} \x1b[27m")
            else:
                tabs.append(f"{DIM} {g} {RESET}")
        tabline = " " + "".join(tabs)

        smooth = _SMOOTH_LEVELS[self.smooth_idx]
        footer = (
            f" {DIM}←/→ pages · ↑/↓ scroll · 1-9 cols ({self.ncols or 'auto'}) · "
            f"s smooth ({smooth:g}) · l log ({'on' if self.logscale else 'off'}) "
            f"· q quit{RESET}"
        )

        body_rows = rows - 4
        if current == "console":
            body = self._render_console(cols, body_rows)
        else:
            body = self._render_charts(current, cols, body_rows)
        body += [""] * (body_rows - len(body))

        lines = [header, tabline, ""] + body[:body_rows] + [footer]
        return "\x1b[H" + "\x1b[K\n".join(_pad(l, 0) for l in lines) + "\x1b[J"

    def _render_console(self, cols: int, rows: int) -> list[str]:
        lines = read_console(self.info, max_lines=2000)
        max_offset = max(0, len(lines) - rows)
        self.scroll = min(self.scroll, (max_offset + 4) // 5)
        offset = min(self.scroll * 5, max_offset)
        end = len(lines) - offset
        body = rows - 1 if offset else rows
        out = [" " + l[: cols - 2] for l in lines[max(0, end - body) : end]]
        if offset:
            out.append(f" {DIM}── ↓ {offset} newer lines ──{RESET}")
        return out

    def _render_charts(self, group: str, cols: int, rows: int) -> list[str]:
        keys = group_keys(self.reader.keys()).get(group, [])
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
            steps, values = self.reader.series[key].points()
            # color by absolute position so colors stay stable while scrolling
            color = _SERIES_COLORS[(start + ki) % len(_SERIES_COLORS)]
            chart = ChartSpec(key, [(self.info.label, steps, values, color)])
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
            tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write("\x1b[?1049h\x1b[?25l")  # alt screen, hide cursor
        try:
            last_refresh = 0.0
            while True:
                now = time.monotonic()
                if now - last_refresh >= self.interval or last_refresh == 0:
                    self.reader.refresh()
                    self.info = _reload_info(self.info)
                    last_refresh = now
                sys.stdout.write(self.render())
                sys.stdout.flush()
                key = self._read_key(0.25)
                on_console = (pages := self._pages()) and pages[self.page % len(pages)] == "console"
                if key in ("q", "Q", "\x03"):
                    break
                elif key in ("\x1b[C", "\x1bOC", "n", "\t"):  # right arrow (CSI/SS3)
                    self.page += 1
                    self.scroll = 0
                elif key in ("\x1b[D", "\x1bOD", "p"):  # left arrow
                    self.page -= 1
                    self.scroll = 0
                elif key in ("\x1b[B", "\x1bOB", "j"):  # down arrow
                    # charts: move down the grid; console: back toward newest
                    self.scroll = self.scroll - 1 if on_console else self.scroll + 1
                    self.scroll = max(0, self.scroll)
                elif key in ("\x1b[A", "\x1bOA", "k"):  # up arrow
                    self.scroll = self.scroll + 1 if on_console else self.scroll - 1
                    self.scroll = max(0, self.scroll)
                elif key == "s":
                    self.smooth_idx = (self.smooth_idx + 1) % len(_SMOOTH_LEVELS)
                elif key == "l":
                    self.logscale = not self.logscale
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


def watch(run: RunInfo, interval: float = 2.0, ncols: int | None = None) -> None:
    WatchApp(run, interval=interval, ncols=ncols).run()
