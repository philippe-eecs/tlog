"""Braille-canvas line charts and the ANSI helpers shared by every terminal
view (the live `tlog watch` dashboard and the one-shot `tlog show`/`review`).

A `Canvas` packs a 2x4 dot grid into each cell via Unicode braille (U+2800+),
so a chart drawn here is just a `list[str]` of ANSI lines that can be printed
to the scrollback or diffed into an alt-screen frame.
"""

from __future__ import annotations

import math
import re

from .store import downsample, ema

# braille dot bits indexed [y % 4][x % 2]
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

# palette index -> ANSI 256 color code
BAND = 1
PALETTE = {1: 238, 2: 75, 3: 209, 4: 114, 5: 176, 6: 221, 7: 80, 8: 168}
SERIES_COLORS = [2, 3, 4, 5, 6, 7, 8]
SMOOTH_LEVELS = [0.0, 0.6, 0.9, 0.99]

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"


def fg(code: int) -> str:
    return f"\x1b[38;5;{code}m"


def run_color(idx: int) -> int:
    """Palette code (a PALETTE key) for a 0-based run index."""
    return SERIES_COLORS[idx % len(SERIES_COLORS)]


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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - visible_len(s))


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
            if color != BAND or self._color[i] == 0:
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
                    parts.append(fg(PALETTE.get(color, 250)))
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
    for _, steps, values, color in chart.series:
        if logscale:
            pts = [(s, v) for s, v in zip(steps, values) if v > 0]
            steps = [s for s, _ in pts]
            values = [math.log10(v) for _, v in pts]
        if not steps:
            continue
        s, mean, lo, hi = downsample(steps, values, plot_cols * 2)
        mean = ema(mean, smooth)
        prepared.append((s, mean, lo, hi, color))
        vmin = min(vmin, min(lo))
        vmax = max(vmax, max(hi))

    # latest value in the title only when a single series is shown
    title_val = ""
    if len(chart.series) == 1 and chart.series[0][2]:
        title_val = fmt_num(chart.series[0][2][-1])
    title = f" {BOLD}{chart.key}{RESET}"
    if title_val:
        pad_n = width - len(chart.key) - len(title_val) - 3
        title += " " * max(1, pad_n) + fg(PALETTE[chart.series[0][3]]) + title_val + RESET

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
                canvas.vline(sx(s), sy(h), sy(l), BAND)
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
