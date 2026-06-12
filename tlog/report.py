"""`tlog report` — render a markdown spec into one self-contained HTML page.

The spec is ordinary markdown; fenced blocks whose info string starts with
`tlog` become live elements filled from local runs:

    ```tlog chart
    key: eval/fid
    runs: baseline, high-lr     # optional, defaults to the report's run set
    smooth: 0.9                 # optional EMA weight
    logy: true                  # optional log-scale y axis
    ```

    ```tlog table
    columns: eval/fid min, loss/total, config.lr
    ```

    ```tlog images
    key: eval/recon
    steps: 500, 1500            # optional, default: all logged steps
    ```

Prose between blocks is rendered as markdown, so a report narrates its own
findings — written by hand, or by an agent that inspected the runs.
"""

from __future__ import annotations

import datetime
import html
import math
import re
import webbrowser
from pathlib import Path

from .export import _data_uri
from .store import MetricsReader, RunInfo, downsample, ema, read_media_index, resolve_run

_PALETTE = [
    "#5aa9e6", "#ff9e57", "#7fc96b", "#ef6292",
    "#b88ee6", "#ffd166", "#4ed0c2", "#e66a6a",
]

MAX_CHART_POINTS = 700


# -- spec parsing ----------------------------------------------------------------


def parse_spec(text: str) -> list[tuple]:
    """Split a spec into ("md", text) and ("block", type, params) elements."""
    elements: list[tuple] = []
    md: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(```+)\s*tlog\b\s*(\w*)", lines[i])
        if not m:
            md.append(lines[i])
            i += 1
            continue
        fence, btype = m.group(1), m.group(2)
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].startswith(fence):
            body.append(lines[i])
            i += 1
        i += 1  # closing fence
        if md:
            elements.append(("md", "\n".join(md)))
            md = []
        params = _parse_params(body)
        elements.append(("block", btype or params.pop("type", ""), params))
    if md:
        elements.append(("md", "\n".join(md)))
    return elements


def _parse_params(body: list[str]) -> dict[str, str]:
    params = {}
    for line in body:
        line = line.split("#", 1)[0].strip()
        if line and ":" in line:
            k, v = line.split(":", 1)
            params[k.strip().lower()] = v.strip()
    return params


def _as_list(v: str) -> list[str]:
    return [s.strip() for s in v.split(",") if s.strip()]


def _as_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


# -- markdown (small subset: headings, lists, code, bold/em/links) -----------------

_INLINE = [
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"\*([^*]+)\*"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)"), r'<a href="\2">\1</a>'),
]


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    for pat, repl in _INLINE:
        text = pat.sub(repl, text)
    return text


def md_html(text: str) -> str:
    out: list[str] = []
    para: list[str] = []
    in_list = in_code = False

    def flush_para():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in text.splitlines():
        if line.startswith("```"):
            flush_para()
            close_list()
            out.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        s = line.strip()
        h = re.match(r"^(#{1,4})\s+(.*)", s)
        if h:
            flush_para()
            close_list()
            n = len(h.group(1))
            out.append(f"<h{n}>{_inline(h.group(2))}</h{n}>")
        elif re.match(r"^(-{3,}|\*{3,})$", s):
            flush_para()
            close_list()
            out.append("<hr>")
        elif re.match(r"^[-*]\s+", s):
            flush_para()
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*]\s+", "", s)
            out.append(f"<li>{_inline(item)}</li>")
        elif not s:
            flush_para()
            close_list()
        else:
            para.append(s)
    flush_para()
    close_list()
    return "\n".join(out)


# -- chart (inline SVG) ------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    span = (hi - lo) or abs(hi) or 1.0
    step = 10 ** math.floor(math.log10(span / n))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= n:
            step *= mult
            break
    t = math.ceil(lo / step) * step
    ticks = []
    while t <= hi + abs(step) * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def svg_chart(
    series: list[tuple[str, int, list[int], list[float]]],
    *,
    smooth: float = 0.0,
    logy: bool = False,
    title: str = "",
) -> str:
    """series: (run_name, color_idx, steps, values) per run."""
    W, H, ML, MR, MT, MB = 720, 240, 56, 10, 10, 24
    pw, ph = W - ML - MR, H - MT - MB

    plotted = []  # (name, color, steps, raw, drawn)
    for name, ci, steps, values in series:
        if logy:
            pts = [(s, v) for s, v in zip(steps, values) if v > 0]
            steps, values = [p[0] for p in pts], [p[1] for p in pts]
        if steps:
            drawn = [math.log10(v) for v in values] if logy else values
            plotted.append((name, _PALETTE[ci % len(_PALETTE)], steps, values, drawn))
    if not plotted:
        return '<div class="note">no data</div>'

    x0 = min(s[0] for _, _, s, _, _ in plotted)
    x1 = max(s[-1] for _, _, s, _, _ in plotted)
    ys = [v for _, _, _, _, d in plotted for v in d]
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y0, y1 = y0 - 0.5, y1 + 0.5
    pad = (y1 - y0) * 0.05
    y0, y1 = y0 - pad, y1 + pad

    def X(s: float) -> float:
        return ML + (s - x0) / (x1 - x0) * pw

    def Y(v: float) -> float:
        return MT + (1 - (v - y0) / (y1 - y0)) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for t in _nice_ticks(y0, y1):
        y = Y(t)
        label = _fmt(10 ** t) if logy else _fmt(t)
        parts.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{ML - 6}" y="{y + 3:.1f}" class="tick" text-anchor="end">{label}</text>'
        )
    for t in _nice_ticks(x0, x1, n=6):
        x = X(t)
        # clamp anchors so edge labels don't clip outside the viewBox
        anchor = "end" if x > W - MR - 18 else ("start" if x < ML + 18 else "middle")
        parts.append(
            f'<text x="{x:.1f}" y="{H - 8}" class="tick" text-anchor="{anchor}">{_fmt(t)}</text>'
        )

    def poly(steps: list[int], drawn: list[float], color: str, opacity: float) -> str:
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, drawn))
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}"'
            f' stroke-width="1.6" opacity="{opacity}"/>'
        )

    legend = []
    for name, color, steps, raw, drawn in plotted:
        if smooth > 0 and len(drawn) > 2:
            parts.append(poly(steps, drawn, color, 0.25))
            parts.append(poly(steps, ema(drawn, smooth), color, 1.0))
        else:
            parts.append(poly(steps, drawn, color, 1.0))
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(name)}'
            f' <b>{_fmt(raw[-1])}</b></span>'
        )
    parts.append("</svg>")
    cap = f'<div class="chart-title">{html.escape(title)}</div>' if title else ""
    return (
        f'<figure class="chart">{cap}{"".join(parts)}'
        f'<div class="legend">{"".join(legend)}</div></figure>'
    )


# -- report builder ----------------------------------------------------------------


class _Ctx:
    """Resolved runs + readers, with stable colors by first appearance."""

    def __init__(self, default_runs: list[RunInfo], root: str):
        self.root = root
        self.order: list[RunInfo] = []
        self.readers: dict = {}
        self.default = [self._track(r) for r in default_runs]

    def _track(self, info: RunInfo) -> RunInfo:
        for r in self.order:
            if r.path == info.path:
                return r
        self.order.append(info)
        return info

    def color(self, info: RunInfo) -> int:
        return next(i for i, r in enumerate(self.order) if r.path == info.path)

    def reader(self, info: RunInfo) -> MetricsReader:
        rd = self.readers.get(info.path)
        if rd is None:
            rd = self.readers[info.path] = MetricsReader(info, include_system=True)
            rd.refresh()
        return rd

    def runs_for(self, params: dict) -> list[RunInfo] | str:
        if "runs" not in params:
            return self.default
        out = []
        for spec in _as_list(params["runs"]):
            info = next(
                (r for r in self.order if spec in (r.name, r.id, r.path.name)), None
            ) or resolve_run(spec, self.root)
            if info is None:
                return f"run not found: {spec}"
            out.append(self._track(info))
        return out


def _note(msg: str) -> str:
    return f'<div class="note">⚠ {html.escape(msg)}</div>'


def _render_chart(ctx: _Ctx, params: dict) -> str:
    key = params.get("key")
    if not key:
        return _note("chart block needs `key:`")
    runs = ctx.runs_for(params)
    if isinstance(runs, str):
        return _note(runs)
    series = []
    for info in runs:
        sr = ctx.reader(info).series.get(key)
        if sr is None or not len(sr):
            continue
        steps, values = sr.points()
        s, mean, _, _ = downsample(steps, values, MAX_CHART_POINTS)
        series.append((info.name, ctx.color(info), s, mean))
    if not series:
        return _note(f"no data for {key!r}")
    return svg_chart(
        series,
        smooth=float(params.get("smooth", 0) or 0),
        logy=_as_bool(params.get("logy", "")),
        title=params.get("title", key),
    )


def _render_table(ctx: _Ctx, params: dict) -> str:
    runs = ctx.runs_for(params)
    if isinstance(runs, str):
        return _note(runs)
    cols = _as_list(params.get("columns", ""))

    def cell(info: RunInfo, spec: str) -> str:
        if spec.startswith("config."):
            v = info.config.get(spec[len("config."):], "—")
            return html.escape(str(v))
        parts = spec.rsplit(None, 1)
        key, agg = (parts[0], parts[1]) if len(parts) == 2 and parts[1] in (
            "min", "max", "last") else (spec, "last")
        sr = ctx.reader(info).series.get(key)
        if sr is None or not len(sr):
            return "—"
        _, values = sr.points()
        v = {"min": min, "max": max, "last": lambda x: x[-1]}[agg](values)
        return _fmt(v)

    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["run", "status", "step"] + cols)
    rows = []
    for info in runs:
        rd = ctx.reader(info)
        dot = f'<i class="dot" style="background:{_PALETTE[ctx.color(info) % len(_PALETTE)]}"></i>'
        tds = [
            f"<td>{dot}{html.escape(info.name)}</td>",
            f"<td>{info.status}</td>",
            f"<td>{rd.last_step if rd.last_step is not None else '—'}</td>",
        ]
        tds += [f"<td>{cell(info, c)}</td>" for c in cols]
        rows.append(f"<tr>{''.join(tds)}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _render_images(ctx: _Ctx, params: dict, max_image_px: int) -> str:
    key = params.get("key")
    if not key:
        return _note("images block needs `key:`")
    runs = ctx.runs_for(params)
    if isinstance(runs, str):
        return _note(runs)

    per_run: list[dict[int, dict]] = []
    steps: set[int] = set()
    for info in runs:
        by_step = {
            int(rec.get("_step", 0)): rec
            for rec in read_media_index(info)
            if rec.get("key", "media") == key
        }
        per_run.append(by_step)
        steps.update(by_step)
    if not steps:
        return _note(f"no images for {key!r}")

    wanted = sorted(steps, reverse=True)
    if "steps" in params:
        ask = [int(float(s)) for s in _as_list(params["steps"])]
        wanted = [s for s in wanted if s in ask]
    if "last" in params:
        wanted = wanted[: int(params["last"])]

    head = "<th></th>" + "".join(
        f'<th><i class="dot" style="background:{_PALETTE[ctx.color(r) % len(_PALETTE)]}"></i>'
        f"{html.escape(r.name)}</th>"
        for r in runs
    )
    rows = []
    for step in wanted:
        tds = [f'<td class="step">{step:,}</td>']
        for info, by_step in zip(runs, per_run):
            rec = by_step.get(step)
            if not rec:
                tds.append("<td></td>")
                continue
            imgs = []
            for rel in rec.get("files", []):
                uri = _data_uri(info.path / "media" / rel, max_image_px)
                if uri:
                    imgs.append(f'<img src="{uri}" alt="">')
            cap = rec.get("caption")
            cap_html = f'<div class="cap">{html.escape(cap)}</div>' if cap else ""
            tds.append(f'<td><div class="imgs">{"".join(imgs)}</div>{cap_html}</td>')
        rows.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f'<table class="media"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


_CSS = """
:root { --bg:#0f1115; --panel:#161a22; --border:#232936; --text:#d7dce5;
        --muted:#8b93a3; --accent:#5aa9e6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 20px 60px; }
h1 { font-size: 26px; margin: 18px 0 6px; }
h2 { font-size: 19px; margin: 26px 0 6px; }
h3, h4 { margin: 20px 0 4px; }
p, ul { color: var(--text); margin: 8px 0; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--border); margin: 22px 0; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px;
  background: var(--panel); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; overflow-x: auto; }
pre code { background: none; padding: 0; }
figure.chart { margin: 14px 0; background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px 8px; }
.chart-title { color: var(--muted); font-size: 12px; margin-bottom: 4px;
  font-family: ui-monospace, Menlo, monospace; }
svg { width: 100%; height: auto; display: block; }
svg .grid { stroke: var(--border); stroke-width: 1; }
svg .tick { fill: var(--muted); font-size: 10px;
  font-family: ui-monospace, Menlo, monospace; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; padding: 6px 2px 4px;
  font-size: 12px; color: var(--muted); }
.legend i, .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 5px; }
.legend b { color: var(--text); font-weight: 600; }
table { border-collapse: collapse; margin: 14px 0; width: 100%;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden; font-size: 13px; }
th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500;
  font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
tr:last-child td { border-bottom: 0; }
table.media td { vertical-align: top; }
td.step { color: var(--muted); font-family: ui-monospace, Menlo, monospace;
  white-space: nowrap; }
.imgs { display: flex; flex-wrap: wrap; gap: 6px; }
.imgs img { max-width: 230px; border-radius: 6px; border: 1px solid var(--border); }
.cap { color: var(--muted); font-size: 11px; margin-top: 4px; }
.note { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  color: var(--muted); padding: 10px 14px; margin: 14px 0; }
footer { color: var(--muted); font-size: 12px; margin-top: 40px;
  border-top: 1px solid var(--border); padding-top: 12px; }
"""

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>{{CSS}}</style></head>
<body><main>
{{BODY}}
<footer>generated by tlog · {{TS}}</footer>
</main></body></html>
"""


def build_report(
    spec_text: str,
    runs: list[RunInfo],
    root: str = "runs",
    max_image_px: int = 512,
    title: str | None = None,
) -> str:
    ctx = _Ctx(runs, root)
    chunks = []
    for el in parse_spec(spec_text):
        if el[0] == "md":
            chunks.append(md_html(el[1]))
            continue
        _, btype, params = el
        if btype == "chart":
            chunks.append(_render_chart(ctx, params))
        elif btype == "table":
            chunks.append(_render_table(ctx, params))
        elif btype == "images":
            chunks.append(_render_images(ctx, params, max_image_px))
        else:
            chunks.append(_note(f"unknown tlog block type: {btype or '(none)'}"))
    if title is None:
        m = re.search(r"^#\s+(.+)$", spec_text, re.M)
        title = m.group(1).strip() if m else "tlog report"
    return (
        _PAGE.replace("{{TITLE}}", html.escape(title))
        .replace("{{CSS}}", _CSS)
        .replace("{{BODY}}", "\n".join(chunks))
        .replace("{{TS}}", datetime.datetime.now().isoformat(timespec="seconds"))
    )


def report_html(
    spec_path: Path,
    runs: list[RunInfo],
    root: str,
    output: Path | None = None,
    max_image_px: int = 512,
    open_browser: bool = False,
) -> Path:
    out = output or spec_path.with_suffix(".html")
    out.write_text(
        build_report(spec_path.read_text(), runs, root, max_image_px), encoding="utf-8"
    )
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
