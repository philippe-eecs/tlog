"""Grid layout: many charts in one group must scroll, and column count must be
controllable. Rendered headlessly (terminal size from COLUMNS/LINES env)."""

import re

import pytest

from tlog.run import Run
from tlog.store import find_runs
from tlog.tui import WatchApp

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
N_METRICS = 9


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setenv("LINES", "30")
    run = Run(project="p", name="many", dir=tmp_path,
              capture_console=False, system_metrics=False)
    for s in range(0, 50, 10):
        run.log({f"loss/metric_{i:02d}": float(i + s) for i in range(N_METRICS)}, step=s)
    run.finish()
    app = WatchApp(find_runs(tmp_path)[0])
    app._refresh()
    return app


def shown_metrics(app):
    text = ANSI.sub("", app.render())
    return sorted(set(re.findall(r"loss/metric_\d\d", text)))


def test_overflow_scrolls_with_indicator(app):
    first = shown_metrics(app)
    assert 0 < len(first) < N_METRICS  # doesn't fit -> partial view, not clipped silently
    assert f"of {N_METRICS}" in ANSI.sub("", app.render())  # position indicator

    app.scroll = 99  # clamped to the last page of charts
    last = shown_metrics(app)
    assert f"loss/metric_{N_METRICS - 1:02d}" in last
    assert app.scroll < 99

    seen = set()
    for app.scroll in range(0, app.scroll + 1):
        seen.update(shown_metrics(app))
    assert len(seen) == N_METRICS  # every chart reachable by scrolling


def test_manual_columns(app):
    app.ncols = 1
    text = ANSI.sub("", app.render())
    for line in text.splitlines():
        assert len(re.findall(r"loss/metric_", line)) <= 1  # one chart per row
    app.ncols = 3
    text = ANSI.sub("", app.render())
    assert any(len(re.findall(r"loss/metric_", line)) == 3 for line in text.splitlines())
    assert "cols (3)" in text


def test_columns_clamped_to_pane_width(app):
    app.ncols = 9  # 100 cols can't fit 9 legible charts -> clamp (100 // 24 = 4)
    text = ANSI.sub("", app.render())
    assert all(len(re.findall(r"loss/metric_", line)) <= 4 for line in text.splitlines())


# -- multi-run compare ---------------------------------------------------------


def make_compare_app(tmp_path, monkeypatch, images="halfblock"):
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "40")
    red = bytes([255, 40, 40] * 4)  # 2x2
    blue = bytes([40, 40, 255] * 4)
    runs = []
    for name, pixels in (("baseline", red), ("variant", blue)):
        run = Run(project="p", name=name, dir=tmp_path,
                  capture_console=False, system_metrics=False)
        for s in range(0, 100, 10):
            run.log({"loss/total": 1.0 / (s + 1), f"only/{name}": float(s)}, step=s)
        run.log_images("eval/recon", [(pixels, 2, 2, 3)], step=50, caption=f"{name}@50")
        (run.dir / "console.log").write_text(f"hello from {name}\n")
        run.finish()
        runs.append(run.dir)
    infos = [find_runs(d)[0] for d in runs]
    app = WatchApp(infos, images=images)
    app._refresh()
    return app


def test_compare_legend_and_overlay(tmp_path, monkeypatch):
    app = make_compare_app(tmp_path, monkeypatch)
    raw = app.render()
    text = ANSI.sub("", raw)
    # legend shows both runs; charts page overlays both run colors
    assert "baseline" in text and "variant" in text
    assert "38;5;75m" in raw and "38;5;209m" in raw  # run 0 + run 1 palette colors
    # union of keys: per-run-only metrics each get a chart group
    assert "only" in app._pages()


def test_compare_media_page_side_by_side(tmp_path, monkeypatch):
    app = make_compare_app(tmp_path, monkeypatch)
    pages = app._pages()
    assert "media" in pages
    app.page = pages.index("media")
    raw = app.render()
    text = ANSI.sub("", raw)
    assert "eval/recon" in text
    assert "▀" in raw  # halfblock thumbnails rendered
    assert "baseline@50" in text and "variant@50" in text  # captions per column
    # both runs' colored column headers present
    header_line = next(l for l in raw.splitlines() if "baseline" in ANSI.sub("", l))
    assert "38;5;75m" in header_line


def test_compare_console_focus_cycles(tmp_path, monkeypatch):
    app = make_compare_app(tmp_path, monkeypatch)
    pages = app._pages()
    app.page = pages.index("console")
    assert "hello from baseline" in ANSI.sub("", app.render())
    app.focus = 1
    assert "hello from variant" in ANSI.sub("", app.render())


def test_media_off_hides_page(tmp_path, monkeypatch):
    app = make_compare_app(tmp_path, monkeypatch, images="off")
    assert "media" not in app._pages()


def test_single_run_has_no_legend(app):
    text = ANSI.sub("", app.render())
    assert text.splitlines()[2].strip() == ""  # line after tabs is blank, no legend
