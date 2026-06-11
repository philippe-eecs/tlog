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
    app.reader.refresh()
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
