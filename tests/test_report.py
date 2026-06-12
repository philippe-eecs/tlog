"""Tests for `tlog report`: spec parsing, markdown subset, and HTML rendering."""

from pathlib import Path

from tlog.report import build_report, md_html, parse_spec, svg_chart
from tlog.run import Run
from tlog.store import find_runs

SPEC = """\
# Comparing two runs

Some **bold** prose with `code` and a [link](https://example.com).

- first point
- second point

```tlog chart
key: loss/total
smooth: 0.9
```

```tlog table
columns: loss/total min, loss/total, config.lr
```

```tlog images
key: eval/recon
steps: 50
```

```python
print("plain code block, untouched")
```
"""


def make_runs(tmp_path) -> list:
    for name, lr in (("a", 1e-3), ("b", 3e-4)):
        run = Run(project="p", name=name, dir=tmp_path, config={"lr": lr},
                  capture_console=False, system_metrics=False)
        for s in range(0, 100, 10):
            run.log({"loss/total": (1.0 if name == "a" else 2.0) / (s + 1)}, step=s)
        run.log_images(
            "eval/recon", [(bytes([0, 128, 255] * 4), 2, 2, 3)], step=50,
            caption="recon!",
        )
        run.finish()
    return sorted(find_runs(tmp_path), key=lambda r: r.name)


def test_parse_spec_blocks():
    els = parse_spec(SPEC)
    blocks = [e for e in els if e[0] == "block"]
    assert [b[1] for b in blocks] == ["chart", "table", "images"]
    assert blocks[0][2] == {"key": "loss/total", "smooth": "0.9"}
    assert blocks[2][2]["steps"] == "50"
    # plain fenced code stays in the markdown stream
    assert any("plain code block" in e[1] for e in els if e[0] == "md")


def test_md_subset():
    html = md_html("# T\n\npara **b** `c`\n\n- x\n- y\n\n---")
    assert "<h1>T</h1>" in html
    assert "<strong>b</strong>" in html and "<code>c</code>" in html
    assert html.count("<li>") == 2
    assert "<hr>" in html


def test_svg_chart_overlays_runs():
    fig = svg_chart(
        [("a", 0, [0, 10, 20], [3.0, 2.0, 1.0]), ("b", 1, [0, 10, 20], [4.0, 3.5, 3.0])],
        smooth=0.5,
    )
    # raw + smoothed polyline per run
    assert fig.count("<polyline") == 4
    assert "a <b>1</b>" in fig and "b <b>3</b>" in fig


def test_build_report_end_to_end(tmp_path):
    runs = make_runs(tmp_path)
    html = build_report(SPEC, runs, root=str(tmp_path))

    assert "<h1>Comparing two runs</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<title>Comparing two runs</title>" in html
    # chart: 2 runs x (raw + smoothed)
    assert html.count("<polyline") == 4
    # table: config values and min aggregation (run a: min of 1/(s+1) at s=90)
    assert "0.001" in html and "0.0003" in html
    assert "0.01099" in html  # 1/91
    # images: embedded with caption, only step 50
    assert html.count("data:image/png;base64,") == 2
    assert "recon!" in html
    # plain code block untouched, no leftover placeholders
    assert "plain code block, untouched" in html
    assert "{{" not in html


def test_block_run_filter_and_errors(tmp_path):
    runs = make_runs(tmp_path)
    html = build_report(
        "```tlog chart\nkey: loss/total\nruns: a\n```", runs, root=str(tmp_path)
    )
    assert html.count("<polyline") == 1

    html = build_report("```tlog chart\nkey: nope/missing\n```", runs, str(tmp_path))
    assert "no data" in html
    html = build_report("```tlog chart\nkey: x\nruns: ghost\n```", runs, str(tmp_path))
    assert "run not found: ghost" in html
    html = build_report("```tlog wat\n```", runs, str(tmp_path))
    assert "unknown tlog block" in html


def test_report_cli(tmp_path, capsys):
    from tlog.cli import main

    make_runs(tmp_path)
    spec = tmp_path / "spec.md"
    spec.write_text("# R\n```tlog table\ncolumns: loss/total\n```\n")
    main(["report", str(spec), "--dir", str(tmp_path)])
    out = tmp_path / "spec.html"
    assert out.exists()
    assert "<table>" in out.read_text()
    assert "wrote" in capsys.readouterr().out
