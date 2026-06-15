import io

from tlog import render
from tlog.media import encode_png
from tlog.run import Run
from tlog.store import _load_run


def _make_run(tmp_path):
    r = Run(project="t", name="a", dir=tmp_path, capture_console=False, system_metrics=False)
    for i in range(50):
        r.log({"loss/total": 1.0 / (i + 1)}, step=i)
    r.log_images("eval/img", [(bytes([200, 100, 50]) * 16, 4, 4, 3)], step=49, caption="cap")
    r.finish()
    return _load_run(r.dir)


def test_md_ansi_basics():
    text = "\n".join(render.md_ansi("# Title\n\nsome **bold** and `code`\n\n- a\n- b\n"))
    assert "Title" in text
    assert "\x1b[1m" in text  # bold/heading styling present
    assert "•" in text       # bullet


def test_ansi_table():
    rows = [(0, "a", "running", 10, ["0.5"]), (1, "b", "finished", 20, ["0.6"])]
    out = "\n".join(render.ansi_table(["loss"], rows))
    assert "run" in out and "loss" in out
    assert "a" in out and "b" in out


def test_emit_image_halfblock(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(encode_png(bytes([10, 20, 30] * 64), 8, 8, 3))
    buf = io.StringIO()
    render.emit_image(p, out=buf, backend="halfblock", width_cells=8)
    assert "▀" in buf.getvalue()  # half-block cell


def test_emit_image_kitty_reserves_and_places(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(encode_png(bytes([10, 20, 30] * 64), 8, 8, 3))
    buf = io.StringIO()
    render.emit_image(p, out=buf, backend="kitty", width_cells=20)
    out = buf.getvalue()
    assert "\x1b_G" in out  # a kitty graphics escape was emitted


def test_emit_markdown_blocks(tmp_path):
    info = _make_run(tmp_path)
    spec = tmp_path / "r.md"
    spec.write_text(
        "# R\n\n```tlog chart\nkey: loss/total\n```\n\n"
        "```tlog table\ncolumns: loss/total min\n```\n"
    )
    buf = io.StringIO()
    render.emit_markdown(spec, [info], str(tmp_path), out=buf, backend="halfblock", width=90)
    txt = buf.getvalue()
    braille = sum(1 for c in txt if 0x2800 <= ord(c) <= 0x28FF)
    assert braille > 0                 # chart drawn
    assert "loss/total min" in txt     # table column header


def test_show_dispatch_image(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(encode_png(bytes([1, 2, 3] * 64), 8, 8, 3))
    buf = io.StringIO()
    assert render.show(str(p), root=str(tmp_path), out=buf, backend="halfblock")
    assert "▀" in buf.getvalue()
    assert render.show(str(tmp_path / "missing.png"), root=str(tmp_path), out=buf) is False
