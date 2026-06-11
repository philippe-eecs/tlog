import base64

import pytest

from tlog.media import encode_png
from tlog import termimg


def gradient_png(w=96, h=96):
    pixels = bytearray()
    for y in range(h):
        for x in range(w):
            pixels += bytes((int(x / w * 255), int(y / h * 255), 128))
    return encode_png(bytes(pixels), w, h, 3), bytes(pixels)


def test_pure_decode_roundtrip_rgb():
    png, pixels = gradient_png(5, 4)
    w, h, rgb = termimg._decode_png_pure(png)
    assert (w, h) == (5, 4)
    assert bytes(rgb) == pixels


def test_pure_decode_gray_and_rgba():
    g = encode_png(bytes(range(12)), 4, 3, 1)
    w, h, rgb = termimg._decode_png_pure(g)
    assert (w, h) == (4, 3)
    assert rgb[:6] == bytearray((0, 0, 0, 1, 1, 1))

    a = encode_png(bytes([10, 20, 30, 255] * 6), 3, 2, 4)
    _, _, rgb = termimg._decode_png_pure(a)
    assert bytes(rgb) == bytes([10, 20, 30] * 6)


def test_pure_decode_pil_written_png():
    """PIL uses scanline filters (1-4); the pure decoder must undo them."""
    Image = pytest.importorskip("PIL.Image")
    import io
    import random

    rng = random.Random(0)
    img = Image.new("RGB", (32, 24))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(32 * 24)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    w, h, rgb = termimg._decode_png_pure(buf.getvalue())
    assert (w, h) == (32, 24)
    assert bytes(rgb) == img.tobytes()


def test_box_downscale():
    # uniform image stays uniform at any scale
    rgb = bytearray([100, 150, 200] * (20 * 10))
    w, h, out = termimg._box_downscale(rgb, 20, 10, 5, 3)
    assert (w, h) == (5, 3)
    assert all(out[i * 3 : i * 3 + 3] == bytearray((100, 150, 200)) for i in range(15))


def test_render_halfblock_geometry():
    png, _ = gradient_png(96, 96)
    lines = termimg.render_halfblock(png, 32)
    assert len(lines) == 16  # 32px wide -> 32px tall (square) -> /2 rows
    assert "▀" in lines[0]
    assert "38;2;" in lines[0] and "48;2;" in lines[0]
    assert lines[0].endswith(termimg.RESET)


def test_detect_backend(monkeypatch):
    for var in ("TMUX", "TERM", "TERM_PROGRAM", "KITTY_WINDOW_ID"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("TERM", "xterm-kitty")
    assert termimg.detect_backend() == "kitty"
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    assert termimg.detect_backend() == "halfblock"  # tmux eats the protocol
    assert termimg.detect_backend(force="kitty") == "kitty"  # explicit override
    monkeypatch.delenv("TMUX")

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert termimg.detect_backend() == "iterm2"
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    assert termimg.detect_backend() == "iterm2"
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert termimg.detect_backend() == "halfblock"
    assert termimg.detect_backend(force="off") == "off"


def test_kitty_emitter_chunking():
    png, _ = gradient_png(96, 96)  # base64 > 4096 -> multiple chunks
    seq = termimg.render_kitty(png, image_id=7, cols=30, rows=15)
    chunks = seq.split("\x1b\\")[:-1]
    assert len(chunks) >= 2
    assert "a=T" in chunks[0] and "i=7" in chunks[0] and "c=30" in chunks[0]
    assert all(c.startswith("\x1b_G") for c in chunks)
    assert ",m=1;" in chunks[0] or "m=1;" in chunks[0]
    assert "m=0" in chunks[-1]
    # payload chunks within kitty's 4096 limit
    for c in chunks:
        assert len(c.split(";", 1)[1]) <= 4096

    place = termimg.render_kitty(None, image_id=7, cols=30, rows=15)
    assert "a=p" in place and "i=7" in place


def test_iterm2_emitter_roundtrip():
    png, _ = gradient_png(8, 8)
    seq = termimg.render_iterm2(png, width_cells=20)
    assert seq.startswith("\x1b]1337;File=inline=1;width=20;")
    payload = seq.split(":", 1)[1].rstrip("\x07")
    assert base64.b64decode(payload) == png


def test_cached_halfblock_reuses(tmp_path):
    png, _ = gradient_png(16, 16)
    p = tmp_path / "img.png"
    p.write_bytes(png)
    first = termimg.cached_halfblock(p, 16)
    second = termimg.cached_halfblock(p, 16)
    assert first is second  # cache hit returns the same object
    missing = termimg.cached_halfblock(tmp_path / "nope.png", 16)
    assert "missing" in missing[0]
