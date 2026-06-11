import json
import struct
import zlib

import pytest

from tlog.media import encode_png, save_image
from tlog.run import Run
from tlog.store import find_runs


def parse_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = {}
    pos = 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + body), f"bad crc for {tag}"
        chunks[tag] = body
        pos += 12 + length
    return chunks


def test_encode_png_roundtrip():
    w, h = 3, 2
    pixels = bytes(range(w * h * 3))
    png = encode_png(pixels, w, h, 3)
    chunks = parse_png(png)
    width, height, depth, color = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (width, height, depth, color) == (3, 2, 8, 2)
    raw = zlib.decompress(chunks[b"IDAT"])
    # filter byte 0 + scanline per row
    assert len(raw) == h * (1 + w * 3)
    assert raw[1 : 1 + w * 3] == pixels[: w * 3]


def test_save_image_numpy(tmp_path):
    np = pytest.importorskip("numpy")
    # float HWC in [0,1]
    arr = np.linspace(0, 1, 4 * 5 * 3).reshape(4, 5, 3)
    save_image(arr, tmp_path / "a.png")
    assert (tmp_path / "a.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # uint8 CHW gets transposed
    chw = (arr.transpose(2, 0, 1) * 255).astype("uint8")
    save_image(chw, tmp_path / "b.png")
    chunks = parse_png((tmp_path / "b.png").read_bytes())
    width, height = struct.unpack(">II", chunks[b"IHDR"][:8])
    assert (width, height) == (5, 4)


def test_export_html_smoke(tmp_path):
    from tlog.export import export_html

    run = Run(project="p", name="r1", dir=tmp_path, config={"lr": 1},
              capture_console=False, system_metrics=False)
    for s in range(0, 100, 10):
        run.log({"loss/total": 1.0 / (s + 1), "eval/fid": 50 - s / 4}, step=s)
    run.log_images("eval/recon", [(bytes([0, 128, 255] * 4), 2, 2, 3)], step=50)
    run.finish()

    info = find_runs(tmp_path)[0]
    out = export_html([info], tmp_path / "report.html")
    html = out.read_text()
    assert "loss/total" in html
    assert "data:image/png;base64," in html
    assert '"mode"' not in html.split("TLOG_MODE")[0]  # sanity: template filled
    assert "{{" not in html.replace("{{}}", "")  # no leftover placeholders

    payload = html.split("window.TLOG_DATA = ", 1)[1].split(";\n", 1)[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert data["runs"][0]["name"] == "r1"
    assert len(data["runs"][0]["metrics"]["loss/total"]["steps"]) == 10
    assert data["runs"][0]["media"][0]["step"] == 50
