"""Render images in the terminal.

Primary backend is half-block rendering — each cell shows two vertical pixels
via `▀` with 24-bit fg/bg colors — which works in every terminal, including
through tmux over SSH. When a capable terminal is confidently detected (and
we're NOT inside tmux, which usually eats the escapes), the kitty or iTerm2
graphics protocols are used instead for true pixel images.

PNG decoding uses PIL when installed; otherwise a pure-stdlib decoder covers
the PNGs tlog itself writes (8-bit gray/RGB/RGBA, non-interlaced) and typical
PIL output.
"""

from __future__ import annotations

import base64
import math
import os
import struct
import zlib
from pathlib import Path

RESET = "\x1b[0m"


# -- decoding ------------------------------------------------------------------


def _decode_png_pure(data: bytes) -> tuple[int, int, bytearray]:
    """Decode a PNG to (w, h, RGB bytes). Supports bit depth 8, color types
    0 (gray) / 2 (RGB) / 6 (RGBA), non-interlaced — the formats tlog writes."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    idat = bytearray()
    w = h = color_type = None
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            w, h, depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or interlace != 0 or color_type not in (0, 2, 6):
                raise ValueError(
                    f"unsupported PNG (depth={depth} color={color_type} interlaced={interlace})"
                )
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        pos += 12 + length
    if w is None:
        raise ValueError("PNG missing IHDR")

    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = w * channels
    raw = zlib.decompress(bytes(idat))
    out = bytearray(h * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ftype = raw[p]
        p += 1
        line = bytearray(raw[p : p + stride])
        p += stride
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa = abs(b - c)
                pb = abs(a - c)
                pc = abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ftype != 0:
            raise ValueError(f"bad PNG filter {ftype}")
        out[y * stride : (y + 1) * stride] = line
        prev = line

    if channels == 3:
        return w, h, out
    rgb = bytearray(w * h * 3)
    if channels == 1:
        for i in range(w * h):
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = out[i]
    else:  # RGBA: drop alpha
        for i in range(w * h):
            rgb[i * 3 : i * 3 + 3] = out[i * 4 : i * 4 + 3]
    return w, h, rgb


def _box_downscale(
    rgb: bytearray, w: int, h: int, tw: int, th: int
) -> tuple[int, int, bytearray]:
    if tw >= w:  # never upscale
        return w, h, rgb
    out = bytearray(tw * th * 3)
    for ty in range(th):
        y0 = ty * h // th
        y1 = max(y0 + 1, (ty + 1) * h // th)
        for tx in range(tw):
            x0 = tx * w // tw
            x1 = max(x0 + 1, (tx + 1) * w // tw)
            rs = gs = bs = n = 0
            for y in range(y0, y1):
                base = y * w * 3
                for x in range(x0, x1):
                    i = base + x * 3
                    rs += rgb[i]
                    gs += rgb[i + 1]
                    bs += rgb[i + 2]
                    n += 1
            o = (ty * tw + tx) * 3
            out[o] = rs // n
            out[o + 1] = gs // n
            out[o + 2] = bs // n
    return tw, th, out


def load_rgb(png_bytes: bytes, max_w_px: int) -> tuple[int, int, bytearray]:
    """Decode a PNG and downscale to at most max_w_px wide (aspect kept)."""
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        if img.width > max_w_px:
            img.thumbnail((max_w_px, max_w_px * 4))
        return img.width, img.height, bytearray(img.tobytes())
    except ImportError:
        pass
    w, h, rgb = _decode_png_pure(png_bytes)
    if w > max_w_px:
        th = max(1, round(h * max_w_px / w))
        return _box_downscale(rgb, w, h, max_w_px, th)
    return w, h, rgb


# -- renderers -----------------------------------------------------------------


def render_halfblock(png_bytes: bytes, width_cells: int) -> list[str]:
    """Render as lines of `▀` cells: one pixel per cell horizontally, two
    vertically. Cell aspect (~1:2) makes the pixels roughly square."""
    w, h, rgb = load_rgb(png_bytes, max_w_px=width_cells)

    def px(x: int, y: int) -> tuple[int, int, int]:
        i = (y * w + x) * 3
        return rgb[i], rgb[i + 1], rgb[i + 2]

    lines = []
    for y in range(0, h, 2):
        parts = []
        last = None
        for x in range(w):
            top = px(x, y)
            bot = px(x, y + 1) if y + 1 < h else (13, 15, 19)
            seq = "\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm" % (*top, *bot)
            if seq != last:
                parts.append(seq)
                last = seq
            parts.append("▀")
        parts.append(RESET)
        lines.append("".join(parts))
    return lines


def kitty_id_for(path: Path) -> int:
    return zlib.crc32(str(path).encode()) & 0x7FFFFFFF or 1


def kitty_delete_all() -> str:
    return "\x1b_Ga=d,d=A,q=2\x1b\\"


def render_kitty(
    png_bytes: bytes | None, image_id: int, cols: int, rows: int
) -> str:
    """Place an image at the cursor, scaled into cols x rows cells.
    png_bytes=None places an already-transmitted image by id."""
    if png_bytes is None:
        return f"\x1b_Ga=p,i={image_id},c={cols},r={rows},q=2\x1b\\"
    payload = base64.b64encode(png_bytes).decode("ascii")
    chunks = [payload[i : i + 4096] for i in range(0, len(payload), 4096)]
    seqs = []
    for j, chunk in enumerate(chunks):
        ctrl = []
        if j == 0:
            ctrl = [f"a=T", "f=100", f"i={image_id}", f"c={cols}", f"r={rows}", "q=2"]
        ctrl.append(f"m={1 if j < len(chunks) - 1 else 0}")
        seqs.append(f"\x1b_G{','.join(ctrl)};{chunk}\x1b\\")
    return "".join(seqs)


def render_iterm2(png_bytes: bytes, width_cells: int) -> str:
    payload = base64.b64encode(png_bytes).decode("ascii")
    return (
        f"\x1b]1337;File=inline=1;width={width_cells};preserveAspectRatio=1;"
        f"size={len(png_bytes)}:{payload}\x07"
    )


# -- backend detection -----------------------------------------------------------

BACKENDS = ("auto", "halfblock", "kitty", "iterm2", "off")


def detect_backend(force: str = "auto") -> str:
    """Pick an image backend. Inside tmux/screen the graphics protocols are
    usually swallowed, so we force half-block there unless overridden."""
    if force != "auto":
        return force
    env = os.environ
    term = env.get("TERM", "")
    if env.get("TMUX") or "screen" in term:
        return "halfblock"
    if "kitty" in term or "ghostty" in term or env.get("KITTY_WINDOW_ID"):
        return "kitty"
    prog = env.get("TERM_PROGRAM", "")
    if prog in ("iTerm.app", "WezTerm"):
        return "iterm2"
    return "halfblock"


# -- cache -----------------------------------------------------------------------

_CACHE: dict = {}
_CACHE_MAX = 64


def cached_halfblock(path: Path, width_cells: int) -> list[str]:
    """Half-block render with an (path, mtime, width) keyed cache."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return [f"(missing: {path.name})"]
    key = (str(path), mtime, width_cells)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        lines = render_halfblock(path.read_bytes(), width_cells)
    except (OSError, ValueError) as e:
        lines = [f"(can't render {path.name}: install pillow?)"]
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = lines
    return lines
