"""Image logging: convert torch tensors / numpy arrays / PIL images to PNG.

torch and numpy are never imported here — they're picked up from sys.modules
only if the training script already imported them. If PIL is installed it is
used for encoding; otherwise a minimal pure-stdlib PNG encoder kicks in, so
the core package stays dependency-free.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path
from typing import Any


def encode_png(pixels: bytes, width: int, height: int, channels: int) -> bytes:
    """Encode raw uint8 pixel bytes (row-major, HWC) as a PNG. Pure stdlib."""
    color_type = {1: 0, 3: 2, 4: 6}[channels]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    stride = width * channels
    raw = b"".join(
        b"\x00" + pixels[y * stride : (y + 1) * stride] for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _to_uint8_hwc(obj: Any) -> Any:
    """Normalize a torch tensor or numpy array to a uint8 numpy array of shape
    (H, W, C) with C in {1, 3, 4}. Returns a numpy array."""
    mod = type(obj).__module__.split(".")[0]

    if mod == "torch":
        t = obj.detach()
        if t.is_floating_point():
            t = t.clamp(0, 1).mul(255).round()
        t = t.to("cpu")
        arr = t.numpy()
    else:
        arr = obj

    np = sys.modules.get("numpy")
    if np is None:
        raise TypeError(
            "numpy is required to log array images (it was not found in sys.modules)"
        )
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        arr = (arr.clip(0.0, 1.0) * 255).round()
    arr = arr.astype("uint8")

    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected 2D or 3D image, got shape {arr.shape}")
    # CHW -> HWC for channel-first tensors
    if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected 1/3/4 channels, got shape {arr.shape}")
    return arr


def save_image(obj: Any, path: Path) -> None:
    """Save an image-like object as PNG. Accepts a PIL Image, torch tensor,
    numpy array, or a raw (pixel_bytes, width, height, channels) tuple."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if (
        isinstance(obj, tuple)
        and len(obj) == 4
        and isinstance(obj[0], (bytes, bytearray))
    ):
        pixels, w, h, c = obj
        path.write_bytes(encode_png(bytes(pixels), w, h, c))
        return

    # PIL image: it knows how to save itself.
    if hasattr(obj, "save") and hasattr(obj, "mode"):
        obj.save(path, format="PNG")
        return

    arr = _to_uint8_hwc(obj)
    h, w, c = arr.shape

    try:
        from PIL import Image  # optional, preferred encoder

        img = Image.fromarray(arr[:, :, 0] if c == 1 else arr)
        img.save(path, format="PNG")
        return
    except ImportError:
        pass

    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy()
    path.write_bytes(encode_png(arr.tobytes(), w, h, c))
