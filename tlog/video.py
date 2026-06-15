"""Optional video playback for `tlog show clip.mp4` (pip install tlog-ml[video]).

Smooth in-place playback works in kitty/Ghostty (and iTerm2/WezTerm); elsewhere
we fall back to a contact sheet of evenly spaced frames, so you always see the
clip even over a plain SSH session.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from . import termimg
from .chart import BOLD, DIM, RESET


def _frames(path: Path, max_frames: int):
    """Yield PNG-encoded bytes for up to `max_frames` evenly sampled frames."""
    try:
        import imageio.v3 as iio
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "video support needs imageio — install with: pip install tlog-ml[video]"
        )
    frames = list(iio.imiter(path))
    if not frames:
        return []
    step = max(1, len(frames) // max_frames)
    out = []
    for fr in frames[::step][:max_frames]:
        buf = io.BytesIO()
        Image.fromarray(fr).convert("RGB").save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def play(
    path: Path,
    out=sys.stdout,
    backend: str | None = None,
    fps: float = 12.0,
    max_frames: int = 240,
    width_cells: int | None = None,
) -> None:
    path = Path(path)
    backend = backend or termimg.detect_backend()
    import shutil

    width_cells = width_cells or min(shutil.get_terminal_size((100, 40)).columns - 2, 100)
    try:
        pngs = _frames(path, max_frames)
    except RuntimeError as e:
        out.write(f"{DIM}{e}{RESET}\n")
        return
    if not pngs:
        out.write(f"{DIM}(no frames in {path.name}){RESET}\n")
        return

    out.write(f"{BOLD}{path.name}{RESET} {DIM}· {len(pngs)} frames{RESET}\n")
    live = backend in ("kitty", "iterm2") and out is sys.stdout and sys.stdout.isatty()
    if not live:  # contact sheet: a handful of frames inline
        from .render import emit_image  # local import avoids a cycle at import time

        picks = pngs[:: max(1, len(pngs) // 6)][:6]
        import tempfile

        for i, data in enumerate(picks):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(data)
                tmp = Path(f.name)
            out.write(f"{DIM}frame {i * max(1, len(pngs) // 6)}{RESET}\n")
            emit_image(tmp, out=out, backend=backend, width_cells=width_cells // 2)
            tmp.unlink(missing_ok=True)
        return

    # live: reserve a box, then redraw frames in place
    from PIL import Image

    w, h = Image.open(io.BytesIO(pngs[0])).size
    rows = max(1, round(width_cells * (h / w) * 0.5))
    img_id = termimg.kitty_id_for(path)
    out.write("\n" * rows + f"\x1b[{rows}A")
    try:
        for data in pngs:
            out.write("\x1b[s")  # save cursor (top-left of the box)
            if backend == "kitty":
                out.write(termimg.kitty_delete_all())
                out.write(termimg.render_kitty(data, img_id, width_cells, rows))
            else:
                out.write(termimg.render_iterm2(data, width_cells))
            out.write("\x1b[u")  # restore cursor to the box top
            out.flush()
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    out.write(f"\x1b[{rows}B\r")
