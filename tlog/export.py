"""`tlog export` — render runs into one self-contained HTML file.

Everything (frontend, uPlot, metric data, images as base64) is inlined, so the
file can be opened in VS Code's preview, scp'd to a laptop, or attached to a
message with no server and no internet access.
"""

from __future__ import annotations

import base64
import datetime
import io
import json
from pathlib import Path

from .payload import run_media, run_metrics, run_summary
from .store import RunInfo, read_console

FRONTEND = Path(__file__).parent / "frontend"


def _data_uri(png_path: Path, max_px: int) -> str | None:
    try:
        raw = png_path.read_bytes()
    except OSError:
        return None
    if max_px > 0:
        try:  # downscale with PIL if available to keep the report small
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            if max(img.size) > max_px:
                img.thumbnail((max_px, max_px))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw = buf.getvalue()
        except ImportError:
            pass
        except Exception:
            pass
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def build_data(runs: list[RunInfo], max_image_px: int = 512) -> dict:
    payload_runs = []
    for info in runs:
        summary = run_summary(info)
        summary["metrics"] = run_metrics(info)
        media = []
        for rec in run_media(info):
            files = []
            for rel in rec["files"]:
                uri = _data_uri(info.path / "media" / rel, max_image_px)
                if uri:
                    files.append(uri)
            if files:
                rec = dict(rec, files=files)
                media.append(rec)
        summary["media"] = media
        summary["console"] = "\n".join(read_console(info, max_lines=300))
        payload_runs.append(summary)
    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "runs": payload_runs,
    }


def render_template(mode: str, data: dict | None, title: str = "tlog") -> str:
    html = (FRONTEND / "index.html").read_text()
    data_json = "null" if data is None else json.dumps(
        data, separators=(",", ":")
    ).replace("</", "<\\/")
    return (
        html.replace("{{TITLE}}", title)
        .replace("{{UPLOT_CSS}}", (FRONTEND / "vendor" / "uplot.min.css").read_text())
        .replace("{{CSS}}", (FRONTEND / "style.css").read_text())
        .replace("{{UPLOT_JS}}", (FRONTEND / "vendor" / "uplot.min.js").read_text())
        .replace("{{MODE}}", mode)
        .replace("{{DATA}}", data_json)
        .replace("{{APP_JS}}", (FRONTEND / "app.js").read_text())
    )


def export_html(
    runs: list[RunInfo], output: Path, max_image_px: int = 512
) -> Path:
    data = build_data(runs, max_image_px=max_image_px)
    title = "tlog — " + ", ".join(r.name for r in runs[:3]) + (
        f" +{len(runs) - 3}" if len(runs) > 3 else ""
    )
    output = Path(output)
    output.write_text(render_template("export", data, title), encoding="utf-8")
    return output
