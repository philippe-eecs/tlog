"""`tlog serve` — live web dashboard on a stdlib HTTP server.

Designed for SLURM clusters: run it on the login/compute node, let VS Code
Remote auto-forward the port (or `ssh -L 8585:localhost:8585 cluster`), and
open it in your laptop browser. Charts poll for new data every few seconds.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .export import render_template
from .payload import metrics_from_reader, run_media, run_summary
from .store import MetricsReader, find_runs, read_console


class _State:
    """Run discovery + per-run incremental readers, shared across requests."""

    def __init__(self, root: str):
        self.root = root
        self.lock = threading.Lock()
        self.readers: dict[str, MetricsReader] = {}
        self.index_html = render_template("serve", None).encode("utf-8")

    def runs(self):
        return find_runs(self.root)

    def find(self, run_id: str):
        for info in self.runs():
            if info.id == run_id:
                return info
        return None

    def metrics(self, info) -> dict:
        with self.lock:
            reader = self.readers.get(info.id)
            if reader is None or reader.run.path != info.path:
                reader = MetricsReader(info, include_system=True)
                self.readers[info.id] = reader
            reader.refresh()
            return metrics_from_reader(reader)


class _Handler(BaseHTTPRequestHandler):
    state: _State  # set on the server class

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            "application/json", status,
        )

    def do_GET(self):  # noqa: N802 (stdlib naming)
        try:
            self._route(unquote(urlparse(self.path).path))
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._json({"error": str(e)}, status=500)
            except Exception:
                pass

    def _route(self, path: str) -> None:
        state = self.state
        if path in ("/", "/index.html"):
            self._send(state.index_html, "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            self._json({"runs": [run_summary(r) for r in state.runs()]})
            return

        parts = [p for p in path.split("/") if p]
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "run":
            info = state.find(parts[2])
            if info is None:
                self._json({"error": "run not found"}, status=404)
                return
            if parts[3] == "metrics":
                self._json({"metrics": state.metrics(info)})
            elif parts[3] == "media":
                self._json({"media": run_media(info)})
            elif parts[3] == "console":
                text = "\n".join(read_console(info, max_lines=500))
                self._send(text.encode("utf-8", "replace"), "text/plain; charset=utf-8")
            else:
                self._json({"error": "unknown endpoint"}, status=404)
            return

        if len(parts) >= 3 and parts[0] == "media":
            info = state.find(parts[1])
            if info is not None:
                media_root = (info.path / "media").resolve()
                target = (media_root / "/".join(parts[2:])).resolve()
                if target.is_file() and target.is_relative_to(media_root):
                    self._send(target.read_bytes(), "image/png")
                    return
            self._send(b"not found", "text/plain", status=404)
            return

        self._send(b"not found", "text/plain", status=404)


def serve(root: str, host: str = "127.0.0.1", port: int = 8585) -> None:
    state = _State(root)
    handler = type("Handler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    n = len(state.runs())
    print(f"tlog: serving {n} run{'s' * (n != 1)} from {Path(root).resolve()}")
    print(f"tlog: open http://{host}:{port}")
    print("tlog: on a cluster, forward the port:  ssh -L "
          f"{port}:localhost:{port} <cluster>  (VS Code forwards it automatically)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ntlog: bye")
