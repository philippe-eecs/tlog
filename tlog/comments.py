"""Unified, append-only comment store — the human↔agent feedback loop.

One file, `<root>/.tlog/comments.jsonl`, holds every comment regardless of what
it targets: a section/line of a review doc, a run (optionally a step/media key),
or a line of source being reviewed. Comments are added from the CLI, from the
live TUI (key `c`), or by an agent; an agent reads open ones with
`tlog comments --json` and closes them with `tlog resolve`.

    {"id":"a1b2c3","_ts":..,"author":"human","status":"open","text":"...",
     "target":{"type":"doc","path":"review.md","anchor":"Results"}}
    {"_ts":..,"resolves":"a1b2c3","by":"claude","note":"fixed"}   # resolution

Folding the log by id (last status wins) gives the current state.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import writer

DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
_C_OPEN = "\x1b[38;5;209m"
_C_DONE = "\x1b[38;5;114m"


def comments_path(root: str | Path) -> Path:
    return Path(root).expanduser() / ".tlog" / "comments.jsonl"


@dataclass
class Comment:
    id: str
    ts: float
    author: str
    status: str
    text: str
    target: dict = field(default_factory=dict)
    resolution: str = ""

    def target_str(self) -> str:
        t = self.target
        kind = t.get("type", "?")
        if kind == "doc":
            a = f"#{t['anchor']}" if t.get("anchor") else ""
            return f"doc:{t.get('path', '?')}{a}"
        if kind == "run":
            step = f"@{t['step']}" if t.get("step") is not None else ""
            key = f":{t['key']}" if t.get("key") else ""
            return f"run:{t.get('run', '?')}{step}{key}"
        if kind == "file":
            line = f":{t['line']}" if t.get("line") is not None else ""
            return f"file:{t.get('path', '?')}{line}"
        return kind


# -- read/write ----------------------------------------------------------------


def _append(root: str | Path, rec: dict) -> None:
    path = comments_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(writer.dumps(rec) + "\n")


def add(root: str | Path, target: dict, text: str, author: str = "human") -> str:
    cid = uuid.uuid4().hex[:6]
    _append(root, {
        "id": cid, "_ts": time.time(), "author": author,
        "status": "open", "text": text, "target": target,
    })
    return cid


def resolve(root: str | Path, cid: str, note: str = "", by: str = "claude") -> bool:
    existing = {c.id for c in load(root)}
    if cid not in existing:
        return False
    _append(root, {"_ts": time.time(), "resolves": cid, "by": by, "note": note})
    return True


def load(root: str | Path) -> list[Comment]:
    """All comments, folded (last status wins), in creation order."""
    path = comments_path(root)
    if not path.is_file():
        return []
    by_id: dict[str, Comment] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "resolves" in rec:
            c = by_id.get(rec["resolves"])
            if c:
                c.status = "resolved"
                c.resolution = rec.get("note", "")
            continue
        if "id" not in rec:
            continue
        if rec["id"] not in by_id:
            order.append(rec["id"])
        by_id[rec["id"]] = Comment(
            id=rec["id"], ts=rec.get("_ts", 0.0), author=rec.get("author", "?"),
            status=rec.get("status", "open"), text=rec.get("text", ""),
            target=rec.get("target", {}),
        )
    return [by_id[i] for i in order]


def select(
    comments: list[Comment],
    *,
    doc: str | None = None,
    run: str | None = None,
    status: str | None = None,
) -> list[Comment]:
    out = comments
    if status:
        out = [c for c in out if c.status == status]
    if doc:
        out = [c for c in out if c.target.get("type") == "doc"
               and Path(c.target.get("path", "")).name == Path(doc).name]
    if run:
        out = [c for c in out if c.target.get("type") == "run"
               and run in c.target.get("run", "")]
    return out


# -- CLI target parsing --------------------------------------------------------


def parse_target(spec: str, root: str | Path = "runs") -> dict:
    """Parse a `tlog comment <target>` spec into a target dict.

    doc:PATH[#ANCHOR] · run:PROJ/NAME[@STEP][:KEY] · file:PATH[:LINE]
    Bare specs are inferred (a .md path → doc, a resolvable run → run, else file)."""
    if spec.startswith("doc:"):
        path, _, anchor = spec[4:].partition("#")
        return {"type": "doc", "path": path, **({"anchor": anchor} if anchor else {})}
    if spec.startswith("run:"):
        return _run_target(spec[4:])
    if spec.startswith("file:"):
        return _file_target(spec[5:])
    if Path(spec).suffix.lower() in (".md", ".markdown"):
        return {"type": "doc", "path": spec}
    from .store import resolve_run

    base = spec.split("@")[0].split(":")[0]
    if resolve_run(base, root):
        return _run_target(spec)
    return _file_target(spec)


def _run_target(rest: str) -> dict:
    rest, _, key = rest.partition(":")
    run, _, step = rest.partition("@")
    t = {"type": "run", "run": run}
    if step.isdigit():
        t["step"] = int(step)
    if key:
        t["key"] = key
    return t


def _file_target(rest: str) -> dict:
    path, _, line = rest.rpartition(":")
    if path and line.isdigit():
        return {"type": "file", "path": path, "line": int(line)}
    return {"type": "file", "path": rest}


# -- $EDITOR -------------------------------------------------------------------


def edit_text(initial: str = "", suffix: str = ".md") -> str:
    """Open $VISUAL/$EDITOR on a temp file seeded with `initial`; return the
    saved contents (empty string if the editor exits non-zero)."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(initial)
        tmp = f.name
    try:
        rc = subprocess.call(shlex.split(editor) + [tmp])
        if rc != 0:
            return ""
        return Path(tmp).read_text(encoding="utf-8")
    finally:
        os.unlink(tmp)


# -- review-doc scaffold (section-anchored comments) ---------------------------

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def doc_headings(text: str) -> list[str]:
    out = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _HEADING.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def review_scaffold(doc_path: Path, doc_text: str) -> str:
    lines = [
        f"# Comments on {doc_path.name}",
        "# Type feedback under any section heading below. Empty sections are",
        "# ignored. Lines starting with '#' are ignored. Save & quit when done.",
        "",
    ]
    for h in doc_headings(doc_text):
        lines += [f"## [{h}]", ""]
    lines += ["## [general]", ""]
    return "\n".join(lines)


def parse_scaffold(text: str) -> list[tuple[str, str]]:
    """Return [(anchor, body)] for sections that got non-comment content."""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = re.match(r"^##\s+\[(.*)\]\s*$", line)
        if m:
            sections.append((m.group(1).strip(), []))
        elif sections and not line.lstrip().startswith("#"):
            sections[-1][1].append(line)
    out = []
    for anchor, body in sections:
        joined = "\n".join(body).strip()
        if joined:
            out.append((anchor, joined))
    return out


def comment_on_doc(root: str | Path, doc_path: Path, author: str = "human") -> int:
    """Open the editor scaffold for a review doc; persist each filled section."""
    doc_path = Path(doc_path)
    text = doc_path.read_text(encoding="utf-8")
    edited = edit_text(review_scaffold(doc_path, text), suffix=".md")
    n = 0
    for anchor, body in parse_scaffold(edited):
        add(root, {"type": "doc", "path": str(doc_path), "anchor": anchor}, body, author)
        n += 1
    return n


# -- display -------------------------------------------------------------------


def format_list(comments: list[Comment], color: bool = True) -> str:
    if not comments:
        return f"{DIM}no comments{RESET}" if color else "no comments"
    out = []
    for c in comments:
        mark = (_C_DONE + "✓" if c.status == "resolved" else _C_OPEN + "○") + RESET
        head = f"{mark} {BOLD}{c.id}{RESET} {DIM}{c.author} · {c.target_str()}{RESET}"
        if not color:
            head = f"[{'x' if c.status == 'resolved' else ' '}] {c.id} {c.author} {c.target_str()}"
        out.append(head)
        for ln in c.text.splitlines():
            out.append(f"    {ln}")
        if c.status == "resolved" and c.resolution:
            out.append(f"    {DIM}↳ {c.resolution}{RESET}")
    return "\n".join(out)


def to_json(comments: list[Comment]) -> str:
    return json.dumps([
        {"id": c.id, "ts": c.ts, "author": c.author, "status": c.status,
         "text": c.text, "target": c.target, "resolution": c.resolution}
        for c in comments
    ], indent=2)
