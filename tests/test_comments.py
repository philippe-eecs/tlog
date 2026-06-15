from pathlib import Path

from tlog import comments


def test_add_load_resolve(tmp_path):
    cid = comments.add(tmp_path, {"type": "run", "run": "p/a"}, "hi", author="human")
    cs = comments.load(tmp_path)
    assert len(cs) == 1
    assert cs[0].id == cid and cs[0].status == "open" and cs[0].text == "hi"

    assert comments.resolve(tmp_path, cid, note="done")
    cs = comments.load(tmp_path)
    assert cs[0].status == "resolved" and cs[0].resolution == "done"
    assert not comments.resolve(tmp_path, "zzzzzz")  # unknown id


def test_select_filters(tmp_path):
    comments.add(tmp_path, {"type": "doc", "path": "r.md", "anchor": "A"}, "x")
    rid = comments.add(tmp_path, {"type": "run", "run": "p/a"}, "y")
    comments.resolve(tmp_path, rid)
    allc = comments.load(tmp_path)
    assert len(comments.select(allc, status="open")) == 1
    assert len(comments.select(allc, status="resolved")) == 1
    assert len(comments.select(allc, doc="r.md")) == 1
    assert len(comments.select(allc, run="p/a")) == 1


def test_parse_target():
    assert comments.parse_target("doc:r.md#Results") == {
        "type": "doc", "path": "r.md", "anchor": "Results"}
    assert comments.parse_target("run:p/a@500:eval/x") == {
        "type": "run", "run": "p/a", "step": 500, "key": "eval/x"}
    assert comments.parse_target("file:foo.py:42") == {
        "type": "file", "path": "foo.py", "line": 42}
    assert comments.parse_target("notes.md")["type"] == "doc"


def test_review_scaffold_roundtrip():
    doc = "# Title\n\n## Results\nbody\n\n## Next steps\n"
    assert comments.doc_headings(doc) == ["Title", "Results", "Next steps"]
    scaffold = comments.review_scaffold(Path("r.md"), doc)
    filled = (scaffold
              .replace("## [Results]\n", "## [Results]\nlooks good\n")
              .replace("## [general]\n", "## [general]\nship it\n"))
    anchors = dict(comments.parse_scaffold(filled))
    assert anchors["Results"] == "looks good"
    assert anchors["general"] == "ship it"
    assert "Title" not in anchors  # untouched sections are dropped
