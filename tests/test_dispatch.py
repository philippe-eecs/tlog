from tlog import store
from tlog.run import Run


def _mk(tmp_path, name, group=None):
    r = Run(project="demo", name=name, dir=tmp_path, group=group,
            capture_console=False, system_metrics=False)
    r.log({"loss": 1.0}, step=0)
    r.finish()
    return r


def test_resolve_runs_variants(tmp_path):
    _mk(tmp_path, "a", group="g1")
    _mk(tmp_path, "b", group="g1")
    _mk(tmp_path, "c")
    root = str(tmp_path)
    assert {r.name for r in store.resolve_runs("a", root)} == {"a"}            # single
    assert {r.name for r in store.resolve_runs("demo", root)} == {"a", "b", "c"}  # project
    assert {r.name for r in store.resolve_runs("g1", root)} == {"a", "b"}     # group
    assert store.resolve_runs("does-not-exist", root) == []


def test_saved_sets_roundtrip(tmp_path):
    _mk(tmp_path, "a")
    _mk(tmp_path, "b")
    root = str(tmp_path)
    added, total = store.link_runs(root, "cmp", ["a", "b"], note="compare a/b")
    assert (added, total) == (2, 2)
    sets = store.list_sets(root)
    assert [s["name"] for s in sets] == ["cmp"] and sets[0]["note"] == "compare a/b"
    assert {r.name for r in store.resolve_runs("cmp", root)} == {"a", "b"}
    # appending the same run is a no-op (dedup)
    added2, total2 = store.link_runs(root, "cmp", ["a"])
    assert (added2, total2) == (0, 2)


def test_sets_and_comments_not_seen_as_runs(tmp_path):
    _mk(tmp_path, "a")
    root = str(tmp_path)
    store.link_runs(root, "cmp", ["a"])
    from tlog import comments
    comments.add(root, {"type": "run", "run": "demo/a"}, "note")
    # .tlog/ must not be discovered as a run
    assert {r.name for r in store.find_runs(root)} == {"a"}
