import json

from tlog.store import (
    JsonlTail,
    Series,
    downsample,
    ema,
    find_runs,
    group_keys,
    last_record,
    read_console,
)


def test_series_keep_last_dedup():
    s = Series()
    s.add(10, 1.0)
    s.add(20, 2.0)
    s.add(10, 9.0)  # re-logged step after a resume: last write wins
    steps, values = s.points()
    assert steps == [10, 20]
    assert values == [9.0, 2.0]


def test_jsonl_tail_incremental_and_partial(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')
    tail = JsonlTail(path)
    assert [r["a"] for r in tail.read_new()] == [1, 2]
    assert tail.read_new() == []
    with open(path, "a") as f:
        f.write('{"a":3}\n{"a":4')  # last line incomplete
    assert [r["a"] for r in tail.read_new()] == [3]
    with open(path, "a") as f:
        f.write("}\n")
    assert [r["a"] for r in tail.read_new()] == [4]


def test_downsample_preserves_spikes():
    steps = list(range(1000))
    values = [1.0] * 1000
    values[500] = 100.0  # single spike must survive downsampling
    s, mean, lo, hi = downsample(steps, values, 50)
    assert len(s) <= 50
    assert max(hi) == 100.0
    assert min(lo) == 1.0


def test_downsample_noop_when_small():
    steps, values = [1, 2, 3], [1.0, 2.0, 3.0]
    s, mean, lo, hi = downsample(steps, values, 10)
    assert s == steps and mean == values


def test_ema_debiased():
    vals = [1.0, 1.0, 1.0]
    assert ema(vals, 0.9) == [1.0, 1.0, 1.0]  # constant stays constant
    assert ema([1.0, 2.0], 0.0) == [1.0, 2.0]


def test_group_keys():
    groups = group_keys(["loss/total", "loss/charb", "eval/fid", "lr"])
    assert groups["loss"] == ["loss/total", "loss/charb"]
    assert groups["eval"] == ["eval/fid"]
    assert groups["metrics"] == ["lr"]


def test_last_record_with_predicate(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"_step":1,"loss/x":3.0}\n{"_step":2,"eval/y":1.0}\n')
    assert last_record(path)["_step"] == 2
    rec = last_record(path, predicate=lambda r: any(k.startswith("loss") for k in r))
    assert rec["_step"] == 1


def test_find_runs_three_layouts(tmp_path):
    run_dir = tmp_path / "proj" / "name__20260611-000000__abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"id": "abc123", "name": "name", "project": "proj", "created_at": 5})
    )
    assert len(find_runs(tmp_path)) == 1  # root of projects
    assert len(find_runs(tmp_path / "proj")) == 1  # project dir
    assert len(find_runs(run_dir)) == 1  # run dir itself
    assert find_runs(tmp_path)[0].id == "abc123"


def test_read_console_handles_cr_and_ansi(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text("{}")
    (run_dir / "console.log").write_text(
        "plain\n10%\r20%\r\x1b[32m100%\x1b[0m done\nlast\n"
    )
    info = find_runs(run_dir)[0]
    lines = read_console(info)
    assert lines == ["plain", "100% done", "last"]
