import json
import threading

from tlog.writer import JsonlWriter, atomic_write_json, dumps


def test_jsonl_append(tmp_path):
    w = JsonlWriter(tmp_path / "m.jsonl")
    w.write({"_step": 1, "loss": 0.5})
    w.write({"_step": 2, "loss": 0.25})
    w.close()
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert [json.loads(l)["_step"] for l in lines] == [1, 2]


def test_jsonl_append_across_reopen(tmp_path):
    path = tmp_path / "m.jsonl"
    JsonlWriter(path).write({"a": 1})
    w2 = JsonlWriter(path)
    w2.write({"a": 2})
    w2.close()
    assert len(path.read_text().splitlines()) == 2


def test_jsonl_concurrent_writes(tmp_path):
    w = JsonlWriter(tmp_path / "m.jsonl")

    def work(k):
        for i in range(50):
            w.write({"t": k, "i": i})

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    w.close()
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert len(lines) == 200
    for line in lines:  # every line is valid JSON (no interleaving)
        json.loads(line)


def test_atomic_write_json(tmp_path):
    path = tmp_path / "meta.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert not list(tmp_path.glob(".*tmp*"))


def test_dumps_handles_weird_values(tmp_path):
    class FakeTensor:
        def item(self):
            return 0.5

    out = json.loads(dumps({"v": FakeTensor(), "p": tmp_path}))
    assert out["v"] == 0.5
    assert out["p"] == str(tmp_path)
