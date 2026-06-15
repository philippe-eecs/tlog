import json

import pytest

import tlog
from tlog.run import Run
from tlog.store import MetricsReader, find_runs


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("RANK", "SLURM_JOB_ID", "SLURM_RESTART_COUNT", "TLOG_DIR"):
        monkeypatch.delenv(var, raising=False)


def make_run(tmp_path, **kw):
    kw.setdefault("capture_console", False)
    kw.setdefault("system_metrics", False)
    return Run(project="t", dir=tmp_path, **kw)


def test_run_creates_files_and_logs(tmp_path):
    run = make_run(tmp_path, name="exp", config={"lr": 1e-3})
    run.log({"loss/total": 1.0}, step=10)
    run.log({"loss/total": 0.5}, step=20)
    run.finish()

    assert (run.dir / "meta.json").is_file()
    assert json.loads((run.dir / "config.json").read_text()) == {"lr": 0.001}
    lines = (run.dir / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == pytest.approx(
        {"_step": 10, "_ts": json.loads(lines[0])["_ts"], "loss/total": 1.0}
    )
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["state"] == "finished"
    assert meta["env"]["hostname"]


def test_auto_step_increments(tmp_path):
    run = make_run(tmp_path)
    run.log({"a": 1.0})
    run.log({"a": 2.0})
    run.log({"a": 3.0}, step=100)
    run.log({"a": 4.0})
    run.finish()
    steps = [json.loads(l)["_step"] for l in (run.dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == [0, 1, 100, 101]


def test_rank_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    run = tlog.init(project="t", dir=str(tmp_path))
    run.log({"a": 1})  # absorbed
    run.finish()
    assert isinstance(run, tlog.NoopRun)
    assert not list(tmp_path.iterdir())


def test_resume_on_slurm_requeue(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    first = make_run(tmp_path, name="job")
    first.log({"loss": 1.0}, step=10)
    first.log({"loss": 0.9}, step=20)
    first.finish()

    # SLURM requeue of the same job -> same run dir, restart recorded
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    resumed = make_run(tmp_path, name="job")
    assert resumed.resumed
    assert resumed.dir == first.dir
    # training restarted from the step-10 checkpoint: re-logs step 20
    resumed.log({"loss": 0.123}, step=20)
    resumed.finish()

    meta = json.loads((resumed.dir / "meta.json").read_text())
    assert len(meta["restarts"]) == 1

    # read side dedups keep-last: step 20 shows the re-logged value
    info = find_runs(resumed.dir)[0]
    reader = MetricsReader(info)
    reader.refresh()
    steps, values = reader.series["loss"].points()
    assert steps == [10, 20]
    assert values[1] == 0.123


def test_reattach_by_name(tmp_path):
    first = make_run(tmp_path, name="exp")
    first.log({"loss": 1.0}, step=0)
    first.finish()
    # default: rerunning the same name re-attaches to the same dir
    again = make_run(tmp_path, name="exp")
    assert again.resumed and again.dir == first.dir
    again.finish()
    assert len(find_runs(tmp_path)) == 1


def test_new_forces_parallel_run(tmp_path):
    first = make_run(tmp_path, name="exp")
    first.finish()
    forked = make_run(tmp_path, name="exp", new=True)
    assert not forked.resumed and forked.dir != first.dir
    forked.finish()
    assert len(find_runs(tmp_path)) == 2


def test_resume_never(tmp_path):
    first = make_run(tmp_path, name="exp")
    first.finish()
    fresh = make_run(tmp_path, name="exp", resume="never")
    assert not fresh.resumed and fresh.dir != first.dir
    fresh.finish()


def test_reset_wipes_prior_data(tmp_path):
    first = make_run(tmp_path, name="exp")
    first.log({"loss": 1.0}, step=0)
    first.log({"loss": 0.5}, step=1)
    first.finish()
    assert (first.dir / "metrics.jsonl").read_text().strip()
    again = make_run(tmp_path, name="exp", reset=True)
    assert again.resumed and again.dir == first.dir
    # prior metrics wiped, then fresh logging appends
    assert (first.dir / "metrics.jsonl").read_text() == ""
    again.log({"loss": 2.0}, step=0)
    again.finish()
    steps = [json.loads(l)["_step"] for l in
             (first.dir / "metrics.jsonl").read_text().splitlines()]
    assert steps == [0]


def test_group_tag_recorded_and_kept(tmp_path):
    first = make_run(tmp_path, name="exp", group="sweepA")
    first.finish()
    assert json.loads((first.dir / "meta.json").read_text())["group"] == "sweepA"
    # re-attach without group keeps it; the run object reflects the stored group
    again = make_run(tmp_path, name="exp")
    assert again.group == "sweepA"
    again.finish()


def test_resume_by_explicit_id(tmp_path):
    first = make_run(tmp_path, name="job")
    rid = first.id
    first.finish()
    again = make_run(tmp_path, id=rid, resume="must")
    assert again.resumed and again.dir == first.dir
    again.finish()


def test_resume_must_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        make_run(tmp_path, resume="must", id="nonexistent")


def test_log_images_raw_tuple(tmp_path):
    run = make_run(tmp_path)
    pixels = bytes([255, 0, 0] * 4)  # 2x2 red
    run.log_images("eval/recon", [(pixels, 2, 2, 3)], step=5, caption="hi")
    run.finish()
    rec = json.loads((run.dir / "media" / "index.jsonl").read_text())
    assert rec["_step"] == 5 and rec["caption"] == "hi"
    png = (run.dir / "media" / rec["files"][0]).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_console_capture(tmp_path):
    run = Run(project="t", dir=tmp_path, capture_console=True, system_metrics=False)
    print("hello from training")
    run.finish()
    assert "hello from training" in (run.dir / "console.log").read_text()


def test_meta_slurm_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-4]")
    run = make_run(tmp_path)
    run.finish()
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["env"]["slurm"]["SLURM_JOB_ID"] == "777"
    assert meta["env"]["slurm"]["SLURM_JOB_NODELIST"] == "node[1-4]"
