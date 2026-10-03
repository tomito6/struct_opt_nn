"""Tests for the unattended-training plumbing behind the time-boxed scripts.

What is covered is what a script launched at midnight relies on without
anybody watching: the argument parsing that decides *when* it starts, the
wait for another run's process, the log tee, and the specs it writes before
waiting. The trainer itself is not run here - the scripts are smoke-tested
with ``--epochs 1`` before every real launch, and the trainer has the
library's own tests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import subprocess
import sys

import pytest

from structsept.app import datasets, hyperparams, unattended

# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def test_parse_start_now_is_now():
    before = dt.datetime.now()
    got = unattended.parse_start(" NOW ")
    assert before <= got <= dt.datetime.now()


def test_parse_start_clock_time_is_today_or_tomorrow():
    now = dt.datetime.now()
    past = (now - dt.timedelta(hours=1)).strftime("%H:%M")
    got = unattended.parse_start(past)
    # The next occurrence of that clock time: ~23 h ahead, whichever date
    # that falls on (between 00:00 and 01:00 it is still today).
    assert dt.timedelta(hours=22, minutes=59) < got - now <= dt.timedelta(hours=23)
    assert (got.hour, got.minute) == tuple(int(x) for x in past.split(":"))

    future = (now + dt.timedelta(hours=1)).strftime("%H:%M")
    got = unattended.parse_start(future)
    assert now < got <= now + dt.timedelta(hours=1, seconds=1)


def test_parse_start_full_date():
    assert unattended.parse_start("2026-09-29 03:15") == dt.datetime(2026, 9, 29, 3, 15)
    assert unattended.parse_start("2026-09-29T03:15") == dt.datetime(2026, 9, 29, 3, 15)


@pytest.mark.parametrize("bad", ["tomorrow", "25:00", "3pm", ""])
def test_parse_start_rejects_nonsense(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        unattended.parse_start(bad)


def test_parse_pid_takes_a_number_or_a_pid_file(tmp_path):
    assert unattended.parse_pid(" 4242 ") == 4242
    pid_file = tmp_path / "run.pid"
    pid_file.write_text("25528\n", encoding="utf-8")
    assert unattended.parse_pid(str(pid_file)) == 25528
    pid_file.write_text("not a pid", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="does not hold"):
        unattended.parse_pid(str(pid_file))
    with pytest.raises(argparse.ArgumentTypeError, match="process id"):
        unattended.parse_pid(str(tmp_path / "missing.pid"))


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def test_tee_writes_to_every_stream_and_skips_none():
    a, b = io.StringIO(), io.StringIO()
    tee = unattended.Tee(a, None, b)
    tee.write("hello\n")
    tee.flush()
    assert a.getvalue() == b.getvalue() == "hello\n"


def test_tee_survives_a_closed_stream():
    closed = io.StringIO()
    closed.close()
    kept = io.StringIO()
    unattended.Tee(closed, kept).write("still here")
    assert kept.getvalue() == "still here"


def test_open_log_tees_and_writes_the_pid(tmp_path, monkeypatch):
    real_out, real_err = sys.stdout, sys.stderr
    try:
        log_file = unattended.open_log(tmp_path, "job")
        try:
            print("to both")
        finally:
            sys.stdout, sys.stderr = real_out, real_err
            log_file.close()
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    assert (tmp_path / "job.log").read_text(encoding="utf-8") == "to both\n"
    import os

    assert (tmp_path / "job.pid").read_text(encoding="utf-8") == str(os.getpid())


# --------------------------------------------------------------------------- #
# waiting for a process
# --------------------------------------------------------------------------- #


def test_wait_for_process_returns_when_it_exits():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
    said = []
    try:
        assert unattended.process_alive(child.pid)
        unattended.wait_for_process(child.pid, log=said.append, poll=0.1)
    finally:
        child.wait(timeout=10)
    assert not unattended.process_alive(child.pid)
    assert said[0].startswith("waiting for process") and said[-1].endswith("finished")


def test_wait_for_process_returns_at_once_for_a_dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    said = []
    started = dt.datetime.now()
    unattended.wait_for_process(child.pid, log=said.append, poll=5.0)
    assert dt.datetime.now() - started < dt.timedelta(seconds=2)
    assert said == [f"process {child.pid} is not running - nothing to wait for"]
    assert not unattended.process_alive(0)


# --------------------------------------------------------------------------- #
# preparing a run
# --------------------------------------------------------------------------- #


def _fake_dataset(root, name="fake", names=("a", "b", "c", "d")):
    """A dataset ``list_datasets`` accepts: empty .npz files and a manifest."""
    folder = root / datasets.SDF_SAMPLES_DIR / name / "plate"
    folder.mkdir(parents=True)
    for n in names:
        (folder / f"{n}.npz").write_bytes(b"")
    (folder.parent / datasets.MANIFEST_NAME).write_text(
        json.dumps({"geom_dimension": 2}), encoding="utf-8"
    )
    splits = root / datasets.SPLITS_DIR
    splits.mkdir()
    (splits / f"{name}.json").write_text(
        json.dumps({name: {"plate": list(names)}}), encoding="utf-8"
    )
    return name


def test_prepare_run_writes_specs_and_clamps_checkpoints(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    name = _fake_dataset(data_root)
    hp = hyperparams.defaults()
    hp.update(
        num_epochs=2, log_frequency=10, snapshot_frequency=100, scenes_per_batch=2
    )
    said = []
    row = unattended.prepare_run(
        tmp_path / "runs" / "r", data_root, name, hp, log=said.append
    )

    assert row["n_instances"] == 4 and row["geom_dimension"] == 2
    specs = json.loads((tmp_path / "runs" / "r" / "specs.json").read_text())
    assert specs["NumEpochs"] == 2
    # a 2-epoch test run must still end with a latest.pth to load
    assert specs["LogFrequency"] == 2 and specs["SnapshotFrequency"] == 2
    meta = json.loads((tmp_path / "runs" / "r" / "metadata.json").read_text())
    assert meta["dataset"] == name and meta["preset"] is False
    assert said[-1].startswith("run r: 4 shapes, geom 2-D")


def test_prepare_run_refuses_a_trained_run_unless_forced(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    name = _fake_dataset(data_root)
    run_dir = tmp_path / "runs" / "r"
    (run_dir / "ModelParameters").mkdir(parents=True)
    (run_dir / "ModelParameters" / "latest.pth").write_bytes(b"")
    hp = hyperparams.defaults()
    with pytest.raises(SystemExit, match="already holds"):
        unattended.prepare_run(run_dir, data_root, name, hp, log=lambda s: None)
    unattended.prepare_run(run_dir, data_root, name, hp, force=True, log=lambda s: None)
    assert (run_dir / "specs.json").is_file()


def test_prepare_run_names_the_missing_dataset_and_how_to_build_it(tmp_path):
    with pytest.raises(SystemExit) as info:
        unattended.prepare_run(
            tmp_path / "r",
            tmp_path,
            "nope",
            hyperparams.defaults(),
            log=lambda s: None,
            build_hint="make it so",
        )
    assert "nope" in str(info.value) and "make it so" in str(info.value)


def test_split_names_follow_the_split_order(tmp_path):
    split = tmp_path / "s.json"
    split.write_text(json.dumps({"ds": {"cls": ["z", "y"], "other": ["x"]}}))
    assert unattended.split_names(split) == ["z", "y", "x"]
    assert unattended.load_codes(tmp_path) is None
    assert unattended.last_epoch(tmp_path) is None


# --------------------------------------------------------------------------- #
# a run copied from another run's specs
# --------------------------------------------------------------------------- #


def test_prepare_run_like_copies_the_recipe_and_replaces_only_the_paths(tmp_path):
    """The reference is a preset on one dataset; the copy trains another one
    with the same specs, and only the data keys and the description differ."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    ref_name = _fake_dataset(data_root, "ref")
    # A second dataset in the same root (the helper makes the splits folder).
    folder = data_root / datasets.SDF_SAMPLES_DIR / "new" / "plate"
    folder.mkdir(parents=True)
    for n in ("p", "q", "r"):
        (folder / f"{n}.npz").write_bytes(b"")
    (folder.parent / datasets.MANIFEST_NAME).write_text(
        json.dumps({"geom_dimension": 2}), encoding="utf-8"
    )
    (data_root / datasets.SPLITS_DIR / "new.json").write_text(
        json.dumps({"new": {"plate": ["p", "q", "r"]}}), encoding="utf-8"
    )
    hp = hyperparams.defaults()
    hp.update(num_epochs=50, latent_dim=1, scenes_per_batch=2)
    like_dir = tmp_path / "runs" / "preset"
    unattended.prepare_run(like_dir, data_root, ref_name, hp, log=lambda s: None)

    said = []
    row, changed = unattended.prepare_run_like(
        tmp_path / "runs" / "copy",
        like_dir,
        data_root,
        "new",
        "the copy",
        log=said.append,
    )
    assert row["name"] == "new" and row["n_instances"] == 3
    assert changed == ["Description", "TestSplit", "TrainSplit"]
    reference = json.loads((like_dir / "specs.json").read_text())
    specs = json.loads((tmp_path / "runs" / "copy" / "specs.json").read_text())
    assert specs["Description"] == "the copy"
    assert specs["TrainSplit"].endswith("new.json")
    for key in reference:
        if key not in unattended.REPLACED_KEYS:
            assert specs[key] == reference[key], key
    meta = json.loads((tmp_path / "runs" / "copy" / "metadata.json").read_text())
    assert meta["dataset"] == "new" and meta["preset"] is False
    assert said[0].startswith("specs.json differs from preset in: Description")

    # A smoke test may shorten the run; the save intervals follow.
    _, changed = unattended.prepare_run_like(
        tmp_path / "runs" / "smoke",
        like_dir,
        data_root,
        "new",
        "smoke",
        epochs=2,
        log=lambda s: None,
    )
    smoke = json.loads((tmp_path / "runs" / "smoke" / "specs.json").read_text())
    assert smoke["NumEpochs"] == 2
    assert smoke["LogFrequency"] == 2 and smoke["SnapshotFrequency"] == 2
    assert "NumEpochs" in changed

    # But the real run refuses a recipe that differs in anything else.
    reference["NumEpochs"] = 60
    (like_dir / "specs.json").write_text(json.dumps(reference), encoding="utf-8")
    (tmp_path / "runs" / "copy" / "ModelParameters").mkdir()
    (tmp_path / "runs" / "copy" / "ModelParameters" / "latest.pth").write_bytes(b"")
    with pytest.raises(SystemExit, match="already holds"):
        unattended.prepare_run_like(
            tmp_path / "runs" / "copy", like_dir, data_root, "new", "x", log=print
        )
    with pytest.raises(SystemExit, match="no readable specs"):
        unattended.prepare_run_like(
            tmp_path / "runs" / "x", tmp_path / "nowhere", data_root, "new", "x"
        )
    with pytest.raises(SystemExit) as info:
        unattended.prepare_run_like(
            tmp_path / "runs" / "x",
            like_dir,
            data_root,
            "nope",
            "x",
            build_hint="make it so",
        )
    assert "nope" in str(info.value) and "make it so" in str(info.value)


# --------------------------------------------------------------------------- #
# the codes against the generating parameter
# --------------------------------------------------------------------------- #


def _fake_run_with_codes(tmp_path, codes, names, values, param):
    """A run directory holding latent codes, plus the split and params.csv."""
    import torch

    run_dir = tmp_path / "runs" / "r"
    (run_dir / "LatentCodes").mkdir(parents=True)
    torch.save(
        {"latent_codes": torch.tensor(codes, dtype=torch.float32)},
        run_dir / "LatentCodes" / "latest.pth",
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"ds": {"plate": names}}), encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    # Rows deliberately out of split order: the match goes through names.
    rows = sorted(zip(names, values), reverse=True)
    (dataset_dir / "params.csv").write_text(
        "\n".join(["name,%s,other" % param] + [f"{n},{v},0" for n, v in rows]),
        encoding="utf-8",
    )
    return run_dir, dataset_dir, split


def test_code_vs_parameter_reads_the_column_and_reports_monotonicity(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    names = ["s0", "s1", "s2", "s3", "s4"]
    h = [0.05, 0.1125, 0.175, 0.2375, 0.3]
    codes = [[0.2], [0.1], [0.0], [-0.1], [-0.2]]  # a straight line down in h
    run_dir, dataset_dir, split = _fake_run_with_codes(tmp_path, codes, names, h, "h")

    said = []
    result = unattended.code_vs_parameter(
        run_dir, dataset_dir, split, "h", label="height h", log=said.append
    )
    assert result["parameter"] == "h" and result["latent_dim"] == 1
    stats = result["components"][0]
    assert stats["pearson"] == pytest.approx(-1.0)
    assert stats["spearman"] == pytest.approx(-1.0)
    assert stats["monotonic"] is True
    assert (stats["min"], stats["max"]) == (pytest.approx(-0.2), pytest.approx(0.2))
    assert (run_dir / "code_vs_h.png").is_file()
    lines = (run_dir / "code_vs_h.csv").read_text().splitlines()
    assert lines[0] == "name,h,code_0"
    assert [l.split(",")[0] for l in lines[1:]] == names  # split order, not csv order
    assert said[0].startswith("code 0 vs h: Pearson -1.000, Spearman -1.000, monotonic")

    # Two components: the line lies along (1, -1); one axis carries everything.
    codes2 = [[c[0], -c[0] + 0.01 * (i % 2)] for i, c in enumerate(codes)]
    run_dir, dataset_dir, split = _fake_run_with_codes(
        tmp_path / "two", codes2, names, h, "h"
    )
    result = unattended.code_vs_parameter(run_dir, dataset_dir, split, "h", log=print)
    assert result["latent_dim"] == 2
    assert result["pc1_variance_share"] > 0.99
    assert result["pc1"]["monotonic"] is True
    assert abs(result["pc1"]["pearson"]) == pytest.approx(1.0, abs=1e-3)
    # h is a linear function of the codes, whatever the basis.
    assert result["linear_fit_r2"] == pytest.approx(1.0, abs=1e-6)
    assert len(result["linear_fit_direction"]) == 2
    assert (run_dir / "code_vs_h.png").is_file()

    # No codes saved: says so, writes nothing.
    said = []
    assert (
        unattended.code_vs_parameter(tmp_path, dataset_dir, split, "h", log=said.append)
        == {}
    )
    assert said == ["no latent codes saved, skipping the code-vs-h check"]
