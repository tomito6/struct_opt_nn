"""Is the ghost hole of the free-hole plate decoder a matter of the seed?

Why
---
``runs/plate_hole_2d_xyr_d3_20260930_0124`` (d = 3, 8 x 256, 800 epochs)
decodes 27 of its 134 *training* shapes with a second, spurious hole in the
middle of the material (audit of 2026-10-02, ``IDEIAS.md``). The data is not
the cause: every shape on disk has the exact SDF of one hole, and the d = 2
run on the byte-identical dataset draws one hole everywhere. Every d = 3
checkpoint ever made - 300, 680 and 800 epochs - is one trajectory with
seed 42, so whether another initialisation draws the ghost too has never
been seen. This trains that other initialisation, and nothing else.

What is trained
---------------
The ``specs.json`` of the 800-epoch run, copied key for key
(``unattended.prepare_run_like``), with three keys changed on purpose:

* ``seed`` - the experiment (default 1, was 42);
* ``NumEpochs`` - 150 instead of 800, where the trajectory is cut;
* ``SnapshotFrequency`` - 50 instead of 200, so there is a decoder to look
  at on epochs 50, 100 and 150 (epoch 1 is kept by ``AdditionalSnapshots``).

The last two do not change what is trained up to the cut. The trainer reads
``NumEpochs`` only to know when to stop and where to save; the step learning
rate halves every ``Interval`` epochs whatever the total, and the code
regulariser ramps with the epoch, not with the epoch count. So epochs 1-150
here are epochs 1-150 of the full 800-epoch recipe, seed apart - the same
argument that makes ``plate2d_xyr_d3_8x256_4h`` (300 epochs) and the
800-epoch run one trajectory, which the audit checked loss for loss.

What it is compared with
------------------------
The seed-42 trajectory, counted with ``check_reconstruction.py`` on the
snapshots of ``plate2d_xyr_d3_8x256_4h`` (:data:`SEED42_GHOSTS`): 0 shapes
with a ghost on epoch 1, 1 on 50, 35 on 100, 20 on 150 - and 27 on 800. The
ghosts are there by epoch 100 and later epochs do not remove them, so 150
epochs are enough to tell.

After training, every snapshot of the new run goes through
``experiments/check_reconstruction.py --checkpoint <epoch>`` (run as a
separate process: scripts in ``experiments/`` are not imported), and the two
trajectories are printed side by side and written to
``ghost_trajectory.json`` in the run directory.

How it runs
-----------
Started once and left alone, like the other ``train_plate_*`` scripts::

    uv run python experiments/train_plate_xyr_seed.py
    uv run python experiments/train_plate_xyr_seed.py --seed 7
    uv run python experiments/train_plate_xyr_seed.py --check-only

It logs to ``outputs/logs/<run>.log`` with the process id in ``<run>.pid``
beside it (``taskkill /PID <pid> /F`` stops it). Measured on this machine,
CPU only, 134 shapes: 33-35 s per epoch with the laptop idle, up to ~50 s in
use - 1.4 to 2.1 h for 150 epochs, plus about 6 min for the checks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time

# Both must be set before torch / DeepSDFStruct are imported: tqdm reads its
# switch at import time, matplotlib picks its backend on first use.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from structsept.app import unattended
from structsept.app.unattended import say

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = "plate_hole_2d_xyr"
LIKE = "plate_hole_2d_xyr_d3_20260930_0124"
LOG_DIR = ROOT / "outputs" / "logs"
CHECK_SCRIPT = ROOT / "experiments" / "check_reconstruction.py"
CHECK_DIR = ROOT / "outputs" / "reconstruction"

#: Training shapes with a ghost hole along the seed-42 trajectory, by epoch.
#: 1-150 from the snapshots of ``plate2d_xyr_d3_8x256_4h``, 800 from
#: ``plate_hole_2d_xyr_d3_20260930_0124`` (audit 2026-10-02; epoch 100
#: recounted 2026-10-04 with ``--checkpoint 100``: 35).
SEED42_GHOSTS = {1: 0, 50: 1, 100: 35, 150: 20, 800: 27}
#: The only specs keys this run may differ in from the reference: the three
#: of the module docstring plus what ``prepare_run_like`` rewrites anyway.
ALLOWED_CHANGES = ("seed", "NumEpochs", "SnapshotFrequency") + tuple(
    unattended.REPLACED_KEYS
)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def prepare_run(run_dir, like_dir, data_root, seed, epochs, every, force=False):
    """Write the reference recipe with another seed, cut at ``epochs``.

    ``unattended.prepare_run_like`` copies every key and handles the epoch
    cut; the seed and the snapshot interval are set on top, and the keys that
    end up different from the reference are printed once more, so the log
    shows exactly what this run changed.
    """
    from structsept.app import training

    like_dir = pathlib.Path(like_dir)
    description = (
        f"Seed check of the ghost hole: the recipe of {like_dir.name} "
        f"(specs.json copied key for key) with seed {seed} instead of 42, "
        f"cut at epoch {epochs} with a snapshot every {every} epochs. The "
        f"first {epochs} epochs are those of the full recipe. Started by "
        f"experiments/train_plate_xyr_seed.py; see ghost_trajectory.json in "
        f"the run directory."
    )
    unattended.prepare_run_like(
        run_dir,
        like_dir,
        data_root,
        DATASET,
        description,
        epochs=epochs,
        force=force,
    )
    path = pathlib.Path(run_dir) / "specs.json"
    specs = training.read_specs(path)
    specs["seed"] = int(seed)
    specs["SnapshotFrequency"] = int(every)
    path.write_text(json.dumps(specs, indent=4), encoding="utf-8")

    reference = training.read_specs(like_dir / "specs.json")
    changed = sorted(
        k for k in set(specs) | set(reference) if specs.get(k) != reference.get(k)
    )
    for key in changed:
        if key != "Description":
            say(f"  {key}: {reference.get(key)} -> {specs.get(key)}")
    if not set(changed) <= set(ALLOWED_CHANGES):
        sys.exit(
            "refusing to train: the recipe itself would differ in "
            + ", ".join(sorted(set(changed) - set(ALLOWED_CHANGES)))
        )


# --------------------------------------------------------------------------- #
# the ghost count along the trajectory
# --------------------------------------------------------------------------- #


def snapshot_epochs(run_dir) -> list[int]:
    """Epochs the run kept a decoder for, ascending (``latest`` left out)."""
    folder = pathlib.Path(run_dir) / "ModelParameters"
    return sorted(int(p.stem) for p in folder.glob("*.pth") if p.stem.isdigit())


def count_ghosts(run_dir, epoch, res=200) -> dict | None:
    """Run ``check_reconstruction.py`` on one snapshot and read its csv.

    Returns ``{"ghost_shapes", "missing_shapes", "shapes"}`` or ``None`` when
    the check failed (its output is in the log either way).
    """
    run = pathlib.Path(run_dir).name
    path = CHECK_DIR / f"{run}_ep{epoch}.csv"
    path.unlink(missing_ok=True)  # never read a csv left by an earlier check
    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--run",
            run,
            "--runs",
            str(pathlib.Path(run_dir).parent),
            "--checkpoint",
            str(epoch),
            "--res",
            str(res),
        ],
        capture_output=True,
        text=True,
    )
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip() and not line.startswith("per-shape numbers"):
            say(f"  {line}")
    if result.returncode != 0 or not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "ghost_shapes": sum(int(r["ghost_voids"]) > 0 for r in rows),
        "missing_shapes": sum(int(r["missing_voids"]) > 0 for r in rows),
        "shapes": len(rows),
    }


def ghost_trajectory(run_dir, seed, res=200) -> dict:
    """Ghost count on every snapshot, next to seed 42; written to the run."""
    say(f"ghost holes along the trajectory of {pathlib.Path(run_dir).name}:")
    counts = {}
    for epoch in snapshot_epochs(run_dir):
        counts[epoch] = count_ghosts(run_dir, epoch, res)
    say(f"{'epoch':>6s}  {'seed 42':>8s}  {f'seed {seed}':>8s}")
    for epoch, c in counts.items():
        ref = SEED42_GHOSTS.get(epoch)
        new = "failed" if c is None else f"{c['ghost_shapes']}/{c['shapes']}"
        say(f"{epoch:>6d}  {'-' if ref is None else ref:>8}  {new:>8s}")
    result = {
        "seed": int(seed),
        "seed42_ghost_shapes": {str(k): v for k, v in SEED42_GHOSTS.items()},
        "ghost_shapes": {
            str(k): (None if c is None else c["ghost_shapes"])
            for k, c in counts.items()
        },
        "missing_shapes": {
            str(k): (None if c is None else c["missing_shapes"])
            for k, c in counts.items()
        },
        "grid_res": int(res),
    }
    (pathlib.Path(run_dir) / "ghost_trajectory.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    what = parser.add_argument_group("what")
    what.add_argument(
        "--seed", type=int, default=1, help="the seed to try (default: 1)"
    )
    what.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="where the trajectory is cut (default: 150)",
    )
    what.add_argument(
        "--every",
        type=int,
        default=50,
        help="snapshot interval in epochs (default: 50)",
    )
    what.add_argument(
        "--run",
        default=None,
        help="run directory name (default: plate_hole_2d_xyr_d3_seed<seed>_ep<epochs>)",
    )
    what.add_argument(
        "--like",
        default=LIKE,
        help=f"run whose specs.json is the recipe (default: {LIKE})",
    )
    what.add_argument("--force", action="store_true", help="train over an existing run")
    what.add_argument(
        "--check-only",
        action="store_true",
        help="skip training; count the ghosts on the snapshots of the existing --run",
    )
    what.add_argument(
        "--res", type=int, default=200, help="grid nodes per axis of the check"
    )
    where = parser.add_argument_group("where")
    where.add_argument("--data-root", default=str(ROOT / "data"))
    where.add_argument("--runs-dir", default=str(ROOT / "runs"))
    where.add_argument("--log-dir", default=str(LOG_DIR))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.every < 1:
        parser.error("--epochs and --every must be at least 1")
    if args.seed == 42:
        parser.error("seed 42 is the trajectory this is compared with")
    args.run = args.run or f"plate_hole_2d_xyr_d3_seed{args.seed}_ep{args.epochs}"
    run_dir = pathlib.Path(args.runs_dir) / args.run
    like_dir = pathlib.Path(args.runs_dir) / args.like
    data_root = pathlib.Path(args.data_root)

    if args.check_only:
        if not (run_dir / "ModelParameters" / "latest.pth").is_file():
            parser.error(f"{run_dir} holds no trained run to check")
        ghost_trajectory(run_dir, args.seed, args.res)
        return

    unattended.open_log(args.log_dir, args.run)
    print(
        f"\n=== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  {args.run}  pid {os.getpid()} ==="
    )
    awake = unattended.KeepAwake()
    awake.hold()
    try:
        prepare_run(
            run_dir,
            like_dir,
            data_root,
            args.seed,
            args.epochs,
            args.every,
            args.force,
        )
        from structsept.app import training

        say("training starts")
        t0 = time.perf_counter()
        # The library's own log handler already prints to the (tee'd) console.
        summary = training.train(run_dir, data_root, log=lambda s: None)
        seconds = time.perf_counter() - t0
        say(f"finished {summary['epochs']} epochs in {seconds / 3600:.2f} h")
        training.write_metadata(
            run_dir,
            dataset=DATASET,
            preset=False,
            last_epoch=unattended.last_epoch(run_dir),
            wall_seconds=round(seconds),
        )
        try:
            ghost_trajectory(run_dir, args.seed, args.res)
        except Exception as exc:  # the trained run matters more than the report
            say(f"ghost check failed: {exc!r}")
    finally:
        awake.clear()
    say("done")


if __name__ == "__main__":
    main()
