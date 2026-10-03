"""Retrain the free-hole plate decoder on the set with the large holes filled in.

Why
---
The Explore 2-D tab shades, on each latent axis, the widest stretch with no
trained code in it. For ``runs/plate_hole_2d_xyr_d3_ep800`` that is 29 % of the
first axis (+0.27 .. +0.47), and the cause is in the data, not in the training:
that axis mostly carries the radius, and the 134 shapes of ``plate_hole_2d_xyr``
hold no hole between ``r = 0.342`` and the prepended extreme at 0.45 - 8 shapes
in all above ``r = 0.25``, the upper half of the radius range. A large hole
needs a centre near the middle of the plate *and* a ``t`` near 1, and a Sobol
draw of 128 points barely reaches that corner of the cube.

What is trained
---------------
``plate_hole_2d_xyr_n166``: the same 134 shapes, file for file and in the same
order, followed by 32 more whose radius is spread evenly over [0.25, 0.45] and
whose centre is free in the room that radius leaves
(``PlateHoleSpace.sample_large_holes``). All three parameters still vary.

The recipe is not written here: it is the ``specs.json`` of the run this one
is compared with (``--like``, by default the ep800 run above), copied key for
key. Only the split, the data source and the description are replaced, and the
script prints which keys differ so that is checked rather than promised.
Nothing cuts the run short either - the epochs of the reference run are the
recipe. Measured on this machine, CPU only: 35 s per epoch for 134 shapes at
night, so about 43 s for 166 and 9.6 h for the 800 epochs - an estimate the
first launch did not meet. It ran 80 epochs in 83 min before its process died
(30/09, 17:28 to 18:51, laptop in use): 62 s per epoch, which is 13.8 h for
the 800. Plan for 14 h.

That first launch is why ``runs/plate_hole_2d_xyr_n166_d3_ep800`` holds a
checkpoint at epoch 80 of 800. It is not the run described here, and it
cannot be continued: ``train_deep_sdf(continue_from=...)`` restores the
decoder and the optimizer but discards the latent codes it loads, so the
codes would restart from their random initialisation under a trained
decoder. Relaunching means ``--force``, from epoch 1.

How it runs
-----------
Started once and left alone, like ``train_plate_xyr_4h.py``::

    uv run python experiments/train_plate_xyr_n166.py
    uv run python experiments/train_plate_xyr_n166.py --start-at 23:30
    uv run python experiments/train_plate_xyr_n166.py --after outputs/logs/<run>.pid
    uv run python experiments/train_plate_xyr_n166.py --epochs 2 --run smoke   # test only
    uv run python experiments/train_plate_xyr_n166.py --check-only

It writes ``runs/<run>/specs.json`` and ``metadata.json`` at once, waits if
asked to (``structsept.app.unattended``; Windows is kept awake), trains with
the Train tab's trainer, and logs to ``outputs/logs/<run>.log`` with the
process id in ``<run>.pid`` beside it (``taskkill /PID <pid> /F`` stops it;
``latest.pth`` is rewritten every ``LogFrequency`` epochs, so a stopped run
is still loadable).

Afterwards it measures what the run was for: per latent component, the widest
gap between neighbouring trained codes as a share of the component's range,
against the threshold past which the GUI shades it
(``structsept.app.viz.GAP_FRACTION``), and which two shapes sit at its ends.
The same is printed for the reference run, and both go to
``latent_coverage.json`` in the run directory. ``--check-only`` does just
this on the existing ``--run``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import time

# Both must be set before torch / DeepSDFStruct are imported: tqdm reads its
# switch at import time, matplotlib picks its backend on first use.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from structsept.app import unattended
from structsept.app.unattended import say

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = "plate_hole_2d_xyr_n166"
LIKE = "plate_hole_2d_xyr_d3_ep800"
RUN_NAME = "plate_hole_2d_xyr_n166_d3_ep800"
LOG_DIR = ROOT / "outputs" / "logs"
BUILD_HINT = (
    "uv run python -m datagen.make_plate_hole --dim 2 --n-uniform 25000 "
    "--n-band 25000 --large-holes 32 --large-holes-from 0.25 "
    "--name plate_hole_2d_xyr_n166 --plot"
)
PARAMS = ("x_c", "y_c", "r")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def prepare_run(run_dir, like_dir, data_root, dataset, epochs=None, force=False):
    """Write the reference run's ``specs.json`` for another dataset.

    ``unattended.prepare_run_like`` (it started life here and moved when the
    triangle family needed the same copy): every key is copied, only
    ``unattended.REPLACED_KEYS`` are pointed at ``dataset``, and the process
    exits if anything else would differ. Returns the dataset row of
    ``datasets.list_datasets``.
    """
    like_dir = pathlib.Path(like_dir)
    description = (
        f"Same recipe as {like_dir.name} (specs.json copied key for key) on "
        f"{dataset}: its 134 shapes plus 32 holes with r spread over "
        f"[0.25, 0.45], to close the untrained gap on the radius axis of the "
        f"latent space. Started by experiments/train_plate_xyr_n166.py; see "
        f"latent_coverage.json in the run directory."
    )
    row, _ = unattended.prepare_run_like(
        run_dir,
        like_dir,
        data_root,
        dataset,
        description,
        epochs=epochs,
        force=force,
        build_hint=BUILD_HINT,
    )
    return row


# --------------------------------------------------------------------------- #
# latent coverage
# --------------------------------------------------------------------------- #


def _shape_table(run_dir):
    """``{latent index: "x_c=.. y_c=.. r=.."}`` of a run, or ``{}``.

    Read from the split the run was trained on and the ``params.csv`` of that
    dataset; a run on a dataset without a parameter table just gets no labels.
    """
    from structsept.app import training

    try:
        split = pathlib.Path(
            training.read_specs(pathlib.Path(run_dir) / "specs.json")["TrainSplit"]
        )
        names = unattended.split_names(split)
        params = split.parents[1] / "SdfSamples" / split.stem / "params.csv"
        with open(params, newline="", encoding="utf-8") as fh:
            table = {row["name"]: row for row in csv.DictReader(fh)}
        return {
            i: "  ".join(f"{p}={float(table[n][p]):.3f}" for p in PARAMS)
            for i, n in enumerate(names)
        }
    except (OSError, KeyError, ValueError):
        return {}


def latent_coverage(run_dir, log=say):
    """Widest untrained gap of every latent component of a run.

    The number behind the shaded band of the Explore tabs' coverage strips:
    the largest distance between two neighbouring trained codes along one
    component, as a share of that component's range. Past
    ``viz.GAP_FRACTION`` the GUI shades it. Returns one dict per component,
    or ``[]`` when the run holds no codes.
    """
    import numpy as np

    from structsept.app import models, viz

    codes = unattended.load_codes(run_dir)
    if codes is None:
        log(f"{pathlib.Path(run_dir).name}: no latent codes saved")
        return []
    shapes = _shape_table(run_dir)
    log(
        f"{pathlib.Path(run_dir).name}: {len(codes)} codes, epoch "
        f"{unattended.last_epoch(run_dir)}, gaps past "
        f"{100 * viz.GAP_FRACTION:.0f} % of the range are shaded in the GUI"
    )
    out = []
    for j in range(codes.shape[1]):
        column = codes[:, j]
        gap, lo, hi = models.largest_gap(column)
        span = max(float(np.ptp(column)), 1e-12)
        ends = [int(np.argmin(np.abs(column - v))) for v in (lo, hi)]
        row = {
            "component": j + 1,
            "min": float(column.min()),
            "max": float(column.max()),
            "gap_from": lo,
            "gap_to": hi,
            "gap_share": gap / span,
            "shaded": bool(gap > viz.GAP_FRACTION * span),
            "shapes_at_the_ends": [shapes.get(i, f"index {i}") for i in ends],
        }
        out.append(row)
        log(
            f"  lambda_{j + 1} in [{row['min']:+.3f}, {row['max']:+.3f}]: widest gap "
            f"{lo:+.3f} .. {hi:+.3f} = {100 * row['gap_share']:.0f} % "
            f"({'SHADED' if row['shaded'] else 'ok'}), between "
            f"[{row['shapes_at_the_ends'][0]}] and [{row['shapes_at_the_ends'][1]}]"
        )
    return out


def compare_coverage(run_dir, like_dir):
    """Coverage of the new run and of the reference; written to the run."""
    result = {
        "run": {
            "name": pathlib.Path(run_dir).name,
            "components": latent_coverage(run_dir),
        },
        "reference": {
            "name": pathlib.Path(like_dir).name,
            "components": latent_coverage(like_dir),
        },
    }
    path = pathlib.Path(run_dir) / "latent_coverage.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    when = parser.add_argument_group("when")
    when.add_argument(
        "--start-at",
        type=unattended.parse_start,
        default="now",
        help="clock time to start training: 'now', 'HH:MM' or 'YYYY-MM-DD HH:MM' "
        "(default: now)",
    )
    when.add_argument(
        "--after",
        type=unattended.parse_pid,
        default=None,
        metavar="PID_OR_FILE",
        help="also wait until this process (a pid, or a .pid file from "
        "outputs/logs) has exited before training",
    )
    what = parser.add_argument_group("what")
    what.add_argument(
        "--run", default=RUN_NAME, help=f"run directory name (default: {RUN_NAME})"
    )
    what.add_argument(
        "--like",
        default=LIKE,
        help=f"run whose specs.json is the recipe (default: {LIKE})",
    )
    what.add_argument("--dataset", default=DATASET, help=f"default: {DATASET}")
    what.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="smoke test only: train this many epochs instead of the recipe's",
    )
    what.add_argument("--force", action="store_true", help="train over an existing run")
    what.add_argument(
        "--check-only",
        action="store_true",
        help="skip training; measure the latent coverage of the existing --run",
    )
    where = parser.add_argument_group("where")
    where.add_argument("--data-root", default=str(ROOT / "data"))
    where.add_argument("--runs-dir", default=str(ROOT / "runs"))
    where.add_argument("--log-dir", default=str(LOG_DIR))
    args = parser.parse_args(argv)
    if isinstance(args.start_at, str):
        args.start_at = unattended.parse_start(args.start_at)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    run_dir = pathlib.Path(args.runs_dir) / args.run
    like_dir = pathlib.Path(args.runs_dir) / args.like
    data_root = pathlib.Path(args.data_root)

    if args.check_only:
        if not (run_dir / "LatentCodes" / "latest.pth").is_file():
            parser.error(f"{run_dir} holds no trained run to check")
        compare_coverage(run_dir, like_dir)
        return

    unattended.open_log(args.log_dir, args.run)
    print(
        f"\n=== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  {args.run}  pid {os.getpid()} ==="
    )
    say(
        f"start {args.start_at:%Y-%m-%d %H:%M}"
        + (f", after process {args.after}" if args.after else "")
        + ", no time limit"
    )

    awake = unattended.KeepAwake()
    awake.hold()
    try:
        prepare_run(run_dir, like_dir, data_root, args.dataset, args.epochs, args.force)
        unattended.wait_until(args.start_at)
        if args.after:
            unattended.wait_for_process(args.after)
        from structsept.app import training

        say("training starts")
        t0 = time.perf_counter()
        # The library's own log handler already prints to the (tee'd) console.
        summary = training.train(run_dir, data_root, log=lambda s: None)
        seconds = time.perf_counter() - t0
        say(f"finished {summary['epochs']} epochs in {seconds / 3600:.2f} h")
    finally:
        awake.clear()

    training.write_metadata(
        run_dir,
        dataset=args.dataset,
        preset=False,
        last_epoch=unattended.last_epoch(run_dir),
        wall_seconds=round(seconds),
    )
    try:
        compare_coverage(run_dir, like_dir)
    except Exception as exc:  # the trained run matters more than the report
        say(f"latent coverage check failed: {exc!r}")
    say("done")


if __name__ == "__main__":
    main()
