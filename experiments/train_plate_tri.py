"""Train a four-triangle plate decoder with the preset recipe, unattended.

What is trained
---------------
One of the two triangle families of ``datagen.make_plate_tri``, chosen with
``--family``:

``h`` (default)
    ``plate_tri_2d_h``: the shared height ``h`` runs from 0.05 to 0.3 in 40
    even steps, base ``2 h``. One generating parameter, code of dimension 1,
    and the question is whether that single number tracks ``h``. Answered
    01/10: a straight line, Pearson -1.000, monotonic
    (``runs/plate_tri_2d_h_d1_4x64``).
``hw``
    ``plate_tri_2d_hw``: height and base width vary independently, Sobol
    128 pairs plus the corners and centre of the feasible set. Two
    generating parameters, code of dimension 2. A learned latent basis is
    arbitrary, so the check is not component by component but whether a
    linear map of the two components recovers ``h`` and recovers ``w``
    (R^2 of each fit), the comparison the free-hole runs still lack.

The recipe is not written here: it is the ``specs.json`` of the family's
preset (``--like``; ``runs/preset_plate2d_tri_h_d1`` or
``runs/preset_plate2d_tri_hw_d2``, written by
``experiments/make_plate_presets.py``) -- 4 x 64 ReLU, 1500 epochs, lr
halved at 500 and 1000, 4096 samples per shape, 5 shapes per batch, codes
from N(0, 0.01) per component. It is copied key for key; only the
description is replaced, and the script prints which keys differ so that is
checked rather than promised. Measured: 20 min for the 40 shapes of ``h``
with other work on the machine; ``hw`` has 133 shapes, so about 30 min
clean.

How it runs
-----------
Started once and left alone, like ``train_plate_r_only_4h.py``::

    uv run python experiments/train_plate_tri.py
    uv run python experiments/train_plate_tri.py --family hw
    uv run python experiments/train_plate_tri.py --family hw --start-at 23:30
    uv run python experiments/train_plate_tri.py --after outputs/logs/<run>.pid
    uv run python experiments/train_plate_tri.py --epochs 2 --run smoke   # test only
    uv run python experiments/train_plate_tri.py --family hw --check-only

It writes ``runs/<run>/specs.json`` and ``metadata.json`` at once, waits if
asked to (``structsept.app.unattended``; Windows is kept awake), trains with
the Train tab's trainer, and logs to ``outputs/logs/<run>.log`` with the
process id in ``<run>.pid`` beside it (``taskkill /PID <pid> /F`` stops it;
``latest.pth`` is rewritten every ``LogFrequency`` epochs, so a stopped run
is still loadable).

Afterwards it compares the learned codes with every generating parameter of
the family, one ``code_vs_<param>.csv`` / ``.png`` per parameter in the run
directory (``unattended.code_vs_parameter``: correlations and monotonicity
per component, and for d > 1 the principal axis and the linear-fit R^2).
Look there first. ``--check-only`` does just this on the existing ``--run``.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
LOG_DIR = ROOT / "outputs" / "logs"
PRESET_HINT = "uv run python experiments/make_plate_presets.py"
BUILD = (
    "uv run python -m datagen.make_plate_tri --dim 2 {flag}--n-uniform 25000 "
    "--n-band 25000 --name {dataset} --plot"
)

#: The two families: dataset, preset, run name, the generating parameters
#: (columns of params.csv) and a line for the run description.
FAMILIES = {
    "h": {
        "dataset": "plate_tri_2d_h",
        "like": "preset_plate2d_tri_h_d1",
        "run": "plate_tri_2d_h_d1_4x64",
        "params": ("h",),
        "what": "four triangular holes, shared height h in [0.05, 0.3], latent d=1",
        "build": BUILD.format(flag="", dataset="plate_tri_2d_h"),
    },
    "hw": {
        "dataset": "plate_tri_2d_hw",
        "like": "preset_plate2d_tri_hw_d2",
        "run": "plate_tri_2d_hw_d2_4x64",
        "params": ("h", "w"),
        "what": (
            "four triangular holes, height h in [0.05, 0.3] and base width w in "
            "[0.1, w_max(h)] varying independently, latent d=2"
        ),
        "build": BUILD.format(flag="--free-width ", dataset="plate_tri_2d_hw"),
    },
}
LABELS = {
    "h": "triangle height h (design units)",
    "w": "triangle base width w (design units)",
}


def prepare_run(run_dir, like_dir, data_root, family, epochs=None, force=False):
    """Write the preset's ``specs.json`` as a run of its own.

    ``unattended.prepare_run_like``: every key copied, only the description
    and the data paths replaced (the preset already points at the dataset,
    so those come out equal), exit if anything else would differ. Returns
    the dataset row of ``datasets.list_datasets``.
    """
    like_dir = pathlib.Path(like_dir)
    if not (like_dir / "specs.json").is_file():
        raise SystemExit(
            f"{like_dir} has no specs.json; write the preset first with\n  {PRESET_HINT}"
        )
    description = (
        f"Same recipe as {like_dir.name} (specs.json copied key for key) on "
        f"{family['dataset']}: {family['what']}. Started by "
        f"experiments/train_plate_tri.py; see code_vs_<param>.png in the run "
        f"directory for the code-vs-parameter check."
    )
    row, _ = unattended.prepare_run_like(
        run_dir,
        like_dir,
        data_root,
        family["dataset"],
        description,
        epochs=epochs,
        force=force,
        build_hint=family["build"],
    )
    return row


def check_codes(run_dir, row, family, log=say):
    """``code_vs_parameter`` for every generating parameter of the family."""
    return {
        p: unattended.code_vs_parameter(
            run_dir, row["path"], row["split"], p, LABELS[p], log=log
        )
        for p in family["params"]
    }


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
        "--family",
        choices=sorted(FAMILIES),
        default="h",
        help="h: shared height, d=1 (default); hw: height and width, d=2",
    )
    what.add_argument(
        "--run", default=None, help="run directory name (default: per family)"
    )
    what.add_argument(
        "--like",
        default=None,
        help="run or preset whose specs.json is the recipe (default: the "
        "family's preset)",
    )
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
        help="skip training; compare the codes of the existing --run with the "
        "family's parameters",
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
    family = FAMILIES[args.family]
    args.run = args.run or family["run"]
    args.like = args.like or family["like"]
    run_dir = pathlib.Path(args.runs_dir) / args.run
    like_dir = pathlib.Path(args.runs_dir) / args.like
    data_root = pathlib.Path(args.data_root)

    if args.check_only:
        from structsept.app import datasets

        if not (run_dir / "LatentCodes" / "latest.pth").is_file():
            parser.error(f"{run_dir} holds no trained run to check")
        rows = {d["name"]: d for d in datasets.list_datasets(data_root)}
        check_codes(run_dir, rows[family["dataset"]], family)
        return

    unattended.open_log(args.log_dir, args.run)
    print(
        f"\n=== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  {args.run}  pid {os.getpid()} ==="
    )
    say(
        f"family {args.family}: {family['what']}; start {args.start_at:%Y-%m-%d %H:%M}"
        + (f", after process {args.after}" if args.after else "")
        + ", no time limit"
    )

    awake = unattended.KeepAwake()
    awake.hold()
    try:
        row = prepare_run(run_dir, like_dir, data_root, family, args.epochs, args.force)
        unattended.wait_until(args.start_at)
        if args.after:
            unattended.wait_for_process(args.after)
        from structsept.app import training

        say("training starts")
        t0 = time.perf_counter()
        # The library's own log handler already prints to the (tee'd) console.
        summary = training.train(run_dir, data_root, log=lambda s: None)
        seconds = time.perf_counter() - t0
        say(f"finished {summary['epochs']} epochs in {seconds / 60:.1f} min")
    finally:
        awake.clear()

    training.write_metadata(
        run_dir,
        dataset=family["dataset"],
        preset=False,
        last_epoch=unattended.last_epoch(run_dir),
        wall_seconds=round(seconds),
    )
    try:
        check_codes(run_dir, row, family)
    except Exception as exc:  # the trained run matters more than the plot
        say(f"code-vs-parameter check failed: {exc!r}")
    say("done")


if __name__ == "__main__":
    main()
