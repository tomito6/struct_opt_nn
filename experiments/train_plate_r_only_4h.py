"""Unattended, time-boxed training of the radius-only plate decoder.

What is trained
---------------
The ``plate_hole_2d_r_only`` dataset: one square plate with a centred circular
hole, 40 radii from 0.07 to 0.45, exact signed distances in the plane. One
generating parameter, so the latent code has dimension 1 by default, and the
question the run answers is whether that single learned number tracks the
radius. ``--latent-dim 2`` trains the same recipe with a two-component code
(more than the geometry needs, on purpose): the run is then named
``plate2d_r_only_d2_8x256_4h`` and the check at the end looks at every
component, and at how one-dimensional the set of codes came out.

This is the "big" version of ``runs/preset_plate2d_r_only_d1`` (4 x 64,
1500 epochs, ~8 min): same dataset, same code dimension and initial spread, but
a decoder the size of the original DeepSDF's smaller cousin and a wall-clock
budget of hours rather than minutes. Sized on this machine (i7-9750H, 6 torch
threads, CPU-only torch) from a 3-epoch benchmark::

    decoder        us / point   s / epoch at 4096 samples x 40 shapes
    6 x 128          38            6.2
    8 x 256         106           17.4     <- chosen
    8 x 512         220           36.0

600 epochs of the 8 x 256 decoder take ~2.9 h at benchmark speed; the default
``--max-hours 4`` leaves room for a laptop that throttles once it is warm.
(Measured afterwards: 1.6 h for the d = 1 run, at night with nothing else
running.)

The recipe (everything else at ``structsept.app.hyperparams`` defaults):

===========================  =====================================================
decoder                      8 layers x 256, ReLU, weight norm, skip into layer 4
dropout                      0.2 on every hidden layer (DeepSDF's own default)
latent                       d = 1 (``--latent-dim``), every component drawn
                             from N(0, 0.1^2), bound 1.0
sampling                     4096 points per shape per epoch, 4 shapes per batch
                             (40 / 4 = 10 optimizer steps per epoch, none dropped)
loss                         clamped L1, delta = 0.1
learning rate                Adam, 5e-4 decoder / 1e-3 codes, halved every 150
epochs                       600 (checkpoints at 1, 100, 200, ... 600; latest
                             every 10)
===========================  =====================================================

How it runs
-----------
The script is meant to be started once and left alone::

    uv run python experiments/train_plate_r_only_4h.py --start-at "2026-09-28 01:10"
    uv run python experiments/train_plate_r_only_4h.py --start-at now --epochs 5
    uv run python experiments/train_plate_r_only_4h.py --latent-dim 2
    uv run python experiments/train_plate_r_only_4h.py --after outputs/logs/<run>.pid

1. It writes ``runs/<run>/specs.json`` and ``metadata.json`` right away, so the
   GUI's run table already lists the run while it waits.
2. It waits: until ``--start-at`` (a clock time; ``now`` for immediately) and,
   with ``--after``, until another run's process has exited - so two long
   runs can be chained without guessing when the first one ends. Windows is
   asked not to sleep in the meantime (``SetThreadExecutionState``, the same
   request a video player makes; the display may still turn off).
3. It trains with ``structsept.app.training.train`` - the trainer the Train
   tab uses, CPU, same seeding - and stops ``--max-hours`` after training
   started, whatever the epoch counter says. The stop reuses the trainer's own
   Ctrl-C handler: a watchdog thread raises SIGINT in the main thread, the
   trainer exits its loop, and ``latest.pth`` (written every ``LogFrequency``
   epochs) plus the numbered snapshots are what remains. ``metadata.json``
   then says ``"stopped_at_deadline": true`` and at which epoch.
4. Afterwards it compares the learned code of every shape with the radius in
   ``params.csv``: Pearson and Spearman correlation, whether the map is
   monotonic, and a scatter plot. With ``--latent-dim 2`` every component is
   checked on its own, plus how much of the codes' variance lies along their
   first principal axis (1.0 = the codes sit on one straight line in latent
   space, which is all a single generating parameter can need) and a plot of
   the codes in the latent plane coloured by radius. Results go into the run
   directory as ``code_vs_r.csv`` and ``code_vs_r.png``. Look there first.

The waiting, the deadline and the log plumbing are
``structsept.app.unattended``, shared with ``train_plate_xyr_4h.py``; what is
here is the recipe and the radius check.

Everything printed goes to ``outputs/logs/<run>.log`` as well as the console
(there is none when launched hidden). The process id is in
``outputs/logs/<run>.pid`` so a running job can be found and killed::

    taskkill /PID <pid> /F

The run is a normal run directory afterwards: open it in the Explore 2-D tab,
or load its hyperparameters in the Train tab with "Start from > run: <run>".
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
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
DATASET = "plate_hole_2d_r_only"
RUN_NAME = "plate2d_r_only_d{d}_8x256_4h"  # d = latent dimension
DEFAULT_LATENT_DIM = 1
#: Variance of every component of the initial codes. The trainer draws each
#: code from N(0, sigma^2 / d), so sigma = sqrt(CODE_VARIANCE * d) keeps this
#: per component whatever d is (the same arithmetic ``hyperparams.from_sheet``
#: does with the supervisor's sheet).
CODE_VARIANCE = 0.01
LOG_DIR = ROOT / "outputs" / "logs"
BUILD_HINT = (
    "uv run python -m datagen.make_plate_hole --dim 2 --radius-only "
    "--n-uniform 25000 --n-band 25000 --name plate_hole_2d_r_only --plot"
)

#: The hyperparameter set, by ``structsept.app.hyperparams`` key. The values
#: not listed take that module's defaults (weight norm on, norm on every
#: layer, xyz only at the input, no tanh, code bound 1.0, lambda 1e-4).
#: ``latent_dim`` and ``code_init_std`` are filled in by ``prepare_run`` from
#: ``--latent-dim``.
HPARAMS = {
    "n_layers": 8,
    "width": 256,
    "latent_in": [4],
    "dropout_layers": "all",
    "dropout_prob": 0.2,
    "loss_function": "clampedL1",
    "clamping_distance": 0.1,
    "samples_per_scene": 4096,
    "scenes_per_batch": 4,
    "num_epochs": 600,
    "lr_dec_type": "Step",
    "lr_dec_initial": 5e-4,
    "lr_dec_interval": 150,
    "lr_dec_factor": 0.5,
    "lr_code_type": "Step",
    "lr_code_initial": 1e-3,
    "lr_code_interval": 150,
    "lr_code_factor": 0.5,
    "log_frequency": 10,
    "snapshot_frequency": 100,
    "additional_snapshots": [1],
    "seed": 42,
}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def prepare_run(
    run_dir, data_root, epochs, force, latent_dim=DEFAULT_LATENT_DIM, log=say
):
    """Assemble the recipe for ``latent_dim`` and write the run's specs.

    Returns the dataset row from ``datasets.list_datasets``; see
    ``unattended.prepare_run`` for what is validated and when it exits.
    """
    from structsept.app import hyperparams

    hp = hyperparams.defaults()
    hp.update(HPARAMS)
    hp["latent_dim"] = latent_dim
    hp["code_init_std"] = math.sqrt(CODE_VARIANCE * latent_dim)
    hp["num_epochs"] = epochs
    hp["description"] = (
        f"Radius-only plate, d={latent_dim}, long CPU run: 8x256 ReLU, dropout "
        f"0.2, weight norm, skip at 4; 4096 samples/shape, 4 shapes/batch "
        f"(10 steps/epoch); {epochs} epochs, lr 5e-4/1e-3 halved every 150; "
        f"codes N(0, {CODE_VARIANCE}) per component. Started unattended by "
        f"experiments/train_plate_r_only_4h.py. See code_vs_r.png in the run "
        f"directory for the code-vs-radius check."
    )
    return unattended.prepare_run(
        run_dir, data_root, DATASET, hp, force=force, log=log, build_hint=BUILD_HINT
    )


# --------------------------------------------------------------------------- #
# code-vs-radius check
# --------------------------------------------------------------------------- #


def compare_code_with_radius(run_dir, dataset_dir, split_path, log=say):
    """Compare the learned code of every shape with its radius.

    ``unattended.code_vs_parameter`` on the ``r`` column of ``params.csv``
    (it started life here and moved when the triangle family needed the same
    check): writes ``code_vs_r.csv`` and ``code_vs_r.png`` into the run
    directory and returns a dict with the numbers.
    """
    return unattended.code_vs_parameter(
        run_dir,
        dataset_dir,
        split_path,
        "r",
        label="hole radius r (design units)",
        log=log,
    )


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
    when.add_argument(
        "--max-hours",
        type=float,
        default=4.0,
        help="hard stop this many hours after training starts (default: 4)",
    )
    what = parser.add_argument_group("what")
    what.add_argument(
        "--latent-dim",
        type=int,
        default=DEFAULT_LATENT_DIM,
        help=f"dimension of the latent code (default: {DEFAULT_LATENT_DIM})",
    )
    what.add_argument(
        "--run",
        default=None,
        help=f"run directory name (default: {RUN_NAME.format(d='<latent-dim>')})",
    )
    what.add_argument(
        "--epochs",
        type=int,
        default=HPARAMS["num_epochs"],
        help=f"epochs to train (default: {HPARAMS['num_epochs']})",
    )
    what.add_argument("--force", action="store_true", help="train over an existing run")
    where = parser.add_argument_group("where")
    where.add_argument("--data-root", default=str(ROOT / "data"))
    where.add_argument("--runs-dir", default=str(ROOT / "runs"))
    where.add_argument("--log-dir", default=str(LOG_DIR))
    args = parser.parse_args(argv)
    if isinstance(args.start_at, str):
        args.start_at = unattended.parse_start(args.start_at)
    if args.latent_dim < 1:
        parser.error("--latent-dim must be at least 1")
    if args.run is None:
        args.run = RUN_NAME.format(d=args.latent_dim)

    unattended.open_log(args.log_dir, args.run)
    run_dir = pathlib.Path(args.runs_dir) / args.run
    data_root = pathlib.Path(args.data_root)
    print(
        f"\n=== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  {args.run}  pid {os.getpid()} ==="
    )
    say(
        f"start {args.start_at:%Y-%m-%d %H:%M}"
        + (f", after process {args.after}" if args.after else "")
        + f", hard stop {args.max_hours:g} h after training starts"
    )

    awake = unattended.KeepAwake()
    awake.hold()
    try:
        row = prepare_run(
            run_dir, data_root, args.epochs, args.force, latent_dim=args.latent_dim
        )
        unattended.wait_until(args.start_at)
        if args.after:
            unattended.wait_for_process(args.after)
        deadline = dt.datetime.now() + dt.timedelta(hours=args.max_hours)
        say(f"training starts, hard stop {deadline:%Y-%m-%d %H:%M}")
        t0 = time.perf_counter()
        summary, stopped = unattended.train_with_deadline(run_dir, data_root, deadline)
        seconds = time.perf_counter() - t0
        epoch = unattended.last_epoch(run_dir)
        if stopped:
            say(f"stopped at the deadline after {seconds / 3600:.2f} h, epoch {epoch}")
        else:
            say(f"finished {summary['epochs']} epochs in {seconds / 3600:.2f} h")
    finally:
        awake.clear()

    from structsept.app import training

    training.write_metadata(
        run_dir,
        dataset=DATASET,
        preset=False,
        stopped_at_deadline=stopped,
        last_epoch=epoch,
        wall_seconds=round(seconds),
    )
    try:
        compare_code_with_radius(run_dir, row["path"], row["split"])
    except Exception as exc:  # the trained run matters more than the plot
        say(f"code-vs-radius check failed: {exc!r}")
    say("done")


if __name__ == "__main__":
    main()
