"""Unattended, time-boxed training of the free-hole plate decoder (d = 3).

What is trained
---------------
The ``plate_hole_2d_xyr`` dataset: one square plate with a circular hole whose
centre ``(x_c, y_c)`` and radius ``r`` all vary - 134 plates (a Sobol set of
128 plus the 6 extremes, margin 0.05), exact signed distances in the plane.
Three generating parameters, so the latent code has dimension 3 by default
(``--latent-dim`` for another), and the question the run answers is whether
the three learned numbers hold the three parameters - the d = 2 runs on the
25-plate set kept ``y_c`` and dropped ``x_c``.

Same recipe as ``train_plate_r_only_4h.py`` (the supervisor's sheet for
``plate2d_r_only_d2_8x256_4h``), on 3.35 x the shapes. That changes one thing:
the clock. An epoch is every shape once, so it costs 134 / 4 = 33 optimizer
steps here against 10 there, and the same 4-hour window holds fewer epochs.
Measured on this machine (i7-9750H, 6 torch threads, CPU-only torch) with
this decoder and 4096 samples per shape:

    run                                  shapes   s / epoch
    plate2d_r_only_d1_8x256_4h (night)      40       9.6
    plate2d_r_only_d2_8x256_4h (evening)    40      13.2
    plate_hole_2d_n134_d2 (daytime use)    134      52.3  (36 when idle)

so 134 shapes run at roughly 32-45 s per epoch when the laptop is otherwise
idle: the sheet's 600 epochs would take 5.5-7.5 h, and 300 fit in 2.7-3.7 h
with the ``--max-hours 4`` stop as the safety net. The learning rate is still
halved four times over the run (every 75 epochs instead of every 150), and
with 33 steps per epoch the run still makes 9 900 optimizer steps against the
6 000 of the 600-epoch radius-only run.

The recipe (everything else at ``structsept.app.hyperparams`` defaults):

===========================  =====================================================
decoder                      8 layers x 256, ReLU, weight norm, skip into layer 4
dropout                      0.2 on every hidden layer (DeepSDF's own default)
latent                       d = 3 (``--latent-dim``), every component drawn
                             from N(0, 0.1^2) - sigma = sqrt(0.01 d) = 0.173 -
                             bound 1.0, lambda 1e-4
sampling                     4096 points per shape per epoch, 4 shapes per batch
                             (134 / 4 = 33 steps per epoch; 2 random shapes sit
                             each epoch out, the trainer drops the partial batch)
loss                         clamped L1, delta = 0.1
learning rate                Adam, 5e-4 decoder / 1e-3 codes, halved every 75
epochs                       300 (checkpoints at 1, 50, 100, ... 300; latest
                             every 10)
===========================  =====================================================

How it runs
-----------
The script is meant to be started once and left alone::

    uv run python experiments/train_plate_xyr_4h.py
    uv run python experiments/train_plate_xyr_4h.py --after outputs/logs/plate2d_r_only_d2_8x256_4h.pid
    uv run python experiments/train_plate_xyr_4h.py --start-at 03:15 --max-hours 4
    uv run python experiments/train_plate_xyr_4h.py --epochs 2 --run smoke   # 1-2 min test
    uv run python experiments/train_plate_xyr_4h.py --check-only --run plate_hole_2d_n134_d2_20260928_1338

1. It writes ``runs/<run>/specs.json`` and ``metadata.json`` right away, so the
   GUI's run table already lists the run while it waits.
2. It waits: until ``--start-at`` (a clock time; ``now`` for immediately) and,
   with ``--after``, until another run's process has exited - the way to
   queue this run behind one that is still training, without guessing when
   that one ends. Windows is asked not to sleep in the meantime.
3. It trains with ``structsept.app.training.train`` - the trainer the Train
   tab uses, CPU, same seeding - and stops ``--max-hours`` after training
   started, whatever the epoch counter says (the trainer's own Ctrl-C handler,
   raised by a watchdog thread; ``latest.pth`` and the numbered snapshots
   remain, ``metadata.json`` says ``"stopped_at_deadline": true`` and at
   which epoch).
4. Afterwards it compares the learned codes with ``params.csv``. The learned
   latent space is an arbitrary basis, so component-by-component correlation
   is only a first look; the real question is whether ``(x_c, y_c, r)`` can
   be read off the code at all. Both are done:

   * Pearson and Spearman correlation of every code component with every
     parameter (a d x 3 table);
   * an affine fit ``(x_c, y_c, r) = W code + b`` by least squares, with the
     R-squared and the RMSE (in design units) of every parameter - R-squared
     near 1 for all three means the code is a linear re-labelling of the
     parameters; a parameter with R-squared near 0 was not learned, or not
     linearly;
   * the reverse fit, code from parameters, saying how linear each code
     component is in the parameters; and the variance share of the codes'
     principal axes (three similar shares = the code really uses three
     directions; one dominant share = it collapsed to fewer).

   Results go into the run directory as ``code_vs_params.csv`` (one row per
   shape: parameters, codes, fitted parameters), ``code_vs_params.json`` (the
   numbers) and ``code_vs_params.png`` (a grid: every code against every
   parameter, then fitted against true parameter). Look there first.
   ``--check-only`` runs just this step on an existing run - any run trained
   on this dataset, e.g. the d = 2 one from the GUI.

The waiting, the deadline and the log plumbing are
``structsept.app.unattended``, shared with ``train_plate_r_only_4h.py``; what
is here is the recipe and the parameter check.

Everything printed goes to ``outputs/logs/<run>.log`` as well as the console
(there is none when launched hidden). The process id is in
``outputs/logs/<run>.pid`` so a running job can be found and killed::

    taskkill /PID <pid> /F

The run is a normal run directory afterwards: open it in the Explore 2-D tab
(three sliders), or load its hyperparameters in the Train tab with
"Start from > run: <run>".
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
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
#: The generating parameters, as columns of the dataset's ``params.csv``.
PARAMS = ("x_c", "y_c", "r")
RUN_NAME = "plate2d_xyr_d{d}_8x256_4h"  # d = latent dimension
DEFAULT_LATENT_DIM = 3
#: Variance of every component of the initial codes. The trainer draws each
#: code from N(0, sigma^2 / d), so sigma = sqrt(CODE_VARIANCE * d) keeps this
#: per component whatever d is (the same arithmetic ``hyperparams.from_sheet``
#: does with the supervisor's sheet).
CODE_VARIANCE = 0.01
LOG_DIR = ROOT / "outputs" / "logs"
BUILD_HINT = (
    "uv run python -m datagen.make_plate_hole --dim 2 "
    "--n-uniform 25000 --n-band 25000 --name plate_hole_2d_xyr --plot"
)

#: The hyperparameter set, by ``structsept.app.hyperparams`` key. The values
#: not listed take that module's defaults (weight norm on, norm on every
#: layer, xyz only at the input, no tanh, code bound 1.0, lambda 1e-4).
#: ``latent_dim``, ``code_init_std``, ``num_epochs`` and the two decay
#: intervals are filled in by ``prepare_run`` from the command line.
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
    "num_epochs": 300,
    "lr_dec_type": "Step",
    "lr_dec_initial": 5e-4,
    "lr_dec_interval": 75,
    "lr_dec_factor": 0.5,
    "lr_code_type": "Step",
    "lr_code_initial": 1e-3,
    "lr_code_interval": 75,
    "lr_code_factor": 0.5,
    "log_frequency": 10,
    "snapshot_frequency": 50,
    "additional_snapshots": [1],
    "seed": 42,
}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def prepare_run(
    run_dir,
    data_root,
    epochs,
    force,
    latent_dim=DEFAULT_LATENT_DIM,
    decay_interval=None,
    log=say,
):
    """Assemble the recipe for ``latent_dim`` / ``epochs`` and write the specs.

    ``decay_interval`` defaults to a quarter of the epochs, which keeps the
    sheet's proportion (600 epochs, halved every 150). Returns the dataset
    row from ``datasets.list_datasets``; see ``unattended.prepare_run`` for
    what is validated and when it exits.
    """
    from structsept.app import hyperparams

    if decay_interval is None:
        decay_interval = max(1, epochs // 4)
    hp = hyperparams.defaults()
    hp.update(HPARAMS)
    hp["latent_dim"] = latent_dim
    hp["code_init_std"] = math.sqrt(CODE_VARIANCE * latent_dim)
    hp["num_epochs"] = epochs
    hp["lr_dec_interval"] = decay_interval
    hp["lr_code_interval"] = decay_interval
    hp["description"] = (
        f"Free-hole plate (x_c, y_c, r), d={latent_dim}, long CPU run: 8x256 "
        f"ReLU, dropout 0.2, weight norm, skip at 4; 4096 samples/shape, 4 "
        f"shapes/batch; {epochs} epochs, lr 5e-4/1e-3 halved every "
        f"{decay_interval}; codes N(0, {CODE_VARIANCE}) per component. Same "
        f"recipe as plate2d_r_only_d2_8x256_4h, epochs sized for 4 h on 134 "
        f"shapes. Started unattended by experiments/train_plate_xyr_4h.py. See "
        f"code_vs_params.png in the run directory for the code-vs-parameter "
        f"check."
    )
    return unattended.prepare_run(
        run_dir, data_root, DATASET, hp, force=force, log=log, build_hint=BUILD_HINT
    )


def dataset_row(data_root, name):
    """The ``datasets.list_datasets`` row for ``name``, or exit."""
    from structsept.app import datasets

    for row in datasets.list_datasets(data_root):
        if row["name"] == name and row["split"]:
            return row
    sys.exit(f"dataset {name} not found under {data_root}")


# --------------------------------------------------------------------------- #
# code-vs-parameter check
# --------------------------------------------------------------------------- #


def affine_fit(X, Y):
    """Least-squares ``Y = X W + b``, column by column.

    Returns ``(fit, r2, rmse, W)``: the fitted ``Y``, the R-squared and the
    root-mean-square residual of every column of ``Y``, and the ``(m + 1, n)``
    coefficient matrix with the intercept as its last row. A constant column
    of ``Y`` gets R-squared ``nan``.
    """
    import numpy as np

    A = np.hstack([X, np.ones((len(X), 1))])
    W, *_ = np.linalg.lstsq(A, Y, rcond=None)
    fit = A @ W
    ss_res = ((Y - fit) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    rmse = np.sqrt(ss_res / len(Y))
    return fit, r2, rmse, W


def compare_codes_with_params(run_dir, dataset_dir, split_path, params=PARAMS, log=say):
    """Compare the learned codes with the generating parameters of every shape.

    See the module docstring for what is computed and why. Writes
    ``code_vs_params.csv``, ``code_vs_params.json`` and ``code_vs_params.png``
    into the run directory and returns the dict that went into the json. The
    latent index is the split order, so the names in the split are matched
    against the ``name`` column of ``params.csv`` rather than trusting row
    order.
    """
    import numpy as np

    run_dir = pathlib.Path(run_dir)
    codes = unattended.load_codes(run_dir)
    if codes is None:
        log("no latent codes saved, skipping the code-vs-parameter check")
        return {}

    names = unattended.split_names(split_path)
    with open(pathlib.Path(dataset_dir) / "params.csv", newline="") as fh:
        table = {
            row["name"]: [float(row[p]) for p in params] for row in csv.DictReader(fh)
        }
    P = np.array([table[n] for n in names])  # (N, p)
    codes = codes[: len(names)]  # (N, d)
    n_shapes, d = codes.shape
    n_params = len(params)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def rank(v):
        return np.argsort(np.argsort(v))

    pearson = [[corr(codes[:, k], P[:, j]) for j in range(n_params)] for k in range(d)]
    spearman = [
        [corr(rank(codes[:, k]), rank(P[:, j])) for j in range(n_params)]
        for k in range(d)
    ]
    fit_p, r2_p, rmse_p, W_p = affine_fit(codes, P)  # parameters from codes
    _, r2_c, _, _ = affine_fit(P, codes)  # codes from parameters
    centred = codes - codes.mean(axis=0)
    # Principal axes by SVD: the squared singular values are the variances
    # of the codes along each axis.
    _, sigma, _ = np.linalg.svd(centred, full_matrices=False)
    total = float(np.sum(sigma**2))
    var_share = (sigma**2 / total).tolist() if total else [float("nan")] * d

    result = {
        "run": run_dir.name,
        "n_shapes": int(n_shapes),
        "latent_dim": int(d),
        "params": list(params),
        "pearson": pearson,  # [code k][param j]
        "spearman": spearman,
        "params_from_codes": {
            "r2": {p: float(v) for p, v in zip(params, r2_p)},
            "rmse": {p: float(v) for p, v in zip(params, rmse_p)},
            "W": W_p.tolist(),  # (d + 1, p); last row is the intercept
        },
        "codes_from_params": {"r2": [float(v) for v in r2_c]},
        "pc_variance_share": var_share,
        "code_min": codes.min(axis=0).tolist(),
        "code_max": codes.max(axis=0).tolist(),
    }

    with open(run_dir / "code_vs_params.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["name"]
            + list(params)
            + [f"code_{k}" for k in range(d)]
            + [f"fit_{p}" for p in params]
        )
        for n, prow, crow, frow in zip(names, P, codes, fit_p):
            writer.writerow(
                [n]
                + [f"{v:.6f}" for v in prow]
                + [f"{v:.6f}" for v in crow]
                + [f"{v:.6f}" for v in frow]
            )
    (run_dir / "code_vs_params.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        d + 1,
        n_params,
        figsize=(3.4 * n_params, 2.9 * (d + 1)),
        squeeze=False,
    )
    for k in range(d):
        for j, p in enumerate(params):
            ax = axes[k, j]
            ax.scatter(P[:, j], codes[:, k], s=12, alpha=0.8)
            ax.set_title(
                f"Pearson {pearson[k][j]:+.2f}   Spearman {spearman[k][j]:+.2f}",
                fontsize=9,
            )
            if j == 0:
                ax.set_ylabel(f"code {k}")
            ax.grid(True, alpha=0.3)
    for j, p in enumerate(params):
        ax = axes[d, j]
        lo, hi = P[:, j].min(), P[:, j].max()
        ax.plot([lo, hi], [lo, hi], "-", color="0.6", lw=1)
        ax.scatter(P[:, j], fit_p[:, j], s=12, alpha=0.8, color="C1")
        ax.set_title(f"R$^2$ {r2_p[j]:.3f}   RMSE {rmse_p[j]:.3f}", fontsize=9)
        ax.set_xlabel(f"{p} (design units)")
        if j == 0:
            ax.set_ylabel("affine fit from code")
        ax.grid(True, alpha=0.3)
    shares = ", ".join(f"{s * 100:.0f} %" for s in var_share)
    fig.suptitle(
        f"{run_dir.name}: {n_shapes} shapes, d = {d}\n"
        f"variance of the codes along their principal axes: {shares}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "code_vs_params.png", dpi=120)
    plt.close(fig)

    for k in range(d):
        cells = "  ".join(
            f"{p} {pearson[k][j]:+.2f}/{spearman[k][j]:+.2f}"
            for j, p in enumerate(params)
        )
        log(
            f"code {k} (from {result['code_min'][k]:+.3f} to "
            f"{result['code_max'][k]:+.3f}) Pearson/Spearman vs {cells}"
        )
    log(
        "parameters from the code (affine fit): "
        + "  ".join(
            f"{p} R2 {r2_p[j]:.3f} RMSE {rmse_p[j]:.4f}" for j, p in enumerate(params)
        )
    )
    log(
        "codes from the parameters (affine fit): "
        + "  ".join(f"code {k} R2 {r2_c[k]:.3f}" for k in range(d))
    )
    log(f"variance of the codes along their principal axes: {shares}")
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
    what.add_argument(
        "--decay-interval",
        type=int,
        default=None,
        help="halve both learning rates every this many epochs (default: a "
        "quarter of --epochs)",
    )
    what.add_argument("--force", action="store_true", help="train over an existing run")
    what.add_argument(
        "--check-only",
        action="store_true",
        help="skip training; run the code-vs-parameter check on the existing --run",
    )
    where = parser.add_argument_group("where")
    where.add_argument("--data-root", default=str(ROOT / "data"))
    where.add_argument("--runs-dir", default=str(ROOT / "runs"))
    where.add_argument("--log-dir", default=str(LOG_DIR))
    args = parser.parse_args(argv)
    if isinstance(args.start_at, str):
        args.start_at = unattended.parse_start(args.start_at)
    if args.latent_dim < 1:
        parser.error("--latent-dim must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.run is None:
        args.run = RUN_NAME.format(d=args.latent_dim)
    run_dir = pathlib.Path(args.runs_dir) / args.run
    data_root = pathlib.Path(args.data_root)

    if args.check_only:
        if not (run_dir / "LatentCodes" / "latest.pth").is_file():
            parser.error(f"{run_dir} holds no trained run to check")
        from structsept.app import training

        # The run may sit on plate_hole_2d_n134, the byte-identical twin of
        # plate_hole_2d_xyr; its metadata says which.
        name = (training.read_metadata(run_dir) or {}).get("dataset") or DATASET
        row = dataset_row(data_root, name)
        compare_codes_with_params(run_dir, row["path"], row["split"])
        return

    unattended.open_log(args.log_dir, args.run)
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
            run_dir,
            data_root,
            args.epochs,
            args.force,
            latent_dim=args.latent_dim,
            decay_interval=args.decay_interval,
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
        compare_codes_with_params(run_dir, row["path"], row["split"])
    except Exception as exc:  # the trained run matters more than the plot
        say(f"code-vs-parameter check failed: {exc!r}")
    say("done")


if __name__ == "__main__":
    main()
