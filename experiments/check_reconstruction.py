"""Does a trained plate decoder reproduce the shapes it was trained on?

Why
---
The loss a run reports cannot answer that. It is an average over sampled
points, taken with dropout on, and a decoder can reach a low, flat loss while
drawing a hole that is not there. ``runs/plate_hole_2d_xyr_d3_20260930_0124``
does exactly that: final loss 0.0058, and in 27 of its 134 *training* shapes
a second, spurious hole of about the size of the real one, in what should be
solid material (found in the audit of 2026-10-02, ``IDEIAS.md``).

So this decodes every training shape with its own trained code, the way the
Explore 2-D tab would show it, and compares the result with the exact field
the dataset was sampled from. Nothing is trained and nothing in ``runs/`` or
``data/`` is written.

What is measured, per training shape
------------------------------------
On a ``res x res`` node grid over [-1, 1]^2, decoder in eval mode:

* **Eq. 38** - the paper's reconstruction error: the mean of ``|SDF_exact|``
  over the decoder's zero level set. 0.0128 in the paper's test case 1,
  0.0156 in test case 2 (Sec. 3.1.1 and 3.2.1).
* **band error** - mean ``|f - SDF_exact|`` where ``|SDF_exact| < 0.1``,
  the band the clamped loss fits.
* **wrong sign** - share of the grid where material and void are swapped.
* **ghost voids** - enclosed void regions of the decoded field that touch no
  hole of the exact one: a hole that is not in the shape.
* **missing voids** - enclosed void regions of the exact field that no
  decoded void touches: a hole that was filled.

A hole that is merely the wrong size, bloated or shifted is neither: it
still overlaps the true one, and the wrong-sign share and Eq. 38 say how far
off it is. A hole shifted clear of the true one counts as both a ghost and a
missing hole, which is what it is. Regions smaller than :data:`MIN_AREA` -
grid specks at a corner of the plate - are not counted as holes at all.

The last two are the ones no averaged number shows. Eq. 38 of the run above
is 0.0136 in the mean, next to the paper's value, but 0.0032 in the median:
the mean is carried by the ghost shapes.

The exact field comes from ``datagen`` (``plate_hole_sdf`` /
``plate_tri_sdf``) with the frame stored in the dataset's manifest, and is
checked against the stored samples of the first shape before it is trusted.

How to run
----------
::

    uv run python experiments/check_reconstruction.py            # every trained 2-D run
    uv run python experiments/check_reconstruction.py --run plate_tri_2d_h_d1_4x64
    uv run python experiments/check_reconstruction.py --run A --run B --res 100 --no-plot
    uv run python experiments/check_reconstruction.py --run A --checkpoint 100

One line per run is printed; ``outputs/reconstruction/<run>.csv`` holds the
per-shape numbers and ``<run>.png`` the eight worst shapes, decoded contour
(red) over the exact one (cyan, dashed). About 10 s per run for the 4 x 64
decoders and 1-2 min for the 8 x 256 ones at the default resolution.

``--checkpoint`` reads an earlier snapshot of the run instead of its
``latest.pth`` - any epoch the run kept under ``ModelParameters/`` - and
writes ``<run>_ep<epoch>.csv`` / ``.png``, so the ghost count can be followed
along one training trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy import ndimage

from datagen.plate_hole_sdf import PlateFrame, plate_hole_sdf
from datagen.plate_tri_sdf import plate_tri_sdf
from structsept.app import models, unattended

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "reconstruction"

#: Clamping distance of the runs: the band in which the field was fitted.
BAND = 0.1
#: Reconstruction error the paper reports for its first test case (Eq. 38).
PAPER_EQ38 = 0.0128
#: Smallest void region counted as a hole, as an area of the [-1, 1]^2 frame
#: (a disc of radius 0.022). The smallest hole of any family is five times
#: that - the triangle of height 0.05, 0.008 - so nothing real is dropped.
MIN_AREA = 1.5e-3
#: Largest disagreement accepted between the recomputed exact field and the
#: samples stored in the dataset (float32 and a 6-decimal params.csv).
STORED_TOL = 1e-4


# --------------------------------------------------------------------------- #
# the run and its dataset
# --------------------------------------------------------------------------- #


def load_run(run_dir, checkpoint="latest"):
    """Decoder, trained codes and exact fields of one run.

    Parameters
    ----------
    run_dir : path-like
    checkpoint : str or int
        ``"latest"`` or the epoch of a snapshot the run kept.

    Returns
    -------
    dict
        ``model`` (eval mode, as ``models.load_model`` returns it), ``codes``
        ``(n, d)``, ``names`` in latent-index order, ``fields`` (one callable
        ``phi(points)`` per shape), ``dataset`` and ``entry``.

    Raises
    ------
    SystemExit
        When the run is not a trained 2-D decoder of a ``datagen`` plate
        family, or the recomputed field does not match the stored samples.
    """
    run_dir = pathlib.Path(run_dir)
    entry = next(
        (e for e in models.list_models(run_dir.parent) if e.name == run_dir.name),
        None,
    )
    if entry is None or entry.source != "run":
        raise SystemExit(f"{run_dir.name}: no trained checkpoint in {run_dir}")
    if entry.geom_dimension != 2:
        raise SystemExit(f"{run_dir.name}: not a 2-D decoder")

    specs = json.loads((run_dir / "specs.json").read_text(encoding="utf-8"))
    split = json.loads(pathlib.Path(specs["TrainSplit"]).read_text(encoding="utf-8"))
    dataset = next(iter(split))
    class_name = next(iter(split[dataset]))
    names = unattended.split_names(specs["TrainSplit"])
    data_dir = pathlib.Path(specs["DataSource"]) / "SdfSamples" / dataset

    manifest = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    with open(data_dir / "params.csv", newline="", encoding="utf-8") as f:
        params = {row["name"]: row for row in csv.DictReader(f)}
    frame = PlateFrame(
        length=manifest["frame"]["length"],
        width=manifest["frame"]["width"],
        pad=manifest["frame"]["pad"],
    )
    fields = [_exact_field(manifest, frame, params[name]) for name in names]

    # The frame and the parameters are read back from files; a mismatch would
    # make every number below meaningless, so the first shape is checked
    # against the samples the network was actually trained on.
    stored = np.load(data_dir / class_name / f"{names[0]}.npz")
    rows = np.vstack([stored["pos"], stored["neg"]])
    worst = float(np.abs(fields[0](rows[:, :2]) - rows[:, 2]).max())
    if worst > STORED_TOL:
        raise SystemExit(
            f"{run_dir.name}: the recomputed field of {names[0]} is {worst:.2e} "
            "off the stored samples - wrong frame or parameters"
        )

    if not (run_dir / "ModelParameters" / f"{checkpoint}.pth").is_file():
        raise SystemExit(f"{run_dir.name}: no checkpoint '{checkpoint}' saved")
    model = models.load_model(entry, checkpoint)
    codes = models.trained_latents(model)
    if len(codes) != len(names):
        raise SystemExit(
            f"{run_dir.name}: {len(codes)} codes for {len(names)} shapes in the split"
        )
    return {
        "model": model,
        "codes": codes,
        "names": names,
        "fields": fields,
        "dataset": dataset,
        "entry": entry,
    }


def _exact_field(manifest, frame, row):
    """``phi(points)`` of one ``params.csv`` row, by the family that made it."""
    maker = str(manifest.get("created_by", ""))
    if maker.endswith("make_plate_hole"):
        return plate_hole_sdf(
            2, frame, float(row["x_c"]), float(row["y_c"]), float(row["r"])
        )
    if maker.endswith("make_plate_tri"):
        # the shared-height family has no width column: base = 2 h
        width = float(row["w"]) if "w" in row else None
        return plate_tri_sdf(2, frame, float(row["h"]), width)
    raise SystemExit(f"no exact field known for datasets made by '{maker}'")


# --------------------------------------------------------------------------- #
# one shape
# --------------------------------------------------------------------------- #


def zero_crossings(field, xs) -> np.ndarray:
    """Points of the zero level set: sign changes along grid edges, placed by
    linear interpolation. ``field`` is ``(row=y, col=x)``; returns ``(m, 2)``."""
    points = []
    left, right = field[:, :-1], field[:, 1:]
    hit = (left < 0) != (right < 0)
    iy, ix = np.nonzero(hit)
    t = left[hit] / (left[hit] - right[hit])
    points.append(np.stack([xs[ix] + t * (xs[ix + 1] - xs[ix]), xs[iy]], axis=1))
    low, high = field[:-1, :], field[1:, :]
    hit = (low < 0) != (high < 0)
    iy, ix = np.nonzero(hit)
    t = low[hit] / (low[hit] - high[hit])
    points.append(np.stack([xs[ix], xs[iy] + t * (xs[iy + 1] - xs[iy])], axis=1))
    return np.vstack(points)


def enclosed_voids(field, node_area):
    """Label image and labels of the void regions that do not reach the border.

    The region that does is the air around the plate; every other one is a
    hole. Regions under :data:`MIN_AREA` are dropped; ``node_area`` is the
    area one grid node stands for.
    """
    labels, n = ndimage.label(field >= 0)
    border = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = [
        k
        for k in range(1, n + 1)
        if k not in border and sizes[k - 1] * node_area >= MIN_AREA
    ]
    return labels, keep


def measure(decoded, exact, xs, phi) -> dict:
    """The numbers of the module docstring for one decoded field.

    Parameters
    ----------
    decoded, exact : numpy.ndarray, shape (res, res)
        Decoder output and exact field on the same node grid.
    xs : numpy.ndarray, shape (res,)
        Node coordinates along either axis.
    phi : callable
        The exact field, for the points of the decoded zero level set.
    """
    band = np.abs(exact) < BAND
    contour = zero_crossings(decoded, xs)
    node_area = float(xs[1] - xs[0]) ** 2
    labels_d, voids_d = enclosed_voids(decoded, node_area)
    labels_e, voids_e = enclosed_voids(exact, node_area)
    # a ghost touches no exact void; a missing hole is touched by no decoded one
    ghosts = [k for k in voids_d if not np.any(exact[labels_d == k] >= 0)]
    missing = [k for k in voids_e if not np.any(decoded[labels_e == k] >= 0)]
    return {
        "eq38": float(np.mean(np.abs(phi(contour)))) if len(contour) else np.nan,
        "band_err": float(np.mean(np.abs(decoded[band] - exact[band]))),
        "wrong_sign": float(np.mean((decoded < 0) != (exact < 0))),
        "voids_decoded": len(voids_d),
        "voids_exact": len(voids_e),
        "ghost_voids": len(ghosts),
        "missing_voids": len(missing),
        # area of the [-1, 1]^2 frame taken by the ghosts, 0 when there is none
        "ghost_area": float(sum(np.sum(labels_d == k) for k in ghosts) * node_area),
    }


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #


def check_run(run_dir, res=200, plot=True, log=print, checkpoint="latest") -> dict:
    """Measure every training shape of a run; write its csv and figure.

    ``checkpoint`` is ``"latest"`` or the epoch of an earlier snapshot; the
    files of an earlier one are named ``<run>_ep<epoch>``. Returns the
    summary row that is also printed.
    """
    run = load_run(run_dir, checkpoint)
    model, codes, names = run["model"], run["codes"], run["names"]
    xs = np.linspace(-1.0, 1.0, int(res))
    grid_x, grid_y = np.meshgrid(xs, xs)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    rows, fields = [], []
    for name, code, phi in zip(names, codes, run["fields"]):
        decoded = models.eval_field_2d(model, code, res)
        exact = phi(points).reshape(decoded.shape)
        rows.append({"name": name, **measure(decoded, exact, xs, phi)})
        fields.append((decoded, exact))

    eq38 = np.array([r["eq38"] for r in rows])
    wrong = np.array([r["wrong_sign"] for r in rows])
    entry = run["entry"]
    latest = str(checkpoint) == "latest"
    summary = {
        "run": pathlib.Path(run_dir).name,
        "dataset": run["dataset"],
        "shapes": len(rows),
        "epoch": entry.epoch if latest else int(checkpoint),
        "planned_epochs": entry.planned_epochs,
        "eq38_mean": float(np.nanmean(eq38)),
        "eq38_median": float(np.nanmedian(eq38)),
        "above_paper": int(np.sum(eq38 > PAPER_EQ38)),
        "band_err": float(np.mean([r["band_err"] for r in rows])),
        "wrong_sign_mean": float(wrong.mean()),
        "wrong_sign_max": float(wrong.max()),
        "ghost_shapes": int(sum(r["ghost_voids"] > 0 for r in rows)),
        "missing_shapes": int(sum(r["missing_voids"] > 0 for r in rows)),
    }

    stem = summary["run"] if latest else f"{summary['run']}_ep{summary['epoch']}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        _plot_worst(OUT_DIR / f"{stem}.png", summary, rows, fields, xs)

    log(_summary_line(summary))
    return summary


def _summary_line(s) -> str:
    epochs = f"{s['epoch']}/{s['planned_epochs']}"
    topology = "topology ok"
    if s["ghost_shapes"] or s["missing_shapes"]:
        topology = (
            f"GHOST HOLE in {s['ghost_shapes']}, hole missing in "
            f"{s['missing_shapes']} of {s['shapes']}"
        )
    return (
        f"{s['run']:<40s} epoch {epochs:>9s}  Eq.38 {s['eq38_mean']:.4f} "
        f"(median {s['eq38_median']:.4f}, {s['above_paper']} > {PAPER_EQ38})  "
        f"band {s['band_err']:.4f}  wrong sign {100 * s['wrong_sign_mean']:.2f} % "
        f"(max {100 * s['wrong_sign_max']:.1f} %)  {topology}"
    )


def _plot_worst(path, summary, rows, fields, xs, n=8):
    """The ``n`` worst shapes: wrong topology first, then by wrong-sign share."""
    import matplotlib.pyplot as plt

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["ghost_voids"] + rows[i]["missing_voids"] > 0,
            rows[i]["wrong_sign"],
        ),
        reverse=True,
    )[:n]
    cols = min(4, len(order))
    lines = -(-len(order) // cols)
    fig, axes = plt.subplots(
        lines, cols, figsize=(3.4 * cols, 3.6 * lines), squeeze=False
    )
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, i in zip(axes.ravel(), order):
        decoded, exact = fields[i]
        ax.set_axis_on()
        ax.contourf(xs, xs, decoded, levels=[-1e9, 0.0], colors=["#444444"])
        ax.contour(xs, xs, decoded, levels=[0.0], colors="red", linewidths=1.2)
        ax.contour(
            xs, xs, exact, levels=[0.0], colors="cyan", linewidths=1.0, linestyles="--"
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        tag = ""
        if rows[i]["ghost_voids"]:
            tag += "  GHOST"
        if rows[i]["missing_voids"]:
            tag += "  MISSING"
        ax.set_title(f"{i}: {rows[i]['name']}{tag}", fontsize=7)
    fig.suptitle(
        f"{summary['run']} - decoded (gray, red) against exact (cyan): "
        f"the {len(order)} worst of {summary['shapes']} training shapes",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        help="run directory name under --runs; repeat for several (default: "
        "every trained 2-D run)",
    )
    parser.add_argument("--runs", default=str(ROOT / "runs"), help="runs directory")
    parser.add_argument("--res", type=int, default=200, help="grid nodes per axis")
    parser.add_argument("--no-plot", action="store_true", help="csv only, no figure")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="'latest' (default) or the epoch of an earlier snapshot of the run",
    )
    args = parser.parse_args(argv)

    runs_dir = pathlib.Path(args.runs)
    names = args.run or [
        e.name
        for e in models.list_models(runs_dir, geom_dimension=2)
        if e.source == "run"
    ]
    if not names:
        raise SystemExit(f"no trained 2-D run under {runs_dir}")
    summaries = []
    for name in names:
        try:
            summaries.append(
                check_run(
                    runs_dir / name,
                    args.res,
                    not args.no_plot,
                    checkpoint=args.checkpoint,
                )
            )
        except SystemExit as exc:
            # one run of another kind must not hide the others
            print(f"skipped - {exc}")
    print(f"per-shape numbers and figures: {OUT_DIR}")
    return summaries


if __name__ == "__main__":
    main()
