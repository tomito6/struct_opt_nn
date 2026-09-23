"""Build a DeepSDF training set of plates with one circular hole.

The experiment this serves
--------------------------
Two families of the same plate, one generating parameter against three:

* ``--radius-only`` -- the hole sits at the plate centre and only its radius
  ``r`` changes. One parameter, so a latent code of dimension 1 should be able
  to hold the family exactly; whether a trained code tracks ``r`` is the
  cleanest test of the whole pipeline.
* the default -- the centre ``(x_c, y_c)`` and the radius all vary, three
  independent parameters. Trained with a 3-dimensional latent code the family
  fits in principle; with 2 it cannot, and which parameter the code gives up
  is the question of the earlier runs.

Either way the parameters are used here to draw the shapes and are then stored
beside the samples in ``params.csv``; they are never an input of the network.
The decoder learns its own code per shape, and the answers above are read
afterwards by comparing the learned codes with that table. The manifest says
which parameters varied, so the two families are told apart on disk.

What it does
------------
1. Draw admissible ``(x_c, y_c, r)`` triples
   (:class:`datagen.plate_hole_params.PlateHoleSpace`: Sobol by default, or
   an even sweep of the radius alone with ``--radius-only``).
2. For each triple, sample the exact signed distance of the plate in the
   normalized frame (:mod:`datagen.plate_hole_sdf`): uniform points plus a band
   around the outer edge and the hole wall.
3. Write the ``.npz`` files, the split, the parameter table and a manifest into
   ``data/`` (:mod:`datagen.dataset`), where the GUI's Train tab lists datasets.

``--dim`` has no default on purpose. ``2`` samples the ``(x, y)`` plane only;
``3`` keeps the plate thickness. See :mod:`datagen.plate_hole_sdf` for why the
2-D version loses nothing for a through-hole of constant thickness.

Training margin vs design margin
--------------------------------
The default ``--margin 0.05`` is looser than the 0.1 ligament the design space
uses (``PlateHoleSpace`` default): train on a slightly larger family than the
optimizer will be allowed to explore, so the design bounds never sit on the
edge of what the decoder has seen. This reproduces the 134-shape set of
``outputs/plate_hole_params.csv`` (``--n 128``, seed 0).

Examples
--------
    uv run python -m datagen.make_plate_hole --dim 2
    uv run python -m datagen.make_plate_hole --dim 2 --radius-only --n 40 --plot
    uv run python -m datagen.make_plate_hole --dim 2 --n 32 --name pilot_2d --plot
    uv run python -m datagen.make_plate_hole --dim 3 --thickness 0.1
    uv run python -m datagen.make_plate_hole --dim 2 --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from datetime import datetime

import numpy as np

from datagen import dataset
from datagen.plate_hole_params import (
    DEFAULT_R_MIN,
    PlateHoleSpace,
    plot_space,
)
from datagen.plate_hole_sdf import (
    DEFAULT_COUNTS,
    DEFAULT_HOLE_FRACTION,
    DEFAULT_PAD,
    DEFAULT_STDS,
    PlateFrame,
    SamplingConfig,
    plate_hole_sdf,
    sample_instance,
)

TRAINING_MARGIN = 0.05
CLASS_NAME = "plate"
PREVIEW_INSTANCES = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample plates with one circular hole into a DeepSDF training "
        "set under data/, ready for the Train tab.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[2, 3],
        required=True,
        help="2: rows (x, y, phi) on the plate plane; 3: rows (x, y, z, phi)",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--name",
        default=None,
        help="dataset name (plate_hole_<dim>d, or plate_hole_<dim>d_r with "
        "--radius-only)",
    )
    out.add_argument("--data-root", default=str(dataset.DEFAULT_DATA_ROOT))
    out.add_argument("--overwrite", action="store_true", help="replace the dataset")
    out.add_argument(
        "--plot", action="store_true", help="also write parameters.png, preview.png"
    )
    out.add_argument(
        "--dry-run", action="store_true", help="report what would be written, stop"
    )

    space = parser.add_argument_group("parameter space (design units)")
    space.add_argument("--length", type=float, default=1.0, help="plate x size")
    space.add_argument("--width", type=float, default=1.0, help="plate y size")
    space.add_argument(
        "--margin",
        type=float,
        nargs="+",
        default=[TRAINING_MARGIN],
        help="smallest ligament: one value, or four (left right bottom top)",
    )
    space.add_argument("--r-min", type=float, default=DEFAULT_R_MIN)
    space.add_argument(
        "--radius-only",
        action="store_true",
        help="fix the hole centre (at --centre) and vary only the radius",
    )
    space.add_argument(
        "--centre",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="fixed hole centre in design units, --radius-only only; "
        "default: the plate centre",
    )
    space.add_argument(
        "--n",
        type=int,
        default=None,
        help="parameter draws (128; 40 radii with --radius-only)",
    )
    space.add_argument(
        "--method",
        default=None,
        choices=["sobol", "lhs", "random", "grid"],
        help="sobol; with --radius-only the default is grid (evenly spaced radii)",
    )
    space.add_argument("--seed", type=int, default=0)
    space.add_argument(
        "--t-power", type=float, default=1.0, help=">1 biases towards large holes"
    )
    space.add_argument(
        "--no-extremes",
        action="store_true",
        help="do not prepend the corners and centre of the design box "
        "(with --radius-only and a drawn --method: the two end radii; a grid "
        "always holds them)",
    )

    geom = parser.add_argument_group("geometry and frame")
    geom.add_argument(
        "--thickness", type=float, default=0.1, help="plate thickness (--dim 3 only)"
    )
    geom.add_argument(
        "--pad",
        type=float,
        default=DEFAULT_PAD,
        help="empty ring around the plate inside [-1, 1]^d",
    )

    samp = parser.add_argument_group("SDF samples per instance")
    samp.add_argument("--n-uniform", type=int, default=None, help="uniform in the box")
    samp.add_argument("--n-band", type=int, default=None, help="near the boundary")
    samp.add_argument(
        "--hole-fraction",
        type=float,
        default=DEFAULT_HOLE_FRACTION,
        help="share of the band on the hole wall",
    )
    samp.add_argument(
        "--stds",
        type=float,
        nargs="+",
        default=list(DEFAULT_STDS),
        help="band offsets along the normal (normalized units)",
    )
    return parser


def make_config(args) -> SamplingConfig:
    n_uniform, n_band = DEFAULT_COUNTS[args.dim]
    return SamplingConfig(
        n_uniform=n_uniform if args.n_uniform is None else args.n_uniform,
        n_band=n_band if args.n_band is None else args.n_band,
        hole_fraction=args.hole_fraction,
        stds=args.stds,
    )


def make_manifest(args, space, params, frame, config, argv) -> dict:
    columns = ["x", "y", "phi"] if args.dim == 2 else ["x", "y", "z", "phi"]
    return {
        "created_by": "datagen.make_plate_hole",
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": "uv run python -m datagen.make_plate_hole " + shlex.join(argv),
        "family": "plate_with_hole",
        "geom_dimension": args.dim,
        "columns": columns,
        "sign": "phi < 0 inside the material (neg), phi >= 0 outside (pos)",
        "dataset": args.name,
        "class": CLASS_NAME,
        "n_instances": len(params),
        "order": "split order = latent code index = params.csv row",
        "parameters": {
            "names": ["x_c", "y_c", "r"],
            "varied": params.varied(),
            "fixed_centre": (
                [float(params.x_c[0]), float(params.y_c[0])]
                if args.radius_only
                else None
            ),
            "table": dataset.PARAMS_NAME,
            "units": "design units, plate frame with origin at the lower-left corner",
            "shown_to_network": False,
        },
        "design_space": {
            "length": space.length,
            "width": space.width,
            "margins": list(space.margins),
            "r_min": space.r_min,
        },
        "parameter_sampling": {
            "family": "radius_only" if args.radius_only else "centre_and_radius",
            "n_requested": args.n,
            "method": args.method,
            "seed": args.seed,
            "t_power": args.t_power,
            "include_extremes": not args.no_extremes,
        },
        "frame": frame.as_dict(args.dim),
        "sdf_sampling": config.as_dict()
        | {"seeding": "instance i uses numpy.random.default_rng([seed, i])"},
    }


def plot_preview(paths, params, frame, dim, n_show=PREVIEW_INSTANCES):
    """Field and stored samples of a few instances, read back from disk.

    Each panel shows the exact field (colour, zero level in black) and, on top,
    a subsample of the rows actually written to the ``.npz`` -- the check that
    the files hold what the field says. For ``dim=3`` it is the mid-plane
    ``z = 0`` and the samples within a thin slab around it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.unique(
        np.linspace(0, len(params) - 1, min(n_show, len(params))).astype(int)
    )
    names = params.names
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4))
    grid = np.linspace(-1.0, 1.0, 241)
    gx, gy = np.meshgrid(grid, grid)
    plane = np.column_stack([gx.ravel(), gy.ravel()])
    if dim == 3:
        plane = np.column_stack([plane, np.zeros(len(plane))])

    for ax, i in zip(axs.ravel(), idx):
        phi = plate_hole_sdf(dim, frame, params.x_c[i], params.y_c[i], params.r[i])
        field = phi(plane).reshape(gx.shape)
        lim = float(np.abs(field).max())
        ax.imshow(
            field,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        ax.contour(gx, gy, field, levels=[0.0], colors="k", linewidths=1.2)

        with np.load(paths["class_dir"] / f"{names[i]}.npz") as npz:
            rows = np.vstack([npz["pos"], npz["neg"]])
        if dim == 3:
            rows = rows[np.abs(rows[:, 2]) < 0.02]
        rng = np.random.default_rng(i)
        rows = rows[rng.permutation(len(rows))[:1500]]
        ax.scatter(
            rows[:, 0],
            rows[:, 1],
            s=1.5,
            c=np.where(rows[:, -1] < 0, "k", "0.55"),
            linewidths=0,
        )
        ax.set_title(
            f"x_c={params.x_c[i]:.3f}  y_c={params.y_c[i]:.3f}  r={params.r[i]:.3f}",
            fontsize=9,
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
    for ax in axs.ravel()[len(idx) :]:
        ax.axis("off")

    where = "plane" if dim == 2 else "mid-plane z = 0"
    fig.suptitle(
        f"{paths['dataset_dir'].name}: exact field on the {where} "
        "(black line: surface) and stored samples (black: inside)"
    )
    fig.tight_layout()
    out = paths["dataset_dir"] / "preview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.centre is not None and not args.radius_only:
        parser.error("--centre only makes sense together with --radius-only")
    suffix = "_r" if args.radius_only else ""
    args.name = args.name or f"plate_hole_{args.dim}d{suffix}"
    # The two families have different natural defaults: a low-discrepancy
    # draw of the box against an even sweep of the one interval. Resolved
    # here, so the manifest records the value that was actually used.
    if args.n is None:
        args.n = 40 if args.radius_only else 128
    if args.method is None:
        args.method = "grid" if args.radius_only else "sobol"

    space = PlateHoleSpace(
        length=args.length,
        width=args.width,
        margin=args.margin if len(args.margin) > 1 else args.margin[0],
        r_min=args.r_min,
    )
    if args.radius_only:
        params = space.sample_radius(
            n=args.n,
            centre=args.centre,
            method=args.method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
            t_power=args.t_power,
        )
    else:
        params = space.sample(
            n=args.n,
            method=args.method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
            t_power=args.t_power,
        )
    # Instance names carry 4 decimals; the de-duplication in sample() works at
    # 9. On a plate measured in small units two distinct holes can share a
    # name and overwrite each other's .npz, so stop before writing anything.
    if len(set(params.names)) != len(params):
        raise ValueError(
            f"{len(params) - len(set(params.names))} instance name(s) collide at "
            "4 decimals. Give the plate in design units nearer 1 (e.g. mm "
            "instead of m) or draw fewer shapes."
        )
    frame = PlateFrame(
        length=args.length,
        width=args.width,
        thickness=args.thickness if args.dim == 3 else None,
        pad=args.pad,
    )
    config = make_config(args)
    n_rows = config.n_uniform + config.n_band
    megabytes = len(params) * n_rows * (args.dim + 1) * 4 / 1e6

    print("=== parameter space ===")
    print(space.summary())
    print("\n=== instances ===")
    print(params.summary())
    print("\n=== samples ===")
    print(
        f"dimension         {args.dim}  (rows {'x, y' if args.dim == 2 else 'x, y, z'}, phi)"
    )
    print(
        f"per instance      {config.n_uniform} uniform + {config.n_band} band "
        f"({100 * config.hole_fraction:.0f} % on the hole wall)"
    )
    print(
        f"frame             plate half-size {frame.half_extents[0]:.3f} x "
        f"{frame.half_extents[1]:.3f}"
        + (f", half-thickness {frame.half_thickness:.3f}" if args.dim == 3 else "")
        + f"  (scale {frame.scale:.3f})"
    )
    paths = dataset.dataset_paths(args.data_root, args.name, CLASS_NAME)
    print(f"output            {paths['dataset_dir']}  (~{megabytes:.0f} MB)")

    if args.dry_run:
        print("\ndry run: nothing written")
        return paths

    paths = dataset.prepare(args.data_root, args.name, CLASS_NAME, args.overwrite)
    started = time.perf_counter()
    inside = []
    for i, (name, x_c, y_c, r) in enumerate(params.rows()):
        rng = np.random.default_rng([args.seed, i])
        rows = sample_instance(rng, args.dim, frame, x_c, y_c, r, config)
        info = dataset.write_instance(paths["class_dir"], name, rows)
        inside.append(info["n_neg"] / (info["n_pos"] + info["n_neg"]))
        if (i + 1) % max(1, len(params) // 10) == 0 or i + 1 == len(params):
            print(f"  {i + 1:4d}/{len(params)} written")

    dataset.write_split(paths, args.name, CLASS_NAME, params.names)
    params.save_csv(paths["params"])
    dataset.write_manifest(
        paths, make_manifest(args, space, params, frame, config, argv)
    )
    print(
        f"\n{len(params)} instances in {time.perf_counter() - started:.1f} s, "
        f"inside share {100 * min(inside):.0f}-{100 * max(inside):.0f} %"
    )
    print(f"split             {paths['split']}")
    print(f"parameters        {paths['params']}")
    print(f"manifest          {paths['manifest']}")

    if args.plot:
        print(
            f"plot              {plot_space(params, paths['dataset_dir'] / 'parameters.png')}"
        )
        print(f"plot              {plot_preview(paths, params, frame, args.dim)}")
    return paths


if __name__ == "__main__":
    main()
