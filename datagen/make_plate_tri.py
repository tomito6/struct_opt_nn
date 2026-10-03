"""Build a DeepSDF training set of square plates with four triangular holes.

The experiment this serves
--------------------------
The second plate family, after the plate with one circular hole
(:mod:`datagen.make_plate_hole`). Four isosceles triangles, one per quarter
of the plate, bases along the edges and tips pointing at the centre, so what
is left is a frame and an X. Their centres are fixed. Two variants, one
generating parameter against two:

* the default -- the triangles' shared height ``h`` is the only parameter,
  between 0.05 and 0.3 on the unit square, and the base is ``2 h``. A latent
  code of dimension 1 should hold the family exactly, the same test as the
  radius-only hole (and it did, 01/10: a straight line in ``h``).
* ``--free-width`` -- the base width ``w`` varies independently of ``h``,
  from 0.1 up to ``w_max(h) = min(0.6, h + 0.5 - sqrt(2) * 0.1)``: the
  ligament between neighbouring triangles is what stops a wide base at a
  small height. Two parameters, latent d = 2, and the question is whether
  the learned plane is spanned by ``(h, w)``.

The details of both -- the ligament arithmetic, the ``(t_h, t_w)`` unit
square the free family is drawn in -- are in :mod:`datagen.plate_tri_params`.

Either way the parameters are used here to draw the shapes and are then
stored beside the samples in ``params.csv``; they are never an input of the
network. The manifest says what varied and what the geometry is, so the two
variants are told apart on disk.

What it does
------------
1. Draw admissible rows (:class:`datagen.plate_tri_params.PlateTriSpace`:
   an even sweep of ``h`` by default, Sobol over ``(h, w)`` with
   ``--free-width``).
2. For each row, sample the exact signed distance of the plate in the
   normalized frame (:mod:`datagen.plate_tri_sdf`): uniform points plus a
   band around the outer edge and the twelve triangle edges.
3. Write the ``.npz`` files, the split, the parameter table and a manifest
   into ``data/`` (:mod:`datagen.dataset`), where the GUI's Train tab lists
   datasets.

``--dim`` has no default on purpose. ``2`` samples the ``(x, y)`` plane only;
``3`` keeps the plate thickness. See :mod:`datagen.plate_hole_sdf` for why the
2-D version loses nothing for through-holes in a plate of constant thickness.

Examples
--------
    uv run python -m datagen.make_plate_tri --dim 2
    uv run python -m datagen.make_plate_tri --dim 2 --n 40 --plot
    uv run python -m datagen.make_plate_tri --dim 2 --n-uniform 25000 --n-band 25000 --name plate_tri_2d_h --plot
    uv run python -m datagen.make_plate_tri --dim 2 --free-width --n-uniform 25000 --n-band 25000 --name plate_tri_2d_hw --plot
    uv run python -m datagen.make_plate_tri --dim 3 --thickness 0.1
    uv run python -m datagen.make_plate_tri --dim 2 --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from datetime import datetime

import numpy as np

from datagen import dataset, preview
from datagen.plate_hole_sdf import (
    DEFAULT_COUNTS,
    DEFAULT_HOLE_FRACTION,
    DEFAULT_PAD,
    DEFAULT_STDS,
    SamplingConfig,
)
from datagen.plate_tri_params import (
    BASE_RATIO,
    DEFAULT_H_MIN,
    DEFAULT_MARGIN,
    DEFAULT_W_CAP,
    DEFAULT_W_MIN,
    SIDES,
    PlateTriSpace,
    plot_space,
)
from datagen.plate_tri_sdf import plate_tri_sdf, sample_instance, square_frame

CLASS_NAME = "plate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample square plates with four triangular holes into a "
        "DeepSDF training set under data/, ready for the Train tab.",
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
        help="dataset name (plate_tri_<dim>d_h, or plate_tri_<dim>d_hw with "
        "--free-width)",
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
    space.add_argument("--size", type=float, default=1.0, help="plate side")
    space.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="smallest ligament anywhere; sets h_max = size/2 - 2 margin and "
        "the width rule",
    )
    space.add_argument("--h-min", type=float, default=DEFAULT_H_MIN)
    space.add_argument(
        "--free-width",
        action="store_true",
        help="let the base width w vary independently of h (latent d = 2 "
        "family) instead of w = 2 h",
    )
    space.add_argument(
        "--w-min", type=float, default=DEFAULT_W_MIN, help="--free-width only"
    )
    space.add_argument(
        "--w-cap",
        type=float,
        default=DEFAULT_W_CAP,
        help="upper limit of the base width, --free-width only",
    )
    space.add_argument(
        "--n",
        type=int,
        default=None,
        help="parameter draws (40 heights; 128 pairs with --free-width)",
    )
    space.add_argument(
        "--method",
        default=None,
        choices=["grid", "sobol", "lhs", "random"],
        help="grid (evenly spaced heights); sobol with --free-width",
    )
    space.add_argument("--seed", type=int, default=0)
    space.add_argument(
        "--t-power",
        type=float,
        default=1.0,
        help=">1 biases a drawn shared-height set towards large triangles",
    )
    space.add_argument(
        "--no-extremes",
        action="store_true",
        help="do not prepend h_min and h_max to a drawn set (a grid always "
        "holds them), or the corners and centre of the (h, w) square with "
        "--free-width",
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
        help="share of the band on the triangle edges",
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
    free = not params.tied
    return {
        "created_by": "datagen.make_plate_tri",
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": "uv run python -m datagen.make_plate_tri " + shlex.join(argv),
        "family": "plate_with_four_triangles",
        "geom_dimension": args.dim,
        "columns": columns,
        "sign": "phi < 0 inside the material (neg), phi >= 0 outside (pos)",
        "dataset": args.name,
        "class": CLASS_NAME,
        "n_instances": len(params),
        "order": "split order = latent code index = params.csv row",
        "parameters": {
            "names": ["h", "w"] if free else ["h"],
            "varied": params.varied(),
            "table": dataset.PARAMS_NAME,
            "units": "design units, plate frame with origin at the lower-left corner",
            "shown_to_network": False,
        },
        "geometry": {
            "triangles": list(SIDES),
            "centres": space.centres.tolist(),
            "shared_height": True,
            "base": ("w, free in [w_min, w_max(h)]" if free else f"{BASE_RATIO:g} h"),
            "base_ratio": None if free else BASE_RATIO,
            "tip": "points at the plate centre; base parallel to the nearest edge",
        },
        "design_space": {
            "size": space.size,
            "margin": space.margin,
            "h_min": space.h_min,
            "h_max": space.h_max,
            "w_min": space.w_min if free else None,
            "w_cap": space.w_cap if free else None,
            "w_max": (
                "min(w_cap, h + size/2 - sqrt(2) margin, size - 2 margin)"
                if free
                else None
            ),
        },
        "parameter_sampling": {
            "family": "height_and_width" if free else "shared_height",
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


def plot_preview(paths, params, frame, dim):
    """Field and stored samples of a few instances, read back from disk."""
    fields = [plate_tri_sdf(dim, frame, h, b) for h, b in zip(params.h, params.base)]
    key = "base" if params.tied else "w"
    titles = [f"h={h:.3f}  {key}={b:.3f}" for h, b in zip(params.h, params.base)]
    return preview.plot_preview(paths, dim, params.names, fields, titles)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    suffix = "_hw" if args.free_width else "_h"
    args.name = args.name or f"plate_tri_{args.dim}d{suffix}"
    # The two families have different natural defaults: an even sweep of the
    # one interval against a low-discrepancy draw of the square. Resolved
    # here, so the manifest records the value that was actually used.
    if args.n is None:
        args.n = 128 if args.free_width else 40
    if args.method is None:
        args.method = "sobol" if args.free_width else "grid"

    space = PlateTriSpace(
        size=args.size,
        margin=args.margin,
        h_min=args.h_min,
        w_min=args.w_min,
        w_cap=args.w_cap,
    )
    if args.free_width:
        params = space.sample_hw(
            n=args.n,
            method=args.method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
        )
    else:
        params = space.sample(
            n=args.n,
            method=args.method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
            t_power=args.t_power,
        )
    # Instance names carry 4 decimals; the de-duplication in the sampler
    # works at 9. Two distinct rows can share a name and overwrite each
    # other's .npz, so stop before writing anything.
    if len(set(params.names)) != len(params):
        raise ValueError(
            f"{len(params) - len(set(params.names))} instance name(s) collide at "
            "4 decimals. Give the plate in design units nearer 1 or draw fewer "
            "shapes."
        )
    frame = square_frame(
        args.size, thickness=args.thickness if args.dim == 3 else None, pad=args.pad
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
        f"({100 * config.hole_fraction:.0f} % on the triangle edges)"
    )
    print(
        f"frame             plate half-size {frame.half_extents[0]:.3f}"
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
    for i, (name, h, base) in enumerate(params.rows()):
        rng = np.random.default_rng([args.seed, i])
        rows = sample_instance(rng, args.dim, frame, h, config, bases=base)
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
