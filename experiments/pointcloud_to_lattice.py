"""Point cloud in, lattice out: fit the latent field to a cloud of the whole part.

What this does
--------------
Takes a point cloud of a part and finds the DeepSDF latent codes whose lattice
reproduces it -- the supervisor's "the code receives a point cloud and the
network figures out the geometry". Concretely, steps 3 and 4 of the pipeline
run backwards: instead of prescribing the latent field and generating a mesh,
the geometry is given and the field is solved for.

The design variables are the control points of the latent B-spline, exactly
the ones the MMA optimizes later, so the result of this script is a legitimate
starting point for an optimization run: it starts from a real part instead of
a constant latent.

How it differs from the library's own path
------------------------------------------
:meth:`DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor.fit_mesh`
does the same thing from a *watertight mesh*, whose sign comes from
``SDFfromMesh``. A cloud has no faces and no sign, so the ground truth here is
:class:`structsept.pointcloud_sdf.PointCloudSDF` (winding number for the sign,
KD-tree for the distance) and the samples come from
:func:`structsept.pointcloud_sdf.sample_cloud_sdf`. Everything downstream -- the
structure, the fit, the export -- is the library's, unchanged.

Note that this is *not* an encoder. DeepSDF has no forward pass from points to
a latent code: the decoder stays frozen and the codes are found by gradient
descent (auto-decoder inference). Same effect seen from outside, but the cost
is minutes per part, not a forward pass.

Examples
--------
Simulate a scan off an STL and fit it (nothing is downloaded, all offline)::

    uv run python experiments/pointcloud_to_lattice.py
    uv run python experiments/pointcloud_to_lattice.py --tiling 4 4 4 --iterations 10
    uv run python experiments/pointcloud_to_lattice.py --noise 0.01 --validate

A real cloud, with or without normals in the file::

    uv run python experiments/pointcloud_to_lattice.py --input scan.ply
    uv run python experiments/pointcloud_to_lattice.py --input scan.npz
"""

import argparse
import logging
import pathlib
import time

import matplotlib
import numpy as np
import torch
import trimesh

import DeepSDFStruct  # noqa: E402
from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor  # noqa: E402
from DeepSDFStruct.pretrained_models import PretrainedModels  # noqa: E402
from DeepSDFStruct.SDF import SDFfromMesh  # noqa: E402
from DeepSDFStruct.utils import configure_logging  # noqa: E402

from structsept.pointcloud_sdf import (  # noqa: E402
    PointCloudSDF,
    cloud_from_mesh,
    estimate_normals,
    load_point_cloud,
    sample_cloud_sdf,
)

logger = logging.getLogger(DeepSDFStruct.__name__)

MESH_SUFFIXES = {".stl", ".obj", ".off", ".3mf"}


def load_target(path, n_points, seed, noise):
    """Get an oriented cloud out of *path*, whatever kind of file it is.

    A mesh is sampled into a cloud (and kept, so the fit can be scored against
    it); a cloud file is read as is, and gets PCA normals only if the file
    carries none.

    Returns
    -------
    points, normals : (N, 3) ndarray
    source_mesh : trimesh.Trimesh or None
        The mesh the cloud came from, when the input was a mesh. This is
        ground truth that a real scan would not have -- used only by
        ``--validate``.
    """
    path = pathlib.Path(path)
    if path.suffix.lower() in MESH_SUFFIXES:
        mesh = trimesh.load_mesh(path)
        points, normals = cloud_from_mesh(
            mesh, n_points=n_points, seed=seed, noise=noise
        )
        print(f"  sampled {len(points)} points off {path.name}")
        if noise > 0:
            print(f"  added isotropic noise, std = {noise}")
        return points, normals, mesh

    points, normals = load_point_cloud(path)
    print(f"  loaded {len(points)} points from {path.name}")
    if normals is None:
        print("  file has no normals -- estimating them by PCA (star-shaped only)")
        normals = estimate_normals(points)
    return points, normals, None


def slice_figure(cloud_sdf, struct, bounds, filename, n=200):
    """Save a mid-height slice of target and fit, side by side."""
    import matplotlib.pyplot as plt

    lo, hi = bounds[0].cpu().numpy(), bounds[1].cpu().numpy()
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    gz = np.full_like(gx, 0.5 * (lo[2] + hi[2]))
    queries = torch.tensor(
        np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1), dtype=torch.float32
    )

    with torch.no_grad():
        target = cloud_sdf(queries).reshape(n, n).cpu().numpy()
        fitted = struct(queries).reshape(n, n).cpu().numpy()

    extent = [lo[0], hi[0], lo[1], hi[1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, field, title in zip(
        axes, [target, fitted], ["target (point cloud)", "fitted lattice"]
    ):
        vmax = np.abs(field).max()
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(gx, gy, field, levels=[0.0], colors="k", linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8, label="phi")
    fig.suptitle("mid-height slice, parameter space")
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)
    print(f"  wrote {filename}")


def report_error(label, sdf_a, sdf_b, bounds, n=4000, seed=0, band=0.05):
    """Score *sdf_a* against *sdf_b* on random points inside *bounds*.

    Three numbers, because no single one is fair here:

    ``IoU``
        Intersection over union of the two solid regions. The headline: it
        asks whether the same points ended up inside, which is the only thing
        a lattice can hope to match.
    ``sign``
        Fraction of points both agree about. Optimistic wherever the part is
        small relative to the box -- both fields are mostly "outside".
    ``band``
        Mean ``|dphi|`` restricted to the target's surface band. Away from the
        surface the two fields *should* disagree: inside a lattice, ``phi`` is
        the distance to the nearest bar, not to the part's boundary. Comparing
        them deep inside measures porosity, not fit quality.
    """
    generator = torch.Generator().manual_seed(seed)
    lo, hi = bounds[0].cpu(), bounds[1].cpu()
    queries = torch.rand((n, 3), generator=generator) * (hi - lo) + lo
    with torch.no_grad():
        a = sdf_a(queries).squeeze()
        b = sdf_b(queries).squeeze()

    in_a, in_b = a < 0, b < 0
    union = (in_a | in_b).sum()
    iou = float((in_a & in_b).sum() / union) if union > 0 else float("nan")
    agree = float(((a < 0) == (b < 0)).float().mean())
    near = b.abs() < band
    band_err = float((a - b).abs()[near].mean()) if near.any() else float("nan")

    print(
        f"  {label}: IoU = {100 * iou:5.1f}%, sign = {100 * agree:5.1f}%, "
        f"band |dphi| = {band_err:.4f}"
    )
    return iou, agree


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        default="DeepSDFStruct/tests/data/cone.stl",
        help="point cloud (.ply/.xyz/.npy/.npz) or a mesh to sample one from",
    )
    parser.add_argument("--n-points", type=int, default=20000)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="scanner noise added to a sampled cloud",
    )
    parser.add_argument("--tiling", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument(
        "--iterations", type=int, default=5, help="epochs over the samples"
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-uniform", type=int, default=20000)
    parser.add_argument("--n-surface", type=int, default=60000)
    parser.add_argument(
        "--stds",
        type=float,
        nargs="+",
        default=[0.02, 0.005],
        help=(
            "surface band widths, in normalized units ([-1, 1] cube). Keep the "
            "finest one above a quarter of the point spacing -- below that the "
            "samples describe detail the cloud does not have"
        ),
    )
    parser.add_argument("--resolution", type=int, default=32, help="FlexiCubes grid")
    parser.add_argument(
        "--model",
        default="Primitives",
        help=f"decoder, one of {[m.name for m in PretrainedModels]}",
    )
    parser.add_argument("--output-dir", default="experiments/outputs/pointcloud")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="score the cloud SDF against the source mesh (needs a mesh input)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matplotlib.use("Agg")
    configure_logging(logging.INFO)
    torch.manual_seed(args.seed)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("\n[1/6] target")
    points, normals, source_mesh = load_target(
        args.input, args.n_points, args.seed, args.noise
    )

    print("\n[2/6] structure")
    recon = LocalShapesReconstructor(
        model=PretrainedModels[args.model], output_dir=out_dir
    )
    # build_struct only needs .copy(), .bounds and .apply_transform, so a
    # PointCloud stands in for the mesh it is typed for. It normalizes the
    # cloud into the [-1, 1] cube the decoder works in, and hands back the map
    # (`scaling`) that takes the result back to the part's own millimetres.
    built = recon.build_struct(trimesh.PointCloud(points), args.tiling)
    struct, scaling, bounds = built.struct, built.scaling, built.bounds
    points_norm = np.asarray(built.mesh_norm.vertices)
    print(
        f"  decoder {args.model}, latent dim {recon.latent_dim}, "
        f"tiling {args.tiling} -> {built.param_spline.torch_spline.control_points.shape[0]} "
        "control points"
    )

    print("\n[3/6] ground truth from the cloud")
    # Normals survive normalization untouched: it is a uniform scale plus a
    # translation, so directions are unchanged.
    cloud_sdf = PointCloudSDF(points_norm, normals, device=recon.device)
    print(f"  point spacing (median nearest neighbour): {cloud_sdf.spacing:.4f}")
    if args.validate:
        if source_mesh is None:
            print("  --validate needs a mesh input, skipping")
        else:
            mesh_norm = source_mesh.copy()
            mesh_norm.apply_translation(-np.asarray(built.shift))
            mesh_norm.apply_scale(1.0 / built.scale)
            report_error(
                "cloud SDF vs source mesh",
                cloud_sdf,
                SDFfromMesh(mesh_norm, scale=False),
                bounds,
            )

    print("\n[4/6] samples")
    t0 = time.time()
    samples = sample_cloud_sdf(
        cloud_sdf,
        bounds,
        n_uniform=args.n_uniform,
        n_surface=args.n_surface,
        stds=tuple(args.stds),
        device=recon.device,
        seed=args.seed,
    )
    inside = (samples.distances < 0).float().mean()
    print(
        f"  {samples.samples.shape[0]} samples in {time.time() - t0:.1f}s "
        f"({100 * inside:.1f}% inside)"
    )

    print("\n[5/6] fit")
    iou_before, _ = report_error("before fit", struct, cloud_sdf, bounds)

    # The library's loss plot smooths with a 41-step window and raises on
    # shorter runs, so it is only asked for when the run is long enough.
    n_steps = args.iterations * max(samples.samples.shape[0] // args.batch_size, 1)
    plot_args = {}
    if n_steps >= 60:
        plot_args = {
            "loss_plot_path": out_dir / "fit_loss.png",
            "loss_csv_path": out_dir / "fit_loss.csv",
        }
    else:
        print(f"  only ~{n_steps} steps, skipping the loss plot (needs ~60)")

    t0 = time.time()
    result = LocalShapesReconstructor.fit_samples(
        struct,
        samples,
        num_iterations=args.iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        **plot_args,
    )
    struct.parametrization.set_param(result["params"][0])
    print(
        f"  {result['num_steps']} steps in {time.time() - t0:.1f}s, "
        f"final loss {result['final_loss']:.5f}"
    )
    iou_after, _ = report_error("after fit ", struct, cloud_sdf, bounds)
    print(f"  IoU went from {100 * iou_before:.1f}% to {100 * iou_after:.1f}%")

    print("\n[6/6] export")
    slice_figure(cloud_sdf, struct, bounds, out_dir / "slice_target_vs_fit.png")
    mesh_path = recon.export(
        struct, scaling, mesh_resolution=args.resolution, output_dir=out_dir
    )
    np.savez(
        out_dir / "latent_control_points.npz",
        control_points=result["params"][0].detach().cpu().numpy(),
        tiling=np.array(args.tiling),
        scale=np.array(built.scale),
        shift=np.asarray(built.shift),
    )
    print(f"  wrote {mesh_path}")
    print(f"  wrote {out_dir / 'latent_control_points.npz'} (the design variables)")
    print(f"\ndone in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
