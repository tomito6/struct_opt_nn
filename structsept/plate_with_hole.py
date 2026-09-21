"""Lattice plate with a circular hole through it -- geometry only, no FEM.

Everything here is geometry: build the SDF, mesh it, look at it. The one
function you are meant to poke at is :func:`plate_with_hole`, whose signature is
the hole itself -- radius and the (x, y) centre of the circle, in metres.

    from structsept.plate_with_hole import plate_with_hole, mesh_and_export

    plate, deformation = plate_with_hole(hole_radius=0.3, hole_center=(0.5, 0.5))
    mesh_and_export(plate, deformation)

Or straight from the shell::

    uv run python -m structsept.plate_with_hole --hole-radius 0.3
    uv run python -m structsept.plate_with_hole --hole-radius 0.15 --hole-center 0.3 0.7

Coordinates are physical metres measured from the plate's lower-left corner, so
on the default 1 x 1 m plate the middle is (0.5, 0.5).
"""

import argparse
import logging
import pathlib

import matplotlib
import numpy as np
import splinepy
import torch

import DeepSDFStruct
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.mesh import create_3D_mesh
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.pretrained_models import PretrainedModels, get_model
from DeepSDFStruct.sdf_primitives import BoxSDF, CylinderSDF
from DeepSDFStruct.SDF import CappedBorderSDF, DifferenceSDF, SDFBase, SDFfromDeepSDF
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.utils import configure_logging

logger = logging.getLogger(DeepSDFStruct.__name__)


class ScaledSpaceSDF(SDFBase):
    """Evaluate an SDF defined in physical metres from parametric queries.

    The lattice lives on the parametric cube [0, 1]^3 and only becomes a
    1 x 1 x 0.1 m plate when the deformation spline is applied to the *mesh
    vertices*, at the very end. So a hole described in metres cannot simply be
    subtracted from it -- the two live in different coordinate systems.

    This wrapper bridges them: it stretches the parametric query points into
    physical space, evaluates the shape there, and divides the resulting
    distance by the same scale on the way back.

    The zero level set -- the surface, which is all the mesher looks for -- is
    exact. The magnitudes are only approximately distances when the plate is
    not square, because a single scalar cannot undo an anisotropic stretch;
    FlexiCubes uses them to position vertices, so they should at least be of
    the right order, which the division ensures.

    Parameters
    ----------
    sdf : SDFBase
        Shape defined in physical coordinates.
    scale : sequence of float
        Physical size of the plate, i.e. the parametric-to-physical factor per
        axis.
    """

    def __init__(self, sdf: SDFBase, scale):
        super().__init__()
        self.sdf = sdf
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=queries.device, dtype=queries.dtype)
        return self.sdf(queries * scale) / scale.mean()

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def plate_with_hole(
    hole_radius=0.25,
    hole_center=(0.5, 0.5),
    length=1.0,
    width=1.0,
    thickness=0.1,
    tiling=(5, 5, 1),
    latent=0.4,
    solid=False,
    device=None,
):
    """Build a plate with one circular hole punched through it.

    Parameters
    ----------
    hole_radius : float
        Radius of the hole in metres. Pass 0 (or less) for no hole.
    hole_center : tuple of float
        (x, y) centre of the hole in metres, from the plate's lower-left
        corner. The hole always runs the full thickness, along z.
    length, width, thickness : float
        Plate size in metres (x, y, z).
    tiling : tuple of int
        Unit cells per axis. Ignored when ``solid``.
    latent : float
        Constant DeepSDF latent value. Stay inside the trained range
        [0.15, 0.75]; outside it the decoder produces meaningless geometry.
        Ignored when ``solid``.
    solid : bool
        Make the plate a plain solid slab instead of a lattice. The neural
        network is not loaded at all in this case -- a solid plate is an
        analytic box, and none of DeepSDFStruct's machinery buys you anything.
        It is here for comparison: the lattice is what gives the optimizer
        something to design.
    device : str, optional
        Defaults to cuda when available.

    Returns
    -------
    plate : SDFBase
        Watertight SDF on the parametric cube [0, 1]^3.
    deformation : TorchSpline
        Parametric-to-physical map. Pass it to ``create_3D_mesh`` as
        ``deformation_function`` -- the SDF alone knows nothing about metres.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if solid:
        # The parametric cube itself: a filled slab once the deformation
        # stretches it to length x width x thickness.
        lattice = BoxSDF(center=[0.5, 0.5, 0.5], extents=[1.0, 1.0, 1.0])
    else:
        model = get_model(PretrainedModels.AnalyticRoundCross, device=device)
        microtile = SDFfromDeepSDF(model)

        lattice = LatticeSDFStruct(
            tiling=list(tiling),
            microtile=microtile,
            parametrization=Constant([latent], device=model.device),
        )

    if hole_radius > 0:
        cx, cy = hole_center
        # Overshoot the plate in z so the cut is a clean through-hole rather
        # than a blind pocket whose end cap lands inside the material.
        hole = CylinderSDF(
            point_a=[cx, cy, -thickness],
            point_b=[cx, cy, 2 * thickness],
            radius=hole_radius,
        )
        body = DifferenceSDF(lattice, ScaledSpaceSDF(hole, [length, width, thickness]))
    else:
        body = lattice

    # Both the hole and the border caps remove material, so their order does
    # not matter -- but the caps are what make the result watertight, so they
    # cannot be skipped.
    plate = CappedBorderSDF(body)

    deformation = TorchSpline(
        splinepy.helpme.create.box(length, width, thickness).bspline,
        device=device,
    )
    return plate, deformation


def mesh_and_export(
    plate,
    deformation,
    resolution=14,
    outdir="experiments/outputs",
    render=True,
    slices=True,
    show=False,
):
    """Mesh the plate, report what came out, and write it to disk.

    Returns the surface mesh so you can keep working with it in a notebook.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = str(plate.get_device())

    if slices:
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (origin, normal, title) in zip(
            axs,
            [
                ((0, 0, 0.5), (0, 0, 1), "mid-plane z (top view)"),
                ((0, 0.5, 0), (0, 1, 0), "section y (side view)"),
            ],
        ):
            plate.plot_slice(
                origin=origin,
                normal=normal,
                res=(300, 300),
                ax=ax,
                show_zero_level=True,
                deformation_function=deformation,
            )
            ax.set_title(title)
        fig.tight_layout()
        slice_file = outdir / "hole_slices.png"
        fig.savefig(slice_file, dpi=150)
        logger.info(f"wrote {slice_file}")

    logger.info("extracting surface mesh ...")
    surf_mesh, _ = create_3D_mesh(
        plate,
        resolution,
        mesh_type="surface",
        differentiate=False,
        device=device,
        deformation_function=deformation,
    )
    verts = surf_mesh.vertices.detach().cpu().numpy()
    trimesh_obj = surf_mesh.to_trimesh()
    logger.info(
        f"surface mesh: {len(verts)} vertices, {surf_mesh.faces.shape[0]} triangles"
    )
    logger.info(f"watertight: {trimesh_obj.is_watertight}")
    logger.info(
        f"bounding box: {np.round(verts.min(axis=0), 4)} "
        f"-> {np.round(verts.max(axis=0), 4)}"
    )
    if trimesh_obj.is_watertight:
        logger.info(f"solid volume: {trimesh_obj.volume:.6f} m^3")

    for name in ("plate_with_hole.vtk", "plate_with_hole.stl"):
        surf_mesh.export(str(outdir / name))
        logger.info(f"wrote {outdir / name}")

    if render:
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        plotter.add_mesh(
            pv.read(str(outdir / "plate_with_hole.stl")),
            color="lightsteelblue",
            smooth_shading=True,
        )
        plotter.add_axes()
        plotter.camera_position = "iso"
        render_file = outdir / "hole_render.png"
        plotter.screenshot(str(render_file))
        logger.info(f"wrote {render_file}")

    if show:
        import gustaf as gus

        gus.show(surf_mesh.to_gus(), axes=1)

    return surf_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hole-radius", type=float, default=0.25, help="[m]")
    parser.add_argument(
        "--hole-center", type=float, nargs=2, default=[0.5, 0.5], help="x y [m]"
    )
    parser.add_argument("--length", type=float, default=1.0, help="x size [m]")
    parser.add_argument("--width", type=float, default=1.0, help="y size [m]")
    parser.add_argument("--thickness", type=float, default=0.1, help="z size [m]")
    parser.add_argument("--tiling", type=int, nargs=3, default=[5, 5, 1])
    parser.add_argument("--latent", type=float, default=0.4)
    parser.add_argument(
        "--solid",
        action="store_true",
        help="plain solid slab instead of a lattice (no neural network involved)",
    )
    parser.add_argument("--resolution", type=int, default=14)
    parser.add_argument("--outdir", default="experiments/outputs")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--show", action="store_true", help="interactive window")
    args = parser.parse_args()

    configure_logging()
    plate, deformation = plate_with_hole(
        hole_radius=args.hole_radius,
        hole_center=tuple(args.hole_center),
        length=args.length,
        width=args.width,
        thickness=args.thickness,
        tiling=tuple(args.tiling),
        latent=args.latent,
        solid=args.solid,
    )
    mesh_and_export(
        plate,
        deformation,
        resolution=args.resolution,
        outdir=args.outdir,
        render=not args.no_render,
        show=args.show,
    )


if __name__ == "__main__":
    main()
