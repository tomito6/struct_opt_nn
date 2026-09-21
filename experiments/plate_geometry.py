"""Generate and visualize a simple lattice plate -- geometry only, no FEM.

The plate is a lattice of DeepSDF unit cells tiled over the parametric cube
[0, 1]^3 and mapped onto a physical box of ``--length`` x ``--width`` x
``--thickness`` metres by a trilinear B-spline (the same FFD map the
optimization loop uses, here just a plain box).

Nothing is optimized and no solver runs: this is step 3-4 of the pipeline
(network inputs + mesh generation) stopped right after meshing.

Examples
--------
    uv run python experiments/plate_geometry.py
    uv run python experiments/plate_geometry.py --tiling 6 6 1 --resolution 14
    uv run python experiments/plate_geometry.py --show          # interactive window
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
from DeepSDFStruct.mesh import create_3D_mesh, export_sdf_grid_vtk
from DeepSDFStruct.parametrization import Constant
from DeepSDFStruct.pretrained_models import PretrainedModels, get_model
from DeepSDFStruct.SDF import CappedBorderSDF, SDFfromDeepSDF
from DeepSDFStruct.torch_spline import TorchSpline
from DeepSDFStruct.utils import configure_logging

logger = logging.getLogger(DeepSDFStruct.__name__)

# Face-sheet thickness for --skins, in parametric units: 0.05 of the plate's
# own thickness on each side.
SKIN_THICKNESS = 0.05


def build_plate(
    length=1.0,
    width=1.0,
    thickness=0.1,
    tiling=(4, 4, 1),
    latent=0.4,
    skins=False,
    device="cpu",
):
    """Assemble the plate SDF and its parametric-to-physical map.

    Returns
    -------
    plate : CappedBorderSDF
        Watertight lattice defined on the parametric cube [0, 1]^3.
    deformation : TorchSpline
        Maps the parametric cube onto the physical plate box.
    """
    model = get_model(PretrainedModels.AnalyticRoundCross, device=device)
    microtile = SDFfromDeepSDF(model)

    # A single latent value held constant over the whole plate: every unit cell
    # is identical. Swapping Constant for SplineParametrization is what turns
    # this into a design vector for the optimizer.
    parametrization = Constant([latent], device=model.device)

    lattice = LatticeSDFStruct(
        tiling=list(tiling), microtile=microtile, parametrization=parametrization
    )

    # Without caps the lattice is cut open at the domain border and meshing
    # yields a non-watertight surface. cap=-1 trims flush with the border,
    # cap=1 adds a solid skin of the given thickness (in parametric units).
    #
    # The two cases differ in nesting, not just in the dict. A cap=1 face fills
    # material with a min(), which also fills the sampling margin outside the
    # domain and leaves the mesh open there -- so the adding pass has to be
    # wrapped in a second, trimming pass (the same nesting the optimization
    # test uses).
    plate = CappedBorderSDF(lattice)  # -> UNIT_CUBE_CAPS_3D, trim all six faces
    if skins:
        add_skins = {
            "z0": {"cap": 1, "measure": SKIN_THICKNESS},
            "z1": {"cap": 1, "measure": SKIN_THICKNESS},
        }
        plate = CappedBorderSDF(CappedBorderSDF(lattice, add_skins))

    # Physical size. splinepy's box has parametric domain [0, 1]^3, so this is
    # just an anisotropic scaling -- but it is a TorchSpline, so it can later
    # be bent into any FFD shape without touching the rest of the script.
    deformation = TorchSpline(
        splinepy.helpme.create.box(length, width, thickness).bspline,
        device=model.device,
    )
    return plate, deformation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=1.0, help="x size [m]")
    parser.add_argument("--width", type=float, default=1.0, help="y size [m]")
    parser.add_argument("--thickness", type=float, default=0.1, help="z size [m]")
    parser.add_argument(
        "--tiling", type=int, nargs=3, default=[4, 4, 1], help="unit cells per axis"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=12,
        help="FlexiCubes grid points per unit cell (cost grows as N^3)",
    )
    parser.add_argument(
        "--latent",
        type=float,
        default=0.4,
        help="constant latent value; stay inside the trained range [0.15, 0.75]",
    )
    parser.add_argument(
        "--skins",
        action="store_true",
        help="add solid face sheets on the top and bottom faces",
    )
    parser.add_argument(
        "--volume",
        action="store_true",
        help="also extract a tetrahedral volume mesh (slower, needed for FEM later)",
    )
    parser.add_argument(
        "--sdf-grid",
        action="store_true",
        help="also export the raw SDF sampled on a structured grid",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip the offscreen pyvista render of the surface mesh",
    )
    parser.add_argument(
        "--show", action="store_true", help="open an interactive gustaf/vedo window"
    )
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("outputs"))
    args = parser.parse_args()

    configure_logging()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    plate, deformation = build_plate(
        length=args.length,
        width=args.width,
        thickness=args.thickness,
        tiling=tuple(args.tiling),
        latent=args.latent,
        skins=args.skins,
        device=device,
    )

    # --- 1. slices through the SDF ------------------------------------------
    # Cheapest sanity check: no meshing involved, just evaluating the network.
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (origin, normal, title) in zip(
        axs,
        [
            ((0, 0, 0.5), (0, 0, 1), "mid-plane z"),
            ((0, 0.5, 0), (0, 1, 0), "section y"),
            ((0.5, 0, 0), (1, 0, 0), "section x"),
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
    fig.suptitle(
        f"SDF slices -- plate {args.length} x {args.width} x {args.thickness} m, "
        f"tiling {args.tiling}, latent {args.latent}"
    )
    fig.tight_layout()
    slice_file = args.outdir / "plate_slices.png"
    fig.savefig(slice_file, dpi=150)
    logger.info(f"wrote {slice_file}")

    # --- 2. surface mesh ----------------------------------------------------
    logger.info("extracting surface mesh with FlexiCubes ...")
    surf_mesh, _ = create_3D_mesh(
        plate,
        args.resolution,
        mesh_type="surface",
        differentiate=False,
        device=device,
        deformation_function=deformation,
    )
    trimesh_obj = surf_mesh.to_trimesh()
    verts = surf_mesh.vertices.detach().cpu().numpy()
    logger.info(
        f"surface mesh: {len(verts)} vertices, {surf_mesh.faces.shape[0]} triangles"
    )
    logger.info(f"watertight: {trimesh_obj.is_watertight}")
    logger.info(f"bounding box min: {np.round(verts.min(axis=0), 4)}")
    logger.info(f"bounding box max: {np.round(verts.max(axis=0), 4)}")
    box_vol = args.length * args.width * args.thickness
    if trimesh_obj.is_watertight:
        logger.info(
            f"solid volume: {trimesh_obj.volume:.6f} m^3 "
            f"({100 * trimesh_obj.volume / box_vol:.1f} % of the bounding box)"
        )

    for name in ("plate_surface.vtk", "plate_surface.stl"):
        out = args.outdir / name
        surf_mesh.export(str(out))
        logger.info(f"wrote {out}")

    # --- 3. offscreen 3D render ---------------------------------------------
    if not args.no_render:
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        plotter.add_mesh(
            pv.read(str(args.outdir / "plate_surface.stl")),
            color="lightsteelblue",
            smooth_shading=True,
        )
        plotter.add_axes()
        plotter.camera_position = "iso"
        render_file = args.outdir / "plate_render.png"
        plotter.screenshot(str(render_file))
        logger.info(f"wrote {render_file}")

    # --- 4. optional extras -------------------------------------------------
    if args.volume:
        logger.info("extracting tetrahedral volume mesh ...")
        vol_mesh, _ = create_3D_mesh(
            plate,
            args.resolution,
            mesh_type="volume",
            differentiate=False,
            device=device,
            deformation_function=deformation,
        )
        logger.info(f"volume mesh: {vol_mesh.volumes.shape[0]} tetrahedra")
        out = args.outdir / "plate_volume.vtk"
        vol_mesh.export(str(out))
        logger.info(f"wrote {out}")

    if args.sdf_grid:
        out = args.outdir / "plate_sdf_grid.vtk"
        export_sdf_grid_vtk(plate, str(out), N=48)

    if args.show:
        import gustaf as gus

        gus.show(surf_mesh.to_gus(), axes=1)
    else:
        logger.info("re-run with --show for an interactive window")


if __name__ == "__main__":
    main()
