"""Plate with a hole whose lattice comes from the neural network -- basic version.

This is the geometry representation of the paper and nothing else: build the
signed distance function, evaluate the trained DeepSDF decoder ``f_theta``, look
at the result. No FEM, no optimization, no design variables being updated.

The shape is assembled from two ingredients, which is exactly the split the
paper makes:

``Omega``
    The **design domain**, written analytically: a rectangular plate with one
    circular hole punched through it. Plain mathematics, no network involved.
``f_theta``
    The **unit cell**, produced by a trained neural network. A single scalar
    latent value ``lambda`` picks one member of the learned shape family --
    here three perpendicular round struts, thin to thick.

The final geometry is the tiled unit cell restricted to the design domain::

    plate = cap_borders( tile(f_theta)  minus  hole )

Three figures come out of a run, into ``experiments/outputs/``:

``sdf_stages.png``
    The three signed distance fields side by side -- design domain, unit cell,
    and the combination of the two. Blue is inside the material, red is
    outside, the black line is the surface (the zero level set).
``latent_sweep.png``
    The same unit cell at several latent values. This is the design space the
    optimizer moves through, in one picture.
``plate_render.png``
    The extracted surface mesh, rendered.

Run it from the repository root::

    uv run python experiments/plate_with_hole_network.py
    uv run python experiments/plate_with_hole_network.py --latent 0.45 --tiling 6 6 1
    uv run python experiments/plate_with_hole_network.py --model AnalyticRoundCross

A note on ``--model``. ``RoundCross`` is the trained network -- 83,576 weights,
the real ``f_theta``, and the default here. ``AnalyticRoundCross`` carries the
same shape family under the same name, but its ``forward`` ignores every weight
and evaluates a closed-form formula instead, so it is the exact reference
geometry. Their latent *scales* are not interchangeable: at ``lambda = 0.6`` the
analytic cell fills 54% of the unit cube while the trained one fills 29%,
because the trained latent codes were learned rather than prescribed. Compare
the shapes, not the numbers.
"""

import argparse
import logging
import pathlib
from typing import NamedTuple

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

# ``ScaledSpaceSDF`` bridges the parametric cube the lattice lives on and the
# physical metres the hole is described in. It is written and explained in the
# library module; there is no reason to keep a second copy of it here.
from structsept.plate_with_hole import ScaledSpaceSDF

logger = logging.getLogger(DeepSDFStruct.__name__)

#: Latent bounds of the library's optimization test, which runs on
#: ``AnalyticRoundCross``: there the latent is the strut radius, and this is
#: the range that test keeps it in. For the *trained* ``RoundCross`` it is
#: only a rule of thumb, not its trained range: that decoder's 20 codes span
#: [-1, 1] with both signs, the cell getting thicker with ``|latent|`` on
#: either side of a minimum near +0.05. The warning in ``build_microtile``
#: therefore also fires for valid codes such as 0.9.
LATENT_RANGE = (0.15, 0.75)


def _latent_tensor(latent, device) -> torch.Tensor:
    """One latent value as the float32 tensor the decoder expects.

    The explicit ``float()`` matters: a numpy scalar -- what a ``linspace``
    hands out -- would produce a float64 tensor, and the decoder's weights are
    float32, so the first matrix multiplication would refuse it.
    """
    return torch.tensor([float(latent)], dtype=torch.float32, device=device)


class Plate(NamedTuple):
    """Everything one run produces, kept together so the figures can share it.

    Attributes
    ----------
    plate : SDFBase
        The final geometry: watertight SDF on the parametric cube [0, 1]^3.
    domain : SDFBase
        The analytic design domain on its own -- a solid plate with the hole
        and no lattice. This is the ``Omega`` of the paper.
    microtile : SDFBase
        ``f_theta``: the decoder as an SDF, on the network's own cube
        [-1, 1]^3. One unit cell, untiled and undeformed.
    deformation : TorchSpline
        Parametric-to-physical map. The SDF knows nothing about metres; this
        is what turns [0, 1]^3 into a length x width x thickness plate.
    """

    plate: SDFBase
    domain: SDFBase
    microtile: SDFBase
    deformation: TorchSpline


# ---------------------------------------------------------------------------
# Step 1 -- the SDF: an analytic plate with a hole
# ---------------------------------------------------------------------------


def build_domain(hole_radius, hole_center, plate_size) -> SDFBase:
    """Signed distance function of the design domain: a slab minus a cylinder.

    Both operands are analytic primitives, so this half of the geometry is
    exact and costs nothing to evaluate.

    Parameters
    ----------
    hole_radius : float
        Radius of the hole in metres. Pass 0 (or less) for a plate with no
        hole at all.
    hole_center : tuple of float
        (x, y) centre of the hole in metres, measured from the plate's
        lower-left corner. The hole always runs the full thickness, along z.
    plate_size : tuple of float
        (length, width, thickness) of the plate in metres.

    Returns
    -------
    SDFBase
        Negative inside the material, positive outside, on [0, 1]^3.
    """
    # The parametric cube itself, filled: it becomes the plate once the
    # deformation stretches it to its physical size.
    slab = BoxSDF(center=[0.5, 0.5, 0.5], extents=[1.0, 1.0, 1.0])
    if hole_radius <= 0:
        return slab
    return DifferenceSDF(slab, _hole(hole_radius, hole_center, plate_size))


def _hole(hole_radius, hole_center, plate_size) -> SDFBase:
    """The cylinder that gets subtracted, expressed in parametric coordinates."""
    cx, cy = hole_center
    thickness = plate_size[2]
    # Overshoot the plate in z so the cut is a clean through-hole rather than
    # a blind pocket whose end cap would land inside the material.
    cylinder = CylinderSDF(
        point_a=[cx, cy, -thickness],
        point_b=[cx, cy, 2 * thickness],
        radius=hole_radius,
    )
    return ScaledSpaceSDF(cylinder, plate_size)


# ---------------------------------------------------------------------------
# Step 2 -- f_theta: the unit cell, straight out of the neural network
# ---------------------------------------------------------------------------


def build_microtile(model_name, latent, device) -> SDFBase:
    """Load a trained decoder and wrap it as an SDF at one fixed latent value.

    This is ``f_theta`` of the paper: a network mapping
    ``(x, y, z, lambda) -> signed distance``, defined on the cube [-1, 1]^3.
    Freezing ``lambda`` to a single number selects one unit cell out of the
    learned family.

    Parameters
    ----------
    model_name : str
        Name of a ``PretrainedModels`` entry, e.g. ``"RoundCross"``.
    latent : float
        The latent value. Stay inside :data:`LATENT_RANGE`.
    device : str or torch.device
        Where to run the decoder.

    Returns
    -------
    SDFfromDeepSDF
        Callable SDF: ``microtile(points)`` with ``points`` of shape (N, 3).
    """
    if not LATENT_RANGE[0] <= latent <= LATENT_RANGE[1]:
        logger.warning(
            f"latent {latent} is outside the trained range {LATENT_RANGE}; "
            "the decoder is extrapolating"
        )
    model = get_model(PretrainedModels[model_name], device=device)
    microtile = SDFfromDeepSDF(model)
    microtile.set_latent_vec(_latent_tensor(latent, model.device))
    return microtile


# ---------------------------------------------------------------------------
# Steps 1 and 2 together -- the plate
# ---------------------------------------------------------------------------


def build_plate(
    model_name="RoundCross",
    latent=0.6,
    hole_radius=0.25,
    hole_center=(0.5, 0.5),
    plate_size=(1.0, 1.0, 0.1),
    tiling=(5, 5, 1),
    device=None,
) -> Plate:
    """Tile ``f_theta`` over the plate and cut the hole out of it.

    Parameters
    ----------
    model_name : str
        Which pretrained decoder to use as the unit cell.
    latent : float
        Constant latent value, applied everywhere. Letting it vary in space --
        ``SplineParametrization`` instead of ``Constant`` -- is what turns this
        into an optimizable design; here it stays constant on purpose.
    hole_radius, hole_center, plate_size
        Passed straight to :func:`build_domain`.
    tiling : tuple of int
        Number of unit cells per axis.
    device : str, optional
        Defaults to cuda when available.

    Returns
    -------
    Plate
        The final SDF plus the pieces it was built from.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    microtile = build_microtile(model_name, latent, device)
    lattice = LatticeSDFStruct(
        tiling=list(tiling),
        microtile=microtile,
        parametrization=Constant([latent], device=microtile.model.device),
    )

    # Keep the lattice only where the domain has material. Cutting the hole and
    # capping the borders both only ever remove material, so their order does
    # not matter -- but the caps are what close the lattice at the plate's
    # faces, and a mesh that is open there is useless downstream.
    if hole_radius > 0:
        body = DifferenceSDF(lattice, _hole(hole_radius, hole_center, plate_size))
    else:
        body = lattice
    plate = CappedBorderSDF(body)

    deformation = TorchSpline(
        splinepy.helpme.create.box(*plate_size).bspline, device=device
    )
    return Plate(
        plate=plate,
        domain=build_domain(hole_radius, hole_center, plate_size),
        microtile=microtile,
        deformation=deformation,
    )


# ---------------------------------------------------------------------------
# Step 3 -- the pictures
# ---------------------------------------------------------------------------


def _fresh_unit_cell(model: Plate, latent) -> SDFBase:
    """A throwaway view of ``f_theta`` at one latent value, safe to plot.

    ``LatticeSDFStruct`` drives its microtile by writing one latent code *per
    query point* into it, and that assignment sticks. Plotting the very same
    object afterwards then fails, because the cached codes still have the shape
    of the last lattice evaluation. Wrapping the decoder again is cheap -- the
    weights are shared, only the thin SDF wrapper is new -- and sidesteps the
    whole problem.
    """
    unit_cell = SDFfromDeepSDF(model.microtile.model)
    unit_cell.set_latent_vec(_latent_tensor(latent, unit_cell.model.device))
    return unit_cell


def figure_sdf_stages(model: Plate, latent, outdir: pathlib.Path, res=(300, 300)):
    """The three fields side by side: domain, unit cell, and their combination.

    Each is cut at its own mid-plane and seen from the top. The colour is the
    signed distance itself; the black line is the surface.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) the analytic design domain, drawn in physical metres
    model.domain.plot_slice(
        origin=(0, 0, 0.5),
        normal=(0, 0, 1),
        res=res,
        ax=axs[0],
        deformation_function=model.deformation,
    )
    axs[0].set_title(r"(a) design domain $\Omega$ — analytic")

    # (b) f_theta on the network's own cube [-1, 1]^3: one cell, untiled
    unit_cell = _fresh_unit_cell(model, latent)
    unit_cell.plot_slice(origin=(0, 0, 0), normal=(0, 0, 1), res=res, ax=axs[1])
    axs[1].set_title(r"(b) unit cell $f_\theta$ — neural network")

    # (c) what the mesher actually sees
    model.plate.plot_slice(
        origin=(0, 0, 0.5),
        normal=(0, 0, 1),
        res=res,
        ax=axs[2],
        deformation_function=model.deformation,
    )
    axs[2].set_title(r"(c) $f_\theta$ tiled over $\Omega$")

    fig.tight_layout()
    path = outdir / "sdf_stages.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {path}")


def figure_latent_sweep(model: Plate, latents, outdir: pathlib.Path, res=(250, 250)):
    """One unit cell per latent value -- the design space in a single row.

    Moving ``lambda`` is the only thing the optimizer ever does, so this is
    worth looking at before running anything expensive.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(latents), figsize=(2.9 * len(latents), 3.4))
    for ax, latent in zip(np.atleast_1d(axs), latents):
        unit_cell = _fresh_unit_cell(model, latent)
        unit_cell.plot_slice(origin=(0, 0, 0), normal=(0, 0, 1), res=res, ax=ax)
        ax.set_title(rf"$\lambda$ = {latent:.2f}")

    fig.suptitle(r"$f_\theta$ across the latent range — mid-plane slice")
    fig.tight_layout()
    path = outdir / "latent_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {path}")


def mesh_and_render(model: Plate, plate_size, resolution=14, outdir=None, render=True):
    """Extract the surface with FlexiCubes, report it, write it out, render it.

    Returns the surface mesh so it can be poked at further in a notebook.
    """
    logger.info("extracting surface mesh ...")
    surf_mesh, _ = create_3D_mesh(
        model.plate,
        resolution,
        mesh_type="surface",
        differentiate=False,
        device=str(model.plate.get_device()),
        deformation_function=model.deformation,
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
        # The volume constraint of the optimization is written on exactly this
        # number, which is why it is worth printing even in a geometry script.
        bbox_volume = float(np.prod(plate_size))
        logger.info(
            f"solid volume: {trimesh_obj.volume:.6f} m^3 "
            f"({100 * trimesh_obj.volume / bbox_volume:.1f}% of the bounding box)"
        )

    stl_path = outdir / "plate_network.stl"
    surf_mesh.export(str(stl_path))
    logger.info(f"wrote {stl_path}")

    if render:
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        plotter.add_mesh(
            pv.read(str(stl_path)), color="lightsteelblue", smooth_shading=True
        )
        plotter.add_axes()
        plotter.camera_position = "iso"
        path = outdir / "plate_render.png"
        plotter.screenshot(str(path))
        logger.info(f"wrote {path}")

    return surf_mesh


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="RoundCross",
        choices=[m.name for m in PretrainedModels],
        help="pretrained decoder used as the unit cell (default: the trained network)",
    )
    parser.add_argument("--latent", type=float, default=0.6, help="latent value")
    parser.add_argument("--hole-radius", type=float, default=0.25, help="[m]")
    parser.add_argument(
        "--hole-center", type=float, nargs=2, default=[0.5, 0.5], help="x y [m]"
    )
    parser.add_argument(
        "--plate-size",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 0.1],
        help="length width thickness [m]",
    )
    parser.add_argument("--tiling", type=int, nargs=3, default=[5, 5, 1])
    parser.add_argument(
        "--resolution", type=int, default=14, help="FlexiCubes cubes per cell per axis"
    )
    parser.add_argument("--outdir", default="experiments/outputs")
    parser.add_argument(
        "--no-mesh", action="store_true", help="figures only, skip meshing entirely"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="skip the pyvista screenshot"
    )
    args = parser.parse_args()

    configure_logging()
    matplotlib.use("Agg")  # write files, never open a window

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"building the plate with {args.model} at latent {args.latent}")
    model = build_plate(
        model_name=args.model,
        latent=args.latent,
        hole_radius=args.hole_radius,
        hole_center=tuple(args.hole_center),
        plate_size=tuple(args.plate_size),
        tiling=tuple(args.tiling),
    )

    figure_sdf_stages(model, args.latent, outdir)
    figure_latent_sweep(model, np.linspace(*LATENT_RANGE, 5), outdir)

    if not args.no_mesh:
        mesh_and_render(
            model,
            plate_size=tuple(args.plate_size),
            resolution=args.resolution,
            outdir=outdir,
            render=not args.no_render,
        )


if __name__ == "__main__":
    main()
