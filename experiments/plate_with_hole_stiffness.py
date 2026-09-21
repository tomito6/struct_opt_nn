"""Lattice plate with a hole, taken through torch-fem up to the stiffness matrix.

This picks up where ``plate_with_hole.py`` stops. That module builds the
geometry; this script turns it into a finite element model and assembles the
global stiffness matrix ``K`` -- and then stops. No boundary conditions, no
loads, no solve. Those are the next step.

It is a *driver*: every piece of real work lives in the library, in
``structsept.plate_with_hole`` (the geometry) and ``structsept.fem`` (the
meshing and the assembly). What is written here is the command line, the
export, and the order in which the pieces are called::

    SDF  --FlexiCubes-->  surface  --tetgen-->  tets  --torch-fem-->  k_e  -->  K
         (fem.tetrahedral_mesh)                  (fem.build_solid) (fem.assemble_stiffness)

Run it from anywhere::

    uv run python experiments/plate_with_hole_stiffness.py
    uv run python experiments/plate_with_hole_stiffness.py --hole-radius 0.15 --tiling 5 5 1 --resolution 14
    uv run python experiments/plate_with_hole_stiffness.py --solid      # plain slab, no lattice

What comes out, in ``experiments/outputs/``:

``plate_with_hole_tets.vtk``
    The tetrahedral mesh torch-fem actually used. Open it in ParaView.
``stiffness_matrix.npz``
    ``K`` as a SciPy CSR matrix. ``scipy.sparse.load_npz`` reads it back.
``stiffness_sparsity.png``
    Sparsity pattern of ``K`` -- the classic picture of a stiffness matrix.

Why ``K`` is singular here, and why that is the point, is explained in the
docstrings of ``structsept.fem``.
"""

import argparse
import logging
import pathlib

import matplotlib
import scipy.sparse
import torch

import DeepSDFStruct
from DeepSDFStruct.optimization import get_mesh_from_torchfem
from DeepSDFStruct.utils import configure_logging

from structsept.fem import (
    assemble_stiffness,
    build_solid,
    check_stiffness,
    default_dtype,
    tetrahedral_mesh,
    to_scipy,
)
from structsept.plate_with_hole import plate_with_hole

logger = logging.getLogger(DeepSDFStruct.__name__)


def export(K, solid, outdir, plot=True):
    """Write the matrix, its picture, and the mesh it came from."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    K_csr = to_scipy(K)
    matrix_file = outdir / "stiffness_matrix.npz"
    scipy.sparse.save_npz(matrix_file, K_csr)
    logger.info(f"wrote {matrix_file}")

    mesh = get_mesh_from_torchfem(solid)
    mesh_file = outdir / "plate_with_hole_tets.vtk"
    mesh.save(mesh_file)
    logger.info(f"wrote {mesh_file}")

    if plot:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.spy(K_csr, markersize=0.05, color="k", rasterized=True)
        ax.set_title(
            f"K, {K_csr.shape[0]} x {K_csr.shape[1]}, {K_csr.nnz} nonzeros", pad=12
        )
        ax.set_xlabel("column (DOF)")
        ax.set_ylabel("row (DOF)")
        fig.tight_layout()
        plot_file = outdir / "stiffness_sparsity.png"
        fig.savefig(plot_file, dpi=150)
        logger.info(f"wrote {plot_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    geometry = parser.add_argument_group("geometry (same as plate_with_hole.py)")
    geometry.add_argument("--hole-radius", type=float, default=0.25, help="[m]")
    geometry.add_argument(
        "--hole-center", type=float, nargs=2, default=[0.5, 0.5], help="x y [m]"
    )
    geometry.add_argument("--length", type=float, default=1.0, help="x size [m]")
    geometry.add_argument("--width", type=float, default=1.0, help="y size [m]")
    geometry.add_argument("--thickness", type=float, default=0.1, help="z size [m]")
    geometry.add_argument(
        "--tiling", type=int, nargs=3, default=[4, 4, 1], help="unit cells per axis"
    )
    geometry.add_argument(
        "--latent",
        type=float,
        default=0.4,
        help="constant latent value, inside the trained range [0.15, 0.75]",
    )
    geometry.add_argument(
        "--solid", action="store_true", help="plain slab instead of a lattice"
    )

    fem = parser.add_argument_group("finite elements")
    fem.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="FlexiCubes grid points per unit cell; element count grows ~N^3",
    )
    fem.add_argument(
        "--youngs-modulus", type=float, default=210e9, help="E [Pa], default steel"
    )
    fem.add_argument("--poisson-ratio", type=float, default=0.3, help="nu [-]")

    output = parser.add_argument_group("output")
    output.add_argument("--outdir", default="experiments/outputs")
    output.add_argument(
        "--no-plot", action="store_true", help="skip the sparsity pattern figure"
    )
    args = parser.parse_args()

    configure_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 3. network inputs -> SDF, 4. mesh generation (float32) --------------
    plate, deformation = plate_with_hole(
        hole_radius=args.hole_radius,
        hole_center=tuple(args.hole_center),
        length=args.length,
        width=args.width,
        thickness=args.thickness,
        tiling=tuple(args.tiling),
        latent=args.latent,
        solid=args.solid,
        device=device,
    )
    vertices, tets = tetrahedral_mesh(
        plate, deformation, resolution=args.resolution, device=device
    )

    # --- 5. forward simulation, up to K (float64) -----------------------------
    with default_dtype(torch.float64):
        solid = build_solid(vertices, tets, args.youngs_modulus, args.poisson_ratio)
        _, K = assemble_stiffness(solid)
        check_stiffness(K, solid)
        export(K, solid, args.outdir, plot=not args.no_plot)

    # From here the next script would: pin the nodes on one face
    # (``solid.constraints[mask, :] = True``), put a load on the opposite one
    # (``solid.forces[mask, 0] = ...``), and call ``solid.solve()``, which
    # re-assembles K with those constraints and solves K u = f.


if __name__ == "__main__":
    main()
