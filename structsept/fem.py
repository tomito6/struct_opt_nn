"""Tetrahedral meshing and stiffness assembly -- the FEM half of the pipeline.

Step 5 of the pipeline in ``CLAUDE.md``, cut short right after the global
stiffness matrix ``K`` exists: no boundary conditions, no loads, no solve::

    SDF  --FlexiCubes-->  surface  --tetgen-->  tets  --torch-fem-->  k_e  -->  K
         (tetrahedral_mesh)                          (build_solid)  (assemble_stiffness)

Everything here takes a geometry and hands back tensors; nothing writes a file
and nothing parses an argument. ``experiments/plate_with_hole_stiffness.py``
is the runnable demonstration that drives these functions end to end, and
``docs/stiffness_theory/`` documents the mathematics behind them.

Two things worth knowing before reading the code:

* **Sizes.** A linear tetrahedron has 4 nodes with 3 displacement DOFs each,
  so every element stiffness matrix ``k_e`` is 12 x 12. The global matrix is
  ``(3 n_nodes) x (3 n_nodes)`` and sparse: a node only talks to the nodes it
  shares an element with.
* **Singularity.** Without boundary conditions ``K`` has exactly six zero
  eigenvalues, one per rigid body mode (three translations, three rotations).
  That is not a bug, it is the check: ``K u = 0`` for every rigid ``u``, which
  :func:`check_stiffness` verifies to machine precision. Pinning nodes is what
  makes ``K`` invertible, and that is where the next step starts.
"""

import contextlib
import logging

import scipy.sparse
import torch
import torchfem.materials
import torchfem.solid

import DeepSDFStruct
from DeepSDFStruct.mesh import create_3D_mesh, tetrahedralize_surface
from DeepSDFStruct.optimization import tet_signed_vol

logger = logging.getLogger(DeepSDFStruct.__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def default_dtype(dtype):
    """Temporarily change torch's default floating point type.

    The two halves of the pipeline disagree about precision. The neural network
    and FlexiCubes run in float32, which is what they were written for.
    torch-fem builds its internal tensors from the *default* dtype and is only
    trustworthy in float64 -- element matrices in single precision lose digits
    during assembly. So the mesh is made in float32 and the FEM part of the
    script runs inside ``with default_dtype(torch.float64):``.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def rigid_body_modes(nodes: torch.Tensor) -> torch.Tensor:
    """The six displacement fields a free body can undergo without straining.

    Returns a ``(3 n_nodes, 6)`` matrix. Columns 0-2 translate every node by
    one along x, y, z. Columns 3-5 are infinitesimal rotations about the
    origin, ``u = omega x r``. Linear elasticity sees no strain in any of them,
    so ``K @ modes`` must vanish for a correctly assembled ``K``.
    """
    n = nodes.shape[0]
    modes = torch.zeros(n, 3, 6, dtype=nodes.dtype)
    for axis in range(3):
        modes[:, axis, axis] = 1.0
        omega = torch.zeros(3, dtype=nodes.dtype)
        omega[axis] = 1.0
        modes[:, :, 3 + axis] = torch.cross(omega.expand_as(nodes), nodes, dim=1)
    return modes.reshape(3 * n, 6)


def to_scipy(K: torch.Tensor) -> scipy.sparse.csr_matrix:
    """Convert a coalesced sparse COO torch tensor to SciPy CSR."""
    K = K.coalesce()
    rows, cols = K.indices().numpy()
    return scipy.sparse.coo_matrix(
        (K.values().numpy(), (rows, cols)), shape=tuple(K.shape)
    ).tocsr()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def tetrahedral_mesh(plate, deformation, resolution=10, device="cpu"):
    """Mesh the plate into tetrahedra fit for FEM.

    Two stages, matching the pipeline in ``CLAUDE.md``:

    1. ``create_3D_mesh(mesh_type="surface")`` runs FlexiCubes on the SDF and
       applies the deformation spline, so the triangles come out in physical
       metres. The result must be watertight -- that is what the
       ``CappedBorderSDF`` in ``plate_with_hole`` is for.
    2. ``tetrahedralize_surface`` hands that closed surface to tetgen, which
       fills the interior with tetrahedra and adds interior nodes as needed.

    Why not ``create_3D_mesh(mesh_type="volume")``, which the library's own
    optimization test uses? Because as of this writing that path, FlexiCubes'
    built-in tetrahedralization, returns a mesh with internal cavities: it
    covers only about 70 % of the volume the surface encloses, on a plain
    unit box as much as on the lattice, at every resolution tried. tetgen
    matches the surface volume to all printed digits. A stiffness matrix is
    only as good as the mesh under it, so the accurate path wins here.

    **Not differentiable.** The surface is extracted with
    ``differentiate=False`` and tetgen works on numpy arrays, so the vertices
    returned here carry no gradient: ``K`` cannot be differentiated with
    respect to the latent through this function. That is enough for a
    stiffness matrix; an optimization loop needs the graph back. tetgen keeps
    the surface nodes as the first nodes of its mesh, unchanged, so the way
    there is to extract the surface with gradients and put its vertices back
    in place of those rows - or to use the paper's boundary-integral
    sensitivities (Eq. 36-37), which need only the surface vertices.

    Two clean-up passes follow, the same the optimization test does. Both are
    no-ops on tetgen output but cost nothing and guard against surprises:
    torch-fem needs every tetrahedron positively oriented (or the Jacobian
    determinant goes negative and the element matrix with it), and a
    zero-volume tetrahedron has a singular Jacobian.

    Returns
    -------
    vertices : torch.Tensor, shape (n_nodes, 3), float64, cpu
    tets : torch.Tensor, shape (n_elements, 4), int64, cpu
    """
    logger.info(f"extracting surface at resolution {resolution} per unit cell ...")
    surface, _ = create_3D_mesh(
        plate,
        resolution,
        mesh_type="surface",
        differentiate=False,
        device=device,
        deformation_function=deformation,
    )
    surface_trimesh = surface.to_trimesh()
    if not surface_trimesh.is_watertight:
        raise RuntimeError(
            "surface mesh is not watertight, tetgen cannot fill it -- "
            "is the SDF wrapped in CappedBorderSDF?"
        )
    logger.info(
        f"surface mesh: {surface.faces.shape[0]} triangles enclosing "
        f"{surface_trimesh.volume:.6f} m^3"
    )

    logger.info("filling with tetrahedra (tetgen) ...")
    volume_mesh, _ = tetrahedralize_surface(surface.to_gus())
    # torch-fem runs on the CPU and, as of this writing, only in float64.
    vertices = torch.as_tensor(volume_mesh.vertices, dtype=torch.float64)
    tets = torch.as_tensor(volume_mesh.volumes, dtype=torch.int64)

    # 1. orientation
    volumes = tet_signed_vol(vertices, tets)
    inverted = volumes < 0
    if inverted.any():
        tets[inverted] = tets[inverted][:, [0, 2, 1, 3]]
        logger.info(f"flipped {int(inverted.sum())} inverted tetrahedra")

    # 2. degenerate elements
    volumes = tet_signed_vol(vertices, tets)
    keep = volumes > 1e-12
    if not keep.all():
        logger.info(f"dropped {int((~keep).sum())} zero-volume tetrahedra")
    tets = tets[keep]

    logger.info(
        f"tetrahedral mesh: {vertices.shape[0]} nodes, {tets.shape[0]} tetrahedra, "
        f"solid volume {volumes[keep].sum():.6f} m^3"
    )
    return vertices, tets


def build_solid(vertices, tets, youngs_modulus=210e9, poisson_ratio=0.3):
    """Wrap the mesh and a linear elastic material into a torch-fem ``Solid``.

    Nothing is computed yet -- the ``Solid`` is the description of the problem:
    where the nodes are, which four of them form each element, and what the
    material does when strained. The default material is steel in SI units,
    so with the plate in metres ``K`` comes out in N/m.

    Must be called inside ``default_dtype(torch.float64)``.
    """
    material = torchfem.materials.IsotropicElasticity3D(
        E=youngs_modulus, nu=poisson_ratio
    )
    solid = torchfem.solid.Solid(vertices, tets, material)
    logger.info(
        f"torch-fem Solid: {solid.n_elem} {solid.etype.__name__} elements, "
        f"{solid.n_nod} nodes x 3 = {solid.n_dofs} degrees of freedom"
    )
    return solid


def assemble_stiffness(solid):
    """Compute the element stiffness matrices and assemble them into ``K``.

    Two calls into torch-fem do the whole job:

    ``solid.k0()``
        Integrates ``B^T C B`` over each element at zero strain, giving one
        12 x 12 matrix per tetrahedron. ``B`` is the strain-displacement
        operator, ``C`` the material stiffness. For a linear tet ``B`` is
        constant, so one integration point is exact.
    ``solid.assemble_matrix(k, con)``
        Scatters each ``k_e`` into the global matrix by adding entry
        ``(i, j)`` of the element to entry ``(dof_i, dof_j)`` of ``K``.
        ``con`` lists constrained DOFs, whose rows and columns get replaced by
        the identity. It is empty here on purpose -- see the module docstring.

    Returns
    -------
    k : torch.Tensor, shape (n_elements, 12, 12)
        Element stiffness matrices, dense.
    K : torch.Tensor, shape (n_dofs, n_dofs), sparse COO
        Global stiffness matrix, unconstrained.
    """
    k = solid.k0()
    logger.info(
        f"element stiffness matrices: {tuple(k.shape)}, "
        f"largest entry {k.abs().max():.3e} N/m"
    )
    constrained_dofs = torch.nonzero(solid.constraints.ravel()).ravel()
    K = solid.assemble_matrix(k, constrained_dofs)
    return k, K


def check_stiffness(K, solid):
    """Report what ``K`` looks like and verify it behaves like one.

    The checks are cheap and each catches a different class of mistake:

    * *symmetry* -- follows from ``B^T C B``; asymmetry means broken assembly.
    * *positive diagonal* -- every DOF resists being moved on its own.
    * *rigid body modes* -- ``K u = 0`` for translations and rotations. This
      is the real test: it exercises the geometry (``B``), the material
      (``C``) and the scatter into ``K`` at once, and only passes if all three
      agree with each other.
    """
    n_dofs = K.shape[0]
    nnz = K._nnz()
    scale = K.values().abs().max()

    rows, cols = K.indices()
    diagonal = K.values()[rows == cols]
    asymmetry = (K - K.t()).coalesce().values().abs().max() / scale
    residual = torch.sparse.mm(K, rigid_body_modes(solid.nodes)).abs().max() / scale

    logger.info("global stiffness matrix K")
    logger.info(f"  size          {n_dofs} x {n_dofs}")
    logger.info(f"  nonzeros      {nnz}  ({100 * nnz / n_dofs**2:.3f} % dense)")
    logger.info(f"  entries       up to {scale:.3e} N/m")
    logger.info(f"  memory        {nnz * 3 * 8 / 1e6:.1f} MB as COO")
    logger.info(f"  asymmetry     {asymmetry:.1e}  (relative, should be ~1e-16)")
    logger.info(f"  min diagonal  {diagonal.min():.3e}  (should be > 0)")
    logger.info(f"  |K u_rigid|   {residual:.1e}  (relative, should be ~1e-16)")

    if asymmetry > 1e-10 or residual > 1e-10 or diagonal.min() <= 0:
        logger.warning("K failed a sanity check -- do not trust it")
