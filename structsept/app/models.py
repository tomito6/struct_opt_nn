"""Model discovery, lattice assembly and field evaluation.

Online stage of the pipeline (steps 3-4 of the paper's method): pick a trained
decoder, interpolate its latent vector over the domain with a B-spline whose
control points are the design variables, tile the unit cell, and evaluate
f_theta(lambda(x), x) either as a 2D slice or as a surface mesh.
"""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from DeepSDFStruct.SDF import CappedBorderSDF, SDFfromDeepSDF
from DeepSDFStruct.geom_reconstruction import build_parameter_spline
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.mesh import create_3D_mesh
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.pretrained_models import (
    PRETRAINED_MODELS_DIR,
    PretrainedModels,
    get_model,
)

ModelEntry = namedtuple("ModelEntry", "name source ref latent_dim n_latents")

_CHECKPOINT = "latest"


def _read_code_length(run_dir: Path) -> int | None:
    """Latent dimension straight from specs.json, without loading any weights."""
    try:
        with open(run_dir / "specs.json", "r", encoding="utf-8") as f:
            return int(json.load(f)["CodeLength"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _count_latents(run_dir: Path) -> int:
    """Number of trained shapes, from the (small) latent code checkpoint."""
    path = run_dir / "LatentCodes" / f"{_CHECKPOINT}.pth"
    if not path.is_file():
        return 0
    try:
        codes = torch.load(path, map_location="cpu", weights_only=True)["latent_codes"]
    except Exception:
        return 0
    if isinstance(codes, torch.Tensor):
        return int(codes.shape[0])
    return int(codes["weight"].shape[0])


def _is_run_dir(path: Path) -> bool:
    return (path / "specs.json").is_file() and (
        path / "ModelParameters" / f"{_CHECKPOINT}.pth"
    ).is_file()


def list_models(runs_dir) -> list[ModelEntry]:
    """Every usable decoder: the shipped pretrained ones plus local training runs."""
    entries = []
    for member in PretrainedModels:
        run_dir = Path(PRETRAINED_MODELS_DIR) / member.value
        entries.append(
            ModelEntry(
                name=member.name,
                source="pretrained",
                ref=member,
                latent_dim=_read_code_length(run_dir),
                n_latents=_count_latents(run_dir),
            )
        )

    runs_dir = Path(runs_dir)
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            if not _is_run_dir(run_dir):
                continue
            entries.append(
                ModelEntry(
                    name=run_dir.name,
                    source="run",
                    ref=str(run_dir),
                    latent_dim=_read_code_length(run_dir),
                    n_latents=_count_latents(run_dir),
                )
            )
    return entries


def load_model(entry: ModelEntry):
    """Load the decoder and its latent codes for a ModelEntry."""
    return get_model(entry.ref, checkpoint=_CHECKPOINT)


def trained_latents(model) -> np.ndarray:
    """The latent codes the decoder was trained on, shape (n_shapes, d)."""
    codes = model._trained_latent_vectors
    if isinstance(codes, (list, tuple)):
        codes = torch.stack(list(codes))
    return codes.detach().cpu().numpy().astype(np.float64)


def latent_dim(model) -> int:
    return int(trained_latents(model).shape[1])


def default_control_points(model, n_ctrl) -> np.ndarray:
    """Mean trained latent replicated on every control point of the spline grid."""
    mean = trained_latents(model).mean(axis=0)
    return np.tile(mean, (int(np.prod(n_ctrl)), 1))


def build_lattice(model, tiling, n_ctrl, control_points, cap=True):
    """Assemble the tiled lattice whose latent field is a trilinear B-spline.

    ``n_ctrl`` fixes the latent spline (degree 1, so one knot span per control
    point pair); ``tiling`` is how many microtiles fill the domain and defaults
    to ``n_ctrl - 1``, i.e. one microtile per knot span.
    """
    n_ctrl = [int(n) for n in n_ctrl]
    if any(n < 2 for n in n_ctrl):
        raise ValueError(f"n_ctrl must be >= 2 in every direction, got {n_ctrl}")
    if tiling is None:
        tiling = [n - 1 for n in n_ctrl]
    tiling = [int(t) for t in tiling]

    d = latent_dim(model)
    control_points = np.asarray(control_points, dtype=np.float64).reshape(-1, d)

    spline = build_parameter_spline([1, 1, 1], [n - 1 for n in n_ctrl], d)
    if spline.control_points.shape[0] != control_points.shape[0]:
        raise ValueError(
            f"expected {spline.control_points.shape[0]} control points for "
            f"n_ctrl={n_ctrl}, got {control_points.shape[0]}"
        )
    spline.control_points = control_points

    parametrization = SplineParametrization(spline, device=model.device)
    lattice = LatticeSDFStruct(
        tiling=tiling,
        microtile=SDFfromDeepSDF(model),
        parametrization=parametrization,
    )
    # Raw lattices are open at the domain faces; capping keeps the mesh watertight.
    sdf = CappedBorderSDF(lattice, None) if cap else lattice
    return sdf, parametrization, lattice


def set_control_points(parametrization, control_points) -> None:
    """Push new design variables into a live parametrization (no rebuild)."""
    param = next(parametrization.parameters())
    values = torch.as_tensor(
        np.asarray(control_points), dtype=param.dtype, device=param.device
    ).reshape(param.shape)
    parametrization.set_param(values)


def _as_bounds(bounds, obj) -> np.ndarray:
    if bounds is None:
        if hasattr(obj, "_get_domain_bounds"):
            bounds = obj._get_domain_bounds()
        else:
            bounds = obj.torch_spline.parametric_bounds
    if isinstance(bounds, torch.Tensor):
        bounds = bounds.detach().cpu().numpy()
    return np.asarray(bounds, dtype=np.float64).reshape(2, 3)


def _plane_points(bounds: np.ndarray, z: float, res: int, device, dtype):
    xs = np.linspace(bounds[0, 0], bounds[1, 0], res)
    ys = np.linspace(bounds[0, 1], bounds[1, 1], res)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, float(z))], axis=1
    )
    return torch.as_tensor(pts, dtype=dtype, device=device)


def eval_sdf_slice(sdf, z, res=120, bounds=None) -> np.ndarray:
    """f_theta on an axis-aligned z-slice, as (row=y, col=x) for imshow."""
    bounds = _as_bounds(bounds, sdf)
    device = sdf.get_device() if hasattr(sdf, "get_device") else "cpu"
    dtype = sdf.get_dtype() if hasattr(sdf, "get_dtype") else torch.float32
    pts = _plane_points(bounds, z, res, device, dtype)
    with torch.no_grad():
        values = sdf(pts)
    return values.detach().cpu().numpy().reshape(res, res).astype(np.float64)


def eval_latent_slice(parametrization, z, res=120, bounds=None) -> np.ndarray:
    """lambda(x) on an axis-aligned z-slice, as (row=y, col=x, component)."""
    bounds = _as_bounds(bounds, parametrization)
    param = next(parametrization.parameters())
    pts = _plane_points(bounds, z, res, param.device, param.dtype)
    with torch.no_grad():
        values = parametrization(pts)
    d = values.shape[1]
    return values.detach().cpu().numpy().reshape(res, res, d).astype(np.float64)


def surface_mesh(sdf, n_base=8):
    """Extract the triangle surface with FlexiCubes and hand it over as trimesh."""
    device = sdf.get_device() if hasattr(sdf, "get_device") else "cpu"
    mesh, _ = create_3D_mesh(
        sdf,
        n_base,
        mesh_type="surface",
        differentiate=False,
        device=device,
    )
    return mesh.to_trimesh()


# --------------------------------------------------------------------------- #
# control-point geometry
# --------------------------------------------------------------------------- #


def flat_index(i, j, k, n_ctrl) -> int:
    """Row of ``control_points`` holding the control point at grid (i, j, k).

    splinepy stores the control net with x running fastest, so the flat index
    is ``i + nx * (j + ny * k)``. Verified against
    ``build_parameter_spline([1,1,1], ...)`` on a non-cubic grid.
    """
    nx, ny, _ = (int(n) for n in n_ctrl)
    return int(i) + nx * (int(j) + ny * int(k))


def control_point_position(i, j, k, n_ctrl) -> tuple[float, float, float]:
    """Where control point (i, j, k) sits in the domain, in [0, 1]^3.

    The latent spline is clamped and degree 1, so its Greville abscissae are
    the knots themselves: control point (i, j, k) sits at
    ``(i/(nx-1), j/(ny-1), k/(nz-1))``. The lattice domain is the unit cube
    because :func:`build_lattice` passes no bounds, so these are the same
    coordinates the slice panels are drawn in.
    """
    out = []
    for index, count in zip((i, j, k), n_ctrl):
        count = int(count)
        out.append(float(index) / (count - 1) if count > 1 else 0.0)
    return tuple(out)


class LatentNeighbors:
    """Nearest-neighbour index over a decoder's trained latent codes.

    Built once when a decoder is loaded, because both halves of the query are
    expensive to redo on every redraw: the dense
    ``(n_cp, n_trained, d)`` difference is 1.1 GB for Primitives2D (19079
    codes of width 16 against 216 control points), and building a k-d tree in
    16 dimensions costs most of a second on its own.
    """

    def __init__(self, trained):
        self.trained = np.asarray(trained, dtype=np.float64)
        self._tree = None
        if self.trained.ndim == 2 and self.trained.shape[0]:
            try:
                from scipy.spatial import cKDTree

                self._tree = cKDTree(self.trained)
            except Exception:
                self._tree = None

    def worst_distance(self, control_points) -> float:
        """Distance from the *furthest* control point to its closest code."""
        cps = np.asarray(control_points, dtype=np.float64)
        if self.trained.size == 0 or cps.size == 0:
            return float("nan")
        if self._tree is not None:
            return float(np.max(self._tree.query(cps, k=1)[0]))
        chunk = max(
            1, int(4_000_000 // max(self.trained.shape[0] * self.trained.shape[1], 1))
        )
        worst = 0.0
        for start in range(0, cps.shape[0], chunk):
            block = cps[start : start + chunk]
            diff = block[:, None, :] - self.trained[None, :, :]
            worst = max(
                worst, float(np.max(np.min(np.linalg.norm(diff, axis=2), axis=1)))
            )
        return worst


def nearest_trained_distance(trained, control_points) -> float:
    """Worst distance from a control point to the closest trained latent code.

    A per-component min/max box says nothing about the holes inside it: the 20
    codes of ``RoundCross`` span [-1, 1] with a 0.4-wide gap in the middle, and
    the mean of them - the app's starting design - falls straight into it. This
    number sees that; the box test does not.

    Convenience wrapper; callers that ask repeatedly should keep a
    :class:`LatentNeighbors` instead of paying for the index every time.
    """
    return LatentNeighbors(trained).worst_distance(control_points)


def largest_gap(values) -> tuple[float, float, float]:
    """Widest empty interval between consecutive sorted ``values``.

    Returns ``(width, lo, hi)``. Used to mark the unsupervised stretch of a
    latent axis on the coverage strip.
    """
    v = np.unique(np.asarray(values, dtype=np.float64).ravel())
    if v.size < 2:
        return 0.0, float("nan"), float("nan")
    gaps = np.diff(v)
    idx = int(np.argmax(gaps))
    return float(gaps[idx]), float(v[idx]), float(v[idx + 1])


# --------------------------------------------------------------------------- #
# extra field evaluations
# --------------------------------------------------------------------------- #


def volume_fraction(sdf, bounds=None, res=20) -> float:
    """Fraction of the domain with ``f_theta < 0``, on a regular grid.

    A coarse Monte-Carlo-free estimate of ``V(lambda_hat)`` - the quantity the
    paper constrains. ``res=20`` is a deliberate compromise: 8000 points cost
    about 100 ms on CPU - fine once a drag has settled, too slow during one.

    The samples are **cell centres**, not grid nodes. A node grid puts
    ``1 - ((res-2)/res)**3`` of its points - 27% at res=20 - exactly on the
    domain faces, and ``CappedBorderSDF`` forces ``phi >= 0`` there, so a node
    grid reports every lattice as roughly a third emptier than it is. Cell
    centres are also the correct midpoint rule for a volume integral.
    """
    bounds = _as_bounds(bounds, sdf)
    device = sdf.get_device() if hasattr(sdf, "get_device") else "cpu"
    dtype = sdf.get_dtype() if hasattr(sdf, "get_dtype") else torch.float32
    res = max(int(res), 1)
    axes = []
    for d in range(3):
        lo, hi = bounds[0, d], bounds[1, d]
        step = (hi - lo) / res
        axes.append(lo + step * (np.arange(res) + 0.5))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = torch.as_tensor(grid, dtype=dtype, device=device)
    with torch.no_grad():
        values = sdf(pts)
    return float((values < 0).double().mean().item())


def unit_cell_sdf(model):
    """A decoder wrapper for previewing one microtile, separate from a lattice.

    ``LatticeSDFStruct`` writes one latent vector *per query point* into its
    microtile and leaves it there, so reusing that same wrapper for a different
    query count raises a shape mismatch. The weights are shared; only the
    wrapper is new.
    """
    return SDFfromDeepSDF(model)


def eval_cell_slice(cell_sdf, latent, z=0.0, res=64) -> np.ndarray:
    """``f_theta(lambda, .)`` on a z-slice of the bare unit cube [-1, 1]^3.

    This is the shape a single latent vector encodes, before the
    transformation function tiles it. Without it a latent value is just a
    number the user has never seen a picture of.
    """
    device = cell_sdf.get_device() if hasattr(cell_sdf, "get_device") else "cpu"
    dtype = cell_sdf.get_dtype() if hasattr(cell_sdf, "get_dtype") else torch.float32
    latent = np.asarray(latent, dtype=np.float64).ravel()
    cell_sdf.latvec = None
    cell_sdf.set_latent_vec(torch.as_tensor(latent, dtype=dtype, device=device))
    xs = np.linspace(-1.0, 1.0, int(res))
    grid_x, grid_y = np.meshgrid(xs, xs)
    pts = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, float(z))], axis=1
    )
    tensor = torch.as_tensor(pts, dtype=dtype, device=device)
    with torch.no_grad():
        values = cell_sdf(tensor)
    return values.detach().cpu().numpy().reshape(int(res), int(res)).astype(np.float64)
