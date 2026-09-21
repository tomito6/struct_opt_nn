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
