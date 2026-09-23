"""Model discovery, lattice assembly and field evaluation.

Online stage of the pipeline (steps 3-4 of the paper's method): pick a trained
decoder, interpolate its latent vector over the domain with a B-spline whose
control points are the design variables, tile the unit cell, and evaluate
f_theta(lambda(x), x) either as a 2D slice or as a surface mesh.

Decoders trained on 2-D samples (``geom_dimension`` 2 in their specs, e.g. the
``datagen`` plate with a hole) describe a whole shape in the plane rather than
a unit cell, so they skip the lattice entirely; the last section of this module
serves the Explore 2-D tab: the field on the plane, the training shapes behind
each latent code, and how well the codes pin down the parameters that made
those shapes.
"""

from __future__ import annotations

import csv
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

# geom_dimension: how many coordinates the decoder takes next to the latent
# code - 3 for the unit-cell decoders of the paper, 2 for a planar shape.
ModelEntry = namedtuple(
    "ModelEntry",
    "name source ref latent_dim n_latents geom_dimension",
    defaults=(3,),
)

_CHECKPOINT = "latest"


def _read_json(path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_code_length(run_dir: Path) -> int | None:
    """Latent dimension straight from specs.json, without loading any weights."""
    try:
        return int(_read_json(run_dir / "specs.json")["CodeLength"])
    except (KeyError, ValueError, TypeError):
        return None


def _read_geom_dimension(run_dir: Path) -> int:
    """Coordinates per query point, from specs.json; 3 when it does not say."""
    network = _read_json(run_dir / "specs.json").get("NetworkSpecs") or {}
    try:
        return int(network.get("geom_dimension", 3))
    except (ValueError, TypeError):
        return 3


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


def list_models(runs_dir, geom_dimension=None) -> list[ModelEntry]:
    """Every usable decoder: the shipped pretrained ones plus local training runs.

    Parameters
    ----------
    runs_dir : path-like
    geom_dimension : int, optional
        Keep only decoders taking this many coordinates. The lattice explorer
        asks for 3, the planar one for 2; ``None`` lists everything.
    """
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
                geom_dimension=_read_geom_dimension(run_dir),
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
                    geom_dimension=_read_geom_dimension(run_dir),
                )
            )
    if geom_dimension is not None:
        entries = [e for e in entries if e.geom_dimension == int(geom_dimension)]
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

    def nearest(self, point) -> tuple[int, float]:
        """Index of the trained code closest to ``point``, and its distance."""
        point = np.asarray(point, dtype=np.float64).ravel()
        if self.trained.size == 0:
            return -1, float("nan")
        if self._tree is not None:
            dist, index = self._tree.query(point, k=1)
            return int(index), float(dist)
        dists = np.linalg.norm(self.trained - point[None, :], axis=1)
        index = int(np.argmin(dists))
        return index, float(dists[index])

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


# --------------------------------------------------------------------------- #
# whole-shape 2-D decoders (Explore 2-D tab)
# --------------------------------------------------------------------------- #

# The planar decoders are trained on [-1, 1]^2, the same box DeepSDF uses in 3-D.
PLANE_BOUNDS = np.array([[-1.0, -1.0], [1.0, 1.0]])


def geom_dimension(model) -> int:
    """Coordinates per query point of a loaded decoder (2 or 3)."""
    return int(getattr(model._decoder, "geom_dimension", 3))


def decode_2d(model, latent, points) -> np.ndarray:
    """``f_theta(lambda, p)`` at planar points ``(n, 2)`` for one latent code.

    Builds the decoder input by hand - code repeated per point, then the
    coordinates - which is the layout ``DeepSDFDecoder`` reads, for any
    ``geom_dimension``.
    """
    param = next(model._decoder.parameters())
    pts = torch.as_tensor(np.asarray(points), dtype=param.dtype, device=param.device)
    lat = torch.as_tensor(
        np.asarray(latent, dtype=np.float64).ravel(),
        dtype=param.dtype,
        device=param.device,
    )
    with torch.no_grad():
        values = model._decoder(torch.cat([lat.expand(len(pts), -1), pts], dim=1))
    return values.detach().cpu().numpy().reshape(-1).astype(np.float64)


def eval_field_2d(model, latent, res=128) -> np.ndarray:
    """``f_theta(lambda, .)`` on a ``res x res`` node grid over [-1, 1]^2.

    Laid out ``(row=y, col=x)`` for ``imshow(origin="lower")``, the same
    convention as :func:`eval_sdf_slice`, so ``viz.draw_sdf_slice`` draws it.
    """
    res = int(res)
    xs = np.linspace(PLANE_BOUNDS[0, 0], PLANE_BOUNDS[1, 0], res)
    grid_x, grid_y = np.meshgrid(xs, xs)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return decode_2d(model, latent, points).reshape(res, res)


def run_shapes(run_dir) -> dict:
    """The training shapes behind a run, one per latent code, in code order.

    The order comes from ``LatentCodes/latent_code_data_map.json``, which the
    trainer writes with the latent index of every ``.npz``; runs without it
    fall back to the split, which is the order the trainer read. When the
    dataset folder carries a ``params.csv`` (``datagen`` writes one), its rows
    are joined on the instance name: the parameters that generated each shape,
    which the decoder itself never saw.

    Returns
    -------
    dict
        ``names`` and ``npz`` (lists, code order), ``dataset_dir`` (Path or
        None), ``manifest`` (dict, empty without ``dataset.json``), ``params``
        (``{column: array}`` aligned with ``names``, NaN where a row is
        missing) and ``param_names`` (the columns worth showing first).
    """
    run_dir = Path(run_dir)
    specs = _read_json(run_dir / "specs.json")
    source = Path(specs.get("DataSource", "."))

    npz = []
    entries = _read_json(run_dir / "LatentCodes" / "latent_code_data_map.json")
    entries = entries.get("latent_codes") or []
    if entries:
        for entry in sorted(entries, key=lambda e: int(e["latent_index"])):
            path = Path(entry.get("npz_filename", ""))
            if not path.is_file() and entry.get("relative_npz_filename"):
                # the absolute path is from the machine that trained; the
                # relative one survives a moved or re-cloned repo
                rel = Path(entry["relative_npz_filename"].replace("\\", "/"))
                path = source / "SdfSamples" / rel
            npz.append(path)
    else:
        split = _read_json(Path(specs.get("TrainSplit", "")))
        for dataset, classes in split.items():
            for class_name, names in classes.items():
                npz += [
                    source / "SdfSamples" / dataset / class_name / f"{n}.npz"
                    for n in names
                ]

    names = [p.stem for p in npz]
    dataset_dir = npz[0].parent.parent if npz else None
    manifest = _read_json(dataset_dir / "dataset.json") if dataset_dir else {}
    params, param_names = {}, []
    if dataset_dir is not None and (dataset_dir / "params.csv").is_file():
        params = _read_params(dataset_dir / "params.csv", names)
        preferred = (manifest.get("parameters") or {}).get("names") or []
        param_names = [c for c in preferred if c in params] or list(params)
    return {
        "names": names,
        "npz": npz,
        "dataset_dir": dataset_dir,
        "manifest": manifest,
        "params": params,
        "param_names": param_names,
    }


def _read_params(path, names) -> dict:
    """Numeric columns of a parameter table, aligned with ``names``."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_name = {row.get("name"): row for row in rows}
    columns = {}
    for column in rows[0].keys() if rows else []:
        if column == "name":
            continue
        values = np.full(len(names), np.nan)
        for i, name in enumerate(names):
            try:
                values[i] = float(by_name[name][column])
            except (KeyError, TypeError, ValueError):
                pass
        if np.isfinite(values).any():
            columns[column] = values
    return columns


def _read_rows(npz_path) -> np.ndarray | None:
    """All stored rows of one instance, ``pos`` and ``neg`` stacked."""
    try:
        with np.load(npz_path) as npz:
            pos = npz["pos"] if "pos" in npz else npz["pos.npy"]
            neg = npz["neg"] if "neg" in npz else npz["neg.npy"]
            return np.vstack([pos, neg]).astype(np.float64)
    except (OSError, KeyError, ValueError):
        return None


def reconstruction_errors(model, trained, npz_paths, n_points=2000, clamp=0.1):
    """How well each training shape is reproduced by its own latent code.

    Mean ``|clamp(f_theta(lambda_i, x)) - clamp(phi_i(x))|`` over a fixed
    subsample of the rows stored for shape ``i``: the training loss, measured
    per shape instead of averaged. A latent space too small for its family
    shows up here - the shapes it had to give up on stand out.

    Returns
    -------
    numpy.ndarray, shape (n_shapes,)
        NaN for a shape whose ``.npz`` cannot be read.
    """
    trained = np.asarray(trained, dtype=np.float64)
    errors = np.full(len(trained), np.nan)
    for i, path in enumerate(list(npz_paths)[: len(trained)]):
        rows = _read_rows(path)
        if rows is None or rows.shape[1] != 3 or not len(rows):
            continue
        take = np.random.default_rng(i).permutation(len(rows))[:n_points]
        rows = rows[take]
        predicted = np.clip(decode_2d(model, trained[i], rows[:, :2]), -clamp, clamp)
        target = np.clip(rows[:, 2], -clamp, clamp)
        errors[i] = float(np.mean(np.abs(predicted - target)))
    return errors


def boundary_samples(npz_path, band=0.02, max_points=4000) -> np.ndarray:
    """Stored samples of one shape that lie within ``band`` of its surface.

    Drawn over the decoded field, they show where the true boundary of that
    training shape is: the honest check of what the decoder made of it.
    """
    rows = _read_rows(npz_path)
    if rows is None or rows.shape[1] != 3:
        return np.zeros((0, 2))
    near = rows[np.abs(rows[:, 2]) < band]
    if len(near) > max_points:
        near = near[np.random.default_rng(0).permutation(len(near))[:max_points]]
    return near[:, :2]


def neighbour_r2(codes, values, k=3) -> float:
    """How well the latent codes pin down one generating parameter.

    Leave-one-out k-nearest-neighbour regression: each shape's parameter is
    predicted as the mean over its ``k`` closest *other* codes, and the score
    is the usual ``R^2 = 1 - SSE / SST``. Near 1: codes that are close belong
    to shapes with close values, so the parameter can be read back from the
    latent vector. Near 0 or below: the latent space does not keep it. Unlike a
    linear fit it does not care how the parameter is laid out in the space.
    """
    codes = np.asarray(codes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    codes, values = codes[ok], values[ok]
    n = len(values)
    k = int(k)
    if n < k + 2 or float(np.var(values)) <= 0.0:
        return float("nan")
    from scipy.spatial import cKDTree

    _, index = cKDTree(codes).query(codes, k=k + 1)
    predicted = np.empty(n)
    for i in range(n):
        # the point itself is usually first, but not when codes coincide
        others = [j for j in np.atleast_1d(index[i]) if j != i][:k]
        predicted[i] = values[others].mean()
    sse = float(np.sum((values - predicted) ** 2))
    sst = float(np.sum((values - values.mean()) ** 2))
    return 1.0 - sse / sst
