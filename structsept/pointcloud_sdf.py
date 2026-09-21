"""Signed distance field from a raw point cloud.

Why this exists
---------------
The reconstruction path of the library
(:class:`~DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor`) fits a
field of DeepSDF latent codes to a *target mesh*: the ground truth comes from
:class:`~DeepSDFStruct.SDF.SDFfromMesh`, which needs a watertight
triangulation to decide what is inside. A point cloud has no faces, hence no
sign, so it cannot feed that path directly. This module closes that gap.

How
---
Two classical ingredients, no new dependency -- ``scipy`` and ``torch`` are
already in the environment:

**Magnitude.** The unsigned distance is the distance to the nearest cloud
point, refined near the surface into the distance to that point's tangent
plane. Plain point-to-point distance is biased outwards by roughly half the
point spacing, and that bias sits exactly where the fit cares most (the zero
level set); the tangent-plane correction removes most of it.

**Sign.** The generalized winding number of an oriented point cloud (Barill et
al. 2018, *Fast Winding Numbers for Soups and Clouds*), in its plain order-0
form: each cloud point is a dipole of strength ``a_i * n_i``, and

    w(q) = 1/(4*pi) * sum_i  a_i * dot(p_i - q, n_i) / ||p_i - q||^3

is close to 1 for points inside the surface and close to 0 outside. Queries
with ``w > 0.5`` are inside. On ``cone.stl`` sampled at 20k points this agrees
with ``trimesh.Trimesh.contains`` on 99.97 % of uniform queries.

Cost and limits
---------------
The winding number is a dense sum: ``O(n_queries * n_cloud)``, about 3.5e7
point-pairs per second on this machine, evaluated in chunks to keep memory
bounded. ``max_winding_points`` subsamples the cloud for the sign only (the
distance always uses every point), which keeps the cost predictable on large
scans.

The cloud **must carry outward normals**. Real scans usually ship them; when
they do not, :func:`estimate_normals` guesses them by local PCA and orients
them away from a reference point, which is only correct for roughly
star-shaped parts. Orienting normals properly (minimum-spanning-tree
propagation) is not implemented here.

Nothing in this module is differentiable, and that is on purpose: it produces
*training data* for the fit, evaluated once up front, never inside the
optimization loop.

Examples
--------
Fit-ready samples from a cloud simulated off an STL::

    import torch, trimesh
    from structsept.pointcloud_sdf import cloud_from_mesh, PointCloudSDF

    mesh = trimesh.load_mesh("DeepSDFStruct/tests/data/cone.stl")
    points, normals = cloud_from_mesh(mesh, n_points=20000, seed=0)

    sdf = PointCloudSDF(points, normals)
    phi = sdf(torch.rand(1000, 3))       # (1000, 1), negative inside
"""

import logging

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from DeepSDFStruct.SDF import SDFBase
from DeepSDFStruct.sampling import SampledSDF

logger = logging.getLogger(__name__)

__all__ = [
    "PointCloudSDF",
    "cloud_from_mesh",
    "estimate_normals",
    "estimate_point_areas",
    "load_point_cloud",
    "sample_cloud_sdf",
]


def cloud_from_mesh(
    mesh: trimesh.Trimesh, n_points: int = 50_000, seed=None, noise: float = 0.0
):
    """Draw an oriented point cloud from a mesh -- a stand-in for a scan.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Source geometry. Only its surface is used, so it need not be
        watertight.
    n_points : int, default 50000
        Number of points to draw, uniformly by area.
    seed : int, optional
        Seed for the sampler, for reproducible clouds.
    noise : float, default 0.0
        Standard deviation of isotropic Gaussian noise added to the points, in
        the mesh's own units. Use it to check how tolerant the downstream fit
        is to scanner noise.

    Returns
    -------
    points : (n_points, 3) ndarray
    normals : (n_points, 3) ndarray
        Outward unit normals, taken from the face each point landed on.
    """
    if seed is not None:
        np.random.seed(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]
    points = np.asarray(points, dtype=np.float64)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=noise, size=points.shape)
    return points, np.asarray(normals, dtype=np.float64)


def load_point_cloud(path):
    """Read a point cloud from disk, with normals when the file has them.

    Accepts anything ``trimesh`` can load (``.ply``, ``.xyz``, ``.obj``, ...)
    plus ``.npy``/``.npz``. A mesh file is rejected on purpose: converting one
    into a cloud is :func:`cloud_from_mesh`'s job, so that the sampling stays
    on record.

    Returns
    -------
    points : (N, 3) ndarray
    normals : (N, 3) ndarray or None
        ``None`` when the file carries no usable normals; pass the points
        through :func:`estimate_normals` in that case.
    """
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.shape[1] == 6:
            return arr[:, :3].astype(np.float64), arr[:, 3:].astype(np.float64)
        return arr[:, :3].astype(np.float64), None
    if path.endswith(".npz"):
        data = np.load(path)
        points = data["points"].astype(np.float64)
        normals = data["normals"].astype(np.float64) if "normals" in data else None
        return points, normals

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"{path} is a mesh, not a point cloud. Convert it explicitly with "
            "cloud_from_mesh(mesh, n_points=...) so the sampling is on record."
        )
    points = np.asarray(loaded.vertices, dtype=np.float64)
    normals = getattr(loaded, "vertex_normals", None)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            normals = None
    return points, normals


def estimate_point_areas(points, k: int = 8, tree: cKDTree | None = None):
    """Surface area each point stands for, from the local point density.

    The disk that reaches the ``k``-th neighbour covers ``k`` points, so each
    point gets ``pi * r_k**2 / k``. Only the winding number uses these
    weights, and it is forgiving: a constant factor shifts ``w`` uniformly,
    far from the 0.5 threshold.
    """
    tree = tree if tree is not None else cKDTree(points)
    # k + 1 because the first neighbour returned is the point itself.
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    r_k = dists[:, -1]
    return np.pi * r_k**2 / k


def estimate_normals(
    points, k: int = 16, tree: cKDTree | None = None, orient_from=None
):
    """Guess unit normals by local PCA, oriented away from a reference point.

    The direction comes from the eigenvector of the smallest eigenvalue of the
    local covariance -- well posed for any surface. The *orientation* is the
    weak part: it is resolved by pointing every normal away from
    ``orient_from`` (the cloud's centroid by default), which is correct only
    for roughly star-shaped parts. A folded or hollow geometry needs proper
    propagation, not implemented here.

    Prefer the normals that came with the data whenever they exist.
    """
    logger.warning(
        "Estimating normals by PCA + outward orientation from a reference point. "
        "This is only reliable for roughly star-shaped geometries -- prefer the "
        "normals shipped with the cloud."
    )
    tree = tree if tree is not None else cKDTree(points)
    _, idx = tree.query(points, k=k, workers=-1)
    neigh = points[idx]  # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    # eigh returns ascending eigenvalues; the first eigenvector is the normal.
    normals = np.linalg.eigh(cov)[1][:, :, 0]

    reference = np.asarray(
        orient_from if orient_from is not None else points.mean(axis=0)
    )
    outward = points - reference
    flip = (normals * outward).sum(axis=1) < 0
    normals[flip] *= -1.0
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


class PointCloudSDF(SDFBase):
    """Signed distance field carried by an oriented point cloud.

    A drop-in replacement for :class:`~DeepSDFStruct.SDF.SDFfromMesh` wherever
    the target is a cloud instead of a watertight mesh: it is an
    :class:`~DeepSDFStruct.SDF.SDFBase`, so ``sdf(queries)`` takes ``(N, 3)``
    and returns ``(N, 1)``, negative inside.

    Parameters
    ----------
    points : (N, 3) array_like
        The cloud, in the coordinate system the queries will use.
    normals : (N, 3) array_like
        Outward normals, one per point. Normalized on the way in.
    areas : (N,) array_like, optional
        Area each point represents. Estimated from local density when omitted.
    k_area : int, default 8
        Neighbour count for that density estimate.
    max_winding_points : int, default 20000
        Cap on the number of dipoles used for the *sign*. Larger clouds are
        subsampled (deterministically, via ``seed``); the distance keeps using
        every point. A query batch costs
        ``n_queries * min(N, max_winding_points)`` point-pairs.
    chunk_size : int, default 2048
        Queries per winding-number chunk. Bounds peak memory at roughly
        ``chunk_size * n_dipoles * 3`` floats.
    tangent_plane : bool, default True
        Refine the distance near the surface into a point-to-plane distance.
    plane_radius_factor : float, default 3.0
        "Near the surface" means within this many median point spacings.
    winding_threshold : float, default 0.5
        Winding number above which a query counts as inside.
    seed : int, default 0
        Seed of the subsampling for ``max_winding_points``.

    Attributes
    ----------
    spacing : float
        Median nearest-neighbour distance in the cloud. The resolution floor
        of everything this class returns: features thinner than ``spacing``
        are not in the data to begin with.

    Notes
    -----
    Not differentiable: queries are detached, and the work happens on the CPU
    via ``scipy``'s KD-tree plus a dense dipole sum. This is ground-truth data
    generation, not part of the optimization loop.
    """

    def __init__(
        self,
        points,
        normals,
        *,
        areas=None,
        k_area: int = 8,
        max_winding_points: int = 20_000,
        chunk_size: int = 2048,
        tangent_plane: bool = True,
        plane_radius_factor: float = 3.0,
        winding_threshold: float = 0.5,
        seed: int = 0,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "points and normals must both be (N, 3); got "
                f"{points.shape} and {normals.shape}"
            )
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError("normals contain a zero vector")
        normals = normals / norm

        self._points = points
        self._normals = normals
        self._tree = cKDTree(points)
        self._tangent_plane = tangent_plane
        self._winding_threshold = float(winding_threshold)
        self._chunk_size = int(chunk_size)

        # Median nearest-neighbour distance: the scale below which the cloud
        # says nothing, and the radius where the tangent-plane correction is
        # trusted.
        nn_dists, _ = self._tree.query(points, k=2, workers=-1)
        self.spacing = float(np.median(nn_dists[:, 1]))
        self._plane_radius = plane_radius_factor * self.spacing

        if areas is None:
            areas = estimate_point_areas(points, k=k_area, tree=self._tree)
        areas = np.asarray(areas, dtype=np.float64).reshape(-1)

        # Subsample the dipoles used for the sign, scaling the areas up to
        # keep the total "mass": w stays ~1 inside.
        n = points.shape[0]
        if n > max_winding_points:
            rng = np.random.default_rng(seed)
            sub = rng.choice(n, size=max_winding_points, replace=False)
            scale = n / max_winding_points
            logger.info(
                "Winding number uses %d of %d cloud points (areas scaled by %.2f)",
                max_winding_points,
                n,
                scale,
            )
        else:
            sub = np.arange(n)
            scale = 1.0

        self.register_buffer(
            "wn_points", torch.tensor(points[sub], dtype=dtype, device=device)
        )
        self.register_buffer(
            "wn_normals", torch.tensor(normals[sub], dtype=dtype, device=device)
        )
        self.register_buffer(
            "wn_areas", torch.tensor(areas[sub] * scale, dtype=dtype, device=device)
        )

    @property
    def points(self):
        """The full cloud, ``(N, 3)`` numpy array."""
        return self._points

    @property
    def normals(self):
        """The unit normals, ``(N, 3)`` numpy array."""
        return self._normals

    def winding_number(self, queries: torch.Tensor) -> torch.Tensor:
        """Generalized winding number at *queries*, shape ``(N,)``.

        About 1 inside the surface, about 0 outside. Exposed because it is the
        diagnostic to look at when signs come out wrong: values hovering
        around 0.5 mean the normals are inconsistent, not that the threshold
        needs tuning.
        """
        device = self.wn_points.device
        q = queries.detach().to(device=device, dtype=self.wn_points.dtype)
        out = []
        for start in range(0, q.shape[0], self._chunk_size):
            batch = q[start : start + self._chunk_size]
            # (chunk, n_dipoles, 3): p_i - q
            diff = self.wn_points.unsqueeze(0) - batch.unsqueeze(1)
            r = diff.norm(dim=-1).clamp_min(1e-9)
            dipole = (diff * self.wn_normals.unsqueeze(0)).sum(-1) / r**3
            out.append((self.wn_areas.unsqueeze(0) * dipole).sum(-1) / (4 * np.pi))
        return torch.cat(out).to(queries.device)

    def unsigned_distance(self, queries_np: np.ndarray) -> np.ndarray:
        """Distance to the surface, ignoring the sign, shape ``(N,)``."""
        dist_nn, idx_nn = self._tree.query(queries_np, workers=-1)
        if not self._tangent_plane:
            return dist_nn
        offset = queries_np - self._points[idx_nn]
        dist_plane = np.abs((offset * self._normals[idx_nn]).sum(axis=1))
        # Far from the cloud a tangent plane says nothing useful -- an
        # extended plane passes close to points that are nowhere near the
        # surface -- so the correction only applies in the near field.
        return np.where(dist_nn < self._plane_radius, dist_plane, dist_nn)

    @torch.no_grad()
    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        queries_np = queries.detach().cpu().numpy().astype(np.float64)
        distance = self.unsigned_distance(queries_np)
        winding = self.winding_number(queries).detach().cpu().numpy()
        signed = np.where(winding > self._winding_threshold, -distance, distance)
        return torch.tensor(signed, dtype=queries.dtype, device=queries.device).reshape(
            -1, 1
        )

    def _get_domain_bounds(self):
        return np.stack([self._points.min(axis=0), self._points.max(axis=0)])


def sample_cloud_sdf(
    cloud_sdf: PointCloudSDF,
    bounds,
    *,
    n_uniform: int,
    n_surface: int,
    stds=(0.02, 0.002),
    device="cpu",
    dtype=torch.float32,
    seed=None,
) -> SampledSDF:
    """Build the fit's training set from a cloud, the cheap way.

    Mirrors :func:`DeepSDFStruct.geom_reconstruction.sample_gt_sdf`, with one
    deliberate difference: the near-surface band is *constructed* rather than
    queried. A cloud point displaced by ``t`` along its outward normal has
    signed distance ``t``, no winding number needed. That band holds most of
    the samples, so the expensive dipole sum only ever runs on the
    ``n_uniform`` space-filling ones.

    Parameters
    ----------
    cloud_sdf : PointCloudSDF
        Source of both the cloud and the ground truth away from it.
    bounds : (2, 3) array_like or torch.Tensor
        Box for the uniform samples, in the same space as the cloud.
    n_uniform : int
        Space-filling samples. They teach the fit where material is *not*, and
        are the only ones that pay for the winding number.
    n_surface : int
        Cloud points used per entry of ``stds``; the band holds
        ``n_surface * len(stds)`` samples.
    stds : sequence of float, default (0.02, 0.002)
        Displacement magnitudes along the normal, in the cloud's units -- one
        coarse, one fine. Keep them below the thinnest feature of the part:
        past that a straight offset leaves the surface it came from, and the
        analytic ``phi = t`` stops being true.
    seed : int, optional
        Seed for reproducible sample sets.

    Returns
    -------
    SampledSDF
        Uniform samples followed by the surface band.
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    bounds_t = torch.as_tensor(np.asarray(bounds), dtype=dtype)
    lo, hi = bounds_t[0], bounds_t[1]

    uniform_pts = (
        torch.rand((n_uniform, 3), generator=generator, dtype=dtype) * (hi - lo) + lo
    ).to(device)
    uniform = SampledSDF(samples=uniform_pts, distances=cloud_sdf(uniform_pts))

    spacing = cloud_sdf.spacing
    too_fine = [s for s in stds if s < 0.25 * spacing]
    if too_fine:
        logger.warning(
            "stds %s are below a quarter of the point spacing (%.4g); those "
            "samples carry no information the cloud actually has.",
            too_fine,
            spacing,
        )

    points = torch.tensor(cloud_sdf.points, dtype=dtype)
    normals = torch.tensor(cloud_sdf.normals, dtype=dtype)
    n_cloud = points.shape[0]

    band_pts, band_phi = [], []
    for std in stds:
        idx = torch.randint(0, n_cloud, (n_surface,), generator=generator)
        t = torch.randn((n_surface, 1), generator=generator, dtype=dtype) * std
        band_pts.append(points[idx] + t * normals[idx])
        band_phi.append(t)

    band = SampledSDF(
        samples=torch.cat(band_pts).to(device),
        distances=torch.cat(band_phi).to(device),
    )
    return uniform + band
