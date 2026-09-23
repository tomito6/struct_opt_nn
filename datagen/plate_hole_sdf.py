"""Exact signed distance of a plate with one circular hole, and samples of it.

Where this sits in the pipeline
-------------------------------
Step 1 of the paper's offline stage: "sample points around the geometry and
compute signed distances". :mod:`datagen.plate_hole_params` says *which* holes
to build; this module turns one ``(x_c, y_c, r)`` into the ``(x, y, [z,] phi)``
rows a DeepSDF decoder is trained on. Nothing here is a network input except the
coordinates: the three hole parameters only shape the field and are never shown
to the decoder, which has to discover its own latent code for each plate.

2-D or 3-D
----------
A plate of constant thickness with a through-hole carries all of its
information in the ``(x, y)`` plane: the 3-D field is the 2-D one extruded,

    phi_3d(x, y, z) = extrude(phi_2d(x, y), |z| - h)

so a 3-D decoder spends most of its samples and capacity learning two flat
faces that are identical for every shape. ``dim=2`` trains on the plane alone
(rows ``(x, y, phi)``); ``dim=3`` keeps the thickness (rows ``(x, y, z, phi)``)
for when the geometry has to go through the 3-D mesher and FEM. The 2-D field
can always be extruded later, so choosing 2-D closes no door.

The field is exact, not a bound
-------------------------------
The usual boolean ``max(plate, -hole)`` is only a lower bound on the distance
near places where the two surfaces meet. Here they never meet: every admissible
hole keeps a ligament of at least the margin to the plate edge. With that, each
of the three regions has a single nearest boundary --

* outside the plate: the outer edge, so ``phi = plate``;
* inside the material: whichever of edge and hole is closer, which is exactly
  ``max(plate, -hole)`` since both are negative-distance there;
* inside the hole: the hole wall, and the outer edge is farther by at least
  the ligament, so ``phi = -hole``;

and the formula is the true Euclidean distance everywhere. The extrusion used
for 3-D is exact for an exact 2-D input (it is the distance to a prism). The
test suite checks both against a brute-force distance to a densely sampled
boundary (outline in 2-D, surface in 3-D).

The normalized frame
--------------------
DeepSDF works in ``[-1, 1]^d``. :class:`PlateFrame` maps the plate's own frame
(origin at the lower-left corner, design units) onto it: centred, uniform
scale, longer side spanning ``[-(1 - pad), 1 - pad]``. The ``pad`` leaves a ring
of empty space inside the sampling box, so the outer edge is learned from both
sides instead of sitting on the border of everything the decoder ever sees.
Uniform scaling keeps the field a true distance (the scale is stored with the
dataset, so distances convert back to design units exactly).

Where the samples go
--------------------
Uniform samples in the box teach the decoder the coarse field; a band of
samples pushed off the boundary along its normal teaches it where the surface
is. The band is *not* spread by arc length: the outer edge is the same for
every shape, while the hole is the only thing that varies. ``hole_fraction``
reserves that share of the band for the hole wall, whatever its size.

Examples
--------
    from datagen.plate_hole_sdf import PlateFrame, SamplingConfig, sample_instance

    frame = PlateFrame()                       # 1 x 1 plate, pad 0.1, 2-D
    rows = sample_instance(np.random.default_rng(0), 2, frame, 0.5, 0.5, 0.2,
                           SamplingConfig(n_uniform=5000, n_band=5000))
    rows.shape                                 # (10000, 3): x, y, phi
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import numpy as np

# Offsets of the band samples along the normal, in normalized units. The same
# two levels DeepSDFStruct's mesh sampler uses by default, so datasets from
# both sources look alike to the trainer (whose clamp is 0.1 by default).
DEFAULT_STDS = (0.05, 0.025)

# Share of the band that goes to the hole wall. Half: the hole is the only
# thing that differs between shapes, but the outer edge still has to be sharp.
DEFAULT_HOLE_FRACTION = 0.5

# Empty ring left between the plate edge and the border of [-1, 1]^d.
DEFAULT_PAD = 0.1

# Samples per instance, uniform + band. 2-D needs far fewer than 3-D for the
# same point density: 10k points in a square are ~100 per side, which a 3-D
# cube only matches at a million.
DEFAULT_COUNTS = {2: (10_000, 10_000), 3: (50_000, 50_000)}


# ------------------------------------------------------------------- the frame


@dataclasses.dataclass(frozen=True)
class PlateFrame:
    """Map between the plate's own frame and the ``[-1, 1]^d`` training frame.

    Parameters
    ----------
    length, width : float
        Plate size along x and y in design units, as in
        :class:`datagen.plate_hole_params.PlateHoleSpace`.
    thickness : float or None
        Plate thickness, needed by ``dim=3`` only. ``None`` for a 2-D frame,
        which then has no :attr:`half_thickness`.
    pad : float
        Fraction of the half-box left empty around the plate, in ``[0, 1)``.

    Notes
    -----
    ``normalized = scale * (design - plate_centre)``, the same scale on every
    axis, so a normalized distance divided by :attr:`scale` is a design-unit
    distance.
    """

    length: float = 1.0
    width: float = 1.0
    thickness: float | None = None
    pad: float = DEFAULT_PAD

    def __post_init__(self):
        if min(self.length, self.width) <= 0:
            raise ValueError("plate length and width must be positive")
        if not 0.0 <= self.pad < 1.0:
            raise ValueError(f"pad must lie in [0, 1), got {self.pad}")
        if self.thickness is not None:
            if self.thickness <= 0:
                raise ValueError("plate thickness must be positive")
            # The faces get the same air as the edges: inside 1 - pad.
            if self.half_thickness > 1.0 - self.pad:
                raise ValueError(
                    f"plate half-thickness {self.half_thickness:.3f} (normalized) "
                    f"exceeds 1 - pad = {1.0 - self.pad:.3f}; the plate does not "
                    "fit the box"
                )

    @property
    def scale(self) -> float:
        """Design units -> normalized units."""
        return 2.0 * (1.0 - self.pad) / max(self.length, self.width)

    @property
    def half_extents(self) -> tuple[float, float]:
        """Plate half-size ``(a, b)`` in the normalized frame."""
        return 0.5 * self.scale * self.length, 0.5 * self.scale * self.width

    @property
    def half_thickness(self) -> float:
        """Plate half-thickness ``h`` in the normalized frame (3-D frames only)."""
        if self.thickness is None:
            raise ValueError("this frame has no thickness; build it with one for 3-D")
        return 0.5 * self.scale * self.thickness

    def hole(self, x_c, y_c, r) -> tuple[np.ndarray, float]:
        """Hole centre and radius in the normalized frame."""
        centre = np.array(
            [
                self.scale * (float(x_c) - 0.5 * self.length),
                self.scale * (float(y_c) - 0.5 * self.width),
            ]
        )
        return centre, self.scale * float(r)

    def as_dict(self, dim: int) -> dict:
        """What a dataset manifest needs to map its samples back to the plate."""
        a, b = self.half_extents
        return {
            "length": self.length,
            "width": self.width,
            "thickness": self.thickness if dim == 3 else None,
            "pad": self.pad,
            "scale": self.scale,
            "plate_half_extents": [a, b],
            "plate_half_thickness": self.half_thickness if dim == 3 else None,
            "mapping": "normalized = scale * (design - plate_centre)",
        }


# ------------------------------------------------------------------- the field


def box_sdf_2d(points, a, b) -> np.ndarray:
    """Exact signed distance to the rectangle ``[-a, a] x [-b, b]``.

    Parameters
    ----------
    points : array_like, shape (n, 2)
    a, b : float
        Half-sizes along x and y.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    q = np.abs(np.asarray(points, dtype=float)) - np.array([a, b])
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def plate_hole_sdf_2d(points, half_extents, centre, radius) -> np.ndarray:
    """Exact signed distance to a rectangle with one circular hole.

    Exact under the condition the design space enforces: the hole lies
    strictly inside the rectangle (see the module docstring).

    Parameters
    ----------
    points : array_like, shape (n, 2)
    half_extents : (float, float)
        Rectangle half-sizes ``(a, b)``.
    centre : array_like, shape (2,)
    radius : float

    Returns
    -------
    numpy.ndarray, shape (n,)
        Negative inside the material, positive outside the plate and in the hole.
    """
    points = np.asarray(points, dtype=float)
    plate = box_sdf_2d(points, *half_extents)
    hole = np.linalg.norm(points - np.asarray(centre, dtype=float), axis=-1) - radius
    return np.maximum(plate, -hole)


def extrude(phi_2d, z, half_thickness) -> np.ndarray:
    """Exact signed distance to the prism ``{phi_2d <= 0} x [-h, h]``.

    Standard extrusion: with ``w = (phi_2d, |z| - h)`` the distance is the
    length of the positive part of ``w`` outside, and ``max(w)`` inside.
    """
    w0 = np.asarray(phi_2d, dtype=float)
    w1 = np.abs(np.asarray(z, dtype=float)) - half_thickness
    inside = np.minimum(np.maximum(w0, w1), 0.0)
    outside = np.hypot(np.maximum(w0, 0.0), np.maximum(w1, 0.0))
    return inside + outside


def plate_hole_sdf(dim, frame: PlateFrame, x_c, y_c, r) -> Callable:
    """The field of one plate as a function of normalized points.

    Parameters
    ----------
    dim : {2, 3}
    frame : PlateFrame
    x_c, y_c, r : float
        The hole, in design units.

    Returns
    -------
    callable
        ``phi(points)`` with ``points`` of shape ``(n, dim)``, returning ``(n,)``.
    """
    _check_dim(dim)
    centre, radius = frame.hole(x_c, y_c, r)
    half_extents = frame.half_extents

    if dim == 2:
        return lambda p: plate_hole_sdf_2d(p, half_extents, centre, radius)

    h = frame.half_thickness

    def phi(points):
        points = np.asarray(points, dtype=float)
        in_plane = plate_hole_sdf_2d(points[:, :2], half_extents, centre, radius)
        return extrude(in_plane, points[:, 2], h)

    return phi


# ----------------------------------------------------------------- the samples


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """How many samples one instance gets, and where.

    Parameters
    ----------
    n_uniform : int
        Uniform samples in ``[-bound, bound]^dim``.
    n_band : int
        Samples pushed off the boundary along its normal.
    hole_fraction : float
        Share of ``n_band`` reserved for the hole wall, in ``[0, 1]``.
    stds : sequence of float
        Offset levels of the band; each band sample draws one of them.
    bound : float
        Half-size of the uniform sampling box.
    """

    n_uniform: int = DEFAULT_COUNTS[2][0]
    n_band: int = DEFAULT_COUNTS[2][1]
    hole_fraction: float = DEFAULT_HOLE_FRACTION
    stds: Sequence[float] = DEFAULT_STDS
    bound: float = 1.0

    def __post_init__(self):
        if self.n_uniform < 0 or self.n_band < 0 or self.n_uniform + self.n_band < 1:
            raise ValueError("need a non-negative number of samples, and at least one")
        if not 0.0 <= self.hole_fraction <= 1.0:
            raise ValueError("hole_fraction must lie in [0, 1]")
        if not len(self.stds) or min(self.stds) <= 0:
            raise ValueError("stds must be a non-empty list of positive values")
        # A tuple, so the frozen instance stays hashable and JSON-friendly.
        object.__setattr__(self, "stds", tuple(float(s) for s in self.stds))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self) | {"stds": list(self.stds)}


def _check_dim(dim):
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")


def _rectangle_perimeter(rng, n, a, b):
    """Points spread by arc length on the rectangle outline, with outward normals."""
    lengths = np.array([2 * a, 2 * b, 2 * a, 2 * b])  # bottom, right, top, left
    s = rng.uniform(0.0, lengths.sum(), n)
    edge = np.searchsorted(np.cumsum(lengths), s, side="right")
    edge = np.minimum(edge, 3)  # s == total on the last edge
    along = s - np.concatenate([[0.0], np.cumsum(lengths)[:-1]])[edge]

    points = np.empty((n, 2))
    normals = np.zeros((n, 2))
    for k, (start, direction, normal) in enumerate(
        [
            ((-a, -b), (1.0, 0.0), (0.0, -1.0)),  # bottom, left to right
            ((a, -b), (0.0, 1.0), (1.0, 0.0)),  # right, bottom to top
            ((a, b), (-1.0, 0.0), (0.0, 1.0)),  # top, right to left
            ((-a, b), (0.0, -1.0), (-1.0, 0.0)),  # left, top to bottom
        ]
    ):
        mask = edge == k
        points[mask] = np.array(start) + along[mask, None] * np.array(direction)
        normals[mask] = normal
    return points, normals


def _circle(rng, n, centre, radius):
    """Points spread by angle on the circle, with normals pointing out of the hole.

    "Out of the hole" is into the material, i.e. the outward normal of the
    material is the opposite; only the line matters for the offset, not the
    direction, because the offsets are symmetric.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return np.asarray(centre) + radius * normals, normals


def _face_points(rng, n, a, b, centre, radius):
    """Points uniform on the plate footprint minus the hole (top/bottom faces)."""
    out = np.empty((0, 2))
    while len(out) < n:
        # Acceptance is 1 - hole area / plate area: one pass for most holes, a
        # few for the largest (down to ~36 % at the 0.05 training margin). It
        # always ends, since a hole never covers the whole footprint.
        take = int(1.5 * (n - len(out))) + 16
        p = rng.uniform([-a, -b], [a, b], size=(take, 2))
        keep = np.linalg.norm(p - centre, axis=1) > radius
        out = np.vstack([out, p[keep]])
    return out[:n]


def band_points(rng, dim, frame: PlateFrame, x_c, y_c, r, config: SamplingConfig):
    """Samples offset from the boundary along its normal.

    Parameters
    ----------
    rng : numpy.random.Generator
    dim : {2, 3}
    frame : PlateFrame
    x_c, y_c, r : float
        The hole, in design units.
    config : SamplingConfig

    Returns
    -------
    numpy.ndarray, shape (config.n_band, dim)
    """
    _check_dim(dim)
    n = int(config.n_band)
    a, b = frame.half_extents
    centre, radius = frame.hole(x_c, y_c, r)
    n_hole = int(round(config.hole_fraction * n))
    n_rest = n - n_hole

    if dim == 2:
        hole_p, hole_n = _circle(rng, n_hole, centre, radius)
        edge_p, edge_n = _rectangle_perimeter(rng, n_rest, a, b)
        points = np.vstack([hole_p, edge_p])
        normals = np.vstack([hole_n, edge_n])
    else:
        h = frame.half_thickness
        # Hole wall: a cylinder of height 2h.
        hole_xy, hole_n2 = _circle(rng, n_hole, centre, radius)
        hole_p = np.column_stack([hole_xy, rng.uniform(-h, h, n_hole)])
        hole_n = np.column_stack([hole_n2, np.zeros(n_hole)])

        # The rest by area between the two faces and the outer side walls.
        face_area = 2.0 * (4.0 * a * b - np.pi * radius**2)
        side_area = 4.0 * (a + b) * 2.0 * h
        n_face = int(round(n_rest * face_area / (face_area + side_area)))
        n_side = n_rest - n_face

        face_xy = _face_points(rng, n_face, a, b, centre, radius)
        up = rng.choice([-1.0, 1.0], n_face)
        face_p = np.column_stack([face_xy, up * h])
        face_n = np.column_stack([np.zeros((n_face, 2)), up])

        side_xy, side_n2 = _rectangle_perimeter(rng, n_side, a, b)
        side_p = np.column_stack([side_xy, rng.uniform(-h, h, n_side)])
        side_n = np.column_stack([side_n2, np.zeros(n_side)])

        points = np.vstack([hole_p, face_p, side_p])
        normals = np.vstack([hole_n, face_n, side_n])

    sigma = np.asarray(config.stds)[rng.integers(len(config.stds), size=n)]
    return points + rng.normal(0.0, 1.0, n)[:, None] * sigma[:, None] * normals


def sample_instance(rng, dim, frame: PlateFrame, x_c, y_c, r, config: SamplingConfig):
    """All training rows of one plate: uniform samples plus the boundary band.

    The distance is evaluated exactly at every point, the band included, so a
    band sample that was pushed past a corner or across a thin ligament still
    carries the right label.

    Parameters
    ----------
    rng : numpy.random.Generator
    dim : {2, 3}
    frame : PlateFrame
    x_c, y_c, r : float
        The hole, in design units.
    config : SamplingConfig

    Returns
    -------
    numpy.ndarray, shape (n_uniform + n_band, dim + 1), float32
        Rows ``(x, y, phi)`` or ``(x, y, z, phi)`` in the normalized frame.
    """
    _check_dim(dim)
    uniform = rng.uniform(-config.bound, config.bound, size=(config.n_uniform, dim))
    band = band_points(rng, dim, frame, x_c, y_c, r, config)
    points = np.vstack([uniform, band])
    phi = plate_hole_sdf(dim, frame, x_c, y_c, r)(points)
    return np.column_stack([points, phi]).astype(np.float32)
