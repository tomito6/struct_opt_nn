"""Exact signed distance of a square plate with four triangular holes, and samples.

Where this sits in the pipeline
-------------------------------
The triangle counterpart of :mod:`datagen.plate_hole_sdf`, and the same step:
:mod:`datagen.plate_tri_params` says *which* triangles to build; this module
turns one row -- a height ``h`` and a base width, or four of each -- into the
``(x, y, [z,] phi)`` rows a DeepSDF decoder is trained on. The frame, the
extrusion to 3-D, the sampling configuration and the plate perimeter are the
hole module's, imported rather than copied: only the holes differ.

The field is exact, not a bound
-------------------------------
``max(plate, -min_i(tri_i))`` with ``tri_i`` the exact signed distance to the
i-th triangle. The argument is the one of the hole module, with four holes
instead of one: the design space keeps every triangle at least the margin
away from the plate edge and from the other triangles, so

* outside the plate the nearest boundary is the outer edge, ``phi = plate``;
* inside the material it is whichever of edge and triangles is closest, and
  all of them are negative-distance there, so ``max(plate, -min tri)``;
* inside a triangle it is that triangle's own wall, ``phi = -tri_i``, and
  the formula picks it because ``plate < 0 < -tri_i``.

The distance to one triangle is the standard exact formula for a convex
polygon (:func:`triangle_sdf_2d`): the distance to the nearest of the three
edges, signed by whether the point is on the inner side of all of them. The
test suite checks the whole field against a brute-force distance to a densely
sampled boundary, in 2-D and in 3-D.

Where the samples go
--------------------
Same recipe as the hole: uniform samples in the box, plus a band pushed off
the boundary along its normal, with ``hole_fraction`` of the band reserved
for the twelve triangle edges (spread by arc length among them) and the rest
on the plate outline. The triangle corners get no special treatment -- the
distance is evaluated exactly at every band point, so a sample pushed past
a corner still carries the right label.

Examples
--------
    from datagen.plate_hole_sdf import SamplingConfig
    from datagen.plate_tri_sdf import plate_tri_sdf, sample_instance, square_frame

    frame = square_frame(1.0)                  # 1 x 1 plate, pad 0.1, 2-D
    phi = plate_tri_sdf(2, frame, 0.3)         # all four: h = 0.3, base 2h = 0.6
    phi = plate_tri_sdf(2, frame, 0.3, 0.4)    # all four: h = 0.3, base 0.4
    rows = sample_instance(np.random.default_rng(0), 2, frame, 0.3,
                           SamplingConfig(n_uniform=5000, n_band=5000))
    rows.shape                                 # (10000, 3): x, y, phi
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from datagen.plate_hole_sdf import (
    DEFAULT_PAD,
    PlateFrame,
    SamplingConfig,
    _check_dim,
    _rectangle_perimeter,
    box_sdf_2d,
    extrude,
)
from datagen.plate_tri_params import BASE_RATIO, SIDES, triangle_vertices

# ------------------------------------------------------------------- the frame


def square_frame(size, thickness=None, pad=DEFAULT_PAD) -> PlateFrame:
    """The :class:`~datagen.plate_hole_sdf.PlateFrame` of a square plate."""
    return PlateFrame(length=size, width=size, thickness=thickness, pad=pad)


def triangles_in_frame(frame: PlateFrame, heights, bases=None):
    """Vertices of the four triangles in the normalized frame, shape ``(4, 3, 2)``.

    Parameters
    ----------
    frame : PlateFrame
        Must be square: the triangle centres sit a quarter of the *side* in
        from each edge, which is only one length on a square.
    heights : float or sequence of 4 float
        Triangle height(s) in design units. One value is shared by all four;
        four values go in the order of :data:`datagen.plate_tri_params.SIDES`
        (top, bottom, left, right).
    bases : float, sequence of 4 float, or None
        Base width(s) in design units, broadcast like ``heights``. ``None``
        is the shared-height family, ``BASE_RATIO * heights``.
    """
    if not np.isclose(frame.length, frame.width):
        raise ValueError(
            f"the four-triangle plate is square; got {frame.length} x {frame.width}"
        )
    hs = np.broadcast_to(np.asarray(heights, dtype=float), (4,))
    if bases is None:
        bs = BASE_RATIO * hs
    else:
        bs = np.broadcast_to(np.asarray(bases, dtype=float), (4,))
    design = np.stack(
        [
            triangle_vertices(frame.length, h, side, b)
            for h, b, side in zip(hs, bs, SIDES)
        ]
    )
    return frame.to_normalized(design)


# ------------------------------------------------------------------- the field


def triangle_sdf_2d(points, vertices) -> np.ndarray:
    """Exact signed distance to one triangle, negative inside.

    Parameters
    ----------
    points : array_like, shape (n, 2)
    vertices : array_like, shape (3, 2)
        Either winding.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    p = np.asarray(points, dtype=float)
    v = np.asarray(vertices, dtype=float)
    # Orientation of the triangle, so the half-plane tests below read
    # "inner side" whichever way the corners are listed.
    e0, e2 = v[1] - v[0], v[0] - v[2]
    orientation = np.sign(e0[0] * e2[1] - e0[1] * e2[0])

    dist2 = np.full(len(p), np.inf)
    inner = np.full(len(p), np.inf)
    for i in range(3):
        edge = v[(i + 1) % 3] - v[i]
        q = p - v[i]
        # Nearest point of the segment, then its distance.
        t = np.clip((q @ edge) / (edge @ edge), 0.0, 1.0)
        pq = q - t[:, None] * edge
        dist2 = np.minimum(dist2, np.einsum("ij,ij->i", pq, pq))
        # Positive on the inner side of this edge's line.
        inner = np.minimum(inner, orientation * (q[:, 0] * edge[1] - q[:, 1] * edge[0]))
    return -np.sqrt(dist2) * np.sign(inner)


def plate_tri_sdf_2d(points, half_extent, triangles) -> np.ndarray:
    """Exact signed distance to a square with four triangular holes.

    Exact under the condition the design space enforces: the triangles lie
    strictly inside the square and apart from each other.

    Parameters
    ----------
    points : array_like, shape (n, 2)
    half_extent : float
        Half-side of the square.
    triangles : array_like, shape (k, 3, 2)
        The holes, any number of them.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Negative inside the material, positive outside the plate and in the
        holes.
    """
    points = np.asarray(points, dtype=float)
    plate = box_sdf_2d(points, half_extent, half_extent)
    holes = np.min([triangle_sdf_2d(points, v) for v in triangles], axis=0)
    return np.maximum(plate, -holes)


def plate_tri_sdf(dim, frame: PlateFrame, heights, bases=None) -> Callable:
    """The field of one plate as a function of normalized points.

    Parameters
    ----------
    dim : {2, 3}
    frame : PlateFrame
        Square, see :func:`triangles_in_frame`.
    heights, bases
        As in :func:`triangles_in_frame`, design units.

    Returns
    -------
    callable
        ``phi(points)`` with ``points`` of shape ``(n, dim)``, returning ``(n,)``.
    """
    _check_dim(dim)
    triangles = triangles_in_frame(frame, heights, bases)
    a = frame.half_extents[0]

    if dim == 2:
        return lambda p: plate_tri_sdf_2d(p, a, triangles)

    h = frame.half_thickness

    def phi(points):
        points = np.asarray(points, dtype=float)
        in_plane = plate_tri_sdf_2d(points[:, :2], a, triangles)
        return extrude(in_plane, points[:, 2], h)

    return phi


# ----------------------------------------------------------------- the samples


def _triangle_edges(rng, n, triangles):
    """Points spread by arc length over every edge of every triangle, with normals.

    The normals are perpendicular to the edge; their direction is
    irrelevant, the band offsets are symmetric.
    """
    tris = np.asarray(triangles, dtype=float)
    starts = tris.reshape(-1, 2)
    ends = np.roll(tris, -1, axis=1).reshape(-1, 2)
    vec = ends - starts
    lengths = np.linalg.norm(vec, axis=1)
    cum = np.cumsum(lengths)

    s = rng.uniform(0.0, cum[-1], n)
    edge = np.minimum(np.searchsorted(cum, s, side="right"), len(lengths) - 1)
    along = s - np.concatenate([[0.0], cum[:-1]])[edge]
    direction = vec / lengths[:, None]
    points = starts[edge] + along[:, None] * direction[edge]
    normals = np.column_stack([-direction[edge, 1], direction[edge, 0]])
    return points, normals


def _face_points(rng, n, a, triangles):
    """Points uniform on the plate footprint minus the holes (top/bottom faces)."""
    out = np.empty((0, 2))
    while len(out) < n:
        # Acceptance is 1 - 2 h w / S^2: 64 % at the largest default triangle
        # (h = 0.3, w = 0.6), so a pass or two. It always ends, since the
        # holes never cover the footprint.
        take = int(1.5 * (n - len(out))) + 16
        p = rng.uniform(-a, a, size=(take, 2))
        keep = np.min([triangle_sdf_2d(p, v) for v in triangles], axis=0) > 0.0
        out = np.vstack([out, p[keep]])
    return out[:n]


def _triangle_area(vertices):
    v = np.asarray(vertices, dtype=float)
    e1, e2 = v[1] - v[0], v[2] - v[0]
    return 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])


def band_points(
    rng, dim, frame: PlateFrame, heights, config: SamplingConfig, bases=None
):
    """Samples offset from the boundary along its normal.

    Parameters
    ----------
    rng : numpy.random.Generator
    dim : {2, 3}
    frame : PlateFrame
    heights, bases
        As in :func:`triangles_in_frame`, design units.
    config : SamplingConfig
        ``hole_fraction`` is the share of the band on the triangle edges.

    Returns
    -------
    numpy.ndarray, shape (config.n_band, dim)
    """
    _check_dim(dim)
    n = int(config.n_band)
    a = frame.half_extents[0]
    triangles = triangles_in_frame(frame, heights, bases)
    n_hole = int(round(config.hole_fraction * n))
    n_rest = n - n_hole

    if dim == 2:
        hole_p, hole_n = _triangle_edges(rng, n_hole, triangles)
        edge_p, edge_n = _rectangle_perimeter(rng, n_rest, a, a)
        points = np.vstack([hole_p, edge_p])
        normals = np.vstack([hole_n, edge_n])
    else:
        h = frame.half_thickness
        # Hole walls: twelve rectangles of height 2h.
        hole_xy, hole_n2 = _triangle_edges(rng, n_hole, triangles)
        hole_p = np.column_stack([hole_xy, rng.uniform(-h, h, n_hole)])
        hole_n = np.column_stack([hole_n2, np.zeros(n_hole)])

        # The rest by area between the two faces and the outer side walls.
        holes_area = sum(_triangle_area(v) for v in triangles)
        face_area = 2.0 * (4.0 * a * a - holes_area)
        side_area = 8.0 * a * 2.0 * h
        n_face = int(round(n_rest * face_area / (face_area + side_area)))
        n_side = n_rest - n_face

        face_xy = _face_points(rng, n_face, a, triangles)
        up = rng.choice([-1.0, 1.0], n_face)
        face_p = np.column_stack([face_xy, up * h])
        face_n = np.column_stack([np.zeros((n_face, 2)), up])

        side_xy, side_n2 = _rectangle_perimeter(rng, n_side, a, a)
        side_p = np.column_stack([side_xy, rng.uniform(-h, h, n_side)])
        side_n = np.column_stack([side_n2, np.zeros(n_side)])

        points = np.vstack([hole_p, face_p, side_p])
        normals = np.vstack([hole_n, face_n, side_n])

    sigma = np.asarray(config.stds)[rng.integers(len(config.stds), size=n)]
    return points + rng.normal(0.0, 1.0, n)[:, None] * sigma[:, None] * normals


def sample_instance(
    rng, dim, frame: PlateFrame, heights, config: SamplingConfig, bases=None
):
    """All training rows of one plate: uniform samples plus the boundary band.

    Parameters
    ----------
    rng : numpy.random.Generator
    dim : {2, 3}
    frame : PlateFrame
    heights, bases
        As in :func:`triangles_in_frame`, design units.
    config : SamplingConfig

    Returns
    -------
    numpy.ndarray, shape (n_uniform + n_band, dim + 1), float32
        Rows ``(x, y, phi)`` or ``(x, y, z, phi)`` in the normalized frame.
    """
    _check_dim(dim)
    uniform = rng.uniform(-config.bound, config.bound, size=(config.n_uniform, dim))
    band = band_points(rng, dim, frame, heights, config, bases)
    points = np.vstack([uniform, band])
    phi = plate_tri_sdf(dim, frame, heights, bases)(points)
    return np.column_stack([points, phi]).astype(np.float32)
