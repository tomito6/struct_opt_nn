"""Design space of a plate with one circular hole -- parameters only, no geometry.

Why this exists
---------------
Training a DeepSDF decoder on a family of "plate with a hole" shapes means
generating one geometry per parameter triple ``(x_c, y_c, r)``. Not every triple
is a plate with a hole: once the circle reaches the plate border the hole stops
being a hole and becomes a notch, the ligament between the two collapses, and
the mesh either breaks outright or carries a stress concentration that has
nothing to do with the design question being asked.

This module decides which triples are admissible and hands back a set of them.
It builds no geometry, loads no network and imports nothing from
``DeepSDFStruct``: it is the step *before* :mod:`datagen.plate_hole_sdf`, which
turns each triple into signed-distance training samples.

The admissibility condition
---------------------------
Let the plate occupy ``[0, L] x [0, W]`` and let ``m`` be the smallest ligament
allowed between the edge of the hole and the edge of the plate. Four conditions,
two per axis -- note that the two far-edge ones are *upper* bounds, which is the
easiest sign error to make here::

    x_c - r >= m_left          L - x_c - r >= m_right
    y_c - r >= m_bottom        W - y_c - r >= m_top

They collapse into a single one. With ``d(x_c, y_c)`` the distance from the hole
centre to the nearest margin-shrunk plate edge::

    d(x_c, y_c) = min(x_c - m_left, L - x_c - m_right,
                      y_c - m_bottom, W - y_c - m_top)

    admissible  <=>  r_min <= r <= d(x_c, y_c)

``-d`` is the signed distance field of the shrunk plate evaluated at the hole
centre, which is why the condition generalizes for free: a non-rectangular plate
only changes ``d``, and a non-circular hole only means reading ``r`` as the
circumradius.

Why the feasible set is a pyramid, and why that matters
-------------------------------------------------------
Fix ``r``: the centre must lie in a rectangle that *shrinks* as ``r`` grows, and
vanishes at ``r = 0.5 * min(L - m_left - m_right, W - m_bottom - m_top)``. So in
``(x_c, y_c, r)`` space the feasible set is a pyramid, not a box. Two
consequences drive the design of this module:

1. The three parameters cannot be drawn independently from fixed intervals.
   Drawing them from the bounding box and rejecting the rest throws away roughly
   two thirds of the proposals (:meth:`PlateHoleSpace.feasible_fraction` reports
   the exact figure for the current settings) and biases what survives -- large
   holes only ever appear near the centre.
2. Box bounds are what MMA wants from a design variable. So instead of sampling
   ``r`` directly, sample a normalized ``t`` in ``[0, 1]`` and let

       r = r_min + t * (d(x_c, y_c) - r_min)

   Every ``(x_c, y_c, t)`` in the unit cube maps to an admissible triple, and
   the inverse map is exact. That unit cube is the natural home of the
   *prescribed latent code* (see :meth:`HoleParameters.latent_codes`).

Note that ``d`` involves a ``min``, so it has kinks on the diagonals of the
plate. Harmless for sampling, but it matters if gradients are later taken
through this map -- a softmin would be needed there.

Why ``r_min`` is not zero
-------------------------
Three independent reasons, any one of which is sufficient:

* ``r = 0`` is the hole disappearing, i.e. a change of topology inside the
  design space. The field stops being smooth in ``r`` exactly there.
* A hole must span several cells of the FlexiCubes grid or it meshes as a
  polygon. :meth:`PlateHoleSpace.cells_across_smallest_hole` turns that into a
  number.
* A small hole is a high-frequency feature and the decoder blurs it.

The one-parameter family
------------------------
:meth:`PlateHoleSpace.sample_radius` fixes the centre -- by default at the
middle of the plate -- and varies only ``r``. It is the same admissibility
condition with two of the three coordinates pinned, so the radius runs from
``r_min`` to ``max_radius(centre)`` and the unit-cube description is
``(u_x, u_y, t)`` with ``u_x`` and ``u_y`` constant. A decoder trained on it
faces one generating parameter; the three-parameter
:meth:`~PlateHoleSpace.sample` is the harder case the same decoder is then
compared against.

Examples
--------
    uv run python -m datagen.plate_hole_params
    uv run python -m datagen.plate_hole_params --n 128 --margin 0.05 --plot
    uv run python -m datagen.plate_hole_params --margin 0.05 --design-margin 0.1
    uv run python -m datagen.plate_hole_params --radius-only --n 40 --plot

    from datagen.plate_hole_params import PlateHoleSpace

    space = PlateHoleSpace(margin=0.1, r_min=0.07)
    params = space.sample(64, method="sobol", seed=0)
    for name, x_c, y_c, r in params.rows():
        ...  # build one geometry per row

    radii = space.sample_radius(40)  # centred hole, r only
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
from typing import Sequence

import numpy as np
from scipy.stats import qmc

EDGE_NAMES = ("left", "right", "bottom", "top")

# A 10 % ligament on a 1 m plate. Absolute, not a fraction of r: what matters
# structurally is the width of the material bridge, not its ratio to the hole.
DEFAULT_MARGIN = 0.1

# 4.5 FlexiCubes cells across the diameter at the N = 32 the meshing scripts
# use, i.e. the default configuration passes its own resolution check with a
# little room to spare. See PlateHoleSpace.cells_across_smallest_hole.
DEFAULT_R_MIN = 0.07

# Mesh resolution the r_min hint is reported against, matching create_3D_mesh.
DEFAULT_MESH_N = 32

# Below this many cells across its diameter the smallest hole meshes as a polygon.
MIN_CELLS_ACROSS = 4

_UNIT_CORNERS = np.array(
    [[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)]
)
# The corners plus the biggest and smallest centred hole: the configurations an
# optimizer drives towards, and where the decoder would otherwise extrapolate.
_UNIT_EXTREMES = np.vstack([_UNIT_CORNERS, [[0.5, 0.5, 1.0], [0.5, 0.5, 0.0]]])


@dataclasses.dataclass(frozen=True)
class PlateHoleSpace:
    """The admissible ``(x_c, y_c, r)`` triples for a plate with one hole.

    All lengths are in the same unit and in the plate's own frame, whose origin
    is the lower-left corner. Nothing here knows about the ``[-1, 1]^d`` training
    frame; :class:`datagen.plate_hole_sdf.PlateFrame` maps the plate onto it.

    Parameters
    ----------
    length, width : float
        Plate size along x and y. The plate is ``[0, length] x [0, width]``.
    margin : float or sequence of 4 float
        Smallest ligament allowed between the hole edge and the plate edge. A
        scalar applies to all four edges; a sequence is read in the order
        ``(left, right, bottom, top)``. Per-edge margins earn their keep when
        loads or supports are applied on a strip of one edge: a hole that is
        legal geometrically can still sit inside the clamped region.
    r_min : float
        Smallest hole radius. Strictly positive, see the module docstring.

    Raises
    ------
    ValueError
        If the plate or the margins are not positive, or if the margins eat so
        much of the plate that no admissible hole is left.
    """

    length: float = 1.0
    width: float = 1.0
    margin: float | Sequence[float] = DEFAULT_MARGIN
    r_min: float = DEFAULT_R_MIN

    def __post_init__(self):
        raw = np.asarray(self.margin, dtype=float).ravel()
        if raw.size not in (1, 4):
            raise ValueError(
                f"margin must be a scalar or 4 values {EDGE_NAMES}, got {raw.size}"
            )
        margins = np.broadcast_to(raw, (4,)) if raw.size == 1 else raw
        # frozen dataclass: the derived tuple has to go in through the back door.
        object.__setattr__(self, "_margins", tuple(float(v) for v in margins))

        if self.length <= 0 or self.width <= 0:
            raise ValueError("plate length and width must be positive")
        if any(v < 0 for v in self._margins):
            raise ValueError("margins must be non-negative")
        if self.r_min <= 0:
            raise ValueError(
                "r_min must be strictly positive, see the module docstring"
            )
        if self.r_max_global <= self.r_min:
            raise ValueError(
                f"no admissible hole: the largest radius that fits anywhere is "
                f"{self.r_max_global:.4f}, which is not above r_min={self.r_min}. "
                "Shrink the margins, shrink r_min, or enlarge the plate."
            )

    # ------------------------------------------------------------------- bounds

    @property
    def margins(self) -> tuple[float, float, float, float]:
        """Per-edge margins in the order ``(left, right, bottom, top)``."""
        return self._margins

    @property
    def r_max_global(self) -> float:
        """Largest radius admissible anywhere, reached at the Chebyshev centre.

        The hole is inscribed in the rectangle the margins leave behind, so this
        is half of that rectangle's shorter side.
        """
        m_l, m_r, m_b, m_t = self._margins
        return 0.5 * min(self.length - m_l - m_r, self.width - m_b - m_t)

    @property
    def center_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Box the hole centre may live in, as ``((x_lo, x_hi), (y_lo, y_hi))``.

        This is the widest the centre can roam, and it is only reachable by a
        hole of radius exactly ``r_min``: at those extremes the pyramid has zero
        height.
        """
        m_l, m_r, m_b, m_t = self._margins
        return (
            (m_l + self.r_min, self.length - m_r - self.r_min),
            (m_b + self.r_min, self.width - m_t - self.r_min),
        )

    # ------------------------------------------------------------ the condition

    def max_radius(self, x_c, y_c):
        """Largest admissible radius for a hole centred at ``(x_c, y_c)``.

        This is the ``d(x_c, y_c)`` of the module docstring. It comes out
        negative for centres outside the margin-shrunk plate, which is what lets
        :meth:`is_valid` reject those without a separate test.

        Parameters
        ----------
        x_c, y_c : array_like
            Hole centre. Broadcast against each other.

        Returns
        -------
        numpy.ndarray
            The broadcast shape of the inputs.
        """
        x_c = np.asarray(x_c, dtype=float)
        y_c = np.asarray(y_c, dtype=float)
        m_l, m_r, m_b, m_t = self._margins
        # Chained rather than np.minimum.reduce: reduce would first stack the
        # four terms into one array, which fails as soon as x_c and y_c are
        # broadcast against each other (a (n, 1) and a (1, m) grid).
        d = np.minimum(x_c - m_l, self.length - x_c - m_r)
        d = np.minimum(d, y_c - m_b)
        return np.minimum(d, self.width - y_c - m_t)

    def clearances(self, x_c, y_c, r):
        """Ligament actually left at each of the four edges, shape ``(4, ...)``.

        Ordered like :attr:`margins`. Every entry must be at least the
        corresponding margin for the triple to be admissible; the minimum over
        the four is the one that decides, and is what
        :meth:`HoleParameters.min_clearance` reports for a whole set.
        """
        # np.stack needs the four terms to agree in shape, so broadcast first.
        x_c, y_c, r = np.broadcast_arrays(
            np.asarray(x_c, dtype=float),
            np.asarray(y_c, dtype=float),
            np.asarray(r, dtype=float),
        )
        return np.stack([x_c - r, self.length - x_c - r, y_c - r, self.width - y_c - r])

    def is_valid(self, x_c, y_c, r, tol=1e-9):
        """Boolean mask of the admissible triples.

        Parameters
        ----------
        x_c, y_c, r : array_like
            Broadcast against each other.
        tol : float
            Slack, so a triple produced by :meth:`from_unit` at ``t = 1`` is not
            rejected by its own round-off.
        """
        r = np.asarray(r, dtype=float)
        return (r >= self.r_min - tol) & (r <= self.max_radius(x_c, y_c) + tol)

    # ------------------------------------------------ the box reparametrization

    def radius_from_t(self, x_c, y_c, t):
        """Map a normalized ``t`` in ``[0, 1]`` to a radius at a given centre.

        ``t = 0`` is always ``r_min``; ``t = 1`` is the largest hole that fits at
        that centre. The span collapses to zero at the corners of
        :attr:`center_bounds`, where only ``r_min`` is admissible -- that taper
        is the pyramid, not a bug.
        """
        span = np.maximum(self.max_radius(x_c, y_c) - self.r_min, 0.0)
        return self.r_min + np.asarray(t, dtype=float) * span

    def t_from_radius(self, x_c, y_c, r, eps=1e-12):
        """Inverse of :meth:`radius_from_t`; returns 0 where the span collapses."""
        span = np.maximum(self.max_radius(x_c, y_c) - self.r_min, 0.0)
        r = np.asarray(r, dtype=float)
        safe = np.where(span > eps, span, 1.0)
        return np.where(span > eps, (r - self.r_min) / safe, 0.0)

    def from_unit(self, u):
        """Map points of the unit cube to admissible ``(x_c, y_c, r)`` triples.

        This is the point of the module: the map is onto the pyramid and every
        input is valid, so no sample is ever wasted and the design space an
        optimizer sees is a plain box.

        Parameters
        ----------
        u : array_like, shape (n, 3)
            Columns are ``(u_x, u_y, t)``, each in ``[0, 1]``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(x_c, y_c, r)``, each of shape ``(n,)``.
        """
        u = np.atleast_2d(np.asarray(u, dtype=float))
        (x_lo, x_hi), (y_lo, y_hi) = self.center_bounds
        x_c = x_lo + u[:, 0] * (x_hi - x_lo)
        y_c = y_lo + u[:, 1] * (y_hi - y_lo)
        return x_c, y_c, self.radius_from_t(x_c, y_c, u[:, 2])

    def to_unit(self, x_c, y_c, r):
        """Inverse of :meth:`from_unit`, as an ``(n, 3)`` array."""
        (x_lo, x_hi), (y_lo, y_hi) = self.center_bounds
        x_c = np.atleast_1d(np.asarray(x_c, dtype=float))
        y_c = np.atleast_1d(np.asarray(y_c, dtype=float))
        return np.stack(
            [
                (x_c - x_lo) / (x_hi - x_lo),
                (y_c - y_lo) / (y_hi - y_lo),
                np.atleast_1d(self.t_from_radius(x_c, y_c, r)),
            ],
            axis=1,
        )

    # ----------------------------------------------------------------- sampling

    def sample(self, n=64, method="sobol", seed=0, include_extremes=True, t_power=1.0):
        """Draw ``n`` admissible parameter triples.

        Sampling happens in the unit cube and is pushed through
        :meth:`from_unit`, so every returned triple is admissible by
        construction -- there is no rejection step.

        Parameters
        ----------
        n : int
            Number of triples. For ``method="grid"`` it is rounded to the
            nearest perfect cube; for ``method="sobol"`` a power of two keeps
            the sequence balanced, and scipy warns when it is not one.
        method : {"sobol", "lhs", "random", "grid"}
            ``sobol`` is the default: a low-discrepancy sequence covers a
            three-parameter box far more evenly than random draws at the
            few-hundred sample sizes this family needs. ``lhs`` guarantees the
            1D marginals, ``random`` is the baseline to compare against, and
            ``grid`` is for a reproducible sweep.
        seed : int
            Seed for the QMC engines; ignored by ``grid``.
        include_extremes : bool
            Prepend the corners and centre of the design box. Worth keeping on,
            see ``_UNIT_EXTREMES``.
        t_power : float
            Radius bias, applied as ``t -> t ** (1 / t_power)``. Uniform ``t``
            is *not* uniform ``r``: the admissible span shrinks towards the edge
            of the centre box, so most centres can only host a small hole and
            the radius histogram piles up near ``r_min``. Values above 1 push
            the sample towards larger holes; 1 leaves it alone.

        Returns
        -------
        HoleParameters
        """
        n = int(n)
        if n < 1:
            raise ValueError("n must be at least 1")

        if method == "sobol":
            u = qmc.Sobol(d=3, scramble=True, seed=seed).random(n)
        elif method == "lhs":
            u = qmc.LatinHypercube(d=3, seed=seed).random(n)
        elif method == "random":
            u = np.random.default_rng(seed).random((n, 3))
        elif method == "grid":
            k = max(2, int(round(n ** (1.0 / 3.0))))
            axis = np.linspace(0.0, 1.0, k)
            u = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
            u = u.reshape(-1, 3)
        else:
            raise ValueError(
                f"unknown method {method!r}, expected one of "
                "'sobol', 'lhs', 'random', 'grid'"
            )

        if t_power != 1.0:
            u = u.copy()
            u[:, 2] = u[:, 2] ** (1.0 / t_power)

        if include_extremes:
            u = np.vstack([_UNIT_EXTREMES, u])

        x_c, y_c, r = self.from_unit(u)

        # Dedupe on the *physical* triple, not on u. At the corners of the
        # centre box the admissible span is zero, so every t collapses onto
        # r_min and the eight unit corners describe only four distinct holes.
        # Two instances with the same name would overwrite each other's .npz.
        _, keep = np.unique(
            np.round(np.stack([x_c, y_c, r], axis=1), 9), axis=0, return_index=True
        )
        keep = np.sort(keep)
        return HoleParameters(
            space=self, x_c=x_c[keep], y_c=y_c[keep], r=r[keep], unit=u[keep]
        )

    def sample_radius(
        self,
        n=40,
        centre=None,
        method="grid",
        seed=0,
        include_extremes=True,
        t_power=1.0,
    ):
        """Draw ``n`` holes at one fixed centre, varying only the radius.

        The one-parameter family: ``(x_c, y_c)`` is pinned and ``r`` runs
        from ``r_min`` to :meth:`max_radius` of that centre. The result is an
        ordinary :class:`HoleParameters` -- same names, same ``unit``
        description ``(u_x, u_y, t)`` with the first two columns now
        constant, same parameter table -- so a dataset built from it differs
        from a three-parameter one only in what its ``params.csv`` says
        varied.

        Parameters
        ----------
        n : int
            Number of radii, at least 1. With ``method="grid"`` the two ends
            are part of the ``n``.
        centre : (float, float) or None
            The hole centre in design units. ``None`` is the plate centre
            (the widest radius range when the margins are symmetric).
        method : {"grid", "sobol", "lhs", "random"}
            ``grid`` -- the default here, unlike :meth:`sample` -- spaces the
            radii evenly between the two ends: for one parameter a sweep is
            more useful than a scrambled sequence, and it needs no seed. The
            other three draw ``t`` in ``[0, 1]`` as :meth:`sample` does.
        seed : int
            QMC/RNG seed; ignored by ``grid``.
        include_extremes : bool
            Make sure ``r_min`` and the largest admissible radius are in the
            set. A grid holds them anyway; the drawn methods get them
            prepended, in that order, like the extremes of :meth:`sample`.
        t_power : float
            Radius bias ``t -> t ** (1 / t_power)`` as in :meth:`sample`,
            applied to the drawn methods only -- a grid stays a grid.

        Returns
        -------
        HoleParameters

        Raises
        ------
        ValueError
            If the centre cannot host any hole above ``r_min``: the span
            ``max_radius(centre) - r_min`` must be positive, or the family is
            a single shape.
        """
        n = int(n)
        if n < 1:
            raise ValueError("n must be at least 1")
        if centre is None:
            centre = (0.5 * self.length, 0.5 * self.width)
        x_c, y_c = (float(v) for v in centre)
        r_max = float(self.max_radius(x_c, y_c))
        if r_max - self.r_min <= 0.0:
            raise ValueError(
                f"no radius range at centre ({x_c}, {y_c}): the largest "
                f"admissible radius there is {r_max:.4f}, not above "
                f"r_min={self.r_min}. Move the centre inwards."
            )

        if method == "grid":
            t = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
        elif method == "sobol":
            t = qmc.Sobol(d=1, scramble=True, seed=seed).random(n)[:, 0]
        elif method == "lhs":
            t = qmc.LatinHypercube(d=1, seed=seed).random(n)[:, 0]
        elif method == "random":
            t = np.random.default_rng(seed).random(n)
        else:
            raise ValueError(
                f"unknown method {method!r}, expected one of "
                "'grid', 'sobol', 'lhs', 'random'"
            )
        if t_power != 1.0 and method != "grid":
            t = t ** (1.0 / t_power)
        if include_extremes and method != "grid":
            t = np.concatenate([[0.0, 1.0], t])

        # Same de-duplication as sample(): a drawn t can land on an end, and
        # two instances with one name would overwrite each other's .npz.
        r = self.radius_from_t(x_c, y_c, t)
        _, keep = np.unique(np.round(r, 9), return_index=True)
        r = r[np.sort(keep)]
        x_all, y_all = np.full(len(r), x_c), np.full(len(r), y_c)
        return HoleParameters(
            space=self, x_c=x_all, y_c=y_all, r=r, unit=self.to_unit(x_all, y_all, r)
        )

    # ------------------------------------------------------------------ reports

    def feasible_fraction(self, n_grid=512):
        """Share of the bounding box that is actually feasible, in ``[0, 1]``.

        This is the acceptance rate rejection sampling would have achieved, and
        the reason :meth:`from_unit` exists. Computed by quadrature rather than
        in closed form so it stays correct for per-edge margins.
        """
        (x_lo, x_hi), (y_lo, y_hi) = self.center_bounds
        xs = np.linspace(x_lo, x_hi, n_grid)
        ys = np.linspace(y_lo, y_hi, n_grid)
        span = np.maximum(self.max_radius(xs[:, None], ys[None, :]) - self.r_min, 0.0)
        return float(span.mean() / (self.r_max_global - self.r_min))

    def contains(self, other):
        """True if every triple admissible in ``other`` is admissible here.

        Meant for checking the training space against the design space: the
        dataset should be generated with *looser* margins than the optimizer
        will respect, so the bounds MMA pushes against still fall well inside
        what the decoder was trained on.
        """
        return (
            bool(np.isclose(self.length, other.length))
            and bool(np.isclose(self.width, other.width))
            and all(a <= b + 1e-12 for a, b in zip(self.margins, other.margins))
            and self.r_min <= other.r_min + 1e-12
        )

    def cells_across_smallest_hole(self, mesh_n=DEFAULT_MESH_N):
        """How many mesh cells the diameter of the smallest hole spans.

        ``create_3D_mesh`` lays ``mesh_n`` grid points across the domain, so the
        cell size is ``min(length, width) / mesh_n``. Below roughly
        ``MIN_CELLS_ACROSS`` the smallest hole comes out of FlexiCubes as a
        visible polygon and the training labels stop matching the mesh.
        """
        cell = min(self.length, self.width) / mesh_n
        return 2.0 * self.r_min / cell

    def summary(self, mesh_n=DEFAULT_MESH_N):
        """Multi-line human-readable report of the space. Returns a string."""
        m_l, m_r, m_b, m_t = self.margins
        (x_lo, x_hi), (y_lo, y_hi) = self.center_bounds
        cells = self.cells_across_smallest_hole(mesh_n)
        verdict = "ok" if cells >= MIN_CELLS_ACROSS else "TOO COARSE"
        frac = self.feasible_fraction()
        return "\n".join(
            [
                f"plate             {self.length} x {self.width}",
                f"margins           left {m_l}  right {m_r}  "
                f"bottom {m_b}  top {m_t}",
                f"conditions        x_c - r >= {m_l}"
                f"     {self.length} - x_c - r >= {m_r}",
                f"                  y_c - r >= {m_b}"
                f"     {self.width} - y_c - r >= {m_t}",
                f"radius            [{self.r_min}, {self.r_max_global:.4f}]"
                "   (upper bound reached at the centre only)",
                f"centre box        x in [{x_lo:.4f}, {x_hi:.4f}]"
                f"   y in [{y_lo:.4f}, {y_hi:.4f}]",
                f"feasible share    {100 * frac:.1f} % of that bounding box"
                f"   -> rejection sampling would discard {100 * (1 - frac):.1f} %",
                f"mesh check        smallest hole spans {cells:.1f} cells at "
                f"N={mesh_n}  ({verdict}, want >= {MIN_CELLS_ACROSS})",
            ]
        )


@dataclasses.dataclass(frozen=True)
class HoleParameters:
    """A concrete set of admissible triples, plus the names to store them under.

    Attributes
    ----------
    space : PlateHoleSpace
        The space they were drawn from.
    x_c, y_c, r : numpy.ndarray, shape (n,)
        The parameters, in the plate's own frame and units.
    unit : numpy.ndarray, shape (n, 3)
        The same points as ``(u_x, u_y, t)`` in the unit cube. This, not
        ``(x_c, y_c, r)``, is the vector that should become the latent code --
        see :meth:`latent_codes`.
    """

    space: PlateHoleSpace
    x_c: np.ndarray
    y_c: np.ndarray
    r: np.ndarray
    unit: np.ndarray

    def __len__(self):
        return len(self.r)

    @property
    def names(self):
        """One DeepSDF instance name per triple, unique and sortable.

        :func:`datagen.dataset.write_instance` writes ``<instance>.npz`` and
        :func:`datagen.dataset.write_split` lists the same string in the split
        json, so the name has to survive a file system: decimal points become
        ``p``.
        """
        return [
            f"hole_x{x:.4f}_y{y:.4f}_r{rad:.4f}".replace(".", "p")
            for x, y, rad in zip(self.x_c, self.y_c, self.r)
        ]

    def latent_codes(self, lo=0.0, hi=1.0):
        """The prescribed latent code of each instance, shape ``(n, 3)``.

        The argument for prescribing rather than learning: in
        ``deep_sdf/training.py`` the latent codes are an ``nn.Embedding``
        initialized with noise and handed to the optimizer, so a normally
        trained decoder comes back with an *arbitrary* latent basis. The
        admissibility condition cannot be written in that basis, which would
        leave MMA with a non-linear constraint instead of box bounds. Freezing
        the codes to :attr:`unit` keeps the design space a box all the way
        through, at the cost of a harder training run.

        Parameters
        ----------
        lo, hi : float
            Range the unit cube is mapped onto. The default leaves it alone.
        """
        return lo + self.unit * (hi - lo)

    def rows(self):
        """Iterate as ``(name, x_c, y_c, r)`` tuples, ready to drive a builder."""
        return zip(self.names, self.x_c, self.y_c, self.r)

    def varied(self, tol=1e-9):
        """Names of the parameters that actually differ across the set.

        ``["x_c", "y_c", "r"]`` for a draw of :meth:`PlateHoleSpace.sample`,
        ``["r"]`` for one of :meth:`PlateHoleSpace.sample_radius`. Read off
        the values, not off which method produced them, so a manifest states
        what is in the files rather than what was asked for.
        """
        return [
            name
            for name, values in (("x_c", self.x_c), ("y_c", self.y_c), ("r", self.r))
            if len(values) > 1 and float(np.ptp(values)) > tol
        ]

    def min_clearance(self):
        """Tightest ligament over the whole set -- the number to sanity-check."""
        return float(self.space.clearances(self.x_c, self.y_c, self.r).min())

    def all_valid(self):
        """True if every triple satisfies the condition. Should never be False."""
        return bool(np.all(self.space.is_valid(self.x_c, self.y_c, self.r)))

    def save_csv(self, path):
        """Write the set to CSV, one row per instance. Returns the path."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clear = self.space.clearances(self.x_c, self.y_c, self.r).min(axis=0)
        r_max = self.space.max_radius(self.x_c, self.y_c)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["name", "x_c", "y_c", "r", "u_x", "u_y", "t", "r_max", "clearance"]
            )
            for i, name in enumerate(self.names):
                writer.writerow(
                    [
                        name,
                        f"{self.x_c[i]:.6f}",
                        f"{self.y_c[i]:.6f}",
                        f"{self.r[i]:.6f}",
                        f"{self.unit[i, 0]:.6f}",
                        f"{self.unit[i, 1]:.6f}",
                        f"{self.unit[i, 2]:.6f}",
                        f"{r_max[i]:.6f}",
                        f"{clear[i]:.6f}",
                    ]
                )
        return path

    def summary(self):
        """Multi-line report of the set itself. Returns a string."""
        return "\n".join(
            [
                f"instances         {len(self)}",
                f"varied            {', '.join(self.varied()) or 'nothing'}",
                f"radius            [{self.r.min():.4f}, {self.r.max():.4f}]"
                f"   mean {self.r.mean():.4f}",
                f"centre x          [{self.x_c.min():.4f}, {self.x_c.max():.4f}]",
                f"centre y          [{self.y_c.min():.4f}, {self.y_c.max():.4f}]",
                f"tightest ligament {self.min_clearance():.4f}"
                f"   (margin {min(self.space.margins)})",
                f"all admissible    {self.all_valid()}",
            ]
        )


def plot_space(params, path):
    """Three diagnostic panels: the holes, the pyramid, the radius spread.

    The plan view is the honest check -- every circle drawn inside the dashed
    margin rectangle means no hole escapes the plate.

    Parameters
    ----------
    params : HoleParameters
    path : str or pathlib.Path
        Output image.

    Returns
    -------
    pathlib.Path
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    space = params.space
    m_l, m_r, m_b, m_t = space.margins
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- plan view: the check that matters ----------------------------------
    ax = axs[0]
    ax.add_patch(
        Rectangle((0, 0), space.length, space.width, fc="0.92", ec="0.3", lw=1.5)
    )
    ax.add_patch(
        Rectangle(
            (m_l, m_b),
            space.length - m_l - m_r,
            space.width - m_b - m_t,
            fc="none",
            ec="tab:red",
            ls="--",
            lw=1.2,
        )
    )
    for x, y, rad in zip(params.x_c, params.y_c, params.r):
        ax.add_patch(Circle((x, y), rad, fc="none", ec="tab:blue", lw=0.7, alpha=0.6))
    ax.set_xlim(-0.05 * space.length, 1.05 * space.length)
    ax.set_ylim(-0.05 * space.width, 1.05 * space.width)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{len(params)} holes (dashed: margin)")

    # --- the pyramid, seen edge on ------------------------------------------
    ax = axs[1]
    (x_lo, x_hi), _ = space.center_bounds
    xs = np.linspace(x_lo, x_hi, 400)
    envelope = space.max_radius(xs, 0.5 * space.width)
    ax.fill_between(
        xs,
        space.r_min,
        np.maximum(envelope, space.r_min),
        color="tab:orange",
        alpha=0.25,
        label="feasible at y_c = W/2",
    )
    scatter = ax.scatter(params.x_c, params.r, s=12, c=params.y_c, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="y_c")
    ax.axhline(space.r_min, color="0.4", ls=":", lw=1)
    ax.set_xlabel("x_c")
    ax.set_ylabel("r")
    ax.set_title("section through the feasible pyramid")
    ax.legend(loc="upper right", fontsize=8)

    # --- radius spread -------------------------------------------------------
    ax = axs[2]
    ax.hist(params.r, bins=20, color="tab:blue", alpha=0.8)
    ax.axvline(space.r_min, color="0.4", ls=":", lw=1.2, label="r_min")
    ax.axvline(space.r_max_global, color="tab:red", ls="--", lw=1.2, label="r_max")
    ax.set_xlabel("r")
    ax.set_ylabel("count")
    ax.set_title("radius distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Define the admissible (x_c, y_c, r) parameters of a plate "
        "with one circular hole, and draw a set of them. Builds no geometry.",
    )
    parser.add_argument("--length", type=float, default=1.0, help="plate x size")
    parser.add_argument("--width", type=float, default=1.0, help="plate y size")
    parser.add_argument(
        "--margin",
        type=float,
        nargs="+",
        default=[DEFAULT_MARGIN],
        help="ligament: one value, or four in the order left right bottom top",
    )
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN)
    parser.add_argument("--n", type=int, default=64, help="number of instances")
    parser.add_argument(
        "--method",
        default=None,
        choices=["sobol", "lhs", "random", "grid"],
        help="sobol; with --radius-only the default is grid (evenly spaced radii)",
    )
    parser.add_argument(
        "--radius-only",
        action="store_true",
        help="fix the hole centre (at --centre) and vary only the radius",
    )
    parser.add_argument(
        "--centre",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="fixed hole centre for --radius-only; default: the plate centre",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--t-power",
        type=float,
        default=1.0,
        help="bias the radius towards larger holes; 1 leaves the sample alone",
    )
    parser.add_argument(
        "--no-extremes",
        action="store_true",
        help="do not prepend the corners and centre of the design box",
    )
    parser.add_argument(
        "--mesh-n",
        type=int,
        default=DEFAULT_MESH_N,
        help="FlexiCubes resolution the r_min check is reported against",
    )
    parser.add_argument(
        "--design-margin",
        type=float,
        default=None,
        help="margin the optimizer will respect; checks this space contains it",
    )
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("outputs"))
    args = parser.parse_args()

    space = PlateHoleSpace(
        length=args.length,
        width=args.width,
        margin=args.margin if len(args.margin) > 1 else args.margin[0],
        r_min=args.r_min,
    )
    print("=== space ===")
    print(space.summary(mesh_n=args.mesh_n))

    if args.design_margin is not None:
        design = PlateHoleSpace(
            length=args.length,
            width=args.width,
            margin=args.design_margin,
            r_min=args.r_min,
        )
        if space.contains(design):
            verdict = "contained -- the decoder has a buffer at the bounds"
        else:
            verdict = "NOT contained -- MMA could leave the trained region"
        print(
            f"design space      margin {args.design_margin}, "
            f"r_max {design.r_max_global:.4f}  ->  {verdict}"
        )

    if args.radius_only:
        method = args.method or "grid"
        params = space.sample_radius(
            n=args.n,
            centre=args.centre,
            method=method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
            t_power=args.t_power,
        )
    else:
        if args.centre is not None:
            parser.error("--centre only makes sense together with --radius-only")
        method = args.method or "sobol"
        params = space.sample(
            n=args.n,
            method=method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
            t_power=args.t_power,
        )
    print("\n=== sample ===")
    print(f"method            {method} (seed {args.seed}, t_power {args.t_power})")
    print(params.summary())

    print("\nfirst 5 instances:")
    for name, x_c, y_c, r in list(params.rows())[:5]:
        print(f"  {name}   x_c={x_c:.4f}  y_c={y_c:.4f}  r={r:.4f}")

    if args.csv is not None:
        print(f"\nwrote {params.save_csv(args.csv)}")
    if args.plot:
        print(f"wrote {plot_space(params, args.outdir / 'plate_hole_params.png')}")


if __name__ == "__main__":
    main()
