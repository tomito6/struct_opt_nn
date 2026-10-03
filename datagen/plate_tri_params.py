"""Design space of a square plate with four triangular holes -- parameters only.

The shape
---------
A square plate of side ``S`` with one isosceles triangle cut out of each
quarter. Each triangle has its base parallel to the nearest plate edge and its
tip pointing at the plate centre, so the material left is a frame plus an X.
The centre of a triangle (the midpoint of its height) sits a quarter of the
way in from its edge -- for the unit square at ``(0.5, 0.75)``, ``(0.5, 0.25)``,
``(0.25, 0.5)`` and ``(0.75, 0.5)`` -- and never moves. For the top triangle
of the unit square, with height ``h`` and base width ``w``::

    base   along  y = 0.75 + h/2,  from  x = 0.5 - w/2  to  x = 0.5 + w/2
    tip    at    (0.5, 0.75 - h/2)

and the other three are the same picture rotated by a quarter turn.

Two families
------------
* **Shared height** (:meth:`PlateTriSpace.sample`): one parameter, ``h``,
  the same for all four triangles, and the base tied to it,
  ``w = BASE_RATIO * h`` with the ratio 2. The triangle scales as a whole,
  the tip angle stays 90 degrees, and the slanted edges of neighbouring
  triangles stay parallel, so each arm of the X has a constant width. This
  is the shape of the reference drawing.
* **Height and width** (:meth:`PlateTriSpace.sample_hw`): two parameters,
  ``h`` and ``w``, varying independently (still the same for all four
  triangles). The tip angle now changes -- needles at small ``w``, flat
  slivers at large ``w`` -- and the arms of the X taper. The shared-height
  family is the line ``w = 2 h`` inside this one.

Why this exists
---------------
Same role as :mod:`datagen.plate_hole_params` for the plate with one circular
hole: decide which parameters are admissible, hand back a set of them, build
no geometry. :mod:`datagen.plate_tri_sdf` turns each row into signed-distance
training samples, :mod:`datagen.make_plate_tri` writes the dataset.

The admissibility condition
---------------------------
Four ligaments of material depend on ``(h, w)``; with ``m`` the smallest one
allowed:

* the **wall** between a base and the plate edge, ``S/4 - h/2``;
* the **tip gap** between a tip and the plate centre, ``S/4 - h/2`` -- the
  same number, because the triangle centre sits halfway between edge and
  plate centre, so what a growing triangle takes from one it takes from the
  other;
* the **arm** of the X between two neighbouring triangles. The top and the
  right triangle are mirror images across the plate diagonal, so their
  distance is twice the distance of the top triangle to that diagonal, which
  a convex polygon attains at a vertex: at the base corner nearest the
  diagonal or at the tip, ``sqrt(2) * min(S/4 + (h - w)/2, S/4 - h/2)``;
* the **side**, between a base corner and the side edges, ``(S - w)/2``.

For the shared-height family (``w = 2h``) the wall is the binding one and
a single interval describes everything::

    h_min <= h <= h_max = S/2 - 2 m            (0.3 for the unit square, m = 0.1)

where the arm is still ``sqrt(2) * m`` at ``h_max``. With the width free the
arm rule binds instead and it couples the two::

    w_min <= w <= w_max(h) = min(w_cap, h + S/2 - sqrt(2) m, S - 2 m)

``w_cap`` is a chosen upper limit (0.6, the base of the largest shared-height
triangle); the middle term is where the base corners of neighbouring
triangles come within ``m`` of each other -- 0.41 at ``h = 0.05``, so a base
of 0.6 is *not* admissible at every height, only from ``h = 0.24`` up. The
feasible set in ``(h, w)`` is therefore a trapezoid, not a box, and as for
the hole's radius the width is sampled through a normalized ``t_w``::

    w = w_min + t_w * (w_max(h) - w_min)

so the unit square ``(t_h, t_w)`` maps onto the whole feasible set, no draw
is wasted, and that square is the natural prescribed latent code
(:meth:`TriParameters.latent_codes`). The map has a kink where the cap takes
over from the arm rule; harmless for sampling, a softmin if gradients are
ever taken through it.

Why ``h_min`` and ``w_min`` are not zero
----------------------------------------
Same reasons as ``r_min`` for the hole: a vanishing triangle is a change of
topology, a small one is a high-frequency feature the decoder blurs, and a
hole spanning less than a few cells of the FlexiCubes grid meshes as a blob.
``h_min = 0.05`` was set by hand and is below the usual mesh rule at
``N = 32`` (it spans 1.6 cells; the hole family asked for 4); ``w_min = 0.1``
is the base of the smallest shared-height triangle. The dataset itself does
not care -- the field is exact -- but meshing the smallest shapes for FEM
later needs a finer grid, ``N >= 80``. :meth:`PlateTriSpace.cells_across_smallest`
reports the number.

Examples
--------
    uv run python -m datagen.plate_tri_params
    uv run python -m datagen.plate_tri_params --n 40 --plot
    uv run python -m datagen.plate_tri_params --free-width --n 128 --plot

    from datagen.plate_tri_params import PlateTriSpace

    space = PlateTriSpace(margin=0.1, h_min=0.05)   # h in [0.05, 0.3]
    params = space.sample(40)                        # an even sweep of h, w = 2h
    both = space.sample_hw(128)                      # Sobol over (h, w)
    for name, h, w in both.rows():
        ...  # build one geometry per row
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib

import numpy as np
from scipy.stats import qmc

# The four triangles, named after the plate edge their base is parallel to.
# This is also the order of everything indexed by triangle (centres, heights
# of a later per-triangle family, vertices).
SIDES = ("top", "bottom", "left", "right")

# Outward unit normal of the edge each triangle sits against; the tip points
# the other way. Same order as SIDES.
SIDE_NORMALS = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])

# base / height of the shared-height family. 2 is a 90 degree tip: the
# triangle of the reference drawing, and the value for which neighbouring
# triangles have parallel edges.
BASE_RATIO = 2.0

# Smallest ligament, absolute, as for the hole family: what matters
# structurally is the width of the material bridge.
DEFAULT_MARGIN = 0.1

# Smallest triangle height. Given, not derived -- see the module docstring.
DEFAULT_H_MIN = 0.05

# Smallest and largest base width of the free-width family. The bounds of
# the shared-height family's bases (2 * 0.05, 2 * 0.3), so that family lies
# inside this one.
DEFAULT_W_MIN = 0.1
DEFAULT_W_CAP = 0.6

# Mesh resolution the h_min hint is reported against, matching create_3D_mesh.
DEFAULT_MESH_N = 32

# Below this many cells across the height the smallest triangle meshes as a
# blob. The hole family's rule, kept for the report.
MIN_CELLS_ACROSS = 4

# Corners and centre of the (t_h, t_w) unit square: the smallest, the needle
# (tall and narrow), the sliver (low and wide), the drawing (h_max, w_cap),
# and the middle. Prepended to a free-width draw, as the hole family does.
_UNIT_EXTREMES_HW = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
)


def triangle_vertices(size, height, side, base=None):
    """Corners of one triangle in the plate's own frame, shape ``(3, 2)``.

    Rows are the two base corners and then the tip, counter-clockwise for
    the top triangle. Design units, origin at the lower-left plate corner.

    Parameters
    ----------
    size : float
        Side of the square plate.
    height : float
        Triangle height, base to tip.
    side : {"top", "bottom", "left", "right"}
        Which triangle.
    base : float or None
        Base width. ``None`` is ``BASE_RATIO * height``.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    height = float(height)
    base = BASE_RATIO * height if base is None else float(base)
    normal = SIDE_NORMALS[SIDES.index(side)]
    tangent = np.array([-normal[1], normal[0]])
    centre = 0.5 * size + 0.25 * size * normal
    base_mid = centre + 0.5 * height * normal
    tip = centre - 0.5 * height * normal
    return np.array(
        [base_mid - 0.5 * base * tangent, base_mid + 0.5 * base * tangent, tip]
    )


@dataclasses.dataclass(frozen=True)
class PlateTriSpace:
    """The admissible ``h`` (and ``w``) for a square plate with four triangles.

    All lengths are in the same unit and in the plate's own frame, whose
    origin is the lower-left corner. Nothing here knows about the ``[-1, 1]^d``
    training frame; :class:`datagen.plate_hole_sdf.PlateFrame` maps the plate
    onto it.

    Parameters
    ----------
    size : float
        Side of the square plate, ``[0, size] x [0, size]``.
    margin : float
        Smallest ligament allowed anywhere: between a triangle and the plate
        edge, between a tip and the plate centre, between two triangles. One
        value: the plate is square and the four triangles are alike, so
        per-edge margins would only shrink the shared range to the tightest
        of them.
    h_min : float
        Smallest triangle height. Strictly positive, see the module docstring.
    w_min, w_cap : float
        Smallest base width, and the chosen upper limit of the base width,
        for the free-width family. The shared-height family ignores both.

    Raises
    ------
    ValueError
        If the plate, the margin or the minima are not positive, or if the
        margin eats so much of the plate that no height above ``h_min`` (or
        no width above ``w_min``) is left.
    """

    size: float = 1.0
    margin: float = DEFAULT_MARGIN
    h_min: float = DEFAULT_H_MIN
    w_min: float = DEFAULT_W_MIN
    w_cap: float = DEFAULT_W_CAP

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("plate size must be positive")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.h_min <= 0 or self.w_min <= 0:
            raise ValueError(
                "h_min and w_min must be strictly positive, see the module docstring"
            )
        if self.h_max <= self.h_min:
            raise ValueError(
                f"no admissible triangle: the largest height that keeps a "
                f"{self.margin} ligament is {self.h_max:.4f}, which is not above "
                f"h_min={self.h_min}. Shrink the margin, shrink h_min, or enlarge "
                "the plate."
            )
        if self.w_cap <= self.w_min:
            raise ValueError(f"w_cap={self.w_cap} must be above w_min={self.w_min}")
        narrowest = float(self.max_width(self.h_min))
        if narrowest <= self.w_min:
            raise ValueError(
                f"no admissible width at h_min={self.h_min}: the widest base that "
                f"keeps a {self.margin} ligament there is {narrowest:.4f}, which is "
                f"not above w_min={self.w_min}"
            )

    # ------------------------------------------------------------------- bounds

    @property
    def h_max(self) -> float:
        """Largest admissible height: the wall ``size/4 - h/2`` equals the margin."""
        return 0.5 * self.size - 2.0 * self.margin

    @property
    def centres(self) -> np.ndarray:
        """Triangle centres, shape ``(4, 2)``, in the order of :data:`SIDES`."""
        return 0.5 * self.size + 0.25 * self.size * SIDE_NORMALS

    def max_width(self, h):
        """Widest admissible base at height ``h`` (free-width family).

        The smallest of the cap, the arm rule (base corners of neighbouring
        triangles ``margin`` apart) and the side rule (base corners
        ``margin`` from the side edges). Broadcasts over ``h``.
        """
        h = np.asarray(h, dtype=float)
        arm_rule = h + 0.5 * self.size - np.sqrt(2.0) * self.margin
        side_rule = self.size - 2.0 * self.margin
        return np.minimum(np.minimum(arm_rule, side_rule), self.w_cap)

    # ------------------------------------------------------------ the condition

    def clearances(self, h, w=None):
        """The four ligaments at ``(h, w)``, shape ``(4, ...)``.

        Rows are ``wall`` (base to plate edge), ``tip`` (tip to plate
        centre), ``arm`` (distance between two neighbouring triangles) and
        ``side`` (base corner to the side edges). The first two are equal by
        construction; the minimum over the four decides admissibility, and
        is what :meth:`TriParameters.min_clearance` reports for a whole set.

        Parameters
        ----------
        h : array_like
        w : array_like or None
            Base width; ``None`` is the shared-height family's ``2 h``.
        """
        h = np.asarray(h, dtype=float)
        w = BASE_RATIO * h if w is None else np.asarray(w, dtype=float)
        wall = 0.25 * self.size - 0.5 * h
        arm = np.sqrt(2.0) * np.minimum(0.25 * self.size + 0.5 * (h - w), wall)
        side = 0.5 * (self.size - w)
        return np.stack(np.broadcast_arrays(wall, wall, arm, side))

    def is_valid(self, h, w=None, tol=1e-9):
        """Boolean mask of the admissible rows.

        Parameters
        ----------
        h : array_like
        w : array_like or None
            Base width; ``None`` checks the height alone (shared-height
            family, whose ``w = 2h`` is admissible whenever ``h`` is).
        tol : float
            Slack, so a row produced by :meth:`from_unit` at ``t = 1`` is not
            rejected by its own round-off.
        """
        h = np.asarray(h, dtype=float)
        ok = (h >= self.h_min - tol) & (h <= self.h_max + tol)
        if w is None:
            return ok
        w = np.asarray(w, dtype=float)
        return ok & (w >= self.w_min - tol) & (w <= self.max_width(h) + tol)

    # ------------------------------------------------ the box reparametrization

    def height_from_t(self, t):
        """Map a normalized ``t`` in ``[0, 1]`` to a height in ``[h_min, h_max]``."""
        return self.h_min + np.asarray(t, dtype=float) * (self.h_max - self.h_min)

    def t_from_height(self, h):
        """Inverse of :meth:`height_from_t`."""
        return (np.asarray(h, dtype=float) - self.h_min) / (self.h_max - self.h_min)

    def from_unit(self, u):
        """Map points of the unit square to admissible ``(h, w)`` pairs.

        The point of the free-width family: the map is onto the feasible
        trapezoid and every input is valid, so no draw is wasted and the
        design space an optimizer sees is a plain box.

        Parameters
        ----------
        u : array_like, shape (n, 2)
            Columns are ``(t_h, t_w)``, each in ``[0, 1]``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(h, w)``, each of shape ``(n,)``.
        """
        u = np.atleast_2d(np.asarray(u, dtype=float))
        h = self.height_from_t(u[:, 0])
        w = self.w_min + u[:, 1] * (self.max_width(h) - self.w_min)
        return h, w

    def to_unit(self, h, w):
        """Inverse of :meth:`from_unit`, as an ``(n, 2)`` array."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        w = np.atleast_1d(np.asarray(w, dtype=float))
        t_w = (w - self.w_min) / (self.max_width(h) - self.w_min)
        return np.stack([self.t_from_height(h), t_w], axis=1)

    # ----------------------------------------------------------------- sampling

    def sample(self, n=40, method="grid", seed=0, include_extremes=True, t_power=1.0):
        """Draw ``n`` admissible heights of the shared-height family.

        One parameter, so the same choices as
        :meth:`datagen.plate_hole_params.PlateHoleSpace.sample_radius`: an
        even sweep by default, the drawn methods for when a scrambled set is
        wanted.

        Parameters
        ----------
        n : int
            Number of heights, at least 1. With ``method="grid"`` the two ends
            are part of the ``n``.
        method : {"grid", "sobol", "lhs", "random"}
            ``grid`` spaces the heights evenly between ``h_min`` and ``h_max``
            and needs no seed. The other three draw ``t`` in ``[0, 1]``.
        seed : int
            QMC/RNG seed; ignored by ``grid``.
        include_extremes : bool
            Make sure ``h_min`` and ``h_max`` are in the set. A grid holds
            them anyway; the drawn methods get them prepended, in that order.
        t_power : float
            Height bias ``t -> t ** (1 / t_power)``, drawn methods only.
            Values above 1 push the sample towards large triangles.

        Returns
        -------
        TriParameters
            With ``w`` unset: the bases are ``2 h``.
        """
        n = int(n)
        if n < 1:
            raise ValueError("n must be at least 1")

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

        # A drawn t can land on an end, and two instances with one name would
        # overwrite each other's .npz: de-duplicate on the height itself.
        h = self.height_from_t(t)
        _, keep = np.unique(np.round(h, 9), return_index=True)
        h = h[np.sort(keep)]
        return TriParameters(space=self, h=h, unit=self.t_from_height(h)[:, None])

    def sample_hw(self, n=128, method="sobol", seed=0, include_extremes=True):
        """Draw ``n`` admissible ``(h, w)`` pairs of the free-width family.

        Sampling happens in the unit square and is pushed through
        :meth:`from_unit`, so every returned pair is admissible by
        construction -- there is no rejection step. Same choices as
        :meth:`datagen.plate_hole_params.PlateHoleSpace.sample`.

        Parameters
        ----------
        n : int
            Number of pairs. For ``method="grid"`` it is rounded to the
            nearest perfect square; for ``method="sobol"`` a power of two
            keeps the sequence balanced.
        method : {"sobol", "lhs", "random", "grid"}
            ``sobol`` is the default: a low-discrepancy sequence covers a
            two-parameter box evenly at the hundred-odd shapes this family
            needs.
        seed : int
            Seed for the QMC engines; ignored by ``grid``.
        include_extremes : bool
            Prepend the corners and centre of the unit square
            (``_UNIT_EXTREMES_HW``): the configurations an optimizer drives
            towards, where the decoder would otherwise extrapolate.

        Returns
        -------
        TriParameters
            With ``w`` set.
        """
        n = int(n)
        if n < 1:
            raise ValueError("n must be at least 1")

        if method == "sobol":
            u = qmc.Sobol(d=2, scramble=True, seed=seed).random(n)
        elif method == "lhs":
            u = qmc.LatinHypercube(d=2, seed=seed).random(n)
        elif method == "random":
            u = np.random.default_rng(seed).random((n, 2))
        elif method == "grid":
            k = max(2, int(round(np.sqrt(n))))
            axis = np.linspace(0.0, 1.0, k)
            u = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
            u = u.reshape(-1, 2)
        else:
            raise ValueError(
                f"unknown method {method!r}, expected one of "
                "'sobol', 'lhs', 'random', 'grid'"
            )
        if include_extremes:
            u = np.vstack([_UNIT_EXTREMES_HW, u])

        h, w = self.from_unit(u)
        # Dedupe on the physical pair: a drawn point can land on an extreme,
        # and two instances with the same name would overwrite each other.
        _, keep = np.unique(
            np.round(np.stack([h, w], axis=1), 9), axis=0, return_index=True
        )
        keep = np.sort(keep)
        return TriParameters(space=self, h=h[keep], unit=u[keep], w=w[keep])

    # ------------------------------------------------------------------ reports

    def cells_across_smallest(self, mesh_n=DEFAULT_MESH_N):
        """How many mesh cells the smallest triangle dimension spans.

        ``create_3D_mesh`` lays ``mesh_n`` grid points across the domain, so
        the cell size is ``size / mesh_n``. The smallest dimension any
        triangle has is ``min(h_min, w_min)``.
        """
        return min(self.h_min, self.w_min) / (self.size / mesh_n)

    def summary(self, mesh_n=DEFAULT_MESH_N):
        """Multi-line human-readable report of the space. Returns a string."""
        cells = self.cells_across_smallest(mesh_n)
        verdict = "ok" if cells >= MIN_CELLS_ACROSS else "TOO COARSE"
        wall, _, arm, _ = self.clearances(self.h_max)
        return "\n".join(
            [
                f"plate             {self.size} x {self.size} (square)",
                f"triangles         4, centres a quarter in from each edge",
                f"margin            {self.margin}",
                f"height            [{self.h_min}, {self.h_max:.4f}]"
                f"   (wall size/4 - h/2 >= {self.margin})",
                f"shared-height     base = {BASE_RATIO:g} h; at h_max wall {wall:.4f}  "
                f"tip gap {wall:.4f}  arm {arm:.4f}",
                f"free-width        w in [{self.w_min}, w_max(h)],  w_max = "
                f"min({self.w_cap}, h + size/2 - sqrt(2) margin, size - 2 margin)"
                f" = {float(self.max_width(self.h_min)):.4f} at h_min, "
                f"{float(self.max_width(self.h_max)):.4f} at h_max",
                f"mesh check        smallest dimension spans {cells:.1f} cells at "
                f"N={mesh_n}  ({verdict}, want >= {MIN_CELLS_ACROSS})",
            ]
        )


@dataclasses.dataclass(frozen=True)
class TriParameters:
    """A concrete set of admissible rows, plus the names to store them under.

    Attributes
    ----------
    space : PlateTriSpace
        The space they were drawn from.
    h : numpy.ndarray, shape (n,)
        The heights, in the plate's own frame and units.
    unit : numpy.ndarray, shape (n, 1) or (n, 2)
        The same rows as ``t`` in the unit interval (shared-height family)
        or ``(t_h, t_w)`` in the unit square (free-width family). This, not
        ``(h, w)``, is the vector that should become the latent code -- see
        :meth:`latent_codes`.
    w : numpy.ndarray or None
        Base widths, shape ``(n,)``, for the free-width family; ``None`` for
        the shared-height family, whose bases are ``BASE_RATIO * h``.
    """

    space: PlateTriSpace
    h: np.ndarray
    unit: np.ndarray
    w: np.ndarray | None = None

    def __len__(self):
        return len(self.h)

    @property
    def tied(self) -> bool:
        """True for the shared-height family (base tied to the height)."""
        return self.w is None

    @property
    def base(self) -> np.ndarray:
        """Base width of every row, whichever family."""
        return BASE_RATIO * self.h if self.w is None else self.w

    @property
    def names(self):
        """One DeepSDF instance name per row, unique and sortable.

        Same convention as the hole family: the name has to survive a file
        system, so decimal points become ``p``. The free-width family adds
        the width, so the two families never collide on disk.
        """
        if self.tied:
            return [f"tri_h{h:.4f}".replace(".", "p") for h in self.h]
        return [
            f"tri_h{h:.4f}_w{w:.4f}".replace(".", "p") for h, w in zip(self.h, self.w)
        ]

    def latent_codes(self, lo=0.0, hi=1.0):
        """The prescribed latent code of each instance, ``(n, 1)`` or ``(n, 2)``.

        See :meth:`datagen.plate_hole_params.HoleParameters.latent_codes` for
        why one would prescribe rather than learn them.
        """
        return lo + self.unit * (hi - lo)

    def rows(self):
        """Iterate as ``(name, h, base)`` tuples, ready to drive a builder."""
        return zip(self.names, self.h, self.base)

    def varied(self, tol=1e-9):
        """Names of the parameters that actually differ across the set.

        ``["h"]`` for the shared-height family (its base varies too, but as
        a function of ``h``), ``["h", "w"]`` for a free-width draw, fewer if
        a column happens to be constant. Read off the values, so a manifest
        states what is in the files rather than what was asked for.
        """
        columns = [("h", self.h)] + ([] if self.tied else [("w", self.w)])
        return [
            name
            for name, values in columns
            if len(values) > 1 and float(np.ptp(values)) > tol
        ]

    def min_clearance(self):
        """Tightest ligament over the whole set -- the number to sanity-check."""
        return float(self.space.clearances(self.h, self.w).min())

    def all_valid(self):
        """True if every row satisfies the condition. Should never be False."""
        return bool(np.all(self.space.is_valid(self.h, self.w)))

    def save_csv(self, path):
        """Write the set to CSV, one row per instance. Returns the path.

        Columns ``name, h, t, base, wall, arm`` for the shared-height family;
        ``name, h, w, t_h, t_w, w_max, wall, arm, side`` for the free-width one.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wall, _, arm, side = self.space.clearances(self.h, self.w)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if self.tied:
                writer.writerow(["name", "h", "t", "base", "wall", "arm"])
                for i, name in enumerate(self.names):
                    writer.writerow(
                        [
                            name,
                            f"{self.h[i]:.6f}",
                            f"{self.unit[i, 0]:.6f}",
                            f"{self.base[i]:.6f}",
                            f"{wall[i]:.6f}",
                            f"{arm[i]:.6f}",
                        ]
                    )
            else:
                w_max = self.space.max_width(self.h)
                writer.writerow(
                    ["name", "h", "w", "t_h", "t_w", "w_max", "wall", "arm", "side"]
                )
                for i, name in enumerate(self.names):
                    writer.writerow(
                        [
                            name,
                            f"{self.h[i]:.6f}",
                            f"{self.w[i]:.6f}",
                            f"{self.unit[i, 0]:.6f}",
                            f"{self.unit[i, 1]:.6f}",
                            f"{w_max[i]:.6f}",
                            f"{wall[i]:.6f}",
                            f"{arm[i]:.6f}",
                            f"{side[i]:.6f}",
                        ]
                    )
        return path

    def summary(self):
        """Multi-line report of the set itself. Returns a string."""
        lines = [
            f"instances         {len(self)}",
            f"varied            {', '.join(self.varied()) or 'nothing'}",
            f"height            [{self.h.min():.4f}, {self.h.max():.4f}]"
            f"   mean {self.h.mean():.4f}",
        ]
        if not self.tied:
            lines.append(
                f"width             [{self.w.min():.4f}, {self.w.max():.4f}]"
                f"   mean {self.w.mean():.4f}"
            )
        lines += [
            f"tightest ligament {self.min_clearance():.4f}"
            f"   (margin {self.space.margin})",
            f"all admissible    {self.all_valid()}",
        ]
        return "\n".join(lines)


def plot_space(params, path):
    """Three diagnostic panels: the triangles, the feasible set, the spread.

    The plan view is the honest check -- every triangle drawn inside the
    dashed margin square, and none reaching the dotted centre square, means
    no shape breaks the ligament at the edge or the centre; the second panel
    is the ligament between triangles (shared-height family) or the feasible
    trapezoid in ``(h, w)`` with the draw on it (free-width family).

    Parameters
    ----------
    params : TriParameters
    path : str or pathlib.Path
        Output image.

    Returns
    -------
    pathlib.Path
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    space = params.space
    s, m = space.size, space.margin
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- plan view: the check that matters ----------------------------------
    ax = axs[0]
    ax.add_patch(Rectangle((0, 0), s, s, fc="0.92", ec="0.3", lw=1.5))
    ax.add_patch(
        Rectangle((m, m), s - 2 * m, s - 2 * m, fc="none", ec="tab:red", ls="--")
    )
    ax.add_patch(
        Rectangle(
            (0.5 * s - m, 0.5 * s - m), 2 * m, 2 * m, fc="none", ec="tab:red", ls=":"
        )
    )
    for h, base in zip(params.h, params.base):
        for side in SIDES:
            ax.add_patch(
                Polygon(
                    triangle_vertices(s, h, side, base),
                    closed=True,
                    fc="none",
                    ec="tab:blue",
                    lw=0.7,
                    alpha=0.6,
                )
            )
    ax.set_xlim(-0.05 * s, 1.05 * s)
    ax.set_ylim(-0.05 * s, 1.05 * s)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{len(params)} shapes (dashed: margin, dotted: centre gap)")

    hs = np.linspace(space.h_min, space.h_max, 200)
    ax = axs[1]
    if params.tied:
        # --- the ligaments as h grows ---------------------------------------
        wall, _, arm, _ = space.clearances(hs)
        ax.plot(hs, wall, label="wall = tip gap", color="tab:orange")
        ax.plot(hs, arm, label="arm of the X", color="tab:green")
        ax.axhline(m, color="tab:red", ls="--", lw=1, label="margin")
        ax.plot(params.h, np.zeros_like(params.h), "|", color="tab:blue", ms=12)
        ax.set_xlabel("h")
        ax.set_ylabel("ligament")
        ax.set_title("material left between the holes")
        ax.legend(fontsize=8)
    else:
        # --- the feasible trapezoid in (h, w) -------------------------------
        ax.fill_between(
            hs,
            space.w_min,
            space.max_width(hs),
            color="tab:orange",
            alpha=0.25,
            label="admissible (h, w)",
        )
        ax.plot(hs, BASE_RATIO * hs, "--", color="0.4", lw=1, label="w = 2 h")
        ax.scatter(params.h, params.w, s=12, color="tab:blue")
        ax.set_xlabel("h")
        ax.set_ylabel("w")
        ax.set_title("feasible set and the draw")
        ax.legend(loc="upper left", fontsize=8)

    # --- spread --------------------------------------------------------------
    ax = axs[2]
    if params.tied:
        ax.hist(params.h, bins=20, color="tab:blue", alpha=0.8)
        ax.axvline(space.h_min, color="0.4", ls=":", lw=1.2, label="h_min")
        ax.axvline(space.h_max, color="tab:red", ls="--", lw=1.2, label="h_max")
        ax.set_xlabel("h")
        ax.set_title("height distribution")
    else:
        ax.hist(params.w, bins=20, color="tab:blue", alpha=0.8)
        ax.axvline(space.w_min, color="0.4", ls=":", lw=1.2, label="w_min")
        ax.axvline(space.w_cap, color="tab:red", ls="--", lw=1.2, label="w_cap")
        ax.set_xlabel("w")
        ax.set_title("width distribution")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Define the admissible parameters of a square plate with four "
        "triangular holes, and draw a set of them. Builds no geometry.",
    )
    parser.add_argument("--size", type=float, default=1.0, help="plate side")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="smallest ligament anywhere: triangle to edge, tip to centre, "
        "triangle to triangle",
    )
    parser.add_argument("--h-min", type=float, default=DEFAULT_H_MIN)
    parser.add_argument(
        "--free-width",
        action="store_true",
        help="draw (h, w) pairs with the base width free, instead of w = 2 h",
    )
    parser.add_argument("--w-min", type=float, default=DEFAULT_W_MIN)
    parser.add_argument(
        "--w-cap", type=float, default=DEFAULT_W_CAP, help="upper limit of w"
    )
    parser.add_argument(
        "--n", type=int, default=None, help="instances (40; 128 with --free-width)"
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=["grid", "sobol", "lhs", "random"],
        help="grid (evenly spaced heights); sobol with --free-width",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--t-power",
        type=float,
        default=1.0,
        help="bias a drawn shared-height set towards large triangles",
    )
    parser.add_argument(
        "--no-extremes",
        action="store_true",
        help="do not prepend the ends of the range (corners with --free-width)",
    )
    parser.add_argument(
        "--mesh-n",
        type=int,
        default=DEFAULT_MESH_N,
        help="FlexiCubes resolution the size check is reported against",
    )
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("outputs"))
    args = parser.parse_args()

    space = PlateTriSpace(
        size=args.size,
        margin=args.margin,
        h_min=args.h_min,
        w_min=args.w_min,
        w_cap=args.w_cap,
    )
    print("=== space ===")
    print(space.summary(mesh_n=args.mesh_n))

    if args.free_width:
        n = args.n or 128
        method = args.method or "sobol"
        params = space.sample_hw(
            n=n, method=method, seed=args.seed, include_extremes=not args.no_extremes
        )
    else:
        n = args.n or 40
        method = args.method or "grid"
        params = space.sample(
            n=n,
            method=method,
            seed=args.seed,
            include_extremes=not args.no_extremes,
            t_power=args.t_power,
        )
    print("\n=== sample ===")
    print(f"method            {method} (seed {args.seed}, t_power {args.t_power})")
    print(params.summary())

    print("\nfirst 5 instances:")
    for name, h, base in list(params.rows())[:5]:
        print(f"  {name}   h={h:.4f}  base={base:.4f}")

    if args.csv is not None:
        print(f"\nwrote {params.save_csv(args.csv)}")
    if args.plot:
        print(f"wrote {plot_space(params, args.outdir / 'plate_tri_params.png')}")


if __name__ == "__main__":
    main()
