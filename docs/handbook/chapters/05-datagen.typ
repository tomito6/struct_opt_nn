#import "../template.typ": *

// ---------------------------------------------------------------- local drawing helpers
// Small hand-drawn sketches (plate, feasible sets). Coordinates are lengths from the
// top-left corner of the enclosing box.
#let _seg(p, q, stroke: 0.6pt + c-ink) = place(top + left, line(start: p, end: q, stroke: stroke))
#let _poly(fill: none, stroke: 0.6pt + c-ink, ..pts) = place(
  top + left, polygon(fill: fill, stroke: stroke, ..pts),
)
#let _disc(c, r, fill: none, stroke: 0.6pt + c-ink) = place(
  top + left, dx: c.at(0) - r, dy: c.at(1) - r, circle(radius: r, fill: fill, stroke: stroke),
)
#let _lab(p, body, dx: 0pt, dy: 0pt, size: 7.4pt, fill: c-ink) = place(
  top + left, dx: p.at(0) + dx, dy: p.at(1) + dy, text(size: size, fill: fill, body),
)
#let _dash(paint, w: 0.6pt) = (paint: paint, thickness: w, dash: "dashed")
#let _dot(paint, w: 0.6pt) = (paint: paint, thickness: w, dash: "dotted")

= Training data: the `datagen` package <ch:datagen>

Step 1 of the paper's offline stage samples points around the training geometries and
computes their signed distance; the paper does this for 3-D unit cells. The current
research in this repo uses 2-D plate families whose signed distance is known in closed
form, so the samples are computed exactly and no mesh is involved. `datagen/` is the
package that does this: it draws admissible shapes from a parametric family, evaluates
their exact SDF at uniform and near-boundary points, and writes a DeepSDF training set
into `data/`. Two families exist, a plate with one circular hole and a square plate with
four triangular holes, and seven datasets built from them are on disk. What the trainer
does with the files is @ch:training[Chapter]; why the generating parameters are stored
next to the samples is the research question of @ch:status[Chapter].

== Why a separate package

`datagen` imports nothing from `structsept`, `experiments` or `DeepSDFStruct`, and no
library or app code imports `datagen`. Only tests do: `tests/test_datagen.py`, and
`tests/test_app_explore2d.py`, which builds a small fixture dataset with it. The boundary
is a folder, as the package docstring (#f("datagen/__init__.py", 1)) puts it:

```
datagen/  --writes-->  data/  <--reads--  Train tab, experiments/train_*.py, DeepSDF trainer
```

This buys three things. The file layout in `datagen/dataset.py` is the whole interface,
so any generator that writes it is a valid data source; the GUI's SDF maker
(@ch:gui[Chapter]) writes the same layout from meshes, without manifest and parameter
table. `datagen` needs only NumPy, SciPy and Matplotlib, never torch, so generating data
does not depend on the training stack. And the shape parameters visibly never reach the
network: they leave `datagen` only through `params.csv` (and encoded in the instance
names, which the trainer uses only as file names), and only analysis code reads them.

#wip[The triangle family, `preview.py` and `--large-holes` are not committed (file list
  under Git state in @ch:repo[Chapter]); the datasets `plate_hole_2d_xyr_n166`,
  `plate_tri_2d_h` and `plate_tri_2d_hw` were built from that code. Commit before you
  build on it.]

== Parameters, field, glue

Every family is split into the same three layers (@fig:datagen-pipeline). A *params*
module (`plate_hole_params`, `plate_tri_params`) decides which parameter tuples are legal
and draws a set of them; it is pure arithmetic in design units and has its own CLI to
inspect the design space. An *sdf* module (`plate_hole_sdf`, `plate_tri_sdf`) computes
the exact field of one tuple in the decoder's $[-1, 1]^d$ frame and the sample points;
the triangle module imports the frame, extrusion and sampling recipe from the hole
module. A *make* script (`make_plate_hole`, `make_plate_tri`) is the CLI glue. Below them
sits `dataset.py`, the on-disk contract, which knows nothing about plates, and
`preview.py`, which draws the field of 6 shapes with the stored rows on top, read back
from disk as a check.

#figure(
  diagram(
    spacing: (7mm, 4mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nmod((0, 0), [`*_params` \ _which shapes_], inset: 5pt),
    ndat((1, 0), [`HoleParameters` \ `TriParameters`], inset: 5pt),
    nmod((2, 0), [`*_sdf` \ `sample_instance`], inset: 5pt),
    nsym((3, 0), [`write_instance` \ split by sign], inset: 5pt),
    ndat((4, 0), [`<ds>/plate/` \ `<name>.npz`], inset: 5pt),
    nsym((3, 1), [`write_split`], inset: 5pt),
    ndat((4, 1), [`splits/` \ `<ds>.json`], inset: 5pt),
    ndat((4, 2), [`params.csv`], inset: 5pt),
    nstep((0, 3), [`make_*.main` \ flags, `argv`], inset: 5pt),
    nsym((3, 3), [`write_manifest`], inset: 5pt),
    ndat((4, 3), [`dataset.json`], inset: 5pt),
    edge((0, 0), (1, 0), "-|>", lbl[draw]),
    edge((1, 0), (2, 0), "-|>", lbl[row $i$]),
    edge((2, 0), (3, 0), "-|>", lbl[rows]),
    edge((3, 0), (4, 0), "-|>"),
    edge((1, 0), (1, 1), (3, 1), "-|>", lbl[names, in order], label-pos: 0.7),
    edge((3, 1), (4, 1), "-|>"),
    edge((1, 0), (1, 2), (4, 2), "-|>", lbl[`save_csv`], label-pos: 0.75),
    edge((0, 3), (3, 3), "-|>", lbl[`make_manifest`: command, frame, sampling]),
    edge((3, 3), (4, 3), "-|>"),
    edge((0, 3), (0, 0), "--|>", lbl[space, sampler], label-side: right),
  ),
  placement: auto,
  caption: [How one dataset is written. The `make_*` script drives every step. The
    parameter set fixes the order of the shapes, and that one order is reused for the
    `.npz` loop (shape $i$ seeded with `[seed, i]`), the split and the parameter table.],
) <fig:datagen-pipeline>

The split pays off because admissibility is arithmetic on parameters: it can be checked,
plotted and unit-tested without computing a single distance. The field module never
decides whether a shape is legal; it assumes that it is, and that assumption is exactly
what makes its formula exact.

== The hole family

=== Which holes are legal

The plate is $[0, L] times [0, W]$ in design units, default $1 times 1$, origin at the
lower-left corner. A hole $(x_c, y_c, r)$ must keep a ligament of at least $m$ to every
edge; otherwise it is a notch, not a hole. The four conditions $x_c - r >= m$,
$L - x_c - r >= m$, $y_c - r >= m$, $W - y_c - r >= m$ collapse into one:

$ r_"min" <= r <= d(x_c, y_c) = min(x_c - m, L - x_c - m, y_c - m, W - y_c - m) $

The margin may also be four values (left, right, bottom, top), for when a strip of one
edge will carry a load or a support.

For a fixed $r$, the centre must lie in a rectangle that shrinks as $r$ grows. In
$(x_c, y_c, r)$ space the feasible set is therefore a *pyramid*, not a box
(@fig:datagen-hole). Drawing the three parameters from their bounding box and rejecting
the illegal ones would discard two thirds of the draws (`feasible_fraction()` gives
0.332, the volume of a pyramid over its box), and the survivors would be biased: large
holes only fit near the centre.

The fix is a map from the unit cube $(u_x, u_y, t) in [0, 1]^3$ onto the pyramid:

$ x_c = x_"lo" + u_x (x_"hi" - x_"lo"), quad
  y_c = y_"lo" + u_y (y_"hi" - y_"lo"), quad
  r = r_"min" + t (d(x_c, y_c) - r_"min") $

The centre box (`center_bounds`) is #box[$[x_"lo", x_"hi"] = [m + r_"min", L - m - r_"min"]$]
in $x$, and likewise in $y$. Every point of the cube is a legal hole, no draw is wasted,
and the inverse map is exact.
The unit coordinates are stored in `params.csv` next to $(x_c, y_c, r)$. They have a
second purpose: a box is what MMA wants as bounds on design variables, so the unit
coordinates are the natural _prescribed_ latent codes (`latent_codes()`). Nothing calls
that method yet; every training so far learns its codes (@ch:status[Chapter]).

$r_"min" = 0.07$ is strictly positive for three independent reasons: $r = 0$ is a change
of topology; a hole must span several FlexiCubes cells to mesh as a circle (the smallest
hole spans 4.48 cells at `N_base` $= 32$ cubes across the plate, the FlexiCubes
resolution of @ch:method[Chapter]; the rule asks for 4); and the decoder blurs small
features.

#let _P = 3.1cm          // plate side in panel (a)
#let _pa(x, y) = (0.15cm + x * _P, 0.15cm + (1 - y) * _P)
#let _Q = 4.6cm          // x_c axis in panel (b)
#let _pb(x, r) = (0.75cm + x * _Q, 0.25cm + (0.5 - r) * 6.4cm)
#figure(
  grid(
    columns: (auto, auto), column-gutter: 1.0cm, align: bottom,
    box(width: 3.5cm, height: 3.45cm, {
      place(top + left, dx: _pa(0, 1).at(0), dy: _pa(0, 1).at(1),
        rect(width: _P, height: _P, fill: luma(222), stroke: 0.6pt + c-ink))
      place(top + left, dx: _pa(0.05, 0.95).at(0), dy: _pa(0.05, 0.95).at(1),
        rect(width: 0.9 * _P, height: 0.9 * _P, stroke: _dash(rgb("#c2410c"))))
      _disc(_pa(0.3, 0.6), 0.25 * _P, stroke: _dash(c-accent))
      _disc(_pa(0.3, 0.6), 0.15 * _P, fill: white)
      _disc(_pa(0.3, 0.6), 0.8pt, fill: c-ink, stroke: none)
      _seg(_pa(0.3, 0.6), _pa(0.05, 0.6), stroke: 0.8pt + c-accent)
      _lab(_pa(0.075, 0.6), text(fill: c-accent)[$d$], dy: 1pt)
      _lab(_pa(0.6, 0.2), text(fill: rgb("#c2410c"))[margin $m$], size: 6.8pt)
    }),
    box(width: 6.0cm, height: 3.85cm, {
      _seg(_pb(0, 0), _pb(1.02, 0))
      _seg(_pb(0, 0), _pb(0, 0.52))
      place(top + left, dx: _pb(0.12, 0.45).at(0), dy: _pb(0.12, 0.45).at(1),
        rect(width: 0.76 * _Q, height: 0.38 * 6.4cm, stroke: _dot(c-muted)))
      _poly(fill: c-data.lighten(80%), stroke: 0.7pt + c-data,
        _pb(0.12, 0.07), _pb(0.88, 0.07), _pb(0.5, 0.45))
      _poly(stroke: _dash(c-ink), _pb(0.17, 0.07), _pb(0.83, 0.07), _pb(0.5, 0.40))
      _seg(_pb(0.3, 0.07), _pb(0.3, 0.25), stroke: 1.4pt + c-accent)
      _lab(_pb(0.3, 0.07), text(fill: c-accent)[$t = 0$], dx: 3pt, dy: -9pt, size: 6.8pt)
      _lab(_pb(0.3, 0.25), text(fill: c-accent)[$t = 1$], dx: 5pt, dy: 0pt, size: 6.8pt)
      _lab(_pb(0, 0.07), [$r_"min"$], dx: -18pt, dy: -5pt)
      _lab(_pb(0, 0.45), [$0.45$], dx: -17pt, dy: -5pt)
      _lab(_pb(0, 0.40), [$0.40$], dx: -17pt, dy: -4pt)
      _lab(_pb(0.12, 0), [$0.12$], dx: -8pt, dy: 3pt)
      _lab(_pb(0.88, 0), [$0.88$], dx: -8pt, dy: 3pt)
      _lab(_pb(1.02, 0), [$x_c$], dx: 2pt, dy: -5pt)
      _lab(_pb(0, 0.52), [$r$], dx: 3pt, dy: -3pt)
    }),
  ),
  caption: [The hole family. Left: one hole of the training family (margin
    $m = 0.05$) and its centre. The largest hole at that centre (dashed) touches the
    margin line; its radius is $d(x_c, y_c)$. Right: section of the feasible set at
    $y_c = 0.5$. Filled: training margin 0.05; dashed: design margin 0.1; dotted: the
    box rejection sampling would draw from. The blue segment is one column of the
    reparametrisation, $t$ running from $r_"min"$ to $d(x_c, y_c)$.],
) <fig:datagen-hole>

=== Three samplers

#[
  #set text(size: 9pt)
  #kv(
    columns: (auto, 1fr, auto),
    [*Sampler*], [*What it draws*], [*Flag, dataset*],
    [`sample(n=64,` \ `method="sobol")`],
    [All three parameters in the unit cube (Sobol, or `lhs`, `random`, `grid`), with 10
      extreme points prepended (`_UNIT_EXTREMES`: the cube corners, the smallest and the
      largest centred hole)],
    [default \ `xyr`, `n134`, `n25`],
    [`sample_radius(n=40,` \ `centre=None)`],
    [A fixed centre (the plate centre by default) and radii evenly spaced from
      $r_"min"$ to `max_radius(centre)`],
    [`--radius-only` \ `r_only`],
    [`sample_large_holes(` \ `n, r_from, seed)`],
    [$r$ uniform in $[r_"from", r_"max"]$, then the centre uniform in the section of the
      pyramid at that height],
    [`--large-holes` \ `xyr_n166`],
  )
]

*Why `--n 128` gives 134 shapes.* At the corners of the centre box the radius span is
zero, so the 8 cube corners give only 4 distinct holes, all of radius $r_"min"$. The
sampler removes duplicates on the physical triple (at 9 decimals), which leaves 6
extremes plus 128 Sobol points. The extremes come first in the order, so they hold latent
indices 0 to 5. With `--n 19` the same rule gives the 25 shapes of `plate_hole_2d_n25`.

*Why `sample_large_holes` exists.* A large hole needs a centre near the middle _and_ a $t$
near 1, a small corner of the cube. The 134-shape Sobol set has only 8 holes with
$r > 0.25$ and none between $r = 0.342$ and the extreme at 0.45. A decoder trained on it
has a stretch of latent space without any shape (the gap in the GUI's latent-coverage
view). The radius bias `t_power` cannot fix this, because it does not move the centres
inwards. `make_plate_hole` appends the large holes with `HoleParameters.extended()` and
seed + 1, so the first 134 shapes keep their index and, thanks to the per-shape seeding
below, their files.

=== Training margin and design margin

`PlateHoleSpace` defaults to $m = 0.1$, the margin an optimizer would respect, but
`make_plate_hole` trains with `TRAINING_MARGIN = 0.05`. The decoder thus sees a slightly
larger family than the optimizer may explore ($r_"max"$ 0.45 instead of 0.40, centre box
$[0.12, 0.88]$ instead of $[0.17, 0.83]$), so the MMA bounds never sit on the edge of the
training data. `PlateHoleSpace.contains()` checks this, and the parameter CLI prints the
verdict. The triangle family has no such buffer: it uses margin 0.1 for both.

```bash
uv run python -m datagen.plate_hole_params --n 128 --margin 0.05 --design-margin 0.1
```

#incode[`PlateHoleSpace` (#f("datagen/plate_hole_params.py", 141)) holds the condition;
  `max_radius` (:231) is $d$; `from_unit` (:308) and `to_unit` (:331) are the cube map;
  the samplers are `sample` (:347), `sample_radius` (:424) and `sample_large_holes`
  (:519). The result is a `HoleParameters` (:650) with `names`, `extended` (:711),
  `latent_codes` (:688) and `save_csv` (:764). `TRAINING_MARGIN` is
  #f("datagen/make_plate_hole.py", 85).]

== The exact signed distance

*The frame.* `PlateFrame` maps design units into the decoder's frame:

$ bold(x)_"norm" = s (bold(x)_"design" - bold(x)_"centre"), quad
  s = (2 (1 - "pad")) / max(L, W) $

With $"pad" = 0.1$ the scale is $s = 1.8$, and the unit plate occupies
$[-0.9, 0.9]^2$ inside the $[-1, 1]^2$ sampling box. The empty ring lets the decoder
learn the outer edge from both sides instead of at the border of everything it ever
sees. The scale is the same on every axis, so normalized distances are true distances;
divide by $s$ to get design units.

*The field.* With $phi_"box"$ the exact distance to the rectangle and $bold(c)$, $R$ the
hole in normalized units,

$ phi(bold(p)) = max(phi_"box" (bold(p)), -(norm(bold(p) - bold(c)) - R)) $

A boolean `max` is in general only a bound on the distance near the places where two
surfaces meet. Here they never meet, because admissibility keeps a ligament of at least
$m$: outside the plate the outer edge is the nearest boundary; in the material both terms
are negative and the `max` picks the nearer surface; inside the hole its wall is nearer
than the edge by at least the ligament. The formula is therefore the exact Euclidean
distance, *but only for admissible parameters*; the tests compare it with brute-force
distances to a densely sampled boundary. Sign convention, as everywhere in DeepSDF:
$phi < 0$ in the material, $phi >= 0$ outside the plate and in the hole.

*2-D or 3-D.* `--dim` has no default, on purpose. With `--dim 3` the 2-D field is
extruded exactly to the plate thickness (default 0.1) and the rows are $(x, y, z, phi)$.
But a through-hole plate of constant thickness carries all its information in the plane;
a 3-D decoder would spend most of its samples on two flat faces that are the same for
every shape. `--dim 2` (rows $(x, y, phi)$) loses nothing, and the 2-D field can still be
extruded later. Every dataset on disk is 2-D; the 3-D path is exercised only by the
tests. A 2-D dataset trains a decoder with `geom_dimension` 2, which the GUI shows in the
Explore 2-D tab (@ch:gui[Chapter]), not in the lattice pipeline.

== Where the samples go

`sample_instance` returns all rows of one shape: `n_uniform + n_band` points with their
exact $phi$, as `float32`.

- *Uniform points* in $[-1, 1]^d$, pad ring included. They teach the coarse field and
  the sign far from the surface.
- *Band points*: points on the boundary, moved along the normal by
  $cal(N)(0, 1) dot sigma$, with $sigma$ drawn per point from `stds = (0.05, 0.025)`,
  the two levels of DeepSDFStruct's mesh sampler. A share `hole_fraction = 0.5` of the
  band goes to the hole wall (spread by angle) whatever the hole's size, the rest by arc
  length to the outer outline: the hole is the only thing that changes between shapes.
  In 3-D the rest is split by area between the faces and the side walls.

The band carries half the budget because the trainer clamps target and prediction to
$plus.minus delta$, $delta = 0.1$ by default (@ch:training[Chapter]): only the band
$|phi| < 0.1$ is learned accurately, and a uniform point far from the plate contributes
little more than its sign. Since $phi$ is evaluated exactly at every point, a band point
pushed past a corner or across a thin ligament still carries the right label; band
points can land slightly outside the box ($|x|$ up to about 1.16 in the datasets on disk).

*Counts.* The defaults are $10^4 + 10^4$ points for 2-D and $5 dot 10^4 + 5 dot 10^4$
for 3-D, but every dataset on disk was built with 25 000 + 25 000, i.e. 50 000 rows per
shape. The trainer then draws `SamplesPerScene` rows per shape and step, half from `pos`
and half from `neg`, whatever the ratio in the file. The inside share of the files (40 to
69 %) therefore never reaches the network.

*Seeding.* Shape $i$ gets its own generator, `default_rng([seed, i])`, instead of sharing
one with the whole loop. A file thus depends only on the seed, its own index, its
parameters and the sampling settings. Appending shapes leaves the earlier files
byte-identical: the first 134 files of `plate_hole_2d_xyr_n166` are bit for bit those of
`plate_hole_2d_xyr`. Changing `--seed` or the counts changes every file; reordering the
shapes changes the file of every shape that moves.

#incode[In #f("datagen/plate_hole_sdf.py"): `PlateFrame` (:100), `box_sdf_2d` (:192),
  `plate_hole_sdf_2d` (:211), `extrude` (:236), `SamplingConfig` (:285), `band_points`
  (:377), `sample_instance` (:434), `DEFAULT_COUNTS` (:93). The per-shape seeding is in
  the loop of `make_plate_hole.main`, #f("datagen/make_plate_hole.py", 407). The trainer's
  half-and-half draw is `unpack_sdf_samples_from_ram` (the trainer always loads the split
  into RAM), #f("DeepSDFStruct/DeepSDFStruct/deep_sdf/data.py", 169).]

== The triangle family

The second family (all of it untracked, see the box at the top of this chapter) is a
square plate of side $S$ (default 1) with one isosceles triangle cut out of each quarter
(@fig:datagen-tri). The base is parallel to the nearest edge, the tip points at the plate
centre, and the centre of each triangle (the midpoint of its height) is fixed a quarter of
the side in from its edge. What remains is a frame plus an X. All four triangles share
height $h$ and base width $w$. Four ligaments depend on $(h, w)$ and must all be at least
$m = 0.1$:

- *wall*, base to plate edge: $S\/4 - h\/2$;
- *tip gap*, tip to plate centre: also $S\/4 - h\/2$, because the triangle centre sits
  halfway between edge and plate centre;
- *arm* of the X, between neighbouring triangles:
  $sqrt(2) min(S\/4 + (h - w)\/2, S\/4 - h\/2)$;
- *side*, base corner to the side edges: $(S - w)\/2$.

There are two families. In the *shared-height* family (default) the base is tied to the
height, $w = 2 h$: a 90° tip, so the slanted edges of neighbours are parallel and every
arm has constant width. The wall binds, so $h in [h_"min", S\/2 - 2 m] = [0.05, 0.3]$, and
`PlateTriSpace.sample` draws an even grid of 40 heights (names like `tri_h0p0500`). In
the *free-width* family (`--free-width`) $w in [w_"min", w_"max" (h)]$ with

$ w_"max" (h) = min(w_"cap", h + S\/2 - sqrt(2) m, S - 2 m),
  quad w_"min" = 0.1, quad w_"cap" = 0.6 $

The arm rule binds below $h approx 0.24$ and the cap above it, so the feasible set is a
rectangle with one corner cut off, not a box. As for the hole, it is sampled through a unit square $(t_h, t_w)$: `sample_hw`
draws 128 Sobol points and prepends the 4 corners and the centre of the square, 133
shapes (names like `tri_h0p0500_w0p1000`). The shared-height family is the diagonal
$w = 2 h$ of this set.

#let _T = 3.3cm
#let _pt(x, y) = (0.15cm + x * _T, 0.15cm + (1 - y) * _T)
#let _tri(h, w, side) = {
  // same construction as triangle_vertices() in datagen/plate_tri_params.py
  let n = (top: (0, 1), bottom: (0, -1), left: (-1, 0), right: (1, 0)).at(side)
  let t = (-n.at(1), n.at(0))
  let c = (0.5 + 0.25 * n.at(0), 0.5 + 0.25 * n.at(1))
  let b = (c.at(0) + 0.5 * h * n.at(0), c.at(1) + 0.5 * h * n.at(1))
  (
    _pt(b.at(0) - 0.5 * w * t.at(0), b.at(1) - 0.5 * w * t.at(1)),
    _pt(b.at(0) + 0.5 * w * t.at(0), b.at(1) + 0.5 * w * t.at(1)),
    _pt(c.at(0) - 0.5 * h * n.at(0), c.at(1) - 0.5 * h * n.at(1)),
  )
}
#let _ph(h, w) = (0.8cm + h * 13cm, 0.2cm + (0.7 - w) * 4.4cm)
#figure(
  grid(
    columns: (auto, auto), column-gutter: 1.0cm, align: bottom,
    box(width: 3.65cm, height: 3.6cm, {
      place(top + left, dx: _pt(0, 1).at(0), dy: _pt(0, 1).at(1),
        rect(width: _T, height: _T, fill: luma(222), stroke: 0.6pt + c-ink))
      for s in ("top", "bottom", "left", "right") {
        _poly(fill: white, stroke: 0.6pt + c-ink, .._tri(0.2, 0.4, s))
      }
      let k = 0.9pt + c-accent
      let lab(p, s, dx: 2pt, dy: -3.5pt) = _lab(p, text(fill: c-accent)[#s], dx: dx,
        dy: dy, size: 6.3pt)
      _seg(_pt(0.5, 0.85), _pt(0.5, 1.0), stroke: k)
      lab(_pt(0.5, 0.925), [wall])
      _seg(_pt(0.5, 0.65), _pt(0.5, 0.5), stroke: k)
      lab(_pt(0.5, 0.58), [tip])
      _seg(_pt(0.6, 0.75), _pt(0.75, 0.6), stroke: k)
      lab(_pt(0.72, 0.765), [arm], dx: 0pt)
      _seg(_pt(0.7, 0.85), _pt(1.0, 0.85), stroke: k)
      lab(_pt(0.8, 0.85), [side], dx: 0pt, dy: -8pt)
    }),
    box(width: 6.0cm, height: 3.4cm, {
      _seg(_ph(0, 0), _ph(0.36, 0))
      _seg(_ph(0, 0), _ph(0, 0.7))
      _poly(fill: c-concept.lighten(85%), stroke: 0.7pt + c-concept,
        _ph(0.05, 0.1), _ph(0.3, 0.1), _ph(0.3, 0.6), _ph(0.2414, 0.6),
        _ph(0.05, 0.4086))
      _seg(_ph(0.05, 0.1), _ph(0.3, 0.6), stroke: _dash(c-step, w: 1.1pt))
      _lab(_ph(0.2, 0.4), text(fill: c-step)[$w = 2 h$], dx: 4pt, dy: -2pt, size: 6.8pt)
      _lab(_ph(0.22, 0.6), [$w_"cap" = 0.6$], dx: -6pt, dy: -11pt, size: 6.8pt)
      _lab(_ph(0.06, 0.52), [arm rule], dx: -2pt, dy: -2pt, size: 6.8pt)
      _lab(_ph(0.1, 0.1), [$w_"min" = 0.1$], dx: 0pt, dy: 1pt, size: 6.8pt)
      _lab(_ph(0.05, 0), [$0.05$], dx: -8pt, dy: 2pt)
      _lab(_ph(0.3, 0), [$0.3$], dx: -6pt, dy: 2pt)
      _lab(_ph(0.36, 0), [$h$], dx: 2pt, dy: -5pt)
      _lab(_ph(0, 0.7), [$w$], dx: 3pt, dy: -2pt)
    }),
  ),
  caption: [The triangle family. Left: $h = 0.2$, $w = 2 h$, with the four ligaments.
    Right: the feasible set of the free-width family in $(h, w)$. The arm rule
    $w <= h + 0.5 - sqrt(2) dot 0.1$ cuts the corner; the shared-height family is the
    dashed diagonal.],
) <fig:datagen-tri>

The field is $max(phi_"box", -min_i phi_("tri", i))$, with `triangle_sdf_2d` the exact
distance to a convex polygon; it is exact for the same reason as the hole. The band's
`hole_fraction` is spread by arc length over the twelve triangle edges. The field
functions also accept four separate heights (one per triangle), but no sampler or CLI
flag produces them yet.

#incode[`triangle_vertices` #f("datagen/plate_tri_params.py", 160), `PlateTriSpace`
  (:192) with `max_width` (:268), `sample` (:365) and `sample_hw` (:424);
  `triangle_sdf_2d` #f("datagen/plate_tri_sdf.py", 115) and `plate_tri_sdf` (:175); the
  CLI is `make_plate_tri.main`, #f("datagen/make_plate_tri.py", 274).]

Unlike $r_"min"$, $h_"min" = 0.05$ was set by hand and spans only 1.6 FlexiCubes cells
at `N_base` $= 32$. The dataset does not care, since the field is exact, but meshing the
smallest triangles for FEM later needs #box[`N_base` $>= 80$].

== The on-disk contract

`datagen/dataset.py` defines the layout; the trainer fixes the name `SdfSamples/`, the
GUI's dataset list the name `splits/`:

```
data/
├── SdfSamples/<dataset>/
│   ├── plate/<instance>.npz   arrays pos (phi >= 0) and neg (phi < 0), float32,
│   │                          rows (x, y, [z,] phi) in the normalized frame
│   ├── params.csv             generating parameters, one row per instance
│   ├── dataset.json           manifest: exact command, geom_dimension, frame, sampling
│   └── preview.png, parameters.png      (with --plot)
└── splits/<dataset>.json      {"<dataset>": {"plate": [instance names, in order]}}
```

*The invariant: split order = latent index = `params.csv` row.* The trainer gives latent
code $i$ to the $i$-th name of the split, and row $i$ of `params.csv` describes the same
shape. The manifest repeats the rule in its `order` key. Analysis code still joins codes
to parameters _by name_ through the split (`split_names`, `code_vs_parameter`) instead of
trusting the row order.

#block(breakable: false)[
  #set text(size: 9pt)
  #kv(
    columns: (auto, auto, 1fr),
    [*File*], [*Written by*], [*Read by*],
    [`<instance>.npz`], [`write_instance`],
    [the DeepSDF loader (half `pos`, half `neg`); the SDF maker's audit
      `validate_dataset`; `preview.py`],
    [`splits/<ds>.json`], [`write_split`],
    [the trainer, through `TrainSplit` in `specs.json`; `split_names`; `list_datasets`
      (only whether it exists)],
    [`params.csv`], [`save_csv`],
    [`code_vs_parameter`; the scripts `train_plate_xyr_4h.py` and
      `train_plate_xyr_n166.py`; the tests],
    [`dataset.json`], [`write_manifest`],
    [only the key `geom_dimension`, by `structsept/app/datasets.py`; everything else is
      for humans],
  )
]

The manifest's most useful key is `command`, the exact CLI that built the dataset. `data/`
is gitignored, so this is how a dataset is reproduced on another machine. The `frame`
block (scale 1.8, pad 0.1) is informational: no code converts decoder coordinates back to
design units for you. Without a manifest (the SDF maker's datasets) the GUI falls back to
the row width of the first `.npz`. The columns of `params.csv` (one schema for holes, two
for triangles) and every manifest key are listed in @app:formats[Appendix].

`prepare()` refuses to write into an existing dataset (any `.npz` or the split file
present) unless `--overwrite` is given. Even then it deletes only the files `datagen` owns
in the `plate` class (`.npz`/`.stl`, manifest, parameter table, previews, split), and it
refuses if another class folder holds samples. How a run's `specs.json` points at the
split and at `data/` is in @ch:training[Chapter].

#incode[Writers in #f("datagen/dataset.py"): `prepare` (:81), `write_instance` (:138),
  `write_split` (:145), `write_manifest` (:153). Readers: `list_datasets`,
  `geom_dimension` and `validate_dataset` in #f("structsept/app/datasets.py") (:44, :82,
  :271); `get_instance_filenames` and `unpack_sdf_samples_from_ram` in
  #f("DeepSDFStruct/DeepSDFStruct/deep_sdf/data.py") (:57, :169); `split_names` and
  `code_vs_parameter` in #f("structsept/app/unattended.py") (:403, :525); `to_specs`,
  #f("structsept/app/hyperparams.py", 1118).]

== The datasets on disk

All seven datasets are 2-D, have the single class `plate`, 25 000 + 25 000 rows per
shape, pad 0.1 and scale 1.8; the hole sets use margin 0.05, the triangle sets 0.1. They
take 16 to 101 MB each. Each one is rebuilt by the common command of its family, plus the
flags in the last column, plus `--name <dataset>`; these are the flags that the `command`
key of its manifest records.

```bash
uv run python -m datagen.make_plate_hole --dim 2 --n-uniform 25000 --n-band 25000 --plot
uv run python -m datagen.make_plate_tri --dim 2 --n-uniform 25000 --n-band 25000 --plot
# add --dry-run to print the space, the instances and the output size, writing nothing
```

#[
  #set text(size: 9pt)
  #kv(
    columns: (auto, 1fr, auto, auto),
    [*Dataset*], [*Family and draw*], [*Shapes*], [*Extra flags*],
    [`plate_hole_2d_n25`], [hole, Sobol 19 + 6 extremes (first pilot set)], [25],
    [`--n 19`],
    [`plate_hole_2d_n134`], [hole, Sobol 128 + 6 extremes], [134], [`--n 128`],
    [`plate_hole_2d_xyr`], [the same draw; `.npz` files byte-identical to `n134`], [134],
    [none],
    [`plate_hole_2d_xyr_n166`], [`xyr` + 32 large holes, $r in [0.25, 0.45]$], [166],
    [`--large-holes 32` \ `--large-holes-from 0.25`],
    [`plate_hole_2d_r_only`], [centred hole, 40 radii from 0.07 to 0.45], [40],
    [`--radius-only`],
    [`plate_tri_2d_h`], [triangles, 40 heights from 0.05 to 0.3, $w = 2 h$], [40], [none],
    [`plate_tri_2d_hw`], [triangles, Sobol $(h, w)$ 128 + 5 extremes], [133],
    [`--free-width`],
  )
]

The training presets written by `experiments/make_plate_presets.py` expect `r_only`,
`xyr`, `tri_h` and `tri_hw`, and the unattended training scripts print the build command
of their dataset when it is missing (@ch:training[Chapter]). A third family means a new
`*_params`/`*_sdf`/`make_*` triple against the same contract; the recipe is in
@ch:workflows[Chapter].

== Gotchas

#watch[
  - *Without `--name` a hole dataset gets a default name* (`plate_hole_2d`,
    `plate_hole_2d_r`) that the presets do not know; they look for `plate_hole_2d_xyr` and
    `plate_hole_2d_r_only`. (The triangle defaults `plate_tri_2d_h` and `plate_tri_2d_hw`
    happen to match.) Without the two count flags you get $10^4 + 10^4$ rows and different
    files. To reproduce a dataset, copy the `command` from its `dataset.json`.
  - *`n134` and `xyr` are twins*: identical `.npz` files and parameter tables, and splits
    that differ only in their dataset key. `n134` and `n25` predate the manifest keys
    `parameters.varied`, `parameters.fixed_centre` and `parameter_sampling.family`; code
    that reads them must tolerate their absence.
  - *The margin default depends on the entry point*: 0.1 in `PlateHoleSpace` and in the
    `plate_hole_params` CLI, 0.05 in `make_plate_hole`. Inspect the training set with
    `--margin 0.05`.
  - *The parameter CLIs plot into `outputs/` under the current directory*, not under the
    repo root (`--outdir`, default `outputs`). The `make_*` scripts put their plots into
    the dataset folder.
  - *A leftover split file blocks a rebuild*: `prepare()` raises `FileExistsError`, so
    delete `splits/<ds>.json` too when you clean up by hand. The other way round,
    `get_instance_filenames` only logs a warning for a split entry without its `.npz`,
    and the trainer's RAM load right after it raises, before the first epoch.
  - *Flag combinations.* `--large-holes` needs `--large-holes-from` and excludes
    `--radius-only`; `--centre` needs `--radius-only`. A Sobol draw whose $n$ is not a power
    of two (`--n 19`) prints a harmless SciPy warning.
]
