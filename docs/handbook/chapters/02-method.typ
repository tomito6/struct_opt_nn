#import "../template.typ": *

= The method, concept by concept <ch:method>

The paper turns a vector of design variables into a compliance value through one long
chain of operations, and every link of the chain is differentiable. This chapter takes
the links one at a time, in the order the code runs them: the idea, the little math you
need, why it is done this way, and an _In the code_ box with the class and the file.
The API around these classes is in @ch:library[Chapter], this project's FEM code in
@ch:fem[Chapter], training in @ch:training[Chapter], paper symbols next to code names
in @app:glossary[Appendix]. Keep three coordinate spaces apart while you read: the
decoder's cube $[-1, 1]^3$ holding _one_ unit cell, the parametric cube $[0, 1]^3$
holding the _whole_ lattice, and the physical domain $overline(Omega)$ in metres.

== Signed distance functions

A solid $Omega$ with surface $Gamma = partial Omega$ is described by its signed
distance function (SDF)
$ phi(bold(x)) = cases(
  -min_(bold(y) in Gamma) norm(bold(x) - bold(y)) & "if" bold(x) in Omega,
  +min_(bold(y) in Gamma) norm(bold(x) - bold(y)) & "otherwise.",
) $
*Negative means inside, i.e. material; positive means void; zero is the surface.* An
exact SDF has $norm(nabla phi) = 1$ almost everywhere. Unlike a mesh, a function can be
evaluated anywhere, combined with other functions and differentiated: if $phi$ depends
on parameters, its zero level set moves continuously with them. Shapes combine with
`min` and `max`:
$ phi_(A union B) = min(phi_A, phi_B), quad
  phi_(A inter B) = max(phi_A, phi_B), quad
  phi_(A without B) = max(phi_A, -phi_B). $
The sign is right everywhere and the zero level set is exact, but away from the surface
$abs(phi)$ is in general only a _lower bound_ of the true distance (inside the overlap of
two shapes, or near a concave corner, the true distance is larger). The mesher needs
only the sign and a roughly right magnitude near the surface, so this is enough. `min`
and `max` are differentiable almost everywhere; the gradient goes to the active
argument.

#incode[Every SDF is a `torch.nn.Module` subclassing `SDFBase`
  (#f("DeepSDFStruct/DeepSDFStruct/SDF.py", 194)): `sdf(points)` maps `(N, 3)` points
  to `(N, 1)` distances, and primitive sizes are `nn.Parameter`s, so gradients reach
  them. `UnionSDF` (line 575) is the `min` above, `DifferenceSDF` (line 660) the `max`
  with the negated subtrahend. API and traps: @ch:library[Chapter].]

== DeepSDF: the unit cell as an auto-decoder

A DeepSDF decoder is a multilayer perceptron $f_theta (lambda, bold(x))$: it takes a
latent code $lambda in RR^d$ concatenated with a point $bold(x) in [-1, 1]^3$ and
returns the signed distance to the shape that $lambda$ encodes. Trained once on a
family of shapes, it lets $lambda$ move continuously through the family. That is the
design space of the method: low-dimensional, continuous, and limited to shapes like the
training shapes, whereas density-based topology optimization can produce anything.

The decoder is an _auto-decoder_ (Park et al., 2019): *there is no encoder*. Each
training shape $j$ gets its own code $lambda_j$, a free parameter in a table, optimized
together with the weights:
$ min_(theta, {lambda_j}) sum_j sum_i abs(c_delta (f_theta (lambda_j, bold(x)_(i j)))
  - c_delta (phi_(i j))) + lambda_"reg" norm(lambda_j), quad
  c_delta (s) = max(-delta, min(delta, s)). $
Clamping to $plus.minus delta$ spends the network's capacity near the surface, where the
zero level set is decided. The norm penalty, weighted by $lambda_"reg"$
(`CodeRegularizationLambda` in the code; the paper calls it $sigma$), keeps the codes
compact around the origin. After training, row $j$ of the table _is_ shape $j$. Two consequences:

- *The codes are coordinates the network invents.* Nothing ties a latent axis to a
  physical parameter such as a radius. Whether the learned axes recover the generating
  parameters is this repo's research question (@ch:status[Chapter]).
- *Reconstruction is optimization.* To represent a new shape, freeze $theta$ and
  minimize the same loss over the code alone, or over the control points of a latent
  field. The structural optimization uses the same mechanism: gradients with respect to
  the latent input, weights fixed.

#incode[`train_deep_sdf` (#f("DeepSDFStruct/DeepSDFStruct/deep_sdf/training.py", 299))
  minimizes this loss over the decoder and the code table. `get_model`
  (#f("DeepSDFStruct/DeepSDFStruct/pretrained_models.py", 98)) loads a trained run and
  `SDFfromDeepSDF` (#f("DeepSDFStruct/DeepSDFStruct/SDF.py", 1199)) wraps it as an SDF
  on $[-1, 1]^3$ whose only parameter is the latent: the weights stay frozen, as
  reconstruction and MMA need. API and traps: @ch:library[Chapter]; training:
  @ch:training[Chapter].]

#watch[`PretrainedModels.AnalyticRoundCross`, used by the library template and by
  `structsept/plate_with_hole.py`, is not a learned network. Its `forward`
  (#f("DeepSDFStruct/DeepSDFStruct/deep_sdf/networks/analytic_round_cross.py", 90))
  ignores the weights and evaluates three axis-aligned cylinders of radius
  $r = lambda$ in closed form: the latent _is_ the strut radius. The trained `RoundCross`
  has codes in $[-1, 1]$ and at the same $lambda = 0.6$ fills 29 % of the cell instead
  of 54 %. Compare shapes, never $lambda$ values across decoders.]

== The transformation function $T(bold(x))$

The decoder knows one cell in $[-1, 1]^3$; the lattice has $t_x times t_y times t_z$
cells in $[0, 1]^3$. The paper maps one onto the other with one global function per
coordinate (its Eq. 18):
$ T(x) = 4 abs((t_x x)/2 - floor((t_x x)/2 + 1/2)) - 1, quad
  phi_"lat" (bold(xi)) = f_theta (lambda(bold(xi)), T(xi_1), T(xi_2), T(xi_3)). $
$T$ is a triangle wave (@fig:method-transformation): it rises from $-1$ to $1$ across
one cell and falls back across the next. *Neighbouring cells are mirror images, not
translated copies.* An asymmetric cell, the normal case for a network geometry, does not
match a translated copy of itself on the shared face: the SDF would jump and the mesh
would crack; a mirrored copy matches by construction. The lattice SDF is continuous if
both $T$ and $lambda(bold(xi))$ are, and the range of $T$ does not depend on the number
of cells. (Deep Local Shapes, by Chabra et al., blends discontinuous per-cell
transformations instead.)

#figure(
  placement: auto,
  image("../../stiffness_theory/figures/transformation.png", width: 62%),
  caption: [$T(x)$ for $t_x = 4$ (figure from `docs/stiffness_theory`, axis labels in
    Portuguese: _entrada da rede_ = network input). Dashed lines are cell interfaces,
    where $T$ mirrors instead of jumping.],
) <fig:method-transformation>

#incode[`transform` (#f("DeepSDFStruct/DeepSDFStruct/lattice_structure.py", 262)) is
  Eq. 18, and `LatticeSDFStruct._compute` (line 172) is $phi_"lat"$: the latent field at
  the _raw_ parametric coordinates, the cell at the transformed ones, and a positive
  distance to the box outside the bounds (default $[0, 1]^d$). The values are in cell
  units (a cell spans 2): the zero level set is exact, the magnitudes are not metric.
  API and traps: @ch:library[Chapter].]

== The latent field $lambda(bold(x))$: the design variables

To grade the lattice, the latent code varies over space as a tensor-product B-spline
(the paper's Eq. 21; here $phi_i$ are basis functions, not the SDF):
$ lambda(bold(xi)) = sum_i phi_i (bold(xi)) hat(lambda)_i, quad hat(lambda)_i in RR^d. $
*The control points $hat(lambda)_i$ are the design variables.* Four B-spline properties
explain the choice:

- *Local support.* A degree-$p$ basis function is non-zero on only $p + 1$ knot spans,
  so a control point changes the lattice only in its neighbourhood.
- *Partition of unity.* $phi_i >= 0$ and $sum_i phi_i = 1$: $lambda(bold(xi))$ is a
  convex combination of nearby control points. Box bounds on $hat(lambda)$ therefore
  bound $lambda(bold(xi))$ everywhere, and MMA's bounds keep every cell inside the
  decoder's trained range.
- *Independent resolution.* The number of control points does not depend on the number
  of cells: the template has $2 times 1 times 1$ cells and $5 times 1 times 3 = 15$
  control points, the paper's cantilever 54 cells and $9 times 2 times 5 = 90$.
- *Continuity.* Degree $p >= 1$ gives a continuous field, hence (with $T$) a continuous
  geometry.

#incode[`SplineParametrization`
  (#f("DeepSDFStruct/DeepSDFStruct/parametrization.py", 133)) wraps a `TorchSpline`
  (#f("DeepSDFStruct/DeepSDFStruct/torch_spline.py", 275)), which copies the splinepy
  control points into an `nn.Parameter` of shape `(n_ctrl, d)` and evaluates the spline
  in torch. That parameter _is_ $hat(lambda)$: `next(parametrization.parameters())`.
  `Constant` is the one-code baseline. API and traps: @ch:library[Chapter].]

#wip[The project's own geometry still uses `Constant([latent])`, i.e. identical cells, in
  `structsept/plate_with_hole.py` and the plate experiments. Switching to
  `SplineParametrization` is item 8 of `experiments/IDEIAS.md`, still open. Only the GUI's
  Explore tab and the point-cloud fit use a spline latent field today.]

== Closing the borders: `CappedBorderSDF`

Cutting a lattice at the domain faces leaves struts open, and an open surface has no
inside: no volume, nothing for tetgen to fill, no FEM. `CappedBorderSDF` intersects the
SDF with planes at the faces of the child's bounds. Each face key (`x0`, `x1`, ...,
`z1`) maps to `{"cap": ±1, "measure": m}`; at the lower face $x = b_0$:

- `cap: -1` trims, $phi <- max(phi, (b_0 + m) - x)$, a flat cut (at the face if $m = 0$);
- `cap: +1` adds a plate of thickness $m$, $phi <- min(phi, x - (b_0 + m))$.

The default `UNIT_CUBE_CAPS_3D` trims all six faces flush. A plate (a `min`) also fills
everything outside the domain, so plates are nested inside a trimming pass. This is how
the template builds its clamping plate at `x0` and its loading plate at `z1`:

```python
plates = {"x0": {"cap": 1, "measure": 0.05}, "z1": {"cap": 1, "measure": 0.1}}
sdf = CappedBorderSDF(CappedBorderSDF(lattice, plates))  # inner adds, outer trims
```

Measures are parametric. Because the caps are `min`/`max`, the gradient reaches
$hat(lambda)$ only where the lattice is the active argument.

#incode[`CappedBorderSDF` (#f("DeepSDFStruct/DeepSDFStruct/SDF.py", 1450)), defaults at
  line 132, nesting at #f("DeepSDFStruct/tests/test_structural_optimization.py", 54).
  `LatticeSDFStruct` already returns a positive distance outside its bounds, so an
  uncapped lattice also closes somewhere near the face; the cap makes the closure an
  exact plane, and every lattice in this repo has one. API and traps:
  @ch:library[Chapter].]

== FlexiCubes: from the SDF to a surface mesh

The SDF is sampled on a regular grid; the surface crosses every grid cube whose corners
differ in sign. Marching cubes puts a vertex on each sign-change edge $(a, b)$ at
$bold(x)_e = (phi_b bold(x)_a - phi_a bold(x)_b) \/ (phi_b - phi_a)$ and connects them
by a case table, which produces sliver triangles; the paper rejected it because slivers
are bad for FEM. FlexiCubes (Shen et al., 2023) is a _dual_ method: one vertex per cut
cube (up to four where a cube holds several surface patches), at a weighted combination
of its edge crossings, with neighbouring cubes joined by
quads split into triangles. The elements are better shaped, and every vertex is a smooth
function of the $phi$ values at its cube's corners, hence of $hat(lambda)$. Only grid
values at cut cubes receive a gradient, and the topology (which cubes are cut, how many
vertices exist) is discrete: the gradient tells how the surface moves, not when a strut
appears or vanishes.

*Resolution.* For an integer `N_base` the grid has
$N_i = ceil(N_"base" t_i e_i \/ e_"max") + 1$ cubes along axis $i$ (at least 4), over the
bounds (extents $e_i$) extended by 5 % per side. On the parametric cube all $e_i$ are
equal, so *`N_base` is roughly the number of cubes per unit cell along every axis* (a
little less, because the grid also covers the margins). The paper's
"resolution 20" is `N_base = 20`; the template's 10 with tiling $[2, 1, 1]$ gives
$21 times 11 times 11$ cubes. Cost grows with $N_"base"^3$ times the number of cells.

#incode[`create_3D_mesh(sdf, N_base, mesh_type, ..., deformation_function=None)`
  (#f("DeepSDFStruct/DeepSDFStruct/mesh.py", 669)); the resolution rule is
  `process_N_base_input` (line 614). `get_verts` (line 995) evaluates `sdf(samples)` with
  autograd on, runs `FlexiCubes`, applies the deformation. `mesh_type="surface"` gives
  triangles, `"volume"` FlexiCubes' interior tets; the vertices keep their `grad_fn`.
  API and traps: @ch:library[Chapter].]

== Freeform deformation: from the parametric cube to metres

The physical domain is reached through a trivariate B-spline,
$overline(bold(x)) = sum_i N_i (bold(xi)) bold(P)_i$. For a box this is a stretch,
$(L xi_1, W xi_2, H xi_3)$; moving the $bold(P)_i$ bends the part without touching
anything upstream. It is the same `TorchSpline` class as the latent field, so it is
differentiable, but its control points are *not* design variables (the paper keeps the
deformation fixed). *The FFD moves mesh vertices after extraction; the SDF never sees
metres.* Anything sized in metres, such as a hole, must be converted to parametric units
before it enters the SDF (`ScaledSpaceSDF`, @ch:fem[Chapter]), cap measures are
parametric, and a strongly stretching FFD gives stretched elements.

#incode[`get_verts` (#f("DeepSDFStruct/DeepSDFStruct/mesh.py", 995)) calls
  `deformation_function.forward(verts)` after FlexiCubes. The template maps $[0, 1]^3$
  onto a $2 times 1 times 1$ beam with
  `TorchSpline(splinepy.helpme.create.box(2, 1, 1).bspline)`.]

== Tetrahedralization: the open trade-off

FEM needs a volume mesh. There are two routes and neither is satisfactory yet:

#[
  #set text(size: 9pt)
  #kv(
    columns: (auto, 1fr, 1fr),
    [*Route*], [*Gain*], [*Cost*],
    [FlexiCubes interior tets\ `mesh_type="volume"`\ (library template)],
    [Differentiable end to end.],
    [*About 30 % of the volume missing.* On a solid unit cube the tets sum to 0.68--0.70 at
      every resolution; their boundary has 13 034 triangles in 5 836 components, against
      1 200 in one for the surface: internal cavities.],
    [Surface, then tetgen\ `tetrahedralize_surface`\ (`structsept/fem.py`)],
    [Volume matches the surface to all printed digits, with fewer elements.],
    [*No gradient.* NumPy and tetgen end the autograd graph at the surface vertices.],
  )
]

The cavity mesh gives the compliance of a body with 30 % voids; the tetgen mesh gives the
right compliance but no sensitivities. The two possible ways out are discussed in
@ch:fem[Chapter] (What is missing before an optimization loop). On either route torch-fem needs positively oriented tets
(the element Jacobian determinant is $6 V_e$): the template and `fem.py` flip negative
tets and drop tets with volume at most $10^(-12)$.

#wip[Unresolved. Until it is, compliance from the library template is that of the cavity
  mesh, not of the geometry the decoder describes.]

== FEM: linear elasticity, stiffness, compliance

On $overline(Omega)$ the structure obeys
$nabla dot bold(sigma) + bold(b) = bold(0)$,
$bold(epsilon) = 1/2 (nabla bold(u) + nabla bold(u)^T)$,
$bold(sigma) = bold(C) : bold(epsilon)$. In the weak form,
$integral bold(epsilon)(bold(v)) : bold(C) : bold(epsilon)(bold(u)) dif x =
ell(bold(v))$, the left side becomes the stiffness matrix. It depends on geometry and
material only, not on loads or supports, which is why `structsept/fem.py` can assemble
$bold(K)$ before choosing any boundary condition. For linear tetrahedra the shape
functions are barycentric, the strain-displacement matrix $bold(B)$ ($6 times 12$) is
constant, and one integration point is exact:
$ bold(k)_e = integral_(Omega_e) bold(B)^T bold(C) bold(B) dif x
  = V_e bold(B)^T bold(C) bold(B) quad (12 times 12), quad quad
  bold(K) = sum_e bold(L)_e^T bold(k)_e bold(L)_e . $
The Boolean $bold(L)_e$ are never formed: each entry of $bold(k)_e$ is scatter-added at
its pair of global degrees of freedom. Without supports $bold(K)$ is singular, with the
six rigid-body modes ($bold(u) = bold(c)$, $bold(u) = bold(omega) times bold(x)$) as
kernel. Checking $bold(K) bold(u)_"rigid" = bold(0)$ tests $bold(B)$ and the assembly,
not the material: $bold(B) bold(u)_"rigid"$ is already zero element by element, so the
product vanishes for any $bold(C)$. Dirichlet constraints remove the kernel.

#block(breakable: false)[The optimization needs the compliance, the volume and the
  volume constraint:
  $ J = bold(f)^T bold(u) = bold(u)^T bold(K) bold(u), quad
    V = sum_e V_e, quad G = V - V_"target" <= 0. $]
$J$ is twice the strain energy (the paper writes
$integral_Omega sigma(u) : epsilon(u) dif x$); minimizing it maximizes stiffness under
the given load. For design-independent loads compliance is self-adjoint,
$dif J \/ dif p = -bold(u)^T (partial bold(K) \/ partial p) bold(u)$, with no second
solve. The paper solved the FEM with MFEM through its Python wrapper and wrote this
sensitivity out as a shape derivative (its Eqs. 36--37); the library and this repo use
torch-fem, where autodiff does it: torch-fem's sparse solve is a custom autograd
function whose backward solves $bold(K)^T bold(g) = partial J \/ partial bold(u)$ (the
adjoint solve) and returns $partial J \/ partial bold(K)$; autograd continues from there
into $bold(k)_e$, $V_e$ and the vertex positions.

#incode[`structsept/fem.py` goes as far as $bold(K)$: torch-fem's `solid.k0()` returns
  the $bold(k)_e$ above, `solid.assemble_matrix` does the scatter-add, and
  `check_stiffness` (#f("structsept/fem.py", 230)) runs the rigid-body test
  (@ch:fem[Chapter]). The library template adds supports, load, solve, $J$ and $G$
  (@ch:library[Chapter]).]

== MMA: updating the design

The Method of Moving Asymptotes (Svanberg, 1987) replaces, at every iteration, the
objective and the constraints by convex, separable approximations around the current
design $bold(x)^((k))$,
$ tilde(F)(bold(x)) = r + sum_j (p_j / (U_j - x_j) + q_j / (x_j - L_j)),
  quad L_j < x_j < U_j, $
with $p_j, q_j >= 0$ set from the gradient so that $tilde(F)$ matches $F$ and its slope at
$bold(x)^((k))$. The asymptotes move closer to $x_j$ when a variable oscillates (more
conservative) and away when it moves steadily (bolder); the subproblem is solved cheaply
(`mmapy` uses a primal-dual Newton method). MMA fits because each evaluation costs a full FEM solve plus a backward
pass, there are many variables and few constraints, and only first derivatives exist.

#incode[`MMA(parameters, bounds, max_step=0.1, n_constraints=1)`
  (#f("DeepSDFStruct/DeepSDFStruct/optimization.py", 170)). `step(F, dF, G, dG)`
  (line 349) takes the objective, the constraints and their gradients, solves one MMA
  subproblem with `mmapy.mmasub` and writes the new $hat(lambda)$ back into
  `parameters`. It normalizes $F$ by its first value but not $G$, so scale the
  constraint yourself ($G = V \/ V_"target" - 1$), and it has no stopping rule. API and
  traps: @ch:library[Chapter].]

*Bounds.* The template bounds every control point to $[0.15, 0.75]$: radii of the
analytic round cross, inside the paper's training range of radii 0.1 to 0.75. Outside
its trained range a real decoder returns meaningless geometry. Trained codes have no
physical scale (the bundled `RoundCross` codes span $[-1, 1]$, `ChiAndCross`
$[-0.95, 0.70]$ and $[-0.98, 0.64]$ in its two components), so derive bounds from the per-component minimum and
maximum of `model._trained_latent_vectors`. By the partition of unity they then hold for
$lambda(bold(xi))$ everywhere.

== The gradient chain

One `create_3D_mesh` call plus one torch-fem solve form a single torch computation from
$hat(lambda)$ to $J$; `torch.autograd.grad(J, param)` walks it backwards
(@fig:method-chain) and MMA turns the result into the next design.

#figure(
  diagram(
    spacing: (11mm, 4.6mm), edge-stroke: 0.7pt, mark-scale: 70%,
    ncon((0, 0), [$hat(lambda)$: spline control points $(n_"ctrl", d)$,
      *the design variables*]),
    ncon((0, 1), [$lambda(bold(xi))$ at every grid point, $(n_"grid", d)$]),
    ndat((0, 2), [$phi$ on the grid, $(n_"grid", 1)$]),
    ndat((0, 3), [mesh vertices in $[0, 1]^3$]),
    ndat((0, 4), [vertices $overline(bold(x))$ in metres]),
    ndat((0, 5), [compliance $J$, volume $V$]),
    next((0, 6), [`MMA.step`]),
    // forward
    edge((0, 0), (0, 1), "-|>", shift: 4pt, label-side: left,
      lbl[B-spline basis: `SplineParametrization`]),
    edge((0, 1), (0, 2), "-|>", shift: 4pt, label-side: left,
      lbl[$T$, $f_theta$, caps: `LatticeSDFStruct`, `CappedBorderSDF`]),
    edge((0, 2), (0, 3), "-|>", shift: 4pt, label-side: left,
      lbl[`FlexiCubes` in `create_3D_mesh`]),
    edge((0, 3), (0, 4), "-|>", shift: 4pt, label-side: left,
      lbl[FFD: `TorchSpline`]),
    edge((0, 4), (0, 5), "-|>", shift: 4pt, label-side: left,
      lbl[torch-fem: $bold(k)_e$, $bold(K)$, solve $bold(K) bold(u) = bold(f)$]),
    edge((0, 5), (0, 6), "-|>", shift: 4pt, label-side: left,
      lbl[$F$, $G$ and their gradients]),
    edge((0, 6), (3, 6), (3, 0), (0, 0), "-|>", corner-radius: 4pt,
      lbl[in-place update of $hat(lambda)$]),
    // backward
    edge((0, 5), (0, 4), "--|>", stroke: c-grad, shift: 4pt, label-side: left,
      lbl[adjoint solve, $partial J \/ partial overline(bold(x))$]),
    edge((0, 4), (0, 3), "--|>", stroke: c-grad, shift: 4pt, label-side: left,
      lbl[$partial J \/ partial bold(v)$]),
    edge((0, 3), (0, 2), "--|>", stroke: c-grad, shift: 4pt, label-side: left,
      lbl[$partial J \/ partial phi$ (cut cubes only)]),
    edge((0, 2), (0, 1), "--|>", stroke: c-grad, shift: 4pt, label-side: left,
      lbl[$partial J \/ partial lambda$ (active branches)]),
    edge((0, 1), (0, 0), "--|>", stroke: c-grad, shift: 4pt, label-side: left,
      lbl[$partial J \/ partial hat(lambda)$]),
  ),
  caption: [The gradient chain of one optimization iteration. Black: the forward pass and
    the class doing each step. Pink dashed: the backward pass of `torch.autograd.grad`.
    The FFD control points and the decoder weights sit on the chain but are not design
    variables.],
) <fig:method-chain>

Every link must be a torch operation recorded on the autograd graph. Hence the rule from
`CLAUDE.md`, for anything between $hat(lambda)$ and $J$ or $V$ inside the loop: *no
`.detach()`* (it returns a tensor cut off from the graph), *no NumPy round trip*
(`.numpy()`, `np.*`, `torch.as_tensor(array)`: NumPy has no autograd, the rebuilt tensor
is a constant), *no `.item()`* or `float()` (a Python number is a constant too), and no
`torch.no_grad()` block. Breaking the rule raises no error: the gradient silently comes
out zero, or misses part of the chain. Some things only look like violations: `.item()`
for logging (the template does it), Boolean masks and index tensors, and a
`torch.autograd.Function` with its own backward, such as torch-fem's solve, which uses
SciPy inside. The tetgen path of `fem.py` does break the rule, and the GUI breaks it on
purpose (@ch:gui[Chapter]).

#tip[After any change inside the loop, check that the gradient is non-zero (the template
  asserts it at the end) and that one component agrees with a finite difference: perturb
  one control point by $plus.minus h$, re-mesh, re-solve. Topology changes make finite
  differences noisy, so keep $h$ small and `N_base` moderate.]
