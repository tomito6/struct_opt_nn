#import "../template.typ": *

// Where the code of a section lives: one small line under the heading, so long file
// references stay out of the justified text.
#let src(body) = block(above: 0.45em, below: 0.75em, width: 100%, sticky: true)[
  #set par(justify: false)
  #text(size: 8.5pt, fill: c-muted)[_Code:_ #body]
]
#let L = "DeepSDFStruct/DeepSDFStruct/"

= The DeepSDFStruct library <ch:library>

`DeepSDFStruct` implements the paper. It is a git submodule (`DeepSDFStruct/` at commit
`bbe9881`, working tree clean; upstream
#link("https://github.com/mkofler96/DeepSDFStruct")[`mkofler96/DeepSDFStruct`] on GitHub)
whose importable package is the inner folder `DeepSDFStruct/DeepSDFStruct/`. The library has two halves
that meet in one object, the `DeepSDFModel`: a decoder network plus the table of latent
codes it learned. The *learning half* turns SDF samples into such a model; the *geometry
and optimisation half* uses it as a geometry generator inside the loop.
@ch:method[Chapter] explains the concepts. This chapter is the API: which class does
what, what a call looks like, and where it bites. Edit the submodule only when you mean
to change the library itself (@ch:repo[Chapter]).

== Module map

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr, auto),
    [*Module*], [*What is in it*], [*Used here by*],
    [`SDF.py`], [`SDFBase`, booleans, `CappedBorderSDF`, `SDFfromDeepSDF`],
    [everything],
    [`sdf_primitives.py`], [Analytic shapes: `SphereSDF`, `BoxSDF`, `CylinderSDF`, ...],
    [`plate_with_hole`],
    [`sdf_operations.py`], [Space warps: `TwistSDF`, `ShellSDF`, `RepeatSDF`, ...], [--],
    [`lattice_structure.py`], [`LatticeSDFStruct`, `transform` $= T(bold(x))$],
    [geometry, GUI],
    [`parametrization.py`], [`Constant`, `SplineParametrization`], [geometry, GUI],
    [`torch_spline.py`], [`TorchSpline` (latent field _and_ FFD), `TorchScaling`],
    [geometry],
    [`mesh.py` \ `flexicubes/` \ `flexisquares/`],
    [`create_3D_mesh`, `create_2D_mesh`, `tetrahedralize_surface`; adapted NVIDIA
      FlexiCubes and its 2-D twin],
    [`fem.py`, GUI],
    [`deep_sdf/models.py`], [`DeepSDFModel`: decoder plus code table], [via `get_model`],
    [`deep_sdf/networks/`], [Decoder classes, chosen by `NetworkArch`],
    [`deep_sdf_decoder`],
    [`deep_sdf/training.py`], [`train_deep_sdf`; DeepLS: `training_latent_field.py`],
    [`app/training.py`],
    [`deep_sdf/workspace.py`], [Folder layout, `init_decoder`, `load_trained_model`],
    [`app/training.py`],
    [`deep_sdf/data.py`], [`SDFSamples`: reads `SdfSamples/` via a split file], [tests],
    [`deep_sdf/reconstruction.py`], [`reconstruct_from_samples`], [indirectly],
    [`sampling.py`], [`SampledSDF`, the dataset writer `SDFSampler`], [SDF maker],
    [`pretrained_models.py`], [`PretrainedModels`, `get_model`, `trained_models/`],
    [everything],
    [`geom_reconstruction.py`], [`build_parameter_spline`, `LocalShapesReconstructor`],
    [GUI, experiments],
    [`optimization.py`], [`MMA`, `tet_signed_vol`, `get_mesh_from_torchfem`],
    [all but `MMA`],
  )
]

The remaining files (`local_shapes.py`, whose `LocalShapesSDF` is an older variant of
`LatticeSDFStruct`;
`design_of_experiments.py`; the `export_*` scripts; `splinepy_unitcells/`;
`visualization/`) are not used here. Library docstrings often lag the code (wrong
defaults, examples that call functions which no longer exist). Trust the code.

== The geometry side

=== Everything is an SDF module

#src[`SDFBase` #f(L + "SDF.py", 194), `UnionSDF` #f(L + "SDF.py", 575),
  `DifferenceSDF` #f(L + "SDF.py", 660)]

`SDFBase` is a `torch.nn.Module`. `sdf(points)` checks that `points` is `(N, 3)` or
`(N, 2)`, calls the subclass's `_compute` and returns `(N, 1)` signed distances,
negative inside; that, plus a check of the output's row count, is all `forward` does. A subclass implements `_compute` and
`_get_domain_bounds`, a `(2, dim)` box the mesher uses by default. A geometry is
therefore one module tree, and `parameters()` on its root finds every differentiable
number in it. The *leaves* are primitives with `nn.Parameter` sizes (plus
`SDFfromMesh`, which goes through NumPy and breaks gradients) and the network wrapper
`SDFfromDeepSDF`. The *combinators* are `UnionSDF` (min), `DifferenceSDF`, `Smooth*SDF`,
`CappedBorderSDF`, `LatticeSDFStruct` and the warps. `a + b` returns `UnionSDF(a, b)`; no
other operator is overloaded.

#watch[`UnionSDF`, `SmoothUnionSDF`, `SmoothIntersectionSDF` and the subtracted
  objects of `DifferenceSDF` and `SmoothDifferenceSDF` keep their children in a plain
  Python list, not as registered
  submodules, so `.to()`, `named_parameters()` and the lattice search in
  `create_3D_mesh` do not see them; a lattice inside a `UnionSDF` is meshed without
  scaling the resolution by its tiling. `DifferenceSDF.base_obj`, `CappedBorderSDF.sdf`
  and `LatticeSDFStruct.microtile` are registered.]

=== From checkpoint to SDF

#src[`get_model` #f(L + "pretrained_models.py", 98), `DeepSDFModel`
  #f(L + "deep_sdf/models.py", 23), `SDFfromDeepSDF` #f(L + "SDF.py", 1199),
  `set_latent_vec` #f(L + "SDF.py", 1261)]

```python
model = get_model(PretrainedModels.AnalyticRoundCross, device="cpu")  # or a run folder
cell = SDFfromDeepSDF(model)          # callable SDF on [-1, 1]^3, latent = code 0
cell.set_latent_vec(torch.tensor([0.4]))
phi = cell(torch.zeros(1, 3))         # (1, 1)
```

`get_model(model, checkpoint="latest", device=None)` accepts an enum entry or the path
(as a `str`) of any experiment folder, so the project's `runs/<run>` load the same way. It reads
`specs.json`, rebuilds the decoder named by `NetworkArch`, loads the `ModelParameters/`
and `LatentCodes/` checkpoints, and returns a `DeepSDFModel`.

*`DeepSDFModel` is a plain object, not an `nn.Module`.* The good consequence: decoder
weights never appear in `sdf.parameters()`, so an optimiser built on a geometry tree
moves only codes or control points, which is exactly what reconstruction and MMA need.
The bad one: `sdf.to(device)` does not move the decoder. `get_model` picks CUDA when it
is available, while `create_3D_mesh` defaults to `"cpu"`; always pass
`device=model.device` to the mesher.

`SDFfromDeepSDF(model, max_batch=32**3)` evaluates the decoder in batches. Its domain is
*$[-1, 1]^3$*, the box the decoder was trained on, not the unit cube. It takes
`geometric_dim` from the decoder, so a 2-D decoder gives a 2-D SDF. `set_latent_vec`
accepts two shapes. A `(d,)` code becomes a `Constant` parametrization: one shape
everywhere. An `(N, d)` tensor is stored as `latvec`, one code per query point; this is
what a lattice writes, and `_compute` uses `latvec` whenever it is set.

=== The pretrained models

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, auto, auto, 1fr),
    [*`PretrainedModels.`*], [*`NetworkArch`*], [*$d$*], [*What it is*],
    [`AnalyticRoundCross`], [`analytic_round_cross`], [1],
    [Three crossing cylinders in closed form; the latent _is_ the radius],
    [`RoundCross`], [`deep_sdf_decoder`], [1],
    [Trained cross cell; 20 codes in $[-1, 1]$ (the analytic model ships a copy of this
      table)],
    [`ChiAndCross`], [`deep_sdf_decoder`], [2], [Chi-shaped cell combined with a diagonal cross],
    [`Primitives` \ (= `PrimitivesCL32`), \ `PrimitivesCL16`, \ `PrimitivesCL08`],
    [`deep_sdf_decoder`], [32, 16, 8], [DeepLS (Deep Local Shapes, @ch:method[Chapter])
      models: one latent _field_ per training scene; the stored code table is zeros],
    [`Primitives2D`], [`deep_sdf_decoder`], [16], [DeepLS on 2-D primitives (still a
      3-D decoder)],
  )
]

#watch[`AnalyticRoundCross` is a formula, not a network (@ch:method[Chapter]): its
  values away from the struts are not exact distances, and the template's bounds
  $[0.15, 0.75]$ are strut radii. For a trained decoder, take bounds from the range of
  its trained codes, `model._trained_latent_vectors`.]

=== `LatticeSDFStruct`: tiling and the latent hand-off

#src[`LatticeSDFStruct` #f(L + "lattice_structure.py", 39), `_compute`
  #f(L + "lattice_structure.py", 172), `transform` #f(L + "lattice_structure.py", 262)]

`LatticeSDFStruct(tiling, microtile, parametrization, bounds=None)` tiles a microtile
over a box that defaults to the *parametric unit cube $[0, 1]^d$*. A `parametrization`
module is required; `None` raises `TypeError`. `_compute` runs five steps:

+ Mask the query points inside the box.
+ Evaluate the latent field there, `(n_in, d)`, and push it into the microtile with
  `microtile._set_param`; for an `SDFfromDeepSDF` this sets `latvec`.
+ Map each coordinate into $[-1, 1]$ with `transform`, the paper's Eq. (18), for $x$
  normalised to $[0, 1]$ and $t$ cells along that axis:
  $ T(x) = 4 abs((t x) / 2 - floor((t x) / 2 + 1 / 2)) - 1 $
  A triangle wave: cell 0 runs from $-1$ to $1$, cell 1 back from $1$ to $-1$. Neighbour
  cells are *mirror images*, not translated copies, so the field stays continuous even
  for an asymmetric cell.
+ Evaluate the microtile at the mapped points.
+ Give points outside the box their positive distance to the box, so the mesher's
  wider grid sees empty space there.

The values are in microtile units (a cell spans 2): the zero level set is right, the
magnitudes are scaled by about $2 t$ per axis.

#watch[Step 2 is shared, mutable state. Each evaluation overwrites `microtile.latvec`
  with one code per point of _that_ query. Two evaluations at once (a worker meshing
  while a slider redraws) collide, and the loser fails with a shape mismatch. A stale
  `latvec` also breaks a later direct call on the microtile with another point count,
  even after a 1-D `set_latent_vec`, because `latvec` wins. The GUI (@ch:gui[Chapter])
  locks the controls
  that touch the SDF while a worker runs (`run_worker`,
  #f("structsept/app/runtime.py", 65)) and previews cells with a separate wrapper
  (`unit_cell_sdf`). Elsewhere: one wrapper per use, or set `latvec = None` first.]

=== The latent field: `SplineParametrization` and `TorchSpline`

#src[`SplineParametrization` #f(L + "parametrization.py", 133), `TorchSpline`
  #f(L + "torch_spline.py", 275), `build_parameter_spline`
  #f(L + "geom_reconstruction.py", 85)]

`SplineParametrization(spline, device)` wraps a `splinepy` B-spline in a `TorchSpline`,
which copies the control points into an `nn.Parameter` of shape `(n_cp, d)` and
evaluates the spline with a vectorised de Boor algorithm in torch. *That parameter is
$hat(lambda)$, the design vector*; `next(parametrization.parameters())` returns it. The
control net is flattened x-fastest, index $i + n_x (j + n_y k)$, which the GUI's
`flat_index` relies on. The spline is evaluated at lattice coordinates, so its
parametric box must equal the lattice box; `build_parameter_spline` makes a clamped
spline with one knot span per cell. `set_param(values)` overwrites the control points
under `no_grad`, without a rebuild. `Constant(value)` is the one-code-everywhere
baseline that `plate_with_hole` still uses.

`TorchSpline` evaluates non-rational B-splines only (the NURBS in its docstring are not
implemented). The same class is the FFD: in the template, a `TorchSpline` of the box
spline `create.box(2, 1, 1)` maps $[0, 1]^3$ onto a $2 times 1 times 1$ beam. Its control
points are not design variables.

=== `CappedBorderSDF`: closing the borders

#src[`CappedBorderSDF` #f(L + "SDF.py", 1450), default dict `UNIT_CUBE_CAPS_3D`
  #f(L + "SDF.py", 132)]

`CappedBorderSDF(sdf, cap_border_dict=None, scale=(1, 1, 1))` combines its child with planes at the
child's box faces. Keys `x0` to `z1` name the faces, each mapped to
`{"cap": c, "measure": m}`. With $c = -1$ the face _trims_ (a `max`, cut $m$ inside the
face); with $c = +1$ it _adds_ a solid plate of thickness $m$ (a `min`). Measures are in
parametric units; the default dict trims all six faces flush. This replaces the jump
from strut (negative) to box distance (positive) at the lattice border by a clean
plane. Without it the surface still closes, but between grid nodes at an uncontrolled
place (@ch:method[Chapter]); the template and every lattice the project builds itself
are capped, so do the same. A plate fills the whole half-space
beyond its face, because a `min` with a plane has no outer end, so nest: the inner
wrapper adds plates, the outer one trims back to the box. The default dict is one shared
module-level object; never mutate a wrapper's `cap_border_dict` in place.

```python
plates = {"x0": {"cap": 1, "measure": 0.05}, "z1": {"cap": 1, "measure": 0.1}}
sdf = CappedBorderSDF(CappedBorderSDF(lattice, plates))   # add plates, then trim
```

#figure(
  diagram(
    spacing: (20mm, 8mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nsym((0, 0), [`CappedBorderSDF` \ default dict: trim all faces]),
    nsym((0, 1), [`CappedBorderSDF` \ plates at `x0` and `z1`]),
    nsym((0, 2), [`LatticeSDFStruct` \ tiling `[2, 1, 1]`, box $[0, 1]^3$]),
    nsym((1, 2), [`SplineParametrization`]),
    ndat((2, 2), [`control_points` \ $(15, 1)$ = $hat(lambda)$]),
    nsym((0, 3), [`SDFfromDeepSDF` \ box $[-1, 1]^3$]),
    next((1, 3), [`DeepSDFModel` \ plain object]),
    nsym((2, 3), [`RoundCrossDecoder`]),
    edge((0, 0), (0, 1), "-|>", lbl[sdf], label-side: left),
    edge((0, 1), (0, 2), "-|>", lbl[sdf], label-side: left),
    edge((0, 2), (1, 2), "-|>", lbl[parametrization], label-side: left),
    edge((1, 2), (2, 2), "-|>", lbl[torch_spline], label-side: left),
    edge((0, 2), (0, 3), "-|>", lbl[microtile], label-side: left),
    edge((0, 3), (1, 3), "--|>", lbl[model], label-side: right),
    edge((1, 3), (2, 3), "--|>", lbl[\_decoder], label-side: right),
    edge((1, 2), (0, 3), "--|>", stroke: c-data, lbl[$lambda(bold(x))$ per point],
      label-side: left, label-pos: 0.4, label-sep: 1em),
  ),
  caption: [The geometry tree of the optimisation template. Solid arrows are registered
    submodules, seen by `parameters()` and `.to()`. Dashed arrows are plain attributes;
    the orange one is the latent hand-off (`_set_param`) at every evaluation.],
) <fig:library-tree>

=== Meshing

#src[`create_3D_mesh` #f(L + "mesh.py", 669), `process_N_base_input`
  #f(L + "mesh.py", 614), `tetrahedralize_surface` #f(L + "mesh.py", 451),
  `create_2D_mesh` #f(L + "mesh.py", 848)]

```python
create_3D_mesh(sdf, N_base, mesh_type, differentiate=False, device="cpu", bounds=None,
               diffmode="fwd", deformation_function=None, use_tiling=True,
               extend_bounds=True)   # -> (torchSurfMesh | torchVolumeMesh, None)
```

The mesher takes the box from `sdf._get_domain_bounds()` (it must be a tensor) and
widens it by 5 % on each side. An integer `N_base` becomes
$ceil(N_"base" dot e_i \/ e_max dot t_i) + 1$ cubes along axis $i$, at least 4, with
$e_i$ the box extents and $t_i$ the tiling of the first lattice in the tree. For the
usual cube-shaped parametric box that is $N_"base" t_i + 1$: *`N_base` is roughly the
number of FlexiCubes cubes per unit cell and axis* (a little less, since the grid also
spans the 5 % margins), whatever the physical shape. The template's 10
on `[2, 1, 1]` gives $21 times 11 times 11$ cubes; the paper's 20 on $6 times 3 times 3$
cells gives $121 times 61 times 61$.

The SDF is evaluated on that grid with autograd on. FlexiCubes extracts triangles
(`"surface"`) or its own tetrahedra (`"volume"`, all reoriented positively), and
`deformation_function` (a `TorchSpline` or `TorchScaling`) then moves the *vertices*
into physical space. The SDF never sees physical coordinates, so anything sized in
metres must be converted first (`ScaledSpaceSDF`, @ch:fem[Chapter]). The returned
vertices carry `grad_fn`. `differentiate=True` does _not_ switch gradients on; it adds
an explicit Jacobian of the vertices with respect to all parameters, which is expensive
and only serves derivative export.

`tetrahedralize_surface(surface_mesh)` fills a closed `gus.Faces` surface with tetgen
(switches `pYq`) and returns the volume mesh plus the indices of the surface vertices
in it. It works in NumPy: *the gradient stops there*. `create_2D_mesh` is the 2-D twin
on FlexiSquares (`"line"`, `"surface"`, `"surface_triangle"`); it ignores the SDF's box
and uses $[-0.05, 1.05]^2$ unless given `bounds`. Nothing in the project's own code
calls it (only a library test does).

#watch[The template's `mesh_type="volume"` is differentiable but fills only about 70 %
  of the enclosed volume; surface plus tetgen is exact but not differentiable, and is
  the path `structsept/fem.py` takes. The trade-off is explained in @ch:method[Chapter].]

== The learning side

=== `train_deep_sdf`, step by step

#src[`train_deep_sdf` #f(L + "deep_sdf/training.py", 299), whose only caller here is
  `train` #f("structsept/app/training.py", 129)]

```python
train_deep_sdf(experiment_directory, data_source, continue_from=None, batch_split=1,
               device=None)        # device: the string "cpu" or "cuda", nothing else
```

The trainer fits a classic auto-decoder: one MLP plus one free latent code per training
shape. The project always runs it on the CPU. In order:

+ Read `specs.json` from the experiment folder; build the decoder (line 409).
+ Seed Python, NumPy and torch with `seed` (default 42, line 415), _after_ the decoder
  already drew its initial weights.
+ Load every sample file of the split `Path(data_source) / TrainSplit` into RAM (line
  433); `datagen/` writes them (@ch:datagen[Chapter]). The split order is the code
  index, saved as `latent_code_data_map.json`.
+ Create the codes, `nn.Embedding(n_scenes, CodeLength, max_norm=CodeBound)`, with
  initial spread $"CodeInitStdDev" \/ sqrt("CodeLength")$ (line 471).
+ Build one Adam optimiser with two groups, decoder and codes, each driven by its own
  entry of `LearningRateSchedule` (line 492).
+ Per batch: `SamplesPerScene` points (half outside, half inside) from each of
  `ScenesPerBatch` shapes; input = code concatenated with coordinates; prediction and
  target clamped to $plus.minus$ `ClampingDistance`; loss = `LossFunction` (default
  `clampedL1`, which clamps again at a fixed 0.1)
  $+ "CodeRegularizationLambda" dot min(1, "epoch" \/ 100) dot "mean" norm(lambda)$.
  Decoder gradients are clipped to norm 1.0 (line 637).
+ Save `<epoch>.pth` at multiples of `SnapshotFrequency` and at `AdditionalSnapshots`,
  `latest.pth` every `LogFrequency` epochs (line 684), `training_summary.json` at the
  end.

=== The experiment folder and its recipe

An experiment is a folder with a `specs.json`; the trainer fills in the rest. The bundled
`trained_models/<name>/` and the project's `runs/<run>/` share this layout (the project
adds `metadata.json` and analysis files, @ch:training[Chapter]):

```
specs.json                             the recipe; the only file you write
ModelParameters/{latest,<ep>}.pth      {"epoch", "model_state_dict"}
OptimizerParameters/{latest,<ep>}.pth  Adam state (needed by continue_from)
LatentCodes/{latest,<ep>}.pth          {"epoch", "latent_codes": {"weight": (n, d)}}
LatentCodes/latent_code_data_map.json  code index -> .npz file
Logs.pth, Logs.png                     loss per batch, learning rates, magnitudes
training_summary.json                  final loss, epochs, duration, host, version
```

The keys that decide a run: `NetworkArch`, `NetworkSpecs` (hidden widths `dims`, skip
layers `latent_in`, `geom_dimension` 2 or 3, ...), `CodeLength` ($d$), `NumEpochs`,
`LearningRateSchedule` (decoder, then codes), `SamplesPerScene`, `ScenesPerBatch`,
`ClampingDistance`, `CodeRegularizationLambda` (default $10^(-4)$, always applied),
`CodeBound`, `CodeInitStdDev`, `TrainSplit` and `seed`. The full schema, its validation
and the spreadsheet import are in @ch:training[Chapter]; formats in
@app:formats[Appendix].

=== Reconstruction

#src[`reconstruct_from_samples` #f(L + "deep_sdf/reconstruction.py", 60),
  `LocalShapesReconstructor` #f(L + "geom_reconstruction.py", 337)]

`reconstruct_from_samples(sdf, sdfSample, ...)` runs Adam or L-BFGS (default
`num_iterations=1000` passes over the samples in batches of 512, learning rate
$5 dot 10^(-4)$) on all `sdf.parameters()` against
clamped-L1 targets from a `SampledSDF`; with the decoder unregistered, those are only
codes or control points.
`LocalShapesReconstructor` packages the "mesh to lattice of local shapes" fit: normalise
the mesh, start a degree-1 latent spline at the mean trained code, sample the target,
fit. `experiments/pointcloud_to_lattice.py` uses it to turn a point cloud into a
starting design (@ch:fem[Chapter]).

== The optimisation side: `MMA`

#src[`MMA` #f(L + "optimization.py", 170), `MMA.step` #f(L + "optimization.py", 349)]

`MMA(parameters, bounds, max_step=0.1, n_constraints=1)` runs Svanberg's MMA (from
`mmapy`) on a torch tensor:

```python
x = next(lattice.parametrization.parameters())     # (n_cp, d); MMA updates it in place
opt = MMA(x, np.tile([0.15, 0.75], (x.numel(), 1)))   # bounds: shape (numel, 2)
for it in range(n_iter):
    F, V = forward()                               # mesh -> FEM, all in torch
    G = V / V_target - 1.0                         # <= 0 means feasible
    dF = torch.autograd.grad(F, x, retain_graph=True)[0]
    dG = torch.autograd.grad(G, x)[0]
    opt.step(F, dF, G, dG)
```

`step` moves everything to NumPy, divides $F$ and $d F$ by $|F|$ of the first
iteration, sets move limits $x plus.minus$ `max_step` (absolute, although the docstring
says "fraction of the range") clipped to the bounds, solves the subproblem, and copies
the new design into `parameters` under `no_grad`; the next forward pass rebuilds the
graph. All four arguments must be tensors. Constraints are not normalised, so scale $G$
yourself. There is *no stopping criterion*: `opt.ch`, the relative change of the mean
design, is only logged.

#wip[`MMA` is not called anywhere in `structsept/` or `experiments/`. The project's
  geometry code still uses a `Constant` latent; a `SplineParametrization` is built only
  by the GUI (`models.build_lattice`) and, inside the library's
  `LocalShapesReconstructor`, by the point-cloud fit. The optimisation loop of this repo
  is still to be built, starting from the template below.]

== The template: `test_structural_optimization.py`

#src[#f("DeepSDFStruct/tests/test_structural_optimization.py", 21)]

This test is the only complete instance of paper steps 3--6 and the starting point for
an optimisation run. It is a scaled-down paper test case 1 (which has
$6 times 3 times 3$ cells, $9 times 2 times 5$ control points, resolution 20). Line by
line:

+ *Decoder* (26--27): `get_model(AnalyticRoundCross)` with no device, so CUDA if
  available; then `SDFfromDeepSDF`.
+ *Plates* (29--32): `x0` 0.05 thick (clamped face), `z1` 0.1 thick (loaded face).
+ *Latent spline* (34--38): degrees `[1, 0, 1]`, knots giving $5 times 1 times 3 = 15$
  control points, all 0.3, a strut radius since $d = 1$.
+ *Parametrization and FFD* (42--48): tiling `[2, 1, 1]`, `SplineParametrization`, and
  a `TorchSpline` box that maps $[0, 1]^3$ onto the $2 times 1 times 1$ beam.
+ *Geometry tree* (51--56): `LatticeSDFStruct` inside two `CappedBorderSDF`
  (@fig:library-tree).
+ *Optimiser* (59--66): `param` is the control-point tensor; `target_vol = 0.5`
  (absolute: a quarter of the beam volume 2); bounds $[0.15, 0.75]$; then `MMA`.
+ *Mesh* (74--92): `create_3D_mesh` at `N_base` 10, `mesh_type="volume"`, with the FFD,
  gives tets in physical space; a second, surface mesh only checks watertightness.
+ *Clean the tets* (99--124): reorient negative tets by swapping two vertices, drop tets
  with float64 volume $<= 10^(-12)$; `vol` is the differentiable sum of the rest.
+ *FEM* (128--150): default dtype float64; a `torchfem.solid.Solid` with
  $E = 1000$, $nu = 0.3$; all DOFs fixed at $x < 10^(-5)$; a total force of $-100$ in
  $z$ spread over the unclamped nodes with $z > 0.9$; `spsolve`.
+ *Update* (153--159): $F = f dot u$, $G = V - 0.5$, both gradients by
  `torch.autograd.grad` with respect to `param`, then `optimizer.step`.
+ *Wrap-up* (162--171): dtype back to float32, `sim_out.vtk` written to the working
  directory, both gradients asserted non-zero. One iteration (`num_iter=1`).

```bash
# from the repo root; writes sim_out.vtk into the current directory
uv run pytest DeepSDFStruct/tests/test_structural_optimization.py -v
```

#figure(
  diagram(
    spacing: (7mm, 9mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nsym((0, 0), [`MMA.step` \ `(F, dF, G, dG)`]),
    ndat((1, 0), [`param` $(15, 1)$ \ $hat(lambda)$]),
    nsym((2, 0), [`SplineParametrization` \ $lambda(bold(x))$ per grid point]),
    nsym((3, 0), [`LatticeSDFStruct`, \ `SDFfromDeepSDF`, \ `CappedBorderSDF` $times 2$]),
    nsym((3, 1), [`create_3D_mesh` \ FlexiCubes + FFD, \ `tet_signed_vol` cleanup]),
    next((2, 1), [`torchfem.solid` \ `.Solid`, `solve`]),
    ncon((1, 1), [$F = f dot u$ \ $G = V - 0.5$]),
    nsym((0, 1), [`torch.autograd` \ `.grad`]),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (2, 0), "-|>"),
    edge((2, 0), (3, 0), "-|>"),
    edge((3, 0), (3, 1), "-|>", lbl[SDF on \ the grid], label-side: left),
    edge((3, 1), (2, 1), "-|>"),
    edge((2, 1), (1, 1), "-|>"),
    edge((1, 1), (0, 1), "-|>"),
    edge((0, 1), (0, 0), "-|>", lbl[`dF`, `dG`], label-side: left),
    edge((1, 1), (1, 0), "--|>", stroke: c-grad, lbl[gradient through \ every box],
      label-side: right),
  ),
  caption: [One iteration of the template with the real names. `MMA.step` writes the new
    design into `param` in place; the pink arrow is the autograd path from $F$ and $G$
    back through FEM, FFD, FlexiCubes, the caps, the decoder and the spline basis.],
) <fig:library-loop>

For a real run, change at least: the iteration count and a stopping rule; a decoder with
matching bounds (`np.tile` for $d > 1$; the test's `np.zeros(param.shape) + [lo, hi]`
works only because $d = 1$); a normalised $G$; the volume path (box above); output per
iteration. If anything raises between lines 128 and 162, float64 stays the default
dtype for the whole process. The recipe is in @ch:workflows[Chapter].

== Library-level traps

Besides the boxes above (volume cavities, analytic `AnalyticRoundCross`, decoder
device, shared `latvec`, unregistered children), two more:

- *Silent mismatch.* Weights load with `strict=False`: a `specs.json` that does not fit
  them leaves layers random, without an error.
- *Where to run.* `uv` from the repo root only: `DeepSDFStruct/` has its own
  `pyproject.toml` and `uv.lock`, and `uv run` inside it builds a second environment.

The trainer's own traps (seed order, hardcoded clipping, lost codes on resume,
`latest.pth` lag, ...) are in the table at the end of @ch:training[Chapter].
