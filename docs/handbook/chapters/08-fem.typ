#import "../template.typ": *

= From geometry to stiffness <ch:fem>

Before an optimization loop can exist, three things are needed: a *test geometry*
to try ideas on, a way to turn it into a *finite element model*, and (the supervisor
asked for this on 13/09) a way to bring in a *real part given as a point cloud*. The
project's own library `structsept/` has one small module for each:

#kv(
  columns: (auto, 1fr, auto),
  [*Module*], [*What it does*], [*Driven by*],
  [`plate_with_hole.py`],
  [The recurring test geometry, a lattice plate with a circular hole, plus
    `ScaledSpaceSDF`, the bridge between metres and the parametric cube.],
  [stiffness and\ network scripts],
  [`fem.py`],
  [Watertight SDF $->$ tetrahedra $->$ torch-fem `Solid` $->$ global stiffness
    matrix $bold(K)$, plus sanity checks. Writes no files, parses no arguments.],
  [stiffness script],
  [`pointcloud_sdf.py`],
  [Signed distance field of an oriented point cloud, used as ground truth when a
    latent field is fitted to a real part.],
  [`pointcloud_to_lattice.py`],
)

The scripts that drive them live in `experiments/` and follow the import rule of
@ch:repo[Chapter]. Neither the GUI nor `datagen` uses these modules, and no test
under `tests/` covers them; the smoke test only covers the `DeepSDFStruct`
primitives underneath.

#watch[There are two "plates with a hole". `datagen` builds plate-with-hole _shapes_
  as training data for a decoder (@ch:datagen[Chapter]). `structsept.plate_with_hole`
  builds a _lattice_ plate with a hole cut through it, as a test case for the FE
  model. They share a name and nothing else.]

== Three coordinate spaces

Most of the confusion in this code comes from mixing up coordinate systems. Keep
three apart (@fig:fem-spaces). The *network cube* $[-1, 1]^3$ is where the decoder
$f_theta (lambda, bold(x))$ lives; it holds one unit cell. The *parametric cube*
$[0, 1]^3$ holds the whole lattice: `LatticeSDFStruct` maps each cell back into the
network cube with the mirrored triangle wave $T(bold(x))$ (@ch:method[Chapter]).
Every SDF in this chapter, including the hole and the border caps, is a function on
this cube. *Physical metres* appear only when the `TorchSpline` box moves the _mesh
vertices_, inside `create_3D_mesh(..., deformation_function=...)`. The SDF never
sees a metre.

#figure(
  diagram(
    spacing: (26mm, 9mm), edge-stroke: 0.7pt, mark-scale: 70%,
    ncon((0, 0), [*Network cube* $[-1, 1]^3$\ one unit cell\ $f_theta (lambda, bold(x))$]),
    ncon((1, 0), [*Parametric cube* $[0, 1]^3$\ lattice, caps, hole:\
      every SDF here]),
    ncon((2, 0), [*Physical metres*\ $L times W times t$ plate\ material, $bold(K)$ in N/m]),
    edge((1, 0), (0, 0), "-|>", lbl[$T(bold(x))$,\ per cell]),
    edge((1, 0), (2, 0), "-|>", lbl[FFD, mesh\ vertices only]),
    ndat((2, 1), [hole: `CylinderSDF`\ given in metres]),
    nsym((1, 1), [`ScaledSpaceSDF`\ $phi(S bold(xi)) \/ "mean"(S)$]),
    edge((2, 1), (1, 1), "-|>", lbl[wrapped by]),
    edge((1, 1), (1, 0), "-|>", lbl[subtracted\ from lattice], label-side: left),
  ),
  caption: [The three spaces. A shape given in metres has to be pulled back into
    the parametric cube through `ScaledSpaceSDF` before it can meet the lattice.],
) <fig:fem-spaces>

`ScaledSpaceSDF` (#f("structsept/plate_with_hole.py", 43)) evaluates the wrapped
shape at $S bold(xi)$, with $S = "diag"(L, W, t)$, and divides the result by the
mean of $S$ (0.7 for the default $1 times 1 times 0.1$ m plate). The zero level set
is exact, and that is all a mesher looks for. Away from the surface the values are
only approximate distances on a non-cubic plate, because one scalar cannot undo a
stretch that differs per axis. The division keeps them at the right order of
magnitude, which is what FlexiCubes needs to place vertices.

#watch[The mesh resolution `N` (`N_base`, @ch:method[Chapter]) counts FlexiCubes
  cubes per unit cell _in parametric space_ ($N dot "tiling" + 1$ cubes per axis,
  spread over the domain plus a 5 % margin on each side). On the
  $4 times 4 times 1$ plate the 0.1 m thickness gets as many intervals as a 0.25 m
  cell in the plane, and the solid slab (no tiling) gets $11^3$ cubes in total at
  `N = 10`.]

== The test geometry: `plate_with_hole()`

The function (#f("structsept/plate_with_hole.py", 83)) takes the hole radius and
centre and the plate size in metres, plus `tiling`, `latent` and `solid`. The
defaults are a $1 times 1 times 0.1$ m plate with a centred hole of radius 0.25 m,
tiling $5 times 5 times 1$ and latent 0.4. It returns `(plate, deformation)` and
builds the SDF in a fixed order (@fig:fem-tree), where each step has a reason:

#[
#set par(justify: false)
+ *Microtile:* `get_model(PretrainedModels.AnalyticRoundCross)` in `SDFfromDeepSDF`.
  This "decoder" ignores its weights: its `forward` computes three perpendicular
  cylinders of radius $r = lambda$ in closed form. Zero reconstruction error is what a
  test case needs: anything wrong in the mesh or in $bold(K)$ is then the fault of
  the mesher or the FE code, not of the network.
+ *Lattice:* `LatticeSDFStruct(tiling, microtile, Constant([latent]))`, so all cells
  are identical. Swapping `Constant` for `SplineParametrization` is the one change
  that turns this into a design problem. Nothing on the plate/FE path does it yet
  (the GUI's `structsept/app/models.py` builds spline lattices, but has no FE model).
+ *Hole:* `DifferenceSDF` with a `CylinderSDF` from $z = -t$ to $z = 2t$, wrapped in
  `ScaledSpaceSDF`. The overshoot gives a clean through-hole, not a pocket whose end
  cap lands inside the material.
+ *Caps:* `CappedBorderSDF(body)` trims all six faces of the unit cube flush, so
  every strut the border cuts ends in a flat face on the plate surface. Without it
  the surface would still close (`LatticeSDFStruct` is positive outside its box and
  the mesher pads the grid by 5 %), but at an uncontrolled place between grid nodes
  (@ch:method[Chapter]). Hole and caps both only remove material, so their order
  does not matter.
+ *Map to metres:* a `TorchSpline` of `splinepy.helpme.create.box(L, W, t)`, returned
  separately for `create_3D_mesh` and `plot_slice`.
]

#figure(
  diagram(
    spacing: (5mm, 5mm), edge-stroke: 0.7pt, mark-scale: 70%,
    next((0, 0), [`get_model(`\ `AnalyticRoundCross)`]),
    nsym((0, 1), [`SDFfromDeepSDF`\ microtile on $[-1, 1]^3$]),
    nsym((0, 2), [`LatticeSDFStruct`\ `tiling`,\ `Constant([latent])`]),
    ncon((1, 1), [`solid=True`:\ `BoxSDF`, no network]),
    nsym((2, 0), [`CylinderSDF`\ $z$ from $-t$ to $2t$,\ metres]),
    nsym((2, 1), [`ScaledSpaceSDF`\ scale $[L, W, t]$]),
    nsym((1, 3), [`DifferenceSDF`\ lattice minus hole]),
    nsym((1, 4), [`CappedBorderSDF`\ trim 6 faces\ flush]),
    ndat((2, 4), [`plate`\ SDF on $[0, 1]^3$]),
    nsym((3, 3), [`TorchSpline`\ `splinepy` box\ $(L, W, t)$]),
    ndat((3, 4), [`deformation`]),
    edge((0, 0), (0, 1), "-|>"),
    edge((0, 1), (0, 2), "-|>"),
    edge((0, 2), (1, 3), "-|>", lbl[base]),
    edge((1, 1), (1, 3), "--|>", lbl[replaces\ lattice], label-side: left),
    edge((2, 0), (2, 1), "-|>"),
    edge((2, 1), (1, 3), "-|>", lbl[subtract], label-side: left),
    edge((1, 3), (1, 4), "-|>"),
    edge((1, 4), (2, 4), "-|>"),
    edge((3, 3), (3, 4), "-|>"),
  ),
  caption: [The SDF tree built by `plate_with_hole()`, read top to bottom. The two
    outputs on the bottom row are returned as a pair.],
) <fig:fem-tree>

`solid=True` replaces the lattice with `BoxSDF` and loads no network: same hole,
same mesher, same FE code, no lattice. It is the comparison case.
`mesh_and_export(plate, deformation, resolution=14)`
(#f("structsept/plate_with_hole.py", 170)) is the geometry-only consumer: it writes
two SDF slices, a FlexiCubes surface in metres (VTK and STL) and a render, and logs
watertightness, volume and bounding box. The module CLI calls these two functions.

== `fem.py`: from an SDF to $bold(K)$

`structsept/fem.py` is step 5 of the pipeline, cut short right after the global
stiffness matrix exists: no supports, no loads, no solve (@fig:fem-chain). Without
the export, the whole driver is six calls:

```python
plate, deformation = plate_with_hole(tiling=(4, 4, 1), device="cpu")
vertices, tets = tetrahedral_mesh(plate, deformation, resolution=10)  # float32 side
with default_dtype(torch.float64):       # torch-fem side
    solid = build_solid(vertices, tets)  # steel: E = 210e9 Pa, nu = 0.3
    k, K = assemble_stiffness(solid)     # k: (n_e, 12, 12) dense, K: sparse COO
    check_stiffness(K, solid)            # logs the checks, warns if one fails
K_csr = to_scipy(K)                      # SciPy CSR, for saving or a sparse solve
```

#figure(
  diagram(
    spacing: (4mm, 8mm), edge-stroke: 0.7pt, mark-scale: 70%,
    ndat((0, 0), [`plate` SDF\ float32, $[0, 1]^3$]),
    nsym((1, 0), [`create_3D_mesh`\ surface: FlexiCubes\ + FFD to metres]),
    ncon((2, 0), [watertight?]),
    nsym((3, 0), [`tetrahedralize_surface`\ tetgen `pYq`]),
    node((2, -1), text(size: 8pt, fill: rgb("#c2410c"))[no: `RuntimeError`,\
      wrap in `CappedBorderSDF`], stroke: none),
    nsym((3, 1), [`tet_signed_vol`\ flip inverted,\ drop $V_e <= 10^(-12)$]),
    nsym((2, 1), [`build_solid`\ `IsotropicElasticity3D`\ $E = 210$ GPa, $nu = 0.3$]),
    nsym((1, 1), [`solid.k0()`\ $bold(k)_e = V_e bold(B)^T bold(C) bold(B)$\
      $(n_e, 12, 12)$]),
    nsym((0, 1), [`assemble_matrix`\ scatter-add,\ `con` empty]),
    ndat((0, 2), [$bold(K)$: $3n times 3n$\ sparse COO, N/m]),
    nsym((1, 2), [`check_stiffness`\ symmetry,\ diagonal $> 0$,\
      $bold(K) bold(u)_"rigid" = bold(0)$]),
    ndat((2, 2), [driver `export()`\ `.npz`, `.vtk`, `.png`]),
    next((3, 2), [not yet: supports,\ loads, solve, $J$, MMA]),
    node(enclose: ((0, 1), (2, 1), (0, 2), (2, 2)), stroke: (paint: c-muted,
      thickness: 0.6pt, dash: "dashed"), fill: none, inset: 5pt, corner-radius: 6pt),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (2, 0), "-|>"),
    edge((2, 0), (2, -1), "-|>"),
    edge((2, 0), (3, 0), "-|>", text(size: 7.4pt, fill: c-grad)[via numpy:\
      gradient lost]),
    edge((3, 0), (3, 1), "-|>", lbl[float64 / int64,\ CPU], label-side: left),
    edge((3, 1), (2, 1), "-|>"),
    edge((2, 1), (1, 1), "-|>"),
    edge((1, 1), (0, 1), "-|>"),
    edge((0, 1), (0, 2), "-|>"),
    edge((0, 2), (1, 2), "-|>"),
    edge((1, 2), (2, 2), "-|>"),
    edge((2, 2), (3, 2), "--|>"),
  ),
  caption: [The chain in `structsept/fem.py`, read as a snake from the top left.
    Everything in the dashed frame runs under `default_dtype(torch.float64)`. The
    pink label marks where the autograd chain from the latent to $bold(K)$ breaks.],
) <fig:fem-chain>

*Meshing.* `tetrahedral_mesh` (#f("structsept/fem.py", 98)) extracts a FlexiCubes
surface already in metres and raises a `RuntimeError` if it is not watertight. The
library's `tetrahedralize_surface` then hands it to tetgen with the switches `pYq`:
the surface triangles are kept as faces and quality refinement adds interior Steiner
points. Inverted tets get two nodes swapped and tets of volume
$<= 10^(-12)$ m#super[3] are dropped. Both passes do nothing on today's tetgen
output; they guard torch-fem, which needs positive orientation (the element Jacobian
determinant is $6 V_e$) and no zero-volume tets.

*Why not the volume path?* `create_3D_mesh(mesh_type="volume")`, FlexiCubes' own
tetrahedralization, leaves internal cavities. Measured on a plain unit cube, its tets
add up to only 0.68–0.70 of the enclosed volume at every resolution tried, and the
lattice plate loses about as much. tetgen reproduces the surface volume to every printed digit, so
`fem.py` uses it. `CLAUDE.md` still describes `create_3D_mesh` as doing the
tetrahedralization; this project's FE path replaces that step.

#watch[`experiments/plate_geometry.py --volume` writes the cavity mesh while its help
  text calls it "needed for FEM later". Do not feed it to an FE model.
  `DeepSDFStruct/tests/test_structural_optimization.py` uses the same path, so its
  compliance values belong to a body with about 30 % voids.]

*Material and precision.* `build_solid` (#f("structsept/fem.py", 176)) wraps the
nodes and tets in a `torchfem.solid.Solid` with an isotropic linear-elastic material,
steel in SI units by default. With the mesh in metres, $bold(K)$ is in N/m. The decoder and
FlexiCubes run in float32, but torch-fem builds its tensors from torch's _default_
dtype and loses digits in float32 assembly. Hence `default_dtype`
(#f("structsept/fem.py", 48)), a context manager around everything from
`build_solid` on.

*Assembly.* `assemble_stiffness` (#f("structsept/fem.py", 197)) makes two calls.
First `solid.k0()` gives one matrix $bold(k)_e = V_e bold(B)^T bold(C) bold(B)$ of
size $12 times 12$ per tet; one integration point is exact for a linear tet. Then
`solid.assemble_matrix(k, con)` scatter-adds them into $bold(K)$. Its argument
`con`, the list of constrained DOFs, is empty because nothing is pinned yet, so
$bold(K)$ is the free stiffness matrix. That is on purpose.

*Checks.* Without supports $bold(K)$ *must* be singular: a free body can translate
three ways and rotate three ways without straining, so $bold(K)$ has exactly six
zero eigenvalues. Never try to invert it. `check_stiffness`
(#f("structsept/fem.py", 230)) turns this into a test. It logs the relative
asymmetry, the smallest diagonal entry, and
$max |bold(K) bold(U)_"rigid"| \/ max |K_(i j)|$, where `rigid_body_modes`
(#f("structsept/fem.py", 66)) builds the $(3n, 6)$ matrix of three unit translations
and three infinitesimal rotations $bold(omega) times bold(r)$. The rigid-body check
is the strongest of the three. A rotation produces zero strain only if every
element's $bold(B)$ is consistent with its node positions, and the
zero survives assembly only if every $bold(k)_e$ landed on the right DOFs. A wrong
geometry can still give a symmetric $bold(K)$ with a positive diagonal; it cannot
pass the rotation test. It says nothing about the material: $bold(B) bold(u)_"rigid" =
bold(0)$ element by element, so $bold(K) bold(u)_"rigid"$ vanishes for any
$bold(C)$. (The docstring at #f("structsept/fem.py", 237) and the Portuguese theory
note claim the test also exercises $bold(C)$; they are wrong there.)

#incode[Thresholds at #f("structsept/fem.py", 260): asymmetry or residual above
  $10^(-10)$, or a diagonal entry $<= 0$, logs the warning
  `K failed a sanity check -- do not trust it`. It never raises, so read the log.]

#figure(
  grid(columns: (1fr, 1.3fr), gutter: 8pt, align: horizon,
    image("../../stiffness_theory/figures/tets_render.png", width: 100%),
    image("../../stiffness_theory/figures/spectra.png", width: 100%)),
  caption: [Left: tet mesh of the default run; the hole removes the four central
    cells of the $4 times 4 times 1$ lattice. Right (titles in Portuguese):
    eigenvalue magnitudes of one $bold(k)_e$ and the ten smallest of the solid
    plate's $bold(K)$. The six rigid-body modes (shaded) sit at about $10^(-4)$ N/m or
    below, the
    seventh at $8.2 times 10^6$ N/m.],
) <fig:fem-mesh>

The two runs below (resolution 10, steel) are documented in the theory PDF. The
matrices saved in `experiments/outputs/` and `experiments/outputs/solid/` match in
shape and nonzero count. Of the 4,667 lattice nodes, 571 are Steiner points; the
tets add up to 0.020241 m#super[3], the volume the surface encloses.

#kv(
  columns: (auto, auto, auto),
  [*Quantity*], [*Lattice plate (default)*], [*Solid plate (`--solid`)*],
  [Surface triangles], [8,192], [1,456],
  [Nodes / tets], [4,667 / 16,015], [757 / 2,478],
  [$bold(K)$: size, nonzeros], [14,001#super[2], 488,007 (0.25 %)],
  [2,271#super[2], 78,147 (1.5 %)],
  [Asymmetry / rigid residual], [$2.2 times 10^(-16)$ / $3.7 times 10^(-16)$],
  [$3.0 times 10^(-16)$ / $2.9 times 10^(-16)$],
  [Wall time], [about 7 s], [about 5 s],
)

== The experiment ladder

Four scripts walk from "a lattice exists" to "$bold(K)$ exists". They are a
*conceptual progression, not a data pipeline*: no script reads another one's output.
Three of them build the plate themselves; only `ScaledSpaceSDF` and
`plate_with_hole()` are shared. Scripts 2--4 write into `experiments/outputs/`,
script 1 into `outputs/` at the repo root.

#[
#set text(size: 9pt)
#set par(justify: false)
#kv(
  columns: (58mm, 1fr, 49mm),
  [*Script* (in `experiments/`)], [*What it adds*], [*Writes*],
  [*1* `plate_geometry.py`],
  [Lattice plate, no hole. Does not import `structsept`. Tiling $4 times 4 times 1$.
    `--skins` adds face sheets; `--volume` writes the cavity mesh.],
  [`plate_slices.png`, `plate_surface.vtk/.stl`, render],
  [*2* `-m structsept.`\ `plate_with_hole`],
  [Adds the hole. Tiling $5 times 5 times 1$. `--solid` gives the slab; the hole
    and plate flags mirror the function arguments.],
  [`hole_slices.png`, `plate_with_hole.vtk/.stl`, render],
  [*3* `plate_with_hole_network.py`],
  [Swaps in the _trained_ `RoundCross` decoder at latent 0.6. Figures of the domain
    $Omega$, the cell $f_theta$ and both combined, plus a 5-value latent sweep.],
  [`sdf_stages.png`, `latent_sweep.png`, `plate_network.stl`, render],
  [*4* `plate_with_hole_stiffness.py`],
  [Thin driver: `plate_with_hole()`, the `fem` chain, then `export()`. Tiling
    $4 times 4 times 1$, resolution 10, about 7 s.],
  [`stiffness_matrix.npz` (CSR), `plate_with_hole_tets.vtk`,
    `stiffness_sparsity.png`],
)
]

```bash
uv run python experiments/plate_geometry.py
uv run python -m structsept.plate_with_hole
uv run python experiments/plate_with_hole_network.py
uv run python experiments/plate_with_hole_stiffness.py
# K back in Python: scipy.sparse.load_npz("experiments/outputs/stiffness_matrix.npz")
```

For the slab, add `--solid --outdir experiments/outputs/solid` to step 2 or 4, or
it overwrites the lattice results. In VS Code, `.vscode/launch.json` has steps 2
and 4 and the solid slab ready on F5.

#tip[Open the `.vtk`/`.stl` outputs in ParaView, or with pyvista, which is already in
  the environment (start `uv run python` from the repo root):
  ```python
  import pyvista as pv
  pv.read("experiments/outputs/plate_with_hole_tets.vtk").plot()
  ```
]

#watch[
  - Step 4 computes $bold(K)$ for the *analytic* cell of step 2, not for the trained
    network of step 3. There is no flag to switch.
  - The two cells read the latent differently: at $lambda = 0.6$ the analytic cell
    fills 54 % of the unit cube, the trained one 29 %. Compare shapes, not $lambda$.
  - Imports work from anywhere, but outputs land relative to the current directory.
    Run the scripts from the repo root.
]

== Point cloud in, lattice out

The library already fits a latent field to a _watertight mesh_
(`LocalShapesReconstructor`, @ch:library[Chapter]), taking its signed ground truth
from `SDFfromMesh`. A point cloud has no faces, hence no inside and no sign.
`PointCloudSDF` (#f("structsept/pointcloud_sdf.py", 216)) fills that gap as a
drop-in `SDFBase`:

- *Magnitude:* distance to the nearest point from a SciPy `cKDTree`; within 3 median
  point spacings, the distance to that point's tangent plane instead. Point-to-point
  distance is biased outwards by about half a spacing, right at the zero level set.
- *Sign:* the generalized winding number (Barill et al. 2018). Each point is a
  dipole of strength $a_i bold(n)_i$:
  $ w(bold(q)) = 1 / (4 pi) sum_i a_i ((bold(p)_i - bold(q)) dot bold(n)_i) / (||bold(p)_i - bold(q)||^3) $
  is about 1 inside and 0 outside, and $w > 0.5$ means inside. It is a dense
  $O(n_"query" times n_"cloud")$ sum, computed in chunks of 2048 queries over at
  most 20,000 dipoles with rescaled areas.

#watch[The cloud *must carry outward normals*, despite `CLAUDE.md` calling it
  "unoriented". `estimate_normals` (PCA, pointed away from the centroid) is only right
  for roughly star-shaped parts. Winding values near 0.5 mean inconsistent normals;
  do not tune the threshold. `PointCloudSDF` is not differentiable and its KD-tree
  runs on the CPU:
  ground truth, computed once, never inside an optimization loop.]

`sample_cloud_sdf` (#f("structsept/pointcloud_sdf.py", 395)) _constructs_ the
near-surface band instead of querying it: a cloud point moved by $t$ along its normal
has signed distance $t$. The band holds 85 % of the samples with the script's
defaults, so the expensive winding number runs only on the uniform ones.

`experiments/pointcloud_to_lattice.py` runs the whole path (@fig:fem-cloud). By
default it simulates a 20,000-point scan of `DeepSDFStruct/tests/data/cone.stl` and
fits a $3 times 3 times 3$ lattice of the `Primitives` decoder (latent dimension 32)
with a degree-1 latent spline started at the mean trained code. This is
*auto-decoder inference, not an encoder*: the decoder stays frozen and Adam moves
the control points. Logged result: IoU from 15.4 % to 99.5 % in 170 steps (18 s).
The whole run takes 146 s at the default `--resolution 32` and 71 s at 16; the
difference is all mesh export. The saved
`latent_control_points.npz` holds a $(64, 32)$ array, $4^3$ control points of 32
components: exactly the design variables MMA would start from.

#figure(
  diagram(
    spacing: (12mm, 7mm), edge-stroke: 0.7pt, mark-scale: 70%,
    ndat((0, 0), [`cone.stl`, or a scan\ `.ply` `.npz` `.npy`]),
    nsym((1, 0), [`load_target`\ sample a mesh, or read\ a cloud (+ PCA normals)]),
    nsym((2, 0), [`build_struct`\ normalize to $[-1, 1]^3$,\ spline at mean code]),
    nsym((2, 1), [`PointCloudSDF`\ KD-tree $|phi|$,\ winding-number sign]),
    nsym((1, 1), [`sample_cloud_sdf`\ uniform + band $phi = t$]),
    nsym((0, 1), [`fit_samples`\ Adam, clamped L1,\ frozen decoder]),
    ndat((0, 2), [`latent_control_points.npz`\ $(64, 32)$: design variables]),
    ndat((1, 2), [meshes, SDF grid,\ slice figure]),
    next((2, 2), [`report_error`\ IoU, sign, band error,\ before and after the fit]),
    edge((0, 0), (1, 0), "-|>", lbl[points,\ normals]),
    edge((1, 0), (2, 0), "-|>"),
    edge((2, 0), (2, 1), "-|>", lbl[normalized\ points], label-side: left),
    edge((2, 1), (1, 1), "-|>"),
    edge((1, 1), (0, 1), "-|>", lbl[`SampledSDF`]),
    edge((0, 1), (0, 2), "-|>"),
    edge((0, 1), (1, 2), "-|>"),
  ),
  caption: [`experiments/pointcloud_to_lattice.py`. Only `PointCloudSDF` and
    `sample_cloud_sdf` are new; structure, fit and export are the library's.
    Outputs go to `experiments/outputs/pointcloud/`.],
) <fig:fem-cloud>

```bash
uv run python experiments/pointcloud_to_lattice.py                   # cone.stl
uv run python experiments/pointcloud_to_lattice.py --noise 0.01 --validate
uv run python experiments/pointcloud_to_lattice.py --input scan.ply  # real cloud
```

`report_error` (#f("experiments/pointcloud_to_lattice.py", 151)) scores with
occupancy IoU as the headline. Comparing $phi$ deep inside the part is meaningless,
because inside a lattice $phi$ is the distance to the nearest strut, not to the
part's boundary; the third number is therefore restricted to the target's
$|phi| < 0.05$ band. Note `--output-dir` (not `--outdir`) and `--stds` in normalized
$[-1, 1]$ units.

#wip[Open (IDEIAS, point-cloud section): if the target part is solid, "imitate the
  cloud" pushes every cell towards its most solid form, and perhaps the lattice
  should only fit _inside_ the part; this needs the supervisor's answer. Real clouds
  without normals need minimum-spanning-tree orientation, not implemented.]

== The theory document

`docs/stiffness_theory.pdf` (Portuguese; source
`docs/stiffness_theory/stiffness_theory.typ`) walks through the stiffness run
function by function with the theory behind each step and the measured numbers:
SDFs and min/max Booleans, the hole and `ScaledSpaceSDF`, DeepSDF and the analytic
round cross, tiling and the latent spline, caps, FlexiCubes, FFD, tetgen and tet
orientation, linear elasticity and the Voigt matrix $bold(C)$, the weak form, shape
functions and $bold(k)_e = V_e bold(B)^T bold(C) bold(B)$, assembly and sparsity,
the three properties of $bold(K)$ with the spectra, and what is still missing. Read
it next to `fem.py`. Only three of its seven images are scripted: `make_figures.py`
writes `transformation.png`, `sdf_1d.png` and `spectra.png` (plus
`numbers_solid.json`) into the directory given as its only argument. The compile command is in the header of the `.typ` file.

```bash
uv run python docs/stiffness_theory/make_figures.py docs/stiffness_theory/figures
```

== What is missing before an optimization loop

$bold(K)$ exists and passes its checks. The path from there to the paper's loop
(theory in @ch:method[Chapter], template in @ch:library[Chapter], recipe in
@ch:workflows[Chapter]):

+ *Supports:* `solid.constraints[mask, :] = True` on one face removes the six
  rigid-body modes and makes $bold(K)$ invertible.
+ *Loads:* `solid.forces[mask, d] = ...` on the opposite face.
+ *Solve:* `solid.solve(...)` re-assembles with the constraints and solves
  $bold(K) bold(u) = bold(f)$, the cost bottleneck (in the paper, FEM plus
  sensitivities take about 1 min 30 s of the 1 min 50 s per iteration).
+ *Objective and constraint:* compliance $J = bold(f)^T bold(u)$ and volume
  $V = sum_e V_e$ against a target.
+ *Update:* `MMA` moves the `SplineParametrization` control points within bounds
  ($[0.15, 0.75]$ for the analytic cell; for a trained decoder the per-component
  range of its codes, @ch:library[Chapter]), using $partial J \/ partial hat(lambda)$ and
  $partial V \/ partial hat(lambda)$ from autodiff.

A comment at #f("experiments/plate_with_hole_stiffness.py", 167) sketches the first
three steps, and `DeepSDFStruct/tests/test_structural_optimization.py` does all five
on a cantilever. The catch is the gradient. In `fem.py` the surface goes through
gustaf and numpy into tetgen and comes back via `torch.as_tensor`, which cuts the
autograd chain from the latent to $bold(K)$. The library test keeps its gradient by
staying in torch, but on the volume path with its cavities. One way out is to accept
that path. Another is to keep tetgen for the connectivity and carry the
surface-vertex positions, which do carry gradients, to the tet nodes through the
`surface_mesh_indices` map that `tetrahedralize_surface` returns and `fem.py`
discards; the interior Steiner points would then need a rule of their own.

#wip[None of the five steps exists in `structsept` yet: IDEIAS queue item 9 ("tet
  mesh + torch-fem") stops at $bold(K)$, and the graded lattice (item 8) is still
  open. A mesh that is both
  differentiable and free of cavities is the open design question of this part. The
  modules themselves are committed and unchanged in the working tree.]
