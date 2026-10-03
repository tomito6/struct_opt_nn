#import "../template.typ": *

= The project at a glance <ch:idea>

== What the project is

Struct\_Sept is a HiWi project at the Institute of Lightweight Design and Structural
Biomechanics, TU Wien. It builds on one paper: M. Kofler, M. Giritsch, S. Elgeti,
_Structural optimization of lattice structures using deep neural networks as geometry
representation_, Graphical Models 142 (2025) 101307, doi:10.1016/j.gmod.2025.101307.
The method of the paper is implemented in the library `DeepSDFStruct`, written by the
paper's first author and included in this repo as a git submodule. Paper notation and
code names line up closely, so reading one helps with the other.

*The problem.* A lattice structure fills a part with many small, repeated unit cells.
Grading the cells (thick struts where the part carries load, thin ones elsewhere) gives
stiff, light parts that additive manufacturing can build. The classical ways to optimize
such a part either _homogenize_ it, which assumes the cells are much smaller than the
part and periodic, or reduce every cell to beams or shells. Both limit which geometries
can appear.

*The method.* The paper changes how the geometry itself is described:

- *The unit cell is a neural network.* A DeepSDF decoder $f_theta (lambda, bold(x))$
  returns the signed distance from a point $bold(x)$ to the cell surface: negative
  inside, positive outside, zero on the surface. A short latent vector $lambda$ says
  _which_ cell. The decoder is trained on a family of cells, so the design space is
  continuous, low-dimensional, and limited to shapes like the training shapes.
- *The latent vector becomes a field.* Instead of one $lambda$ for the whole part,
  $lambda$ varies over space as a B-spline,
  $lambda(bold(x)) = sum_i phi_i (bold(x)) hat(lambda)_i$. The spline control points
  $hat(lambda)_i$ are the *design variables*. A transformation function $T(bold(x))$
  tiles the unit cell over the domain so that the geometry stays continuous from cell
  to cell.
- *Full-scale FEM in every iteration.* The whole lattice is meshed (FlexiCubes, then a
  freeform deformation, then tetrahedra) and linear elasticity is solved on that mesh.
  There is no homogenization and no periodicity assumption.
- *MMA updates the design.* The Method of Moving Asymptotes moves $hat(lambda)$ to
  minimize the compliance $J$ under a volume constraint $V <= V_"target"$.
- *Everything is differentiable.* The gradient $partial J \/ partial hat(lambda)$ is
  carried back through the whole chain
  $hat(lambda) -> lambda -> f_theta -> Omega -> overline(Omega) -> J$, mesh extraction
  included (in the paper an adjoint sensitivity analysis gives the FEM link, automatic
  differentiation all the others). This is why every stage is written in PyTorch, and why code inside the loop
  must never call `.detach()` or `.item()` or take a detour through NumPy.

@ch:method[Chapter] explains each of these concepts and ties it to the code.

== The pipeline in six steps

The paper splits the method into an _offline_ part, done once per family of unit cells,
and an _online_ part, repeated in every optimization iteration (@fig:idea-pipeline).
Steps 1 and 2 produce a trained decoder. Steps 3 to 6 use it as a geometry generator:
evaluate the latent field, mesh the SDF, simulate, update the design variables.

#figure(
  diagram(
    spacing: (7mm, 9mm), edge-stroke: 0.7pt, mark-scale: 70%,
    // offline: once per family of cells
    nstep((0, 0), [*1 Training data*\ exact SDF samples\ `datagen/`]),
    edge("-|>"),
    nstep((1, 0), [*2 Train decoder*\ one code per shape\ `train_deep_sdf`]),
    edge("-|>"),
    ndat((2, 0), [*trained decoder* $f_theta$\ + learned codes\ `runs/<run>/`]),
    node((3, 0), text(size: 8pt, fill: c-muted)[*offline*\ once per family], stroke: none),
    // online: every iteration
    ncon((0, 1), [$hat(lambda)$: spline\ control points\ = *design variables*]),
    nmod((1, 1), [*3 Network inputs*\ $lambda(bold(x))$ and $T(bold(x))$\
      `SplineParametrization`]),
    nmod((2, 1), [*4 Mesh*\ FlexiCubes, FFD, tets\ `create_3D_mesh`]),
    nmod((3, 1), [*5 FEM*\ $K u = f$, then $J$, $V$\ `torch-fem`]),
    next((3, 2), [*6 MMA update*\ `optimization.MMA`]),
    node((1.5, 1.62), text(size: 8pt, fill: c-muted)[*online*: every iteration],
      stroke: none),
    // forward pass
    edge((2, 0), (2, 1), "-|>", lbl[$f_theta$ evaluated\ on the grid], label-side: left),
    edge((0, 1), (1, 1), "-|>", shift: 3pt),
    edge((1, 1), (2, 1), "-|>", shift: 3pt),
    edge((2, 1), (3, 1), "-|>", shift: 3pt),
    edge((3, 1), (3, 2), "-|>", lbl[$J$, $V$], label-side: left),
    edge((3, 2), (0, 2), (0, 1), "-|>", lbl[new $hat(lambda)$], label-pos: 0.3),
    // backward pass: autodiff
    edge((3, 1), (2, 1), "--|>", stroke: c-grad, shift: 3pt),
    edge((2, 1), (1, 1), "--|>", stroke: c-grad, shift: 3pt),
    edge((1, 1), (0, 1), "--|>", stroke: c-grad, shift: 3pt, label-side: left,
      label-sep: 21pt,
      text(size: 7.4pt, fill: c-grad)[$partial J \/ partial hat(lambda)$]),
  ),
  placement: auto,
  caption: [The method of the paper. Green: the main activity of this repo. Blue: library
    code that this repo calls or partly re-implements. Grey dashed: only in the library.
    Black arrows are the forward pass; the pink dashed arrows are the gradient
    $partial J \/ partial hat(lambda)$, which is carried back through steps 5, 4
    and 3 to the control points, where MMA reads it.],
) <fig:idea-pipeline>

The cost sits in step 5. In the paper's cantilever test case (about 200 000 tetrahedra)
one iteration takes about 1 min 50 s, of which about 1 min 30 s is FEM plus sensitivity
analysis. When you work on something unrelated to the physics, reduce the mesh
resolution and the tiling first.

== Where this repo stands

The library covers all six steps. This repo uses them unevenly:

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1.25fr, 1fr),
    table.header([*Paper step*], [*Where it lives*], [*State in this repo*]),
    [1 Training data],
    [`datagen/` (exact 2-D SDFs of plate families); the SDF maker window samples meshes
      with the library's `SDFSampler` (`structsept/app/sdf_maker.py`, `datasets.py`)],
    [*Main activity.* Seven datasets, all 2-D plates, on the author's machine: `data/`
      is gitignored, so a clone rebuilds them (@ch:workflows[Chapter])],
    [2 Train decoder],
    [library `train_deep_sdf`, wrapped by `structsept.app.training.train` (which fixes
      the seed order); called by the GUI Train tab and `experiments/train_plate_*.py`],
    [*Main activity.* 13 run folders in `runs/` (gitignored too; smoke tests and
      unfinished runs included) plus 6 untrained presets],
    [3 Network inputs],
    [library `SplineParametrization`, `LatticeSDFStruct`; assembled by `build_lattice`
      in `structsept/app/models.py`],
    [Moved by hand in the GUI Explore tab. It lists only 3-D decoders, so in practice
      the library's pretrained ones],
    [4 Mesh],
    [library `create_3D_mesh`, `tetrahedralize_surface`; `structsept/fem.py`],
    [Works: surfaces in the Explore tab, tetrahedra in experiment scripts; never inside
      a loop],
    [5 FEM], [`torch-fem`, driven by `structsept/fem.py`],
    [*Stops at the global stiffness matrix $K$.* No supports, no loads, no solve],
    [6 MMA update], [library `optimization.MMA`],
    [*Not called anywhere in this repo.* Only in the library tests; the full loop is
      `test_structural_optimization.py`],
  )
]

Two points in this table matter for a newcomer.

*The decoders trained here are 2-D shapes, not 3-D unit cells.* The current work is
steps 1 and 2 on planar plate families: a plate with a circular hole and a plate with
four triangular holes. These decoders take 2-D points (`geom_dimension` 2 in
`specs.json`, the recipe file of a training run, @ch:training[Chapter]), and each one describes a _whole plate_, not a cell to be tiled. They cannot be plugged into
`LatticeSDFStruct` and the MMA loop as they are. The GUI views them in a separate
Explore 2-D tab, and the docstring of #f("structsept/app/models.py", 1) says that they
skip the lattice entirely.

*There is no optimization loop in this repo yet.* `structsept/fem.py` stops at $K$ on
purpose: `check_stiffness` verifies symmetry, a positive diagonal and the six rigid-body
modes in the kernel, but nothing applies supports or loads or solves $K u = f$. The only
complete loop is the library test
#f("DeepSDFStruct/tests/test_structural_optimization.py", 21), which is also the
template for a first optimization run (@ch:library[Chapter], @ch:workflows[Chapter]).

#wip[Most of the newest work is on branch `gui-train-tab` (not merged into `main`) or not
  committed at all; @ch:repo[Chapter] (Git state) lists it. Run `git status` before you
  build on any of it.]

== The current research question

The paper treats the latent space as a design space: MMA moves $hat(lambda)$ inside box
bounds, and the library test uses $[0.15, 0.75]$ for every control point. But a DeepSDF
decoder is an _auto-decoder_. It has no encoder and never sees the parameters that
generated its training shapes. Each shape's code is a free variable, optimized together
with the network weights, so the network invents its own coordinates. Before any
optimization, this repo therefore asks: *does the latent space that an auto-decoder
learns on its own recover the parameters that generated the shapes, and how many latent
dimensions does it need for that?* If it does, box bounds on the latent axes have a
physical meaning, and an optimized latent field can be read back as geometry.

The test bed is 2-D plates with exact signed distance functions, chosen because both
their exact SDF and their generating parameters are known, which makes the comparison
clean; how such whole-plate decoders lead back to the paper's 3-D unit cells is still
open (@ch:status[Chapter]). The two families are a hole with parameters
$(x_c, y_c, r)$ and four triangular holes with height $h$ and base width $w$. After
training, the learned codes are compared with the dataset's parameter table `params.csv`
(row $i$ belongs to code $i$). The answer so far: when the latent dimension $d$ equals
the number of parameters, they come back up to an arbitrary affine change of basis
(fully with one or two parameters, partly with three); with fewer dimensions than
parameters the network keeps what dominates the loss (the hole position) and drops the
rest. @ch:status[Chapter] has the numbers, the rules of thumb and the open threads.

== The code bases at a glance

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1.45fr, 1fr),
    [*Folder*], [*What it holds*], [*You go there to*],
    [`DeepSDFStruct/`],
    [Git submodule, the upstream library: SDF classes, lattice, spline parametrization,
      FlexiCubes meshing, FFD, the DeepSDF trainer, MMA, pretrained decoders],
    [use any step of the method; edit only for a change to the library itself],
    [`structsept/`],
    [This project's library: `plate_with_hole.py` (test geometry), `fem.py` (mesh to
      $K$), `pointcloud_sdf.py` (SDF of a point cloud)],
    [reuse geometry, FEM or point-cloud code],
    [`structsept/app/`],
    [The Tkinter GUI (explorer window `main.py` with Explore, Explore 2-D and Train tabs;
      the separate `sdf_maker.py`) and the headless helpers `training.py`,
      `unattended.py`],
    [train and inspect decoders, by hand or overnight],
    [`datagen/`],
    [Generators of the 2-D plate datasets with exact SDFs; writes to `data/`],
    [build or extend a training set],
    [`experiments/`],
    [Runnable one-off scripts (never imported) and `IDEIAS.md`, the lab notebook in
      Portuguese],
    [run an experiment; read why things are the way they are],
    [`tests/`],
    [pytest suite; `test_deepsdfstruct_env.py` is the offline smoke test],
    [check the environment and your changes],
    [`docs/`],
    [This handbook, `paper_context.md`, `structure.md`, `gui.md`, `stiffness_theory/`],
    [read the background],
    [`data/`, `runs/`, `outputs/`],
    [Generated datasets, training runs, logs; all gitignored],
    [find results (they live only on disk)],
  )
]

One rule keeps this legible: `experiments/` imports from `structsept/`, which imports
from `DeepSDFStruct`, never the other way round. `datagen/` imports nothing from the
others; it talks to the rest only through the files it writes into `data/`. Because
results in `data/`, `runs/` and `outputs/` are not versioned, findings are written down
in `experiments/IDEIAS.md`. @ch:repo[Chapter] has the annotated tree, the editable
installs and where a new file goes.

== How to read this handbook

The appendix is for lookup: glossary (paper symbol to code name), file formats, commands,
gotchas. You do not need to read in order; pick a path:

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Your goal*], [*Suggested path*],
    [Understand the method],
    [@ch:method[Ch.], @ch:library[Ch.], @ch:fem[Ch.]; then `docs/paper_context.md` and
      the paper itself (doi:10.1016/j.gmod.2025.101307; the PDF is not in git because the
      repo is public)],
    [Train a decoder on a new family],
    [@ch:datagen[Ch.], @ch:training[Ch.], the recipes in @ch:workflows[Ch.], then
      @ch:status[Ch.] for what earlier runs taught; formats in @app:formats[Appendix]],
    [Run an optimization],
    [@ch:method[Ch.], the template walkthrough in @ch:library[Ch.], what is missing in
      @ch:fem[Ch.], the recipe in @ch:workflows[Ch.]],
    [Work on the GUI],
    [@ch:repo[Ch.], @ch:gui[Ch.], the Train tab internals in @ch:training[Ch.], testing
      conventions in @ch:workflows[Ch.]],
    [Get going on day one],
    [the setup recipe in @ch:workflows[Ch.], @app:commands[Appendix] and
      @app:gotchas[Appendix]],
  )
]

*The companion board.* `docs/handbook/board.html` shows the same material as a zoomable
map, like a Miro board. It is a single file: open it in a browser, no server needed. It
is organized in numbered frames, read in order, with cards inside them and arrows
between the cards. Click a card for its explanation, the files behind it and the cards it
connects to; when the board is opened from its place in the repo, each file has a button
that opens VS Code at the line. A *Tour* button walks through it step by step.

*Conventions.* A file reference such as #f("structsept/fem.py", 197) is relative to the
repo root and points at the definition; library files start with
`DeepSDFStruct/DeepSDFStruct/`. Boxes mark four kinds of side notes: _In the code_ ties
theory to a file, _Watch out_ is a trap that has bitten someone, _Tip_ is a shortcut,
and _In progress_ is work that is unfinished or not committed; in @ch:library[Chapter]
most sections open with a small _Code:_ line naming the class and file instead of an
In-the-code box. `CLAUDE.md` in the repo root is the one-page briefing that the AI coding
assistant (Claude Code) loads in every session; several chapters quote it, and correct it
where it has fallen behind (@ch:repo[Chapter]). Every Python command runs through `uv`,
never bare `python`. To see the project working before you read on:

```bash
# offline smoke test of the environment (downloads nothing)
uv run pytest tests/test_deepsdfstruct_env.py -v
# the explorer window: Explore, Explore 2-D and Train tabs
uv run python -m structsept.app.main
```
