# Struct_Sept — HiWi project, TU Wien

## What this project is

Structural optimization of **lattice structures** where the geometry is represented
**implicitly by a neural network** (DeepSDF), not by a mesh or a density field.
The latent vector of the network varies continuously in space via a B-spline, and
those spline control points *are* the design variables of the optimization.

This folder is the working environment: experiments, scripts and optimization runs
live here. The library itself lives in the `DeepSDFStruct/` submodule.

**Reference paper — this repo implements its method:**
Kofler M., Giritsch M., Elgeti S., *"Structural optimization of lattice structures
using deep neural networks as geometry representation"*, Graphical Models 142 (2025)
101307, doi:10.1016/j.gmod.2025.101307. Institute of Lightweight Design and
Structural Biomechanics, TU Wien. Michael Kofler is also the author of
`DeepSDFStruct`, so paper notation and code names line up closely.

The PDF sits one level **above** this folder (`../Struct_Opt_Neural_Networks.pdf`),
outside the VS Code workspace, so you cannot read it from here. A distilled version
of the method, notation and test cases is in **`docs/paper_context.md`** — read that
before touching anything in the optimization loop.

## Layout

```
code/                        <- VS Code workspace root (this folder)
├── pyproject.toml           <- deps + editable install of structsept and DeepSDFStruct
├── structsept/              <- THIS PROJECT'S LIBRARY. Importable; nothing has to run it
│   ├── plate_with_hole.py   <- the recurring test geometry (+ ScaledSpaceSDF)
│   ├── plate_hole_params.py <- admissible (x_c, y_c, r) design space of that geometry
│   ├── pointcloud_sdf.py    <- SDF from an unoriented point cloud (winding number)
│   ├── fem.py               <- tetrahedral meshing + stiffness assembly (torch-fem)
│   └── app/                 <- two Tkinter windows: explorer (main.py) + sdf_maker.py
├── experiments/             <- RUNNABLE ONE-OFFS. Nothing here may be imported
│   └── IDEIAS.md            <- Portuguese notebook: queue of experiments and findings
├── tests/                   <- pytest; test_deepsdfstruct_env.py is the smoke test
├── docs/                    <- structure.md, paper_context.md, stiffness_theory/
├── outputs/ experiments/outputs/ data/ runs/   <- all generated, all gitignored
└── DeepSDFStruct/           <- git submodule, github.com/mkofler96/DeepSDFStruct
```

**The one structural rule:** `experiments/` imports from `structsept/`, never the
reverse. When a second caller needs something that lives in `experiments/`, move it
into `structsept/` instead of importing across. Both packages are installed editable,
so no file needs a `sys.path` hack and every script runs from any directory.
`docs/structure.md` is the long version of this.

Editing files under `DeepSDFStruct/` edits the upstream submodule. Do that only when
the intent is a change to the library itself; otherwise put new work in `code/`.

## Running things

@DeepSDFStruct/AGENT_INSTRUCTIONS.md

Environment: `uv`, `.venv/` here, CPython 3.12.7 on Windows.

```bash
uv run python experiments/plate_geometry.py     # never plain `python`
uv run python -m structsept.plate_with_hole     # library modules with a CLI
uv run pytest tests/ -v                         # smoke test, offline
uv run pytest DeepSDFStruct/tests/test_structural_optimization.py -v
```

`tests/test_deepsdfstruct_env.py` is the fastest way to confirm the environment still works
after a dependency change. It downloads nothing; the pretrained-model check is opt-in
with `DEEPSDF_TEST_PRETRAINED=1`.

## The pipeline, in the order the code runs it

Offline (done once, pretrained weights already exist):

1. Sample points around the unit-cell geometries, compute signed distances.
2. Train the DeepSDF decoder — a latent vector encodes which unit cell.

Online (every optimizer iteration):

3. **Network inputs** — `SplineParametrization` interpolates the latent vector over
   the domain from the spline control points; the coordinate transformation tiles the
   unit cell.
4. **Mesh generation** — `create_3D_mesh` evaluates the SDF and extracts a surface
   with FlexiCubes (differentiable dual marching cubes), then FFD via `TorchSpline`,
   then tetrahedralization.
5. **Forward simulation** — full-scale FEM with `torch-fem`. No homogenization, no
   scale separation, no periodicity assumed.
6. **Update** — `MMA` (Method of Moving Asymptotes) updates the latent-spline control
   points from gradients obtained by autodiff. FFD control points are **not** in the
   design vector.

`DeepSDFStruct/tests/test_structural_optimization.py` is a complete, runnable
instance of steps 3–6 — the best starting template for a new optimization run.
`structsept/fem.py` is this project's own version of step 5, stopping at `K`.

## Core API

| Concept | Class / function |
|---|---|
| Pretrained decoder | `pretrained_models.get_model(PretrainedModels.AnalyticRoundCross)` |
| Network as an SDF | `SDF.SDFfromDeepSDF` |
| Analytic shapes | `sdf_primitives.*SDF` (Sphere, Box, CornerSpheres, …) |
| Booleans | `SDF.UnionSDF`, `DifferenceSDF`, `Smooth*SDF` |
| Tiled lattice | `lattice_structure.LatticeSDFStruct(tiling, microtile, parametrization)` |
| Latent field | `parametrization.SplineParametrization` (or `Constant` for a fixed one) |
| FFD / deformation | `torch_spline.TorchSpline` |
| Meshing | `mesh.create_3D_mesh`, `mesh.tetrahedralize_surface` |
| Optimizer | `optimization.MMA` |

Every SDF is a `torch.nn.Module` subclassing `SDFBase` and is **callable**:
`sdf(points)` with `points` of shape `(N, 3)` returns `(N, 1)` signed distances.

## Conventions and gotchas

- **Differentiability is the whole point.** Anything inserted into the loop between
  design variables and objective must stay in torch and keep gradients. No `.detach()`,
  no numpy round-trips, no `.item()` inside the loop.
- Negative SDF = inside, positive = outside. Domain of the network is the unit cube;
  the transformation function maps it onto the tiled/deformed domain.
- Raw lattices are **not watertight** at the domain borders. Wrap in
  `SDF.CappedBorderSDF` before meshing, or FEM will fail on a broken mesh.
- FEM is run at full scale every iteration, so it is the cost bottleneck. When
  iterating on something unrelated to the physics, cut the mesh resolution `N` and the
  tiling first.
- Bounds on the latent control points matter — outside the trained latent range the
  decoder produces meaningless geometry. The existing tests use `[0.15, 0.75]`.
- `git log` here is one commit and everything except the submodule is still
  untracked. Commit before large refactors.

## Working style

- Look at `DeepSDFStruct/tests/` before writing new code — most functionality has a
  test showing the intended call pattern.
- When a change touches meshing or FEM, run the smoke test before claiming it works.
- Ask before adding a dependency: the env is heavy already (torch, splinepy, vtk,
  pyvista, mlflow, tetgenpy) and resolution is slow.

## Keeping this file current

`docs/context-maintenance.md` is the procedure for updating this file: what earns a
place here, what gets removed, and the 150-line budget. It runs daily as a scheduled
task, and on demand with `/update-context`. `docs/context-log.md` records each pass.

If you learn something during a session that a future session would need, say so —
but put it through that procedure rather than appending to this file directly.
