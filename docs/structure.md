# Repository structure

What lives where, and — more useful — **how to decide where a new file goes**.

## The one rule

```
experiments/  ──imports──▶  structsept/  ──imports──▶  DeepSDFStruct/
```

Arrows point one way and never bend back:

- `structsept/` **must not** import from `experiments/`.
- `experiments/` files **must not** import each other.
- Nothing in this repo modifies `DeepSDFStruct/`; it is an upstream submodule.

The rule has a practical consequence you can act on: the moment a second caller
needs something that lives in `experiments/`, that something **moves into
`structsept/`**. You never fix it by importing sideways.

Training data sits outside that chain on purpose:

```
datagen/  ──writes──▶  data/  ◀──reads──  structsept.app (Train tab)
```

`datagen/` draws parametric shapes and samples their signed distance;
`structsept/` is about the network and assumes the samples exist (its one
sampler, the app's SDF maker, turns ready-made meshes into a dataset). Neither
imports the other — the file layout under `data/` is the whole interface,
documented in `datagen/dataset.py`.

## Where does my new file go?

| If the thing you are writing… | …it belongs in |
|---|---|
| generates a parametric SDF training set or the parameters behind it | `datagen/` |
| will be called by more than one other file | `structsept/` |
| is a geometry, a solver, a data structure, a transform | `structsept/` |
| produces a figure, a mesh, a number you want to look at once | `experiments/` |
| answers one question and is then finished | `experiments/` |
| asserts that something is true and must stay true | `tests/` |
| explains *why* rather than *what* | `docs/` |

Unsure? Put it in `experiments/`. Promoting a file later is one `git mv` plus
an import line; demoting a tangled library module is not.

## The tree

```
code/                        <- VS Code workspace root
│
├── structsept/              LIBRARY — importable, no side effects on import
│   ├── __init__.py
│   ├── plate_with_hole.py       the recurring test geometry + ScaledSpaceSDF
│   ├── pointcloud_sdf.py        SDF from an unoriented point cloud
│   ├── fem.py                   tetrahedral meshing + stiffness assembly
│   └── app/                     Tkinter applications (two windows)
│       ├── main.py                  shell: Explore, Explore 2-D, Train tabs
│       ├── tab_explore.py           drive f_theta by moving control points
│       ├── tab_explore2d.py         planar decoders: latent map vs parameters
│       ├── tab_train.py             dataset -> decoder, loss curve, run list
│       ├── hparam_window.py         the "All hyperparameters..." window
│       ├── hyperparams.py           every specs.json key: schema, checks, I/O
│       ├── sdf_maker.py             separate window: meshes -> dataset
│       ├── theme.py                 ttk styles, palette, matplotlib rcParams
│       ├── widgets.py               cards, log boxes, control-point grid
│       ├── runtime.py               worker threads, queue, debounce
│       ├── datasets.py              meshes -> SdfSamples dataset  (step 1)
│       ├── training.py              drives the DeepSDF trainer    (step 2)
│       ├── models.py                lattice assembly + evaluation (steps 3-4)
│       └── viz.py                   drawing only; no torch imported here
│
├── datagen/                TRAINING DATA — writes data/, imports no structsept
│   ├── plate_hole_params.py     admissible (x_c, y_c, r) design space
│   ├── plate_hole_sdf.py        exact 2-D/3-D SDF of the plate + its samples
│   ├── dataset.py               the on-disk contract: npz, split, params, manifest
│   └── make_plate_hole.py       CLI: parameters -> samples -> data/
│
├── experiments/            RUNNABLE ONE-OFFS — never imported by anything
│   ├── plate_geometry.py            lattice plate, geometry only
│   ├── plate_with_hole_network.py   the plate driven by the trained decoder
│   ├── plate_with_hole_stiffness.py the plate taken to the stiffness matrix K
│   ├── pointcloud_to_lattice.py     point cloud in, latent field out
│   ├── IDEIAS.md                    Portuguese notebook: queue + findings
│   └── outputs/                     what those four scripts write (gitignored)
│
├── tests/
│   ├── test_deepsdfstruct_env.py    offline smoke test of env + core API
│   ├── test_datagen.py              exact field, file contract, 2-D training
│   ├── test_app_explore.py          Explore tab regressions (off-screen Tk)
│   ├── test_app_explore2d.py        2-D dataset -> Train tab -> Explore 2-D
│   └── test_app_hyperparams.py      hyperparameter schema, trainer, window
│
├── docs/
│   ├── structure.md                 this file
│   ├── gui.md                       the app, panel by panel, tied to the paper
│   ├── paper_context.md             the reference paper, distilled
│   ├── context-maintenance.md       procedure for keeping CLAUDE.md current
│   ├── context-log.md               one entry per maintenance pass
│   └── stiffness_theory/            Typst source + figures -> stiffness_theory.pdf
│
├── DeepSDFStruct/          GIT SUBMODULE — the library the project is built on
│
├── data/  runs/  outputs/           generated, all gitignored
├── CLAUDE.md                        instructions loaded into every AI session
└── pyproject.toml                   deps + the two editable installs
```

## Why there are no `sys.path` hacks any more

`pyproject.toml` installs **two** packages editable into `.venv`:

| Package | Source | Declared by |
|---|---|---|
| `structsept` | this folder | `[tool.uv] package = true` + `[tool.setuptools] packages` |
| `DeepSDFStruct` | the submodule | `[tool.uv.sources]` |

So `import structsept.fem` resolves from any working directory, and a script
in `experiments/` can be run as a path, as a module, or from the VS Code Run
panel without caring where the shell happens to be.

Before this, four files carried `sys.path.insert(...)` lines, one of them with
a **relative** path (`docs/stiffness_theory/make_figures.py`), which meant the
figure build silently only worked when launched from the repo root. All four
are gone.

## Running things

Always through `uv run` — never a bare `python`.

```bash
# experiments: plain scripts
uv run python experiments/plate_geometry.py
uv run python experiments/plate_with_hole_stiffness.py --solid --resolution 8

# library modules that also carry a CLI: run them as modules
uv run python -m structsept.plate_with_hole --hole-radius 0.3

# training data: design space alone, or the whole dataset into data/
uv run python -m datagen.plate_hole_params --n 128 --margin 0.05 --plot
uv run python -m datagen.make_plate_hole --dim 2 --plot

# the GUI: explorer (Explore + Train)
uv run python -m structsept.app.main

# the GUI: dataset builder, a separate window
uv run python -m structsept.app.sdf_maker

# tests
uv run pytest tests/ -v
uv run pytest DeepSDFStruct/tests/test_structural_optimization.py -v
```

A library module having a `if __name__ == "__main__":` block is allowed — it
is a convenient way to eyeball what the module builds. What is *not* allowed is
the reverse direction: importing a file out of `experiments/`.

## Where output lands

| Path | Written by | Tracked? |
|---|---|---|
| `experiments/outputs/` | the four experiment scripts (`--outdir` overrides) | no |
| `outputs/` | older runs of the same scripts | no |
| `data/` | `datagen` and the app's mesh sampler: training sets | no |
| `runs/` | the app: training runs (weights, latent codes, specs) | no |
| `docs/stiffness_theory/figures/` | `make_figures.py` | yes — they go in the PDF |

Everything under the first three is regenerable. If a result matters, write it
down in `experiments/IDEIAS.md` rather than relying on the file surviving.

## Naming conventions

- Folder names, module names and the app's user-facing strings are **English**.
  `IDEIAS.md` stays Portuguese; it is a lab notebook, not an interface.
- Modules are named for the object they describe (`plate_with_hole`, `fem`),
  not for the action (`build_plate`, `run_fem`).
- Experiment scripts read as a sentence about what they produce:
  `plate_with_hole_stiffness.py` = that plate, taken as far as the stiffness.
