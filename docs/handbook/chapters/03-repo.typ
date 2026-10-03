#import "../template.typ": *

= The repository <ch:repo>

This chapter is the map: what lives where, the one rule that keeps the folders apart, how
the packages are installed and why, where generated files land, and what state git is in.

== The map

The repo root is the VS Code workspace. You edit code in four folders: `structsept/`,
`datagen/`, `experiments/` and `tests/`. `DeepSDFStruct/` is the library, a git submodule.
Everything in `data/`, `runs/` and `outputs/` is generated and not versioned.

```text
code/                           repo root = VS Code workspace
├── pyproject.toml  uv.lock     one distribution "struct-sept"; the locked environment
├── CLAUDE.md                   briefing loaded into every Claude Code session
├── install_shortcuts.bat       fresh clone -> uv sync -> two desktop shortcuts
├── structsept/                 LIBRARY: importable, nothing runs on import
│   ├── plate_with_hole.py        recurring test geometry + ScaledSpaceSDF
│   ├── fem.py                    SDF -> tetrahedral mesh -> global stiffness K
│   ├── pointcloud_sdf.py         SDF of a point cloud with normals
│   └── app/                      the GUI and the training drivers
│       ├── main.py  sdf_maker.py                            the two windows
│       ├── tab_explore.py  tab_explore2d.py  tab_train.py   the three tabs
│       ├── hyperparams.py  hparam_window.py  xlsx.py        specs.json schema, sheets
│       ├── training.py  unattended.py                       trainer wrapper; headless runs
│       ├── datasets.py  models.py  viz.py                   dataset list, lattice, drawing
│       ├── launcher.py  icons/                              desktop shortcut start-up
│       └── runtime.py  widgets.py  theme.py  run_editor.py  GUI plumbing
├── datagen/                    TRAINING DATA: writes data/, imports nothing here
│   ├── plate_hole_params.py  plate_hole_sdf.py    circular hole family
│   ├── plate_tri_params.py   plate_tri_sdf.py     four triangles (untracked)
│   ├── make_plate_hole.py    make_plate_tri.py    CLIs: parameters -> data/
│   ├── dataset.py                THE on-disk contract of a dataset
│   └── preview.py                preview.png of a written dataset
├── experiments/                RUNNABLE ONE-OFFS: never imported
│   ├── plate_geometry.py  plate_with_hole_network.py           early studies of
│   ├── plate_with_hole_stiffness.py  pointcloud_to_lattice.py  the library
│   ├── make_plate_presets.py     untrained preset runs -> runs/
│   ├── train_plate_*.py          four unattended training scripts
│   └── IDEIAS.md                 lab notebook (Portuguese): every finding
├── tests/                      pytest; test_deepsdfstruct_env.py = smoke test
├── docs/                       background documents + this handbook
├── DeepSDFStruct/              git submodule; the package is DeepSDFStruct/DeepSDFStruct/
├── data/  runs/  outputs/      generated, gitignored
└── .vscode/  .claude/          F5 configs + interpreter; the /update-context command
```

Two things surprise people. `DeepSDFStruct/` is the submodule checkout and the Python
package sits one level down, so library paths in this handbook start with
`DeepSDFStruct/DeepSDFStruct/`. And `structsept/app/` is more than the GUI: `training.py`
and `unattended.py` drive the trainer for the headless scripts too
(@ch:training[Chapter]). The gitignored files in the root (a copy of the paper and its
text `.paper_text.txt`, `np.py`, `round_cross_00000.npz`, `struct_sept.egg-info/`) are
not used by any code.

== The one import rule

Imports point one way and never bend back (@fig:repo-imports):

- `experiments/` imports `structsept/`, which imports `DeepSDFStruct` (experiments may also
  use the library directly). `structsept/` never imports from `experiments/`, and
  experiment scripts never import each other, not even for a small helper.
- `datagen/` and `structsept/` never import each other. `datagen` writes `data/`, and the
  readers (Train tab, training scripts, library trainer) know only the file layout defined
  in `datagen/dataset.py` (@ch:datagen[Chapter]). Any generator that writes this layout is
  a valid data source; the SDF maker window writes it from meshes.
- `tests/` may import anything except `experiments/`. The tests are where `datagen` and
  `structsept.app` meet: `test_datagen.py` and `test_app_explore2d.py` write a tiny dataset
  and train a decoder on it.

#figure(
  diagram(
    spacing: (17mm, 8mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nstep((0, 0), [`experiments/`\ runnable one-offs,\
      #text(size: 7.4pt, fill: c-muted)[never import each other]]),
    nmod((1, 0), [`structsept/`\ `fem`, `plate_with_hole`,\ `pointcloud_sdf`]),
    nmod((1, 1), [`structsept.app`\ GUI, `training`,\ `unattended`]),
    next((2, 0.5), [`DeepSDFStruct`\ (submodule)]),
    nstep((0, 2), [`datagen/`\ plate families]),
    ndat((1, 2), [`data/`\ `SdfSamples/`, `splits/`]),
    edge((0, 0), (1, 0), "-|>", lbl[imports]),
    edge((0, 0), (1, 1), "-|>"),
    edge((1, 0), (2, 0.5), "-|>"),
    edge((1, 1), (2, 0.5), "-|>"),
    edge((0, 2), (1, 2), "-|>", lbl[writes]),
    edge((1, 1), (1, 2), "-|>", lbl[reads], label-side: left),
    edge((2, 0.5), (1, 2), "-|>", lbl[trainer reads\ (paths in\ `specs.json`)],
      label-side: left),
    edge((1, 0), (0, 0), "--|>", stroke: rgb("#c2410c"), bend: -45deg, lbl[never]),
    edge((0, 2), (1, 1), "--", stroke: rgb("#c2410c"), lbl[no imports]),
  ),
  caption: [The import rule; an arrow means "depends on", red dashed means forbidden.
    `datagen` reaches the rest only through the folder `data/` (which the SDF maker in
    `structsept.app` also writes). Inside `structsept/`, `app` and the three modules do
    not import each other.],
) <fig:repo-imports>

*Why.* An experiment answers one question and is then finished. If library or GUI code
depended on it, editing a one-off would break something unrelated. Keeping `datagen`
behind a folder means the network side never cares how a dataset was made, and the
generators stay free of torch.

*Promote, never import sideways.* When a second file needs a helper that an experiment
script defined, the helper moves into `structsept/`. The precedent: when `train_plate_tri.py` needed the
code-versus-parameter check of `train_plate_r_only_4h.py`, the function moved to
`code_vs_parameter` (#f("structsept/app/unattended.py", 525)) and the r-only script became
a thin wrapper (not committed yet). Nothing enforces the rule automatically; today a
search for imports in the forbidden directions finds none.

== Where does my new file go?

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (1fr, 1.25fr),
    [*If you are writing …*], [*… it belongs in*],
    [a new parametric shape family for training data],
    [`datagen/`: `<family>_params.py`, `<family>_sdf.py`, `make_<family>.py`
      (@ch:datagen[Chapter])],
    [something a second file needs; a geometry, solver, transform, data structure],
    [`structsept/`, or `structsept/app/` if it is about training or the GUI],
    [a GUI tab or window], [`structsept/app/tab_<name>.py` or `<name>_window.py`],
    [a training run left alone for hours],
    [`experiments/train_<...>.py` on top of `structsept.app.unattended`],
    [a figure, mesh or number you look at once], [`experiments/`],
    [an assertion that must stay true],
    [`tests/test_<area>.py`; GUI tests are `test_app_<part>.py`],
    [a result, a number, a decision],
    [`experiments/IDEIAS.md`: `data/`, `runs/`, `outputs/` are not versioned],
    [an explanation of _why_], [`docs/`],
    [a fix or feature of the method itself],
    [`DeepSDFStruct/`, deliberately: it edits the upstream submodule],
  )
]

Unsure? Put it in `experiments/`. Promoting a script later costs a `git mv` and an import
line; untangling a library module that grew around one experiment costs much more.

== Packages and installs

`pyproject.toml` is 31 lines, and each setting fixes a specific problem:

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Setting*], [*What it does, and why*],
    [`name = "struct-sept"`\ #f("pyproject.toml", 2)],
    [One distribution. Its only declared dependency is `DeepSDFStruct`; torch, torch-fem,
      splinepy, pyvista, mlflow and even pytest come in through the library's own list.],
    [`packages = [...]`\ #f("pyproject.toml", 17)],
    [The import packages `structsept`, `structsept.app`, `datagen`. Listed by hand, since
      auto-discovery would trip over the other top-level folders. `experiments/` and
      `tests/` are deliberately not packages.],
    [`package = true`\ #f("pyproject.toml", 22)],
    [Installs this folder editable into `.venv`: `import structsept` and `import datagen`
      work from any directory, with no `sys.path` hacks.],
    [`[tool.uv.sources]`\ #f("pyproject.toml", 30)],
    [`DeepSDFStruct` from the submodule folder, editable too: an edit there is live.],
    [`editable_mode = "compat"`\ #f("pyproject.toml", 25)],
    [Installs `DeepSDFStruct` through a plain path `.pth` instead of setuptools' import
      hook, so the real package beats the same-named checkout folder (below).],
    [`link-mode = "copy"`\ #f("pyproject.toml", 28)],
    [The repo lives under OneDrive, where uv's hardlinks fail (os error 396) when it
      rebuilds the submodule. Copying is slower but works.],
  )
]

*Why compat mode.* Without it, a script run from the repo root would import the checkout
folder `DeepSDFStruct/` (it has no `__init__.py`) as a namespace package: imports of
submodules still work, but the library's own `__init__.py` never runs. A test guards this
case: `test_package_resolves_with_repo_root_on_path` (#f("tests/test_deepsdfstruct_env.py", 64)).

*In practice.* Edits in `structsept/`, `datagen/` or `DeepSDFStruct/` are live at the next
import, and so is a new module inside those packages; only a new _top-level_ package must
be added to `packages`. Run `uv sync` after any change to `pyproject.toml`, `uv.lock` or
the submodule pointer (the desktop launcher re-syncs by itself when `pyproject.toml`,
`uv.lock` or `DeepSDFStruct/pyproject.toml` changed, @ch:gui[Chapter]). Ask
before adding a dependency: the environment is heavy (216 locked packages, torch
`2.13.0+cpu`, so training runs on the CPU) and resolution is slow. Always go through
`uv run`; in VS Code, `.vscode/settings.json` pins `.venv/Scripts/python.exe` and every
F5 configuration runs from the repo root.

The library takes its version from git metadata (setuptools-scm), so the submodule must
be a real checkout: clone with `--recursive`, never download a zip. In short (the full
recipe is in @ch:workflows[Chapter]):

```bash
git clone --recursive -b gui-train-tab https://github.com/tomito6/struct_opt_nn.git code
cd code
uv sync
```

== Where output lands

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr, auto),
    [*Path*], [*Written by*], [*Tracked*],
    [`data/`], [`make_plate_hole`, `make_plate_tri`, the SDF maker window], [no],
    [`runs/`], [Train tab, `train_plate_*.py`, `make_plate_presets.py`], [no],
    [`outputs/logs/`], [unattended scripts (`<run>.log`, `<run>.pid`); the launcher
      (`explorer.log`, `sdf_maker.log`) for shortcut starts], [no],
    [`outputs/`], [`plate_geometry.py`, the `datagen.*_params` CLIs (`--outdir`
      default)], [no],
    [`experiments/outputs/`], [the other three early experiments and the
      `structsept.plate_with_hole` CLI], [no],
    [`docs/stiffness_theory/figures/`], [`make_figures.py` plus renders; used by the
      PDF], [yes],
    [`docs/figures/`], [GUI screenshots (`gui.md` uses two of the three)], [yes],
  )
]

`data/` and `runs/` are located from the code's own path (#f("structsept/app/main.py", 49),
#f("datagen/dataset.py", 48)), so they always land in the repo root. All of it can be
regenerated, which is also why a result only counts once it is written in
`experiments/IDEIAS.md`.

#watch[The `--outdir` defaults (`outputs`, `experiments/outputs`) are relative to the
  _current directory_: run scripts from the repo root, or output lands wherever your shell
  happens to be. And every `runs/<run>/specs.json` stores _absolute_ paths to `data/`, so
  `runs/` does not survive a moved folder or a new machine; rebuild the datasets and rerun
  `experiments/make_plate_presets.py` (@ch:training[Chapter]).]

== Naming conventions

#[
#set par(justify: false)
- *Language.* Folders, modules and GUI strings are English. `IDEIAS.md` has been written
  in Portuguese so far (@ch:status[Chapter] summarises its findings in English); agree
  with the team on the language of new entries.
- *Modules* are named for the object they describe (`plate_with_hole`, `fem`), not an
  action. Experiment scripts read as a sentence about their product:
  `plate_with_hole_stiffness.py` is that plate, taken as far as the stiffness.
- *Data families* come as a triple `<family>_params.py`, `<family>_sdf.py`,
  `make_<family>.py`, with `<family>` = `plate_hole` or `plate_tri`. *Datasets* are
  `<family>_2d_<variant>` (`plate_hole_2d_r_only`, `plate_hole_2d_xyr_n166`,
  `plate_tri_2d_hw`); shapes inside are named by their parameters, with `p` for the decimal
  point: `hole_x0p1200_y0p1200_r0p0700`.
- *Runs.* The GUI names a run `<dataset>_d<d>_<YYYYmmdd_HHMM>`
  (#f("structsept/app/tab_train.py", 716)); scripts name it after the recipe, as in
  `plate2d_r_only_d1_8x256_4h` (latent dimension, layers × width, time budget). Untrained
  recipes start with `preset_plate2d_`; those of `make_plate_presets.py` are
  `preset_plate2d_<variant>_d<d>` (`r_only`, `xyr`, `tri_h`, `tri_hw`). The run name ties together `runs/<run>/`,
  `outputs/logs/<run>.log` and `docs/hyperparameters/NN_Training_Hyperparameters_<run>.xlsx`.
]

== The `docs/` folder

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*File*], [*What it is*],
    [`paper_context.md`], [The paper, distilled: notation-to-code table, transformation
      function, offline/online loop, test cases. Read it before touching an optimization
      loop.],
    [`structure.md`], [The short version of this chapter; partly outdated (see the last
      section).],
    [`gui.md`], [The GUI panel by panel, screenshots in `figures/` (@ch:gui[Chapter]).],
    [`stiffness_theory/`\ `stiffness_theory.pdf`], [Theory note in Portuguese (Typst):
      from the network to the stiffness matrix, tied to `structsept/fem.py`
      (@ch:fem[Chapter]).],
    [`hyperparameters/`], [The supervisor's xlsx template and filled sheets for the three
      `*_8x256_4h` runs; imported by the GUI via `from_sheet`
      (#f("structsept/app/hyperparams.py", 1426)).],
    [`context-maintenance.md`\ `context-log.md`], [The procedure that keeps `CLAUDE.md`
      small and true, and the log of each pass.],
    [`handbook/`], [This handbook (`handbook.typ`, `chapters/`, `build.py`) and the
      companion `board.html`.],
  )
]

The root file `CLAUDE.md` is the briefing that Claude Code loads into every session,
together with the library's `AGENT_INSTRUCTIONS.md`; it is also a fair one-page summary
for humans. It has a budget of 150 lines and is never appended to directly: update it only
through `/update-context` (procedure in `docs/context-maintenance.md`).

== Git state

#figure(
  diagram(
    spacing: (9mm, 6mm), edge-stroke: 0.7pt, mark-scale: 70%,
    ndat((0, 0), [`c13ee3d` 31 Aug\ initial commit]),
    edge("-|>"),
    ndat((1, 0), [`84be646` 21 Sep\ *`main`*]),
    edge("-|>"),
    ndat((2, 0), [4 commits 23--24 Sep\ GUI tabs, `datagen`]),
    edge("-|>"),
    ndat((3, 0), [`6c07cfc` 29 Sep\ *`gui-train-tab`*]),
    edge((3, 0), (3, 1), "--|>"),
    ncon((3, 1), [working tree:\ uncommitted work\ (`git status`)]),
    next((0.5, 1), [submodule pinned at `bbe9881`\ since the first commit]),
  ),
  caption: [The history: `main` is five commits behind `gui-train-tab` (pushed, not
    merged), and the newest work is not committed at all.],
) <fig:repo-git>

The remote `origin` is `github.com/tomito6/struct_opt_nn`, a public repository (hence the
ignored paper PDF). `gui-train-tab` holds everything from 23--29 September: GUI redesign,
`datagen` and 2-D decoders, the radius-only family and presets, launcher and unattended
training. The submodule points at `bbe9881` ("Update Readme (\#77)" on upstream `main`)
and has no local changes.

#watch[A plain `git clone` checks out `main`, which has no `datagen/`, no launcher and only
  six files in `structsept/app/`. Clone with `-b gui-train-tab`, or check it out first.]

#wip[Not committed (2 October 2026; run `git status` for the current list). _Modified_,
  among others: the `--large-holes` option in
  `datagen/make_plate_hole.py` and `plate_hole_params.py`; `prepare_run_like` and
  `code_vs_parameter` in `structsept/app/unattended.py` with the thinned
  `train_plate_r_only_4h.py`; two triangle presets in `make_plate_presets.py`; smaller
  edits to other `datagen/` and `structsept/app/` modules; `IDEIAS.md`,
  `docs/structure.md`, `docs/paper_context.md` and the tests. _Untracked_: the triangle family
  (`datagen/make_plate_tri.py`, `plate_tri_params.py`, `plate_tri_sdf.py`, `preview.py`),
  `experiments/train_plate_tri.py`, `experiments/train_plate_xyr_n166.py`,
  `docs/handbook/`. Commit before you build on any of it.]

== Docs that have drifted

`CLAUDE.md` predates most of `gui-train-tab`, and `docs/structure.md` is current only in
part. Where they disagree with the code (the git history, the contents of
`structsept/app/` and `datagen/`, the installed packages, where the paper PDF is), trust
the code and this handbook. `CLAUDE.md` is also off on four method
details, each explained where it comes up: the decoder lives on $[-1, 1]^3$ and only the
lattice on the unit cube; an uncapped lattice does close at the borders, the caps just
make the faces flat and exact, so keep them (both @ch:method[Chapter]); this repo
tetrahedralizes with tetgen rather than inside `create_3D_mesh`, and `PointCloudSDF`
needs outward normals (both @ch:fem[Chapter]). And the `check_stiffness` docstring and the
theory note credit the rigid-body test with checking the material $bold(C)$ too; since
$bold(B) bold(u)_"rigid" = bold(0)$, it tests only $bold(B)$ and the assembly. Claude Code
reads `CLAUDE.md` at every start, so it repeats these claims until the file is fixed.
