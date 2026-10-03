#import "../template.typ": *

= Recipes <ch:workflows>

Seven tasks, each as numbered steps with the exact commands. The other chapters
explain why the steps are what they are; here they are only referenced. Two rules
hold for every command: run it from the repo root `code/` (the folder with
`pyproject.toml`), and run Python only through `uv run`. Every flag shown was checked
against the script's `argparse` code; if `--help` ever disagrees, `--help` wins.

== Set up a fresh clone

`data/`, `runs/` and `outputs/` are gitignored, so a clone has the code but no
datasets, no trained decoders and no presets. Set up the environment first, then the
data.

+ *Environment.* You need `git` and `uv`; uv fetches its own Python (other install
  options: #link("https://docs.astral.sh/uv/")[docs.astral.sh/uv]). The project has
  been developed and run only on Windows 11 with CPython 3.12.7; there is no
  `.python-version`; `uv python pin 3.12` keeps uv on that minor version. The desktop
  shortcuts, `KeepAwake` and the PowerShell commands of this chapter are
  Windows-only: elsewhere `launcher --install` refuses and `KeepAwake` does nothing.
  ```powershell
  winget install --id=astral-sh.uv -e              # Windows; once per machine
  git clone --recursive https://github.com/tomito6/struct_opt_nn.git code
  cd code
  git checkout gui-train-tab                       # work since 23/09, not in main
  git submodule update --init --recursive
  uv sync                                          # .venv, editable installs, CPU torch
  uv run pytest tests/test_deepsdfstruct_env.py -v   # offline smoke test
  ```
  The submodule needs its git history (it reads its version from it), so a zip copy
  does not install. Run `uv` only from `code/`: inside `DeepSDFStruct/` it would build
  a second, unrelated environment.
+ *Desktop shortcuts (optional).* Double-click `install_shortcuts.bat`. It checks for
  `uv` and the submodule, runs `uv sync`, then writes _Lattice explorer_ and _SDF maker_ into
  `Desktop\NN\` (@ch:gui[Chapter]). Run it again after moving the folder: the
  shortcuts hold absolute paths.
+ *Data.* Rebuild the datasets, then the presets. The presets come last because
  their `specs.json` store the absolute path of the dataset split (which is also why
  a `runs/` folder does not work on another machine).
  ```bash
  uv run python -m datagen.make_plate_hole --dim 2 --radius-only --n-uniform 25000 --n-band 25000 --name plate_hole_2d_r_only --plot
  uv run python -m datagen.make_plate_hole --dim 2 --n-uniform 25000 --n-band 25000 --name plate_hole_2d_xyr --plot
  uv run python -m datagen.make_plate_tri --dim 2 --n-uniform 25000 --n-band 25000 --name plate_tri_2d_h --plot
  uv run python -m datagen.make_plate_tri --dim 2 --free-width --n-uniform 25000 --n-band 25000 --name plate_tri_2d_hw --plot
  uv run python experiments/make_plate_presets.py
  ```
  For the 166-shape set, add `--large-holes 32 --large-holes-from 0.25` to the
  second command and name it `plate_hole_2d_xyr_n166`. Each dataset keeps its exact
  command under `"command"` in its `dataset.json`. Always pass `--name` and both
  sample counts: with the hole family's default names (`plate_hole_2d`,
  `plate_hole_2d_r`) the presets do not find the dataset, and the default 10~000 +
  10~000 samples differ from every existing dataset.

#wip[A clone of `gui-train-tab` lacks the triangle family (`datagen/make_plate_tri.py`
  and its modules), `--large-holes`, `unattended.prepare_run_like` and
  `code_vs_parameter`, `experiments/train_plate_tri.py`,
  `experiments/train_plate_xyr_n166.py` and the two triangle presets: all uncommitted
  on the author's machine (2 October 2026). Until they are committed, a clone can
  rebuild only the two hole datasets and their presets.]

== The daily research loop

The research question (@ch:status[Chapter]) is answered by one loop, run again and
again: make shapes whose generating parameters are known, train an auto-decoder that
never sees them, check whether its codes recover them (@fig:workflows-loop).

#figure(
  diagram(
    spacing: (22mm, 9mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nstep((0, 0), [*1 Dataset*\ `datagen` writes `data/`\ incl. `params.csv`]),
    nstep((1, 0), [*2 Recipe*\ preset, sheet\ or `HPARAMS`]),
    nstep((2, 0), [*3 Train*\ Train tab or\ `train_plate_*.py`]),
    nstep((2, 1), [*4 Analyse*\ codes vs `params.csv`\ (linear-fit $R^2$)]),
    nstep((1, 1), [*5 Inspect*\ Explore 2-D tab]),
    nstep((0, 1), [*6 Record*\ `experiments/IDEIAS.md`]),
    edge((0, 0), (1, 0), "-|>", lbl[dataset]),
    edge((1, 0), (2, 0), "-|>", lbl[`specs.json`]),
    edge((2, 0), (2, 1), "-|>", lbl[`runs/<run>/`], label-side: left),
    edge((2, 1), (1, 1), "-|>", lbl[`code_vs_*`]),
    edge((1, 1), (0, 1), "-|>", lbl[finding]),
    edge((0, 1), (0, 0), "--|>", lbl[new shapes,\ other $d$,\ more epochs],
      label-side: left),
  ),
  caption: [The loop behind every result so far. The dashed arrow is the decision a
    finding leads to; the coverage gap on the radius axis, for example, led to the
    166-shape dataset.],
) <fig:workflows-loop>

+ *Dataset* in `data/SdfSamples/<dataset>/` (@ch:datagen[Chapter]). Everything
  below relies on one invariant: split order = latent code index = `params.csv` row.
+ *Recipe* from a preset `runs/preset_*` or a supervisor sheet in
  `docs/hyperparameters/` (both loaded in the Train tab's _All hyperparameters..._
  window), or a script's `HPARAMS` dict or `--like` preset (@ch:training[Chapter]).
  The presets set $d$ to the number of generating parameters.
+ *Train*: runs of minutes in the GUI Train tab, runs of hours headless (recipe
  below), which survives a closed window and keeps Windows awake.
+ *Analyse.* The training scripts end by writing `code_vs_<param>.csv/.png`,
  `code_vs_params.*` or `latent_coverage.json` into the run folder. For a run trained
  in the GUI, rerun only that step: add `--check-only --run <run>` to the triangle,
  free-hole or n166 script. Otherwise call `unattended.code_vs_parameter`
  (#f("structsept/app/unattended.py", 525)) with the run folder, the dataset folder,
  the split and a `params.csv` column; it matches codes to rows by shape name. Judge a
  parameter by the linear-fit $R^2$ over _all_ components, since the learned basis is
  arbitrary: in `plate_tri_2d_hw` no single component is monotonic in $h$, yet the
  fit gives $R^2 = 1.000$.
+ *Inspect.* Double-click the run in the Train tab's run table. A planar run opens in
  Explore 2-D: one slider per latent component, gaps wider than 12 % of an axis
  shaded (@ch:gui[Chapter]).
+ *Record* a section `## <title> (dd/mm)` in `experiments/IDEIAS.md`, the lab
  notebook (Portuguese so far): the request, what was built, the exact command,
  launch time and pid, what you expect, later a `**Resultado**` paragraph with the
  numbers. It is the only versioned record of a result; run folders are gitignored.

== Add a new parametric family

A new family (four independent triangle heights, an elliptic hole, ...) gets the same
three `datagen/` modules as the existing two, then a preset and a training script.
Copy the triangle family: it is the newer one and imports the hole family's frame,
extrusion and sampling settings instead of repeating them.

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*File*], [*What it must provide*],
    [`datagen/<family>_params.py`],
    [The admissible parameter set, a map from the unit cube onto it (`from_unit`,
      `to_unit`), samplers (grid for one parameter, Sobol plus extremes for several),
      and a parameters class with `names`, `rows()`, `save_csv()`. Template:
      `datagen/plate_tri_params.py`.],
    [`datagen/<family>_sdf.py`],
    [The exact signed distance in the $[-1, 1]^d$ frame and
      `sample_instance(rng, dim, frame, ..., config)` returning float32 rows
      `(x, y, [z,] phi)`. Reuse `PlateFrame`, `extrude`, `SamplingConfig` from
      `plate_hole_sdf.py`.],
    [`datagen/make_<family>.py`],
    [The CLI (`--dim` required, `--name`, `--overwrite`, `--plot`, `--dry-run`, sample
      counts); instance $i$ seeded with `default_rng([seed, i])`; all writing through
      `datagen/dataset.py` (`prepare`, `write_instance`, `write_split`,
      `write_manifest` with `geom_dimension` and `command`) plus `save_csv`.
      Template: `main` in `datagen/make_plate_tri.py`.],
    [`tests/test_datagen.py`],
    [Exactness against a brute-force distance to a densely sampled boundary; the
      file layout written through the CLI into `tmp_path`.],
    [`experiments/make_plate_presets.py`],
    [One more `PRESETS` entry (#f("experiments/make_plate_presets.py", 70)): run name,
      dataset, `latent_dim`, description.],
    [`experiments/train_<family>.py`],
    [A copy of `train_plate_tri.py` with its own `FAMILIES` entry (dataset, preset,
      run name, the `params.csv` columns to check) and a `LABELS` entry for each of
      those columns. `prepare_run_like` and `code_vs_parameter` do the rest.],
  )
]

Three rules carry over from the existing families (@ch:datagen[Chapter]). First,
the field is exact only for _admissible_ parameters: the boolean
`max(plate, -hole)` is a true distance only while the ligament rule holds. Second,
latent indices must stay stable: prepend the extremes and append extra shapes at the
end, as `HoleParameters.extended` does. Third, `datagen` imports nothing from the
rest of the repo. Then run the CLI with `--dry-run`, then for real with `--plot`
(look at `preview.png`), rerun `make_plate_presets.py`, smoke-test the training
script, and list the new modules in `datagen/__init__.py` and `docs/structure.md`.

== Train overnight without the GUI

Torch runs on the CPU here, so the 8×256 decoder trains for hours. Long runs go
through the headless scripts in `experiments/`, which run the Train tab's trainer
with the helpers in `structsept/app/unattended.py` (lifecycle in @ch:training[Chapter]).

Which script trains which dataset, with which recipe and how long it took: table in
@ch:training[Chapter]; every flag: @app:commands[Appendix].

+ *Smoke-test first.* Two epochs run the whole lifecycle (specs, training,
  metadata, analysis); the log must end with the analysis numbers and `done`.
  ```bash
  uv run python experiments/train_plate_tri.py --family hw --epochs 2 --run smoke
  ```
  In the triangle and n166 scripts `--epochs` exists only for this; in the `_4h`
  scripts it is the real epoch count, so never forget `--run smoke` there. A missing
  dataset or preset stops the script with the command that builds it; the n166
  script also needs its `--like` run, which exists only where it was trained.
  Delete `runs/smoke/` and `outputs/logs/smoke.*` afterwards.
+ *Launch hidden* (PowerShell), so that closing the terminal does not kill the run.
  `specs.json` is written at once, so the run is in the GUI table (after _Refresh_)
  while it waits.
  ```powershell
  $cmd = "run python experiments/train_plate_tri.py --family hw"
  Start-Process uv -ArgumentList $cmd -WindowStyle Hidden
  ```
+ *Schedule or chain.* `--start-at` takes `now`, `HH:MM` (next occurrence) or
  `"YYYY-MM-DD HH:MM"`; `--after` takes a pid or `.pid` file and waits until that
  process has exited.
  ```bash
  uv run python experiments/train_plate_xyr_4h.py --start-at 23:30
  uv run python experiments/train_plate_tri.py --family hw --after outputs/logs/<run>.pid
  ```
  `<run>` is the name of the first run. Start it before the second: `--after` reads
  the pid file once, and a
  leftover file from an older run holds a dead pid, so the second would start at once.
+ *Follow and stop* (PowerShell). The Train tab's loss curve follows only a run
  started there, so follow a headless run in its log.
  ```powershell
  Get-Content outputs/logs/<run>.log -Wait -Tail 30     # follow the log
  taskkill /PID <pid> /F                                # pid: outputs/logs/<run>.pid
  ```
  The two `_4h` scripts also stop by themselves, cleanly, once `--max-hours` have
  passed, and say so in `metadata.json` (`stopped_at_deadline`). At the end every
  script adds `last_epoch` and `wall_seconds` there and writes its analysis files.

#watch[*One training at a time*: torch uses every core, so two trainings (two
  scripts, or a script and the Train tab) slow each other down; chain them with
  `--after`. *No resume*: `structsept.app.training.train` never passes
  `continue_from`, and `--force` starts from scratch. `plate_hole_2d_xyr_n166_d3_ep800`
  died silently at epoch 80 of 800, and its `latest.pth` loads like a finished one.
  The run table shows it as `80/800` and the explorer pickers as _epoch 80 of 800_
  (uncommitted code); in a script, `unattended.last_epoch(run_dir)` gives the epoch.]

== Compute the stiffness matrix of a geometry

The FEM of this repo stops at the unconstrained global stiffness matrix $K$ on
purpose (@ch:fem[Chapter]).

+ *The demonstration*: a 4×4×1 lattice of the analytic round cross (latent 0.4)
  with a 0.25 m hole, resolution 10 (about 10 FlexiCubes cubes per cell), steel. About 7 s; writes
  `stiffness_matrix.npz`, `plate_with_hole_tets.vtk` and `stiffness_sparsity.png`
  into `experiments/outputs/`.
  ```bash
  uv run python experiments/plate_with_hole_stiffness.py
  uv run python experiments/plate_with_hole_stiffness.py --solid --outdir experiments/outputs/solid
  ```
  Further flags: `--tiling`, `--latent`, `--resolution`, `--hole-radius`,
  `--hole-center`, `--length`, `--width`, `--thickness`, `--youngs-modulus`,
  `--poisson-ratio`, `--no-plot`. The log reports asymmetry, smallest diagonal entry
  and rigid-body residual (`check_stiffness`).
+ *Your own geometry.* The functions of `structsept/fem.py` take any SDF on the
  parametric cube $[0, 1]^3$ together with a `TorchSpline` that maps the cube to
  metres. Wrap the SDF in `CappedBorderSDF`: it closes the surface with flat faces at
  the domain border. A lattice alone also closes (it is positive outside its box),
  but an SDF still negative at the edge of the meshing grid gives an open surface,
  and `tetrahedral_mesh` raises. `plate_with_hole`
  (#f("structsept/plate_with_hole.py", 83)) returns such a pair:
  ```python
  import torch
  from DeepSDFStruct.utils import configure_logging
  from structsept.fem import (assemble_stiffness, build_solid, check_stiffness,
                              default_dtype, tetrahedral_mesh, to_scipy)
  from structsept.plate_with_hole import plate_with_hole

  configure_logging()                             # otherwise the checks are silent
  plate, deformation = plate_with_hole(hole_radius=0.15, tiling=(4, 4, 1), device="cpu")
  vertices, tets = tetrahedral_mesh(plate, deformation, resolution=10)
  with default_dtype(torch.float64):              # torch-fem needs float64
      solid = build_solid(vertices, tets)         # steel, E = 210e9 Pa, nu = 0.3
      k, K = assemble_stiffness(solid)            # k_e (n_el, 12, 12), K sparse
      check_stiffness(K, solid)
  K_csr = to_scipy(K)
  ```

#watch[$K$ is singular by design (six rigid-body modes): pin nodes before solving.
  The demo cell is the _analytic_ round cross, not a trained decoder. And the
  surface-to-tetgen path goes through NumPy: no gradient reaches $K$ from the latent.]

== Start a real optimization run

No optimization loop exists in `structsept/` or `experiments/` yet. The only one is
the library test #f("DeepSDFStruct/tests/test_structural_optimization.py", 21),
walked through in @ch:library[Chapter]. Copy it into `experiments/`, so the
submodule stays untouched, and run the copy from `code/`:

```bash
cp DeepSDFStruct/tests/test_structural_optimization.py experiments/optimize_cantilever.py
uv run python experiments/optimize_cantilever.py
```

Its `__main__` block turns every warning into an error; drop that line, and send
`sim_out.vtk` (written to the current directory) to `experiments/outputs/`. Then
change what defines the problem (line numbers of the original):

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, auto, 1fr),
    [*What*], [*Line*], [*Change, and why*],
    [Decoder], [26],
    [`AnalyticRoundCross` is a formula (latent = radius). For a learned cell use
      `RoundCross` ($d = 1$), `ChiAndCross` ($d = 2$) or `get_model("runs/<run>")`
      for a 3-D decoder trained here.],
    [Domain], [42, 47],
    [`tiling = [2, 1, 1]` and the FFD `box(2, 1, 1)` that sets the physical size.],
    [Control net], [34--38],
    [Degrees `[1, 0, 1]` and the knots give 5×1×3 = 15 control points of length $d$,
      starting at 0.3: the design variables.],
    [Bounds], [65],
    [`[0.15, 0.75]` are radii of the analytic cell. For a trained decoder use the range
      of `model._trained_latent_vectors`, as an array of shape `(numel, 2)`
      (`np.tile([lo, hi], (param.numel(), 1))`; the template's broadcast only works for
      $d = 1$).],
    [Supports, loads], [29--32, \ 133--145],
    [Caps at the clamped (`x0`) and loaded (`z1`) faces; $E = 1000$, $nu = 0.3$;
      nodes at $x approx 0$ fixed; total force $-100$ in $z$ on nodes with $z > 0.9$.],
    [Volume], [63, 156],
    [$G = V - V_"target"$ with an absolute $V_"target" = 0.5$ (domain volume 2). MMA
      normalizes only the objective, so scale $G$ yourself, e.g. $V \/ V_"target" - 1$.],
    [Resolution], [76],
    [$N = 10$ per cell; the paper's cantilever uses 20 (about 200~000 tets, 1 min 50 s
      per iteration).],
    [Iterations], [21],
    [`num_iter=1`; `MMA` has no stopping rule, so add one.],
  )
]

#watch[Between control points and $F$, $G$ everything stays in torch: no `.detach()`,
  no `.item()`, no NumPy, or the gradient silently vanishes. The template meshes with
  `mesh_type="volume"`, which loses about 30 % of the volume to cavities; the
  surface-plus-tetgen path of `fem.py` is accurate but not differentiable. Which one
  an optimization should use is open (@ch:fem[Chapter]).]

The 2-D plate decoders of this repo do not fit this template: they have
`geom_dimension` 2 and each one is a whole plate, not a cell for `LatticeSDFStruct`
to tile; a `--dim 3` plate decoder is still a whole plate. A first run uses a
shipped 3-D decoder, or one trained here on 3-D unit-cell data (SDF maker).

== Testing and conventions

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr, auto),
    [*File in `tests/`*], [*What it covers*], [*Needs*],
    [`test_deepsdfstruct_env.py`],
    [Package import, SDF values, booleans, a gradient, FlexiCubes, lattice tiling;
      pretrained check only with `DEEPSDF_TEST_PRETRAINED=1`], [torch, fast],
    [`test_datagen.py`],
    [Design spaces, exact fields, file layout, overwrite safety, both families],
    [one 2-epoch training],
    [`test_app_unattended.py`],
    [Argument parsing, log tee, pid wait, `prepare_run(_like)`, `code_vs_parameter`],
    [no trainer],
    [`test_app_hyperparams.py`],
    [Schema, validator vs real decoder, sheet import, 4-epoch training, window],
    [torch, Tk],
    [`test_app_runs.py`], [Rename, notes, delete; run lists], [Tk],
    [`test_app_explore.py`], [Explore tab regressions], [torch, Tk],
    [`test_app_explore2d.py`], [Tiny dataset, 2-epoch Train-tab run, Explore 2-D],
    [torch, Tk],
    [`test_app_launcher.py`],
    [Console-less launch (real `pythonw`), env sync, instance lock, `.lnk` files],
    [Windows],
  )
]

Tk tests run hidden off-screen and skip without a display; the tests write into
pytest's temporary folders, never into the real `data/` or `runs/`.

```bash
uv run pytest tests/test_deepsdfstruct_env.py -v   # after any dependency change
uv run pytest tests/ -v                            # everything in this repo
uv run pytest DeepSDFStruct/tests/test_structural_optimization.py -v
```

One sheet-import test in `test_app_hyperparams.py` would fail as written (not run):
it expects 300 epochs from the `xyr_d3` sheet, which says 600 (@ch:status[Chapter]).

Conventions:

#[
  #set par(justify: false)
  - *Always `uv run`, from `code/`*, so the locked environment is the one in use.
  - *Format what you touch*: `uvx black <files>`. Black is not a project dependency;
    `uvx` runs it in a throwaway environment.
  - *Differentiability* between design variables and objective (box above). The GUI
    and the $K$ path of `fem.py` are outside that loop.
  - *Never import from `experiments/`.* When a second caller needs script code, move
    it into `structsept/`; that is how `prepare_run_like` and `code_vs_parameter` got
    into `unattended.py`.
  - *Leave `DeepSDFStruct/` alone* unless the change is meant for upstream, and ask
    before adding a dependency: the environment is heavy already.
  - *Record every result* in `experiments/IDEIAS.md`.
  - *`CLAUDE.md` is curated, not appended to.* It is the context of AI sessions, kept
    under 150 lines by `docs/context-maintenance.md` (run as `/update-context`, each
    pass logged in `docs/context-log.md`).
  - *Start new work from `gui-train-tab`*, not `main` (which has no `datagen/`,
    launcher or 2-D tabs), until the branch is merged; ask before pushing to the
    shared remote.
  - *Commit before a big refactor*: the tree holds much uncommitted work, and a
    refactor on top of it cannot be undone cleanly.
]
