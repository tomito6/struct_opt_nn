#import "../template.typ": *

// Chapter pointer at the end of a gotcha, flush right. When it does not fit, it wraps
// to a line of its own, still flush right: the first fill ends the broken line, the
// empty box keeps the second fill from being trimmed at the start of the new line.
#let see(name) = [#h(0.6em)#h(1fr)#sym.zws#box(width: 0pt)#sym.wj#h(1fr)#sym.wj#box[→ #ref(label(name), supplement: [Chapter])]]

= Glossary and notation <app:glossary>

The paper, the library and this repository name the same things in three ways. The
first table maps the paper's symbols to the code; the second explains the words used in
dataset names, run names, scripts and `experiments/IDEIAS.md`.

== Paper symbols and their names in the code

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr, 1.3fr),
    [*Symbol*], [*Meaning*], [*In the code*],
    [$f_theta (lambda, bold(x))$, $theta$],
    [DeepSDF decoder: signed distance at $bold(x)$ of the shape with code $lambda$;
      $theta$ its weights],
    [`NetworkArch` `deep_sdf_decoder`; weights in `ModelParameters/<epoch>.pth`;
      `get_model` loads a `DeepSDFModel` (not an `nn.Module`), `SDFfromDeepSDF` wraps
      it as an SDF],
    [$lambda in RR^d$, $lambda_j$], [latent code; $lambda_j$ the code of training shape
      $j$],
    [row $j$ of `latent_codes["weight"]` in `LatentCodes/<epoch>.pth`; $j$ = position in
      the split = row $j$ of `params.csv`],
    [$d$], [latent dimension],
    [`CodeLength` (specs), `latent_dim` (hyperparameters, metadata), `_d1`, `_d2`,
      `_d3` in run names],
    [$phi$], [signed distance, $phi < 0$ inside the material],
    [last column of every npz row; `sdf(points)` returns shape `(N, 1)`],
    [$delta$, $c_delta$], [clamp of the training loss],
    [`ClampingDistance`; `clampedL1` also clamps at its own 0.1],
    [$sigma$], [initial spread, $lambda_j tilde.op cal(N)(0, sigma^2 \/ d)$ per
      component],
    [`CodeInitStdDev`; presets and script runs: $sqrt(0.01 d)$],
    [$lambda_"reg"$], [weight of the code-norm penalty (the paper's $sigma$)],
    [`CodeRegularizationLambda`, default `1e-4`, ramped in over 100 epochs],
    [$hat(lambda)_i$], [spline control points: the design variables],
    [parameters of `SplineParametrization`, flattened $x$-fastest, row
      $i + n_x (j + n_y k)$],
    [$lambda(bold(x))$, $phi_i$], [latent field $sum_i phi_i (bold(x)) hat(lambda)_i$,
      B-spline basis $phi_i$],
    [`SplineParametrization` (or `Constant`), evaluated by a `TorchSpline`],
    [$T(bold(x))$], [tiling map, parametric $[0,1]^3$ to decoder cube $[-1,1]^3$, cells
      mirrored], [`LatticeSDFStruct(tiling, microtile, parametrization)`],
    [$overline(Omega)$], [deformed, physical domain (FFD)],
    [a `TorchSpline` as `deformation_function` of `create_3D_mesh`; moves vertices only],
    [$bold(K)$, $bold(u)$, $bold(f)$], [stiffness matrix, displacements, forces],
    [`assemble_stiffness` in `structsept/fem.py`; `solve(...)` in the library template],
    [$J$, $V$, $V_"target"$], [compliance $bold(f)^T bold(u)$; volume and its
      bound],
    [`torch.inner(f.ravel(), u.ravel())`; `G = vol - target_vol`, `target_vol = 0.5`;
      `optimization.MMA(param, bounds)`],
    [$x_c, y_c, r$; $h, w$], [hole centre and radius; triangle height and base width
      (design units)],
    [`params.csv` columns; `u_x, u_y, t` and `t_h, t_w` are the same shape as a point of
      the unit cube],
  )
]

== Project jargon

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Term*], [*Meaning*],
    [run],
    [A folder `runs/<run>/` with a `specs.json`. It counts as _trained_ once
      `ModelParameters/latest.pth` exists, which says nothing about the epochs it holds.],
    [preset],
    [An untrained run: only `specs.json` and `metadata.json` (`"preset": true`). The four
      written by `experiments/make_plate_presets.py` are named
      `preset_plate2d_<variant>_d<d>` (`r_only`, `xyr`, `tri_h`, `tri_hw`).],
    [recipe],
    [A `specs.json` minus `DataSource`, `TrainSplit`, `TestSplit`, `Description`.
      `prepare_run_like` copies a recipe and refuses to train if any other key differs.],
    [family],
    [A parametric shape family with a design space and an exact SDF: `plate_hole` (one
      circular hole) or `plate_tri` (four triangular holes); in `datagen/` the triple
      `<family>_params.py`, `<family>_sdf.py`, `make_<family>.py`.],
    [dataset, class, \ instance],
    [`data/SdfSamples/<dataset>/<class>/<instance>.npz` + `data/splits/<dataset>.json`.
      Datasets are named `<family>_2d_<variant>`; the class is always `plate`; one
      instance (the trainer says _scene_) = one shape = one latent code.],
    [geom dimension],
    [Coordinates per sample, `geom_dimension` 2 or 3, taken from the dataset. Every
      decoder trained here is 2 (a _planar_ decoder of a whole plate, shown in Explore
      2-D, not tileable); every shipped one is 3.],
    [shipped decoder],
    [A pretrained model in `DeepSDFStruct/DeepSDFStruct/trained_models/`, named by
      `PretrainedModels`: `AnalyticRoundCross`, `RoundCross`, `ChiAndCross`,
      `Primitives`, ...],
    [dataset variants],
    [`r_only`, `xyr`, `xyr_n166`, `n25`, `n134` (holes); `tri_h`, `tri_hw` (triangles).
      What each draws and how many shapes: the table in @ch:datagen[Chapter]. The
      _extremes_ of the design box come first in the latent order, the _large holes_ of
      `xyr_n166` last.],
    [design units, \ normalized frame],
    [`params.csv` uses plate units, origin at the lower-left corner. Samples live in
      $[-1,1]^d$: normalized $= 1.8 dot ("design" - "plate centre")$, a 0.1 pad.],
    [margin, ligament],
    [The thinnest material allowed between a hole and an edge or another hole. Holes:
      0.05 for training, 0.1 for the design space; triangles: 0.1.],
    [`8x256`, `4x64`, \ skip],
    [Hidden layers × width (`NetworkSpecs.dims`). The long runs use 8×256 with the skip
      (`latent_in`: where $(lambda, bold(x))$ re-enter) at layer 4, the presets 4×64 with
      the skip at layer 2.],
    [`4h`, unattended],
    [`experiments/train_plate_*.py` wait, train, analyse and log to `outputs/logs/`;
      the two `4h` ones also stop at a deadline (`--max-hours`, default 4).],
    [`ep800`],
    [The 800-epoch $d = 3$ recipe of `runs/plate_hole_2d_xyr_d3_ep800` (that run stopped
      at 680; the finished one is `plate_hole_2d_xyr_d3_20260930_0124`).],
    [smoke test],
    [A 2-epoch throwaway run (`--epochs 2 --run smoke`) before every real launch; also
      the environment check `tests/test_deepsdfstruct_env.py`.],
    [code-vs-parameter],
    [Comparing learned codes with `params.csv`: correlation per component and, for
      $d > 1$, the $R^2$ of an affine fit of each parameter on all components.],
    [trained range, \ latent coverage],
    [Per-component min/max of a run's codes (GUI sliders span it ±10 %; optimization
      bounds must come from it). Coverage: the widest gap between neighbouring codes as
      a share of the range; above 12 % (`viz.GAP_FRACTION`) the GUI shades it.],
  )
]

= On-disk formats <app:formats>

Checked against the files on disk on 2 October 2026. The layout of `data/` is the only
interface between `datagen` and everything downstream (@ch:datagen[Chapter]); a run
folder is the library trainer's layout plus a few files of this repo
(@ch:training[Chapter]). `data/`, `runs/` and `outputs/` are gitignored.

== Datasets: `data/`

#[
#set par(justify: false)
From each name in the split the trainer builds the path
`<DataSource>/SdfSamples/<dataset>/<class>/<instance>.npz`, so `SdfSamples` and
`splits` are fixed names (the tree: @ch:datagen[Chapter]). Writers: `write_instance`,
`write_split`, `write_manifest` (#f("datagen/dataset.py", 138) and below). Instance names carry the parameters with
four decimals and `p` for the point (`hole_x0p1200_y0p1200_r0p0700`,
`tri_h0p0500_w0p1000`). A name in the split without its file makes the trainer log a
warning, then fail when it loads the samples into RAM.
]

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*File, key*], [*Content*],
    [npz `pos`, `neg`],
    [float32 arrays of shape `(n, dim + 1)`, one row per sample: the coordinates, then
      $phi$. `pos` holds $phi >= 0$ (void, boundary included), `neg` $phi < 0$
      (material). Coordinates in $[-1,1]^d$; band samples stray slightly past it (up to
      1.16 seen, the SDF maker's _Validate_ allows 1.2). Every dataset so far: 25~000
      uniform + 25~000 band rows per shape. Legacy keys `pos.npy`, `neg.npy` are also
      read.],
    [`splits/<dataset>.json`], [`{"<dataset>": {"plate": [names]}}`, names in latent
      order.],
    [`params.csv`, holes], [`name, x_c, y_c, r, u_x, u_y, t, r_max, clearance`],
    [`params.csv`, `tri_h`], [`name, h, t, base, wall, arm`],
    [`params.csv`, `tri_hw`], [`name, h, w, t_h, t_w, w_max, wall, arm, side`],
  )
]

In `params.csv` all lengths are design units; join through `name`, not row order.
`u_x, u_y, t` (`t`; `t_h, t_w`) are the same shape as a point of the unit cube the
sampler draws from: the box an optimizer would see. `r_max` and `w_max` are the largest
admissible radius or base there; `clearance`, `wall`, `arm`, `side` are ligament widths.
The trainer ignores the file's inside/outside ratio: it draws `SamplesPerScene / 2` rows
from each array (#f("DeepSDFStruct/DeepSDFStruct/deep_sdf/data.py", 169)).

=== `dataset.json`

#[
#set par(justify: false)
The two oldest datasets, `n25` and `n134`, lack `parameters.varied`, `fixed_centre` and
`parameter_sampling.family`: code that reads a manifest must tolerate missing keys.
]

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Key*], [*Content*],
    [`created_by`, `created`, `command`], [Writer, ISO time, and the exact command line
      that rebuilds the dataset.],
    [`geom_dimension`], [2 or 3. *The only key any consumer reads* (`geom_dimension` in
      `structsept/app/datasets.py`; without a manifest the npz row width decides).],
    [`family`, `dataset`, `class`, \ `n_instances`, `columns`, `sign`, \ `order`],
    [`plate_with_hole` or `plate_with_four_triangles`; name; `plate`; shape count;
      `["x", "y", "phi"]`; the sign rule; "split order = latent code index = params.csv
      row".],
    [`parameters`], [`names`, `varied`, `fixed_centre` (holes), `table`, `units`,
      `shown_to_network: false`.],
    [`geometry`], [Triangles only: `triangles`, `centres`, `shared_height`, `base`,
      `base_ratio`, `tip`.],
    [`design_space`], [Holes: `length`, `width`, `margins` (four), `r_min`. Triangles:
      `size`, `margin`, `h_min`, `h_max` (+ `w_min`, `w_cap`, `w_max`).],
    [`parameter_sampling`], [`family`, `n_requested`, `method`, `seed`, `t_power`,
      `include_extremes`; `n166` adds `large_holes`.],
    [`frame`, `sdf_sampling`], [`pad` 0.1, `scale` 1.8, `mapping` (information only:
      nothing converts back to design units for you); `n_uniform`, `n_band`,
      `hole_fraction`, `stds` (0.05, 0.025), `seeding` (shape $i$ uses
      `default_rng([seed, i])`).],
  )
]

== Runs: `runs/<run>/`

A run folder holds `specs.json`, `metadata.json`, the trainer's files and the scripts'
analysis files, in this order below (the tree: @ch:library[Chapter]); a preset holds only
the first two. `latest.pth` and `Logs.pth` are rewritten every `LogFrequency` epochs; numbered snapshots are kept at the multiples of
`SnapshotFrequency` and at the epochs in `AdditionalSnapshots`.

=== `specs.json`

#[
#set par(justify: false)
Written by `write_specs` via `to_specs` (#f("structsept/app/training.py", 28),
#f("structsept/app/hyperparams.py", 1118)) or copied by `prepare_run_like`; read by
`train_deep_sdf`. Examples from `plate2d_r_only_d1_8x256_4h`.
]

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Key*], [*Meaning (example)*],
    [`DataSource`, `TrainSplit`], [*Absolute* paths of `data/` and of the split.
      `TestSplit` (a copy) and `ReconstructionSplit` (`""`) are not read.],
    [`Description`, `NetworkArch`], [Free text (run editor); `deep_sdf_decoder`.],
    [`NetworkSpecs`], [`dims` (width per hidden layer, `[256]*8`), `latent_in` (skip,
      `[4]`), `dropout` (layer indices), `dropout_prob` (0.2), `norm_layers`,
      `weight_norm`, `xyz_in_all`, `use_tanh`, `latent_dropout`, `geom_dimension` (2).],
    [`CodeLength`, `CodeInitStdDev`, \ `CodeBound`], [$d$ (1); $sigma$ of the initial
      codes (0.1); largest code norm (1.0), applied by `nn.Embedding(max_norm=...)`.],
    [`CodeRegularization`, \ `CodeRegularizationLambda`], [The bool is never read; the
      lambda (`1e-4`) alone sets the penalty.],
    [`NumEpochs`, `LogFrequency`, \ `SnapshotFrequency`, \ `AdditionalSnapshots`],
    [600, 10, 100, `[1]`.],
    [`LearningRateSchedule`], [Two entries, decoder then codes, e.g. `"Type": "Step"`,
      `"Initial": 0.0005`, `"Interval": 150`, `"Factor": 0.5`; types `Step`, `Warmup`,
      `Constant`.],
    [`SamplesPerScene`, \ `ScenesPerBatch`], [4096 points per shape and epoch (must be
      even); 4 shapes per batch.],
    [`ClampingDistance`, \ `LossFunction`], [0.1; `clampedL1` (or `leakyClampedL1`,
      `L1`, `MSE`, `huber`).],
    [`DataLoaderThreads`, `seed`], [0 (workers would re-import the app on Windows); 42,
      applied by `training.train` before the library builds the decoder.],
  )
]

=== `metadata.json`

Written by `write_metadata` (#f("structsept/app/training.py", 185)), read by the runs
table (`list_runs`). Keys: `timestamp` (when written; a run without the file is dated
by its `specs.json` and shown with a `~`), `dataset`, `split_path`, `data_source`,
`latent_dim`, `geom_dimension`, `arch`, `dims`, `epochs` (asked for, not reached),
`final_loss` (from `training_summary.json`, else `null`), `deepsdfstruct_version`.
Presets and scripts add `preset`; at the end the scripts add `last_epoch`,
`wall_seconds` and (`4h` only) `stopped_at_deadline`. The GUI writes the file after a
successful training; a script writes it before training (so the run shows while it
waits) and again at the end, so a killed run keeps `final_loss` = `null`.

=== Files the trainer writes

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*File*], [*Content*],
    [`ModelParameters/*.pth`], [`{"epoch", "model_state_dict"}`; 1.86 MB for 8×256.],
    [`LatentCodes/*.pth`], [`{"epoch", "latent_codes": {"weight": (n_shapes, d)}}`; row
      $i$ = split entry $i$. The latent-field decoders `primitives_cl*` store zeros here
      and their codes under `latent_fields_state_dict`.],
    [`OptimizerParameters/*.pth`], [`{"epoch", "optimizer_state_dict"}`, Adam with two
      groups (decoder, codes); 3.7 MB for 8×256. Only for resuming, which nothing here
      does.],
    [`LatentCodes/` \ `latent_code_data_map.json`], [Which file each code was fitted to,
      written at the start: `data_source`, `sdf_samples_subdir`, `latent_codes` (`latent_index`,
      `relative_npz_filename`, absolute `npz_filename`).],
    [`Logs.pth`, `Logs.png`], [`epoch`; `loss`, one float per step (600 epochs × 10
      steps = 6000); per epoch `learning_rate` (decoder, codes), `timing`,
      `latent_magnitude`, `param_magnitude` (per parameter). The Train tab polls it every
      1.5 s. `Logs.png`, the loss plot, is redrawn at each snapshot.],
    [`training_summary.json`], [`loss` (mean batch loss of the last epoch), `num_epochs`,
      `timestamp`, `host_name`, `device`, `training_duration` (`"1:35:59"`), `data_dir`,
      `version`. Only after the last epoch: a stopped or killed run has none.],
  )
]

=== Analysis files

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*File*], [*Content*],
    [`code_vs_<p>.csv`, `.png`], [`unattended.code_vs_parameter`: columns `name`,
      `<p>`, `code_0`, `code_1`, ...; the older `plate2d_r_only_d1_8x256_4h` has
      `name`, `r`, `code`.],
    [`code_vs_params.*`], [`train_plate_xyr_4h.py`: CSV of parameters, codes and fits;
      JSON with `pearson`, `spearman`, `params_from_codes` (`r2`, `rmse`, `W`),
      `codes_from_params`, `pc_variance_share`, `code_min`, `code_max`.],
    [`latent_coverage.json`], [`train_plate_xyr_n166.py`: `run` and `reference`, each a
      list of `components` with `min`, `max`, `gap_from`, `gap_to`, `gap_share`,
      `shaded`, `shapes_at_the_ends`. Not produced yet.],
  )
]

== Logs: `outputs/logs/`

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*File*], [*Content*],
    [`<run>.log`], [Appended; per launch a header with date, time, run and pid, then
      `HH:MM:SS` lines and the analysis numbers. Tee of stdout and stderr by
      `unattended.open_log` (#f("structsept/app/unattended.py", 73)).],
    [`<run>.pid`], [The process id as text, never removed. Used by `--after` and by
      `taskkill`.],
    [`explorer.log`, `sdf_maker.log`], [Shortcut launches only: interpreter, `uv sync`
      output, tracebacks, the output of a GUI training. Moved to `.log.1` above 2 MB.],
  )
]

= Command cheat sheet <app:commands>

Run everything from the repository root, through `uv run`, never with a bare `python`:
several scripts write to paths relative to the working directory, and `uv` started inside
`DeepSDFStruct/` builds a second environment. Type each command on one line.

== Setup

```powershell
git clone --recursive https://github.com/tomito6/struct_opt_nn.git code
cd code
git checkout gui-train-tab                  # the work since 23/09; main is older
git submodule update --init --recursive     # if cloned without --recursive
uv sync                                     # .venv from pyproject.toml + uv.lock
uv run pytest tests/test_deepsdfstruct_env.py -v      # offline smoke test
.\install_shortcuts.bat                     # or double-click: uv sync + Desktop\NN
```

== Data

Each `dataset.json` stores its rebuild command under `"command"`. All share one stem
(`<module>` is `make_plate_hole` or `make_plate_tri`), plus `--plot` for the preview
PNGs; the extra flags of each dataset are in the table of @ch:datagen[Chapter].
`--dry-run` prints the plan; `--help` lists the rest.

```bash
uv run python -m datagen.<module> --dim 2 --n-uniform 25000 --n-band 25000 --name <ds>
```

== Training

```powershell
uv run python experiments/make_plate_presets.py         # presets; after the datasets
uv run python experiments/train_plate_tri.py --epochs 2 --run smoke   # smoke test
uv run python experiments/train_plate_tri.py --family hw --start-at 23:30
uv run python experiments/train_plate_r_only_4h.py --latent-dim 2 --start-at now
uv run python experiments/train_plate_xyr_4h.py --after outputs/logs/<run>.pid
uv run python experiments/train_plate_xyr_4h.py --check-only --run <run>
uv run python experiments/train_plate_xyr_n166.py --force   # retrain from scratch
uv run python experiments/train_plate_xyr_n166.py --check-only
Get-Content outputs/logs/<run>.log -Wait -Tail 20       # follow a log
taskkill /PID <pid> /F                                  # pid in outputs/logs/<run>.pid
```

To keep a job alive after the terminal closes, start it hidden from PowerShell:

```powershell
$cmd = "run python experiments/train_plate_tri.py --family hw"
Start-Process uv -ArgumentList $cmd -WindowStyle Hidden
```

All four scripts take `--start-at` (`now`, `HH:MM`, or date and time
in quotes), `--after` (a pid or a `.pid` file), `--run`, `--epochs`, `--force`, `--data-root`,
`--runs-dir`, `--log-dir`. The rest differs:

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    [*Flag*], [`train_plate_` \ `r_only_4h`], [`train_plate_` \ `xyr_4h`],
    [`train_plate_` \ `tri`], [`train_plate_` \ `xyr_n166`],
    [`--max-hours`], [default 4], [default 4], [--], [--],
    [`--latent-dim`], [default 1], [default 3], [--], [--],
    [`--decay-interval`], [--], [default epochs/4], [--], [--],
    [`--family`], [--], [--], [`h` (default), `hw`], [--],
    [`--like`], [--], [--], [the family's preset], [`..._xyr_d3_ep800`],
    [`--dataset`], [--], [--], [--], [`..._xyr_n166`],
    [`--check-only`], [--], [yes], [yes], [yes],
    [`--epochs`], [changes the recipe], [changes the recipe], [smoke test only],
    [smoke test only],
  )
]

== GUI

```bash
uv run python -m structsept.app.main               # Explore, Explore 2-D, Train
uv run python -m structsept.app.sdf_maker          # meshes -> SdfSamples dataset
uv run python -m structsept.app.launcher explorer  # as the shortcut: splash first
```

== Experiments: geometry, stiffness, point clouds

```bash
uv run python experiments/plate_geometry.py
uv run python -m structsept.plate_with_hole
uv run python experiments/plate_with_hole_network.py
uv run python experiments/plate_with_hole_stiffness.py
uv run python experiments/pointcloud_to_lattice.py --noise 0.01 --validate
```

What each one shows and writes: @ch:fem[Chapter]; `--help` lists the flags.

== Tests and docs

```bash
uv run pytest tests/ -v                             # all project tests, offline
uv run python tests/test_deepsdfstruct_env.py       # smoke test without pytest
uv run pytest tests/test_datagen.py -v              # includes a 2-epoch training
uv run pytest tests/test_app_hyperparams.py tests/test_app_runs.py -v
uv run pytest tests/test_app_launcher.py tests/test_app_unattended.py -v
uv run pytest tests/test_app_explore.py tests/test_app_explore2d.py -v  # need a display
uv run pytest DeepSDFStruct/tests/test_structural_optimization.py -v   # sim_out.vtk
uv run --project .. pytest tests/test_mesh_functions.py -v   # inside DeepSDFStruct/
uv run --no-project --with typst python docs/handbook/build.py     # handbook.pdf
uv run python docs/stiffness_theory/make_figures.py docs/stiffness_theory/figures
```

The smoke test's pretrained-decoder check is opt-in (PowerShell:
`$env:DEEPSDF_TEST_PRETRAINED = "1"`). The `--project ..` line is for library tests that
open `tests/data` relative to the working directory.

= Gotchas checklist <app:gotchas>

The traps that have cost time in this project, each with the chapter that explains it.

== Environment and repository

1. Run `uv` only from the repository root. Inside `DeepSDFStruct/` it builds a second
  environment from the submodule's `pyproject.toml`, without the root's
  `link-mode = "copy"` (on OneDrive hardlinks fail with os error 396); use
  `uv run --project ..` there. #see("ch:repo")
2. A clone has no data: `data/`, `runs/` and `outputs/` are gitignored. Rebuild every
  dataset from its `command`, then the presets. #see("ch:workflows")
3. `specs.json` and `latent_code_data_map.json` store absolute Windows paths: a run
  folder's data paths break on another machine or after moving the repo, and presets
  must be regenerated per machine. #see("ch:training")
4. `main` predates `datagen`, the GUI tabs and the launcher; the work is on
  `gui-train-tab` plus uncommitted files, so some commands of @app:commands[Appendix]
  are missing on a fresh clone. Clone with `--recursive`: the submodule needs its git
  history for its version number. #see("ch:repo")
5. The geometry and FEM scripts' outputs and the default `--input` of
  `pointcloud_to_lattice.py` are relative to the working directory (the training
  scripts and `datagen` resolve theirs from the repo root). #see("ch:repo")

== Data

6. `--dim` has no default, and the hole family's default names (`plate_hole_2d`,
  `plate_hole_2d_r`) and sample counts (10~000 + 10~000) match no existing dataset:
  pass `--name` and both counts. #see("ch:datagen")
7. Latent index = split order = `params.csv` row; extremes come first, large holes
  last. Join codes and parameters by `name`, never by assumption. #see("ch:datagen")
8. The plate-with-hole SDF is exact only for admissible parameters (every ligament at
  least the margin); for a notch it is only a bound. #see("ch:datagen")
9. The trainer rebalances every shape to half inside, half outside; the file's ratio
  never reaches the network. #see("ch:datagen")

== Training

10. Training always runs on CPU (`device="cpu"` in `training.train`). Two trainings at
  once share every core and both slow down: chain them with `--after`.
  #see("ch:training")
11. Trainer traps, each a row of the trap table: `latest.pth`, the checkpoint everything
  loads, is written only at multiples of `LogFrequency` (leave the frequencies on
  _auto_); `SamplesPerScene` must be even; `clampedL1` clamps at its own 0.1 (use `L1`
  for a wider band); `GradientClipNorm` and the `CodeRegularization` bool are ignored;
  `drop_last` leaves out `n_shapes % ScenesPerBatch` shapes per epoch.
  #see("ch:training")
12. _Trained_ only means `latest.pth` exists, so a dead run loads like a finished one.
  Check the checkpoint epoch (`unattended.last_epoch`): the `n166` run stopped at epoch
  80 of its planned 800. #see("ch:status")
13. There is no resume. `--force` retrains from scratch, and the library's
  `continue_from` would restart the codes from random. #see("ch:training")
14. Reusing a run name rewrites `specs.json` (the scripts also `metadata.json`) and
  nothing else: old snapshots, `Logs.pth` and, in the GUI, `metadata.json` stay until
  replaced. #see("ch:training")
15. `--epochs` is a smoke-test override in `train_plate_tri.py` and
  `train_plate_xyr_n166.py` (it skips the recipe check) but changes the recipe in the
  `4h` scripts. Only the `4h` scripts have a deadline; stopping there loses up to
  `LogFrequency` epochs. #see("ch:training")
16. Under `pythonw` (desktop shortcut) `sys.stderr` is `None` and tqdm crashes the
  trainer. Scripts set `TQDM_DISABLE` and `MPLBACKEND=Agg` before importing torch; the
  GUI forces the `Agg` backend. #see("ch:gui")
17. Auto run names have minute resolution, so two windows training the same dataset in
  the same minute collide, and a window does not see the other's new runs until
  _Refresh_. #see("ch:gui")

== Latent space and analysis

18. The learned latent basis is arbitrary. Judge whether a parameter is encoded by an
  affine fit on all components ($R^2$), never component by component.
  #see("ch:status")
19. The bounds $[0.15, 0.75]$ belong to `AnalyticRoundCross`, which is no network: its
  latent _is_ the strut radius. Trained codes sit near 0 (`r_only`, $d = 1$: $-0.17$ to
  $+0.22$), so take bounds from the trained range. #see("ch:library")
20. The mean of the trained codes can lie in an untrained gap while the min/max box test
  passes. Look at the coverage strip and the nearest-code distance. #see("ch:gui")
21. Every decoder trained here is a planar whole-plate decoder: it cannot be tiled by
  `LatticeSDFStruct` or handed to MMA as it is. #see("ch:idea")

== Geometry, FEM and optimization

22. Three coordinate spaces: decoder cube $[-1,1]^3$, parametric $[0,1]^3$, metres. The
  FFD moves only mesh vertices, so anything sized in metres goes through
  `ScaledSpaceSDF`; wrap the lattice in `CappedBorderSDF` before meshing for FEM.
  #see("ch:fem")
23. `create_3D_mesh(mesh_type="volume")` loses about 30 % of the volume to cavities;
  `fem.py` uses surface + tetgen instead, which breaks the gradient chain. Neither path
  is ready for an optimization loop. #see("ch:method")
24. `LatticeSDFStruct` writes per-point codes into its microtile: never evaluate one
  lattice SDF twice at once, and re-wrap an `SDFfromDeepSDF` after using it in a
  lattice. #see("ch:library")
25. `DeepSDFModel` is not an `nn.Module`: `sdf.to(device)` does not move the decoder,
  and `get_model` picks CUDA when it exists. Pass the device to `create_3D_mesh`.
  #see("ch:library")
26. MMA bounds need shape `(numel, 2)`: use `np.tile([lo, hi], (n, 1))`, the template's
  broadcast works only for $d = 1$. MMA scales only the objective (normalize the
  constraint yourself) and has no stopping rule. #see("ch:library")
27. torch-fem needs float64: call it inside `default_dtype(torch.float64)`. The
  unconstrained $bold(K)$ is singular by design (six rigid-body modes).
  #see("ch:fem")
