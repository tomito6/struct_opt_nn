#import "../template.typ": *

= Training a decoder <ch:training>

Before any optimization can happen, the project needs a trained DeepSDF decoder
$f_theta (lambda, bold(x))$: a network that maps a latent code $lambda$ of length $d$
and a query point $bold(x)$ to a signed distance. @ch:method[Chapter] explains the
auto-decoder idea. This chapter shows how a decoder gets trained in this repository,
what ends up on disk, and how you judge the result.

The training loop is not ours. It is the library function `train_deep_sdf`, and apart
from the data folder it takes nothing but a directory: every setting comes from the
`specs.json` inside it. So
the code in this chapter does three things: write a correct `specs.json`, refuse one that would crash
hours later, and record and analyse what the trainer leaves behind. There are two front
ends, and both call the same three functions of `structsept/app/training.py`
(`write_specs`, `train`, `write_metadata`; the two `--like` scripts copy a `specs.json`
instead of calling `write_specs`): the GUI's *Train* tab for runs you watch,
and the headless scripts `experiments/train_plate_*.py` for runs of hours.

== What a run is

A *run* is a folder `runs/<name>/` that holds a `specs.json`; that is the whole
definition (`training.list_runs`). The app or a script writes `specs.json` and a
`metadata.json`, the trainer fills the folder with checkpoints and logs, and an
analysis step may add result files. A run counts as _trained_ once
`ModelParameters/latest.pth` exists. A _preset_ is a run folder with only the two JSON
files: a recipe waiting to be used.

What the trainer optimises: every training shape $i$ gets its own code $lambda_i$, one
row of a `torch.nn.Embedding` of shape $(N_"shapes", d)$. The codes start as
$cal(N)(0, sigma^2 slash d)$ per component ($sigma$ = `CodeInitStdDev`) and their norm
is capped at `CodeBound`. Adam optimises the decoder weights $theta$ and all codes
together, with one learning-rate schedule per parameter group. For a batch $B$ of
(shape, point) samples at epoch $e$ the loss is

$ cal(L) = 1/abs(B) sum_((i,j) in B) abs("cl"(f_theta (lambda_i, bold(x)_j)) - "cl"(s_(i j)))
  + lambda_"reg" min(1, e/100) dot 1/abs(B) sum_((i,j) in B) norm(lambda_i) $

with $s_(i j)$ the true signed distance and $"cl"$ a clamp to $[-delta, delta]$,
$delta$ = `ClampingDistance`. The network never sees the parameters that generated its
shapes, only points and distances.

== The hyperparameter schema

`structsept/app/hyperparams.py` describes every `specs.json` key the app lets you
change. It contains no Tk code, so tests and scripts import it freely. Each key is one
frozen `Field` (#f("structsept/app/hyperparams.py", 86)) with a key, label, group,
kind (`int`, `float`, `bool`, `choice`, `int_list`, `layers`, `opt_int`, `opt_float`,
`text`), default, range, help text and the `specs.json` key it lands in. `FIELDS`
(#f("structsept/app/hyperparams.py", 292)) holds 36 of them in six `GROUPS`. The
hyperparameter window builds its form from this table, the Train tab its one-line
summary, and the `specs.json` writer and reader their key lists. Adding a
hyperparameter is one `Field` entry plus one line each in `to_specs` and `from_specs`.

#[
  #set text(size: 9.5pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Group*], [*Fields and defaults*],
    [Architecture (9)], [`n_layers` 6, `width` 128, `latent_in` [2] (skip connection),
      `weight_norm` on, `norm_layers` all, `dropout_layers` all, `dropout_prob` 0.2,
      `xyz_in_all` off, `use_tanh` off],
    [Latent codes (5)], [`latent_dim` 1, `code_init_std` 1.0, `code_bound` 1.0,
      `code_reg_lambda` 1e-4, `latent_dropout` off],
    [Loss (2)], [`loss_function` clampedL1, `clamping_distance` 0.1],
    [Learning rate (12)], [six per parameter group (decoder, codes): type `Step`,
      initial 5e-4 / 1e-3, interval 500, factor 0.5, `Warmup` target, length 100],
    [Sampling (2)], [`samples_per_scene` 8000 (half inside, half outside),
      `scenes_per_batch` 10],
    [Budget (6)], [`num_epochs` 200, `seed` 42, `log_frequency` auto,
      `snapshot_frequency` auto, `additional_snapshots` [1], `description`],
  )
]

The defaults copy the shipped `RoundCross` and `ChiAndCross` decoders on a smaller CPU
budget: 8000 samples per shape instead of 16 000, and 200 epochs. So with the defaults
the 500-epoch learning-rate step never fires. And `code_init_std` 1.0 gives each code
component a variance of $1 slash d$, while the paper uses 0.01; the runs trained here
set $sigma = sqrt(0.01 dot d)$, except one early $d = 2$ run
(`plate_hole_2d_n134_d2_20260928_1338`, $sigma = 0.1$). The Train tab card has spinboxes
for four fields: `latent_dim`, `n_layers`, `width`, `num_epochs`. The rest is edited in
the _All hyperparameters..._ window.

=== Validation: error, warning, note

`validate` takes a set, the dataset's shape count and its geometry dimension, and
returns its findings as `Issue` objects, worst first (#f("structsept/app/hyperparams.py", 862)). It
runs while you type in the window, when you press _Train_, inside `write_specs`, and in
`unattended.prepare_run` before a script writes a run. It does not run for the two
`--like` scripts: `prepare_run_like` copies another run's `specs.json` key for key and
checks only that the recipe is unchanged and the geometry dimension matches, not that
the recipe suits the new dataset.

#[
  #set text(size: 9.5pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 0.8fr, 1.4fr),
    [*Level*], [*Meaning*], [*Real examples*],
    [error], [The library crashes, or the run ends with nothing to load. Blocks
      _Apply_, _Train_ and `write_specs`.],
    [skip connection at layer 0 or at `n_layers + 1`; a skip with `width` $<=$
      $d$ + geometry dimension; odd `samples_per_scene`; `log_frequency` longer than
      the run],
    [warning], [Legal, but almost certainly not meant.],
    [batch larger than the dataset; epochs not a multiple of the save interval, so
      `latest.pth` holds an earlier epoch],
    [note], [An effect that is easy to misread.],
    ["Decoder learning rate steps every 500 epochs, so it stays at 0.0005 for the
      whole 200-epoch run"; clampedL1 with $delta > 0.1$; "2 of the 134 shapes sit out
      every epoch"],
  )
]

The errors are not guesses: `tests/test_app_hyperparams.py` builds a working
`DeepSDFDecoder` from a list of sets the validator accepts, checks over a grid of
architectures that every decoder the library rejects is refused, and shows that an odd
sample count really crashes the trainer.

`to_specs` (#f("structsept/app/hyperparams.py", 1118)) turns a set into the
`specs.json` dict. It adds what is not a hyperparameter: `NetworkArch`
`deep_sdf_decoder`, `geom_dimension` (always taken from the dataset, never typed),
absolute `DataSource` and `TrainSplit` paths, and `DataLoaderThreads` 0. It resolves
`auto`: `LogFrequency` is the first of 10, 5, 2, 1 that divides `NumEpochs`, and
`SnapshotFrequency` is `NumEpochs // 4`. `from_specs`
(#f("structsept/app/hyperparams.py", 1193)) reads a run's or a shipped decoder's specs
back key by key and adds a note for every value it drops. It does not carry over
`Description` or `geom_dimension`. What cannot be changed at all is listed in `FIXED`
(#f("structsept/app/hyperparams.py", 624)) and shown read-only in the window: the
architecture, the geometry dimension (from the dataset), Adam, gradient clipping and the regularisation ramp (both hardcoded in
the trainer), the CPU, and `DataLoaderThreads` 0 (a loader process re-imports the
whole app on Windows).

== The supervisor's sheet

The folder `docs/hyperparameters/` holds the supervisor's Excel template for reporting
training settings (one hyperparameter per row, the value in column B) and one filled
copy for each of the three 8×256 "4h" runs (`r_only_d1`, `r_only_d2`, `xyr_d3`). The
window's _Import sheet..._ button reads such a file. `xlsx.read_rows`
(#f("structsept/app/xlsx.py", 30)) parses the workbook with the standard library
(zip and XML), because `openpyxl` is not worth adding to a slow-resolving environment
for one column of values.

`from_sheet` (#f("structsept/app/hyperparams.py", 1426)) maps ten rows directly
(geometries per batch, latent dimension, hidden layers, neurons per layer, dropout,
epochs, both learning rates, clamp value, initial latent regularization; the trainer
ramps that weight up over 100 epochs, while the paper's Eq. 9 prints a decay,
@ch:status[Chapter]). Three rows
need arithmetic, because the sheet counts differently from the trainer. With the sheet
of `plate2d_r_only_d1_8x256_4h`:

- *Points per training step* counts the whole batch. Samples per shape =
  16 384 / 4 geometries per batch = 4096, rounded down to an even number. The paper's
  16 000 (and the shipped decoders' `SamplesPerScene`) count per shape, so the two
  numbers are not comparable (open question, @ch:status[Chapter]).
- *Latent initialization variance* is per component, and the trainer draws from
  $cal(N)(0, sigma^2 slash d)$, so $sigma = sqrt(0.01 dot 1) = 0.1$.
- *Decay factor* 0.5 and *interval* 150 go into both schedules, set to `Step`. (The
  `xyr_d3` sheet has 75 instead; see the box below.)

Rows about the data (number of geometries, samples per geometry, sampling strategy) and
rows the app cannot change (activation, optimizer, mean) are checked and reported,
never applied.

#watch[
  - The template has no row for the skip connection, so an imported set keeps
    `latent_in` [2]; the 8×256 runs used layer 4. To reproduce a run exactly, use
    _Start from > run: ..._ in the window, which reads the run's `specs.json`.
  - The `xyr_d3` sheet says 600 epochs with the decay every 75; the run it documents
    trained 300 epochs with the same decay. @ch:status[Chapter] lists this, and the
    test it breaks, under known issues.
]

== What pressing _Train_ does

The Train tab keeps the full set in its state dict (`st["tr_hparams"]`; the dict is
explained in @ch:gui[Chapter]) and lays the four card
spinboxes over it. @ch:gui[Chapter] explains the worker thread and the queue it relies
on. Pressing _Train_ calls `_start_training` (#f("structsept/app/tab_train.py", 692)):

#figure(
  diagram(
    spacing: (12mm, 4mm), edge-stroke: 0.7pt, mark-scale: 70%,
    node(enclose: ((0, 0), (1, 0), (0, 3), (1, 3), (0, 5), (1, 5)), stroke: (paint: c-muted, thickness: 0.6pt, dash: "dashed"),
      corner-radius: 6pt, inset: 6pt),
    node(enclose: ((3, 0), (3, 4)), stroke: (paint: c-muted, thickness: 0.6pt, dash: "dashed"),
      corner-radius: 6pt, inset: 6pt),
    node((0.5, -0.85), text(size: 8pt, weight: "bold", fill: c-muted)[Tk main thread], stroke: none),
    node((2, -0.85), text(size: 8pt, weight: "bold", fill: c-muted)[run folder], stroke: none),
    node((3, -0.85), text(size: 8pt, weight: "bold", fill: c-muted)[worker thread], stroke: none),
    nui((0, 0), [press _Train_]),
    nsym((1, 0), [`_start_training` \ validate, name, \ overwrite prompt]),
    nsym((3, 0), [`runtime.run_worker`]),
    nsym((3, 1), [`write_specs` \ validates again]),
    nsym((3, 2), [`training.train` \ seeds, then calls]),
    next((3, 3), [`train_deep_sdf` \ library, CPU]),
    nsym((3, 4), [`write_metadata`]),
    ndat((2, 1), [`specs.json`]),
    ndat((2, 3), [`Logs.pth`, \ checkpoints]),
    ndat((2, 4), [`metadata.json`]),
    nsym((1, 3), [`_poll_progress` \ every 1.5 s]),
    nui((0, 3), [loss curve, \ `epoch e/N`]),
    nsym((1, 5), [`_training_done`]),
    nui((0, 5), [`runs_changed`: \ table, pickers]),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (3, 0), "-|>", lbl[no errors]),
    edge((3, 0), (3, 1), "-|>"),
    edge((3, 1), (3, 2), "-|>"),
    edge((3, 2), (3, 3), "-|>"),
    edge((3, 3), (3, 4), "-|>"),
    edge((3, 1), (2, 1), "-|>"),
    edge((3, 3), (2, 3), "-|>"),
    edge((3, 4), (2, 4), "-|>"),
    edge((2, 3), (1, 3), "-|>"),
    edge((1, 3), (0, 3), "-|>"),
    edge((1, 0), (1, 3), "--|>", lbl[starts]),
    edge((3, 4), (3, 5), (1, 5), "-|>", lbl[`on_done`, via the queue], label-pos: 0.75),
    edge((1, 5), (0, 5), "-|>"),
  ),
  caption: [Pressing _Train_. The worker is inside the library trainer for the whole
    run, so the Tk thread reads progress back from `Logs.pth`, which the trainer
    rewrites every `LogFrequency` epochs.],
) <fig:training-press>

+ *Check.* The dataset needs a split file. The set is validated against the dataset's
  shape count and geometry dimension (`datasets.geom_dimension` reads the
  `dataset.json` manifest or the width of the npz rows). On errors a dialog lists
  them and nothing is written.
+ *Name.* An empty name becomes `<dataset>_d<d>_<YYYYmmdd_HHMM>`. An existing
  `specs.json` under that name triggers an overwrite prompt.
+ *Worker.* `runtime.run_worker` starts a daemon thread that makes three calls.
  `write_specs` validates once more and writes the file; the non-default values and
  all issues go to the log. The worker calls `training.train` inside
  `runtime.signals_off()`, because the trainer installs a Ctrl-C handler and that
  raises off the main thread. `write_metadata` records dataset, epochs and final
  loss. The three are at lines 28, 129 and 185 of `structsept/app/training.py`.
+ *Progress.* The trainer reports progress only through a tqdm bar, and the worker is
  stuck inside it. So `_poll_progress` (#f("structsept/app/tab_train.py", 770)) loads
  `Logs.pth` every 1.5 s; the curve moves in steps of `LogFrequency` epochs. It
  follows only the run this window started; a headless run is followed in its log
  file. In the committed code a run that stopped early also looks finished in the
  runs table (the `80/800` display is uncommitted, see the box under Unattended
  training).
+ *Done.* `_training_done` calls `run_editor.runs_changed(st)`, which refreshes the
  runs table and both explorer pickers.

`training.train` also fixes a library bug: it seeds Python, NumPy and torch _before_
it calls the trainer. The library builds the decoder, which draws its initial weights
from the global torch RNG, a few lines before it seeds (lines 409 and 415 of
`deep_sdf/training.py`). In a long-lived GUI process that RNG has been advanced by every
earlier run and every decoder the Explore tab built. Without the fix, two runs with
identical settings start from different weights, and an A/B test of one hyperparameter
silently compares two initialisations too. The fix is `_seed_everything`
(#f("structsept/app/training.py", 172)); `test_same_seed_gives_the_same_network` pins
it. Any new code should therefore call `training.train`, never `train_deep_sdf`
directly.

#watch[Overwriting a run name rewrites only `specs.json`. The trainer deletes nothing:
  old snapshots, `Logs.pth` and `metadata.json` stay until replaced, and if the new run
  crashes the table still shows the old final loss. Use a new name.]

== Anatomy of a run folder

`runs/plate2d_r_only_d1_8x256_4h` is a complete run: 40 plates with a centred hole of
radius 0.07 to 0.45, $d = 1$, trained by `experiments/train_plate_r_only_4h.py`.
@app:formats[Appendix] lists every key of every file.

#[
  #set text(size: 9.5pt)
  #set par(justify: false)
  #kv(
    columns: (auto, auto, 1fr),
    [*File*], [*Written by*], [*In this run*],
    [`specs.json`], [`write_specs`], [8×256, skip at 4, dropout 0.2, `geom_dimension` 2,
      $d = 1$, 600 epochs, `Step` every 150, 4096 samples per shape, 4 shapes per
      batch, $sigma = 0.1$; absolute data paths],
    [`metadata.json`], [`write_metadata`], [dataset, $d$, dims, epochs, `final_loss`
      0.00431; from the script also `preset`, `stopped_at_deadline`, `last_epoch` 600,
      `wall_seconds` 5764],
    [`training_summary.json`], [trainer], [only after the last epoch: loss, duration
      1:35:59, host, device],
    [`Logs.pth`, `Logs.png`], [trainer], [per-batch loss (6000 = 600 epochs × 10
      steps), learning rates per epoch, timings, code magnitude; the PNG at snapshots],
    [`ModelParameters/`], [trainer], [decoder weights $theta$ at epochs 1, 100, ..., 600
      and `latest.pth` (1.86 MB each)],
    [`LatentCodes/`], [trainer], [the $40 times 1$ codes at the same epochs, and
      `latent_code_data_map.json` (code index $arrow$ npz file)],
    [`OptimizerParameters/`], [trainer], [Adam state (3.7 MB each); only for resuming,
      which nothing here does],
    [`code_vs_r.csv`, `.png`], [the script], [code +0.216 at $r = 0.07$ falling
      monotonically to $-0.173$ at $r = 0.45$],
  )
]

Two rules follow from how the trainer saves. `latest.pth`, the file the explorers
load, is written only when the epoch is a multiple of `LogFrequency`; numbered
snapshots come from `SnapshotFrequency` and `AdditionalSnapshots`. And the latent index
is the order of the split file: code $i$ belongs to the $i$-th name in
`data/splits/<dataset>.json`. The absolute paths in `specs.json` and
`latent_code_data_map.json` tie a run folder to the machine it was trained on.

== Presets

`experiments/make_plate_presets.py` writes four untrained run folders, one per dataset
variant (`r_only`, `xyr`, `tri_h`, `tri_hw`),
each validated against its dataset. They share one recipe, `COMMON`
(#f("experiments/make_plate_presets.py", 56)): 4×64 ReLU, skip at layer 2, no dropout,
1500 epochs, 4096 samples per shape, 5 shapes per batch, clampedL1 at 0.1, seed 42, and
the default learning rates (halved at epochs 500 and 1000). Only $d$ and
$sigma = sqrt(0.01 dot d)$ differ: `preset_plate2d_r_only_d1`, `_xyr_d3`, `_tri_h_d1`,
`_tri_hw_d2`. In the GUI you load a preset with _Start from > run: ..._ or with _Load
hyperparameters_ on the runs table, then pick the dataset and train. Headless,
`train_plate_tri.py` copies a preset's `specs.json` as its recipe (`--like`). The specs
hold absolute paths and `runs/` is gitignored, so regenerate the presets on every new
machine once the datasets exist:

```bash
uv run python experiments/make_plate_presets.py
```

The two other presets in `runs/` (`preset_plate2d_30min`, `preset_plate2d_sheet_quick`)
are older and not written by the script.

== Unattended training

A long run takes hours on this laptop's CPU (torch is installed CPU-only). The GUI is
the wrong tool: closing it kills the run, Windows may go to sleep, and nobody is there
to start the next one. The scripts `experiments/train_plate_*.py` use the same
`training.train` (CPU, same seeding) and share their machinery in
`structsept/app/unattended.py`. All four follow one lifecycle:

#figure(
  diagram(
    spacing: (6mm, 7mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nstep((0, 0), [script starts]),
    nsym((1, 0), [`open_log`, \ `KeepAwake.hold()`]),
    nsym((2, 0), [`prepare_run` or \ `prepare_run_like`]),
    nsym((3, 0), [`wait_until`, \ then `--after` pid]),
    ndat((1, -1), [`outputs/logs/` \ `<run>.log`, `.pid`]),
    ndat((2, -1), [`specs.json`, \ `metadata.json` \ (listed in the GUI)]),
    nsym((3, 1), [`train_with_deadline` \ or `training.train`]),
    nsym((2, 1), [`write_metadata`]),
    nsym((1, 1), [analysis: codes \ vs `params.csv`]),
    ndat((0, 1), [`code_vs_*`, \ `latent_coverage` \ `.json`]),
    ndat((2, 2), [checkpoints, \ `Logs.pth`]),
    nsym((3, 2), [watchdog thread \ (`_4h` scripts): \ `interrupt_main()`]),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (2, 0), "-|>"),
    edge((2, 0), (3, 0), "-|>"),
    edge((3, 0), (3, 1), "-|>"),
    edge((3, 1), (2, 1), "-|>"),
    edge((2, 1), (1, 1), "-|>"),
    edge((1, 1), (0, 1), "-|>"),
    edge((1, 0), (1, -1), "-|>"),
    edge((2, 0), (2, -1), "-|>"),
    edge((3, 1), (2, 2), "-|>"),
    edge((3, 2), (3, 1), "--|>"),
    edge((0, 0), (0, 0.5), (1, 0.5), (1, 1), "--|>", lbl[`--check-only`],
      label-pos: 0.5, label-side: center),
  ),
  caption: [Lifecycle of an unattended run. The run folder is written first, so the GUI
    lists the run while it waits. `--check-only` skips straight to the analysis.],
) <fig:training-unattended>

+ *Before any import* each script sets `TQDM_DISABLE=1` and `MPLBACKEND=Agg`: tqdm
  reads its switch at import time, and the trainer plots with pyplot.
+ *Log and keep awake.* `open_log` (#f("structsept/app/unattended.py", 73)) tees
  stdout and stderr into `outputs/logs/<run>.log` and writes the process id to
  `<run>.pid` beside it. `KeepAwake.hold()` calls `SetThreadExecutionState`, a
  per-process request that changes no power setting.
+ *Write the run.* `prepare_run` (#f("structsept/app/unattended.py", 247)) takes
  `hyperparams.defaults()` updated with the script's `HPARAMS`, clamps the log and
  snapshot intervals to the epoch count (so a 2-epoch smoke test still ends with a
  `latest.pth`), validates against the dataset and writes both JSON files. If the
  dataset is missing it prints the `datagen` command that builds it.
  `prepare_run_like` (#f("structsept/app/unattended.py", 422)) instead copies another
  run's `specs.json` key for key. Only `DataSource`, `TrainSplit`, `TestSplit` and
  `Description` may differ, otherwise it exits: "same recipe" is checked, not
  promised. Both refuse a folder that already holds `ModelParameters/latest.pth`
  unless `--force` is given.
+ *Wait.* `--start-at` takes `now`, `HH:MM` or `YYYY-MM-DD HH:MM`. `--after` takes a pid
  or a `.pid` file and polls every 30 s until that process is gone.
+ *Train.* In the `_4h` scripts `train_with_deadline`
  (#f("structsept/app/unattended.py", 319)) starts a watchdog thread. At `--max-hours`
  it calls `_thread.interrupt_main()`, which fires the library's own Ctrl-C handler
  (`sys.exit(0)`). The loop stops, and `latest.pth` (at most `LogFrequency` epochs old)
  and the snapshots remain. The other two scripts train without a limit.
+ *Record and analyse.* `write_metadata` adds `last_epoch`, `wall_seconds` and, in the
  `_4h` scripts, `stopped_at_deadline`. Then the codes are compared with the dataset's
  `params.csv`, or, in `train_plate_xyr_n166.py`, measured for latent coverage (next
  section). An analysis error is only logged.

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1.15fr, 1fr),
    [*Script and dataset*], [*Recipe*], [*Measured*],
    [`train_plate_r_only_4h.py` \ `plate_hole_2d_r_only` (40)],
    [`HPARAMS` in the script: 8×256, skip at 4, dropout 0.2, 600 epochs, LR halved
      every 150, 4 shapes per batch; $d = 1$ (`--latent-dim`)],
    [$d = 1$: 1.60 h, code vs $r$ Pearson $-1.000$. \ $d = 2$: 1.95 h, 84.8 % of the
      variance on one axis],
    [`train_plate_xyr_4h.py` \ `plate_hole_2d_xyr` (134)],
    [same, but 300 epochs and LR halved every `epochs // 4` = 75
      (`--decay-interval`); $d = 3$],
    [2.79 h; affine $R^2$ of $(x_c, y_c, r)$: 0.988, 0.741, 0.857],
    [`train_plate_tri.py --family h|hw` \ `plate_tri_2d_h` (40), `_hw` (133)],
    [the family preset's `specs.json` (`--like`): 4×64, 1500 epochs; $d = 1$ or 2],
    [`h`: 20.6 min, Pearson $-1.000$. \ `hw`: 49.9 min, linear $R^2$ of $h$ 1.000,
      of $w$ 0.996],
    [`train_plate_xyr_n166.py` \ `plate_hole_2d_xyr_n166` (166)],
    [`plate_hole_2d_xyr_d3_ep800` copied (`--like`): 8×256, 800 epochs, LR halved
      every 75; $d = 3$],
    [died at epoch 80 of 800 after 83 min (62 s per epoch: plan 14 h); no result],
  )
]

All four take `--start-at`, `--after`, `--run`, `--epochs`, `--force`, `--data-root`,
`--runs-dir` and `--log-dir`; only the `_4h` scripts take `--max-hours` (default 4).
`--check-only` re-runs just the analysis on an existing run; `train_plate_r_only_4h.py`
lacks it. In the two `--like` scripts `--epochs` is a smoke-test override that also
skips the recipe check; in the `_4h` scripts it sets the real epoch count.

```powershell
# 1. smoke-test first, into a throwaway run (delete runs/smoke afterwards)
uv run python experiments/train_plate_tri.py --family hw --epochs 2 --run smoke
# 2. the real run: writes its specs now, trains at 23:30
uv run python experiments/train_plate_tri.py --family hw --start-at 23:30
# 3. queue another run behind the first one's process
uv run python experiments/train_plate_xyr_4h.py --after outputs/logs/<run>.pid
# 4. only the analysis, on a run that exists
uv run python experiments/train_plate_tri.py --family hw --check-only
# 5. stop a run; its pid is in outputs/logs/<run>.pid
taskkill /PID <pid> /F
```

The long runs recorded in `experiments/IDEIAS.md` were launched hidden, with
PowerShell's `Start-Process -WindowStyle Hidden`, so
closing the terminal does not kill them. Follow a run in its `.log` file or in the GUI
(press _Refresh_: run lists update only on the app's own events). Chain long runs with
`--after` instead of running them side by side: torch uses every core, and two runs at
once slow each other down.

#watch[
  - Give a smoke test `--run smoke`. Without it, `--epochs 2` writes into the real
    run's folder, and the real launch then refuses to start without `--force`.
  - `--force` retrains *from scratch*; there is no resume. `training.train` never passes
    the library's `continue_from`, and that path would not help anyway: it restores
    the decoder and Adam but drops the saved codes (see the trap table).
  - A hard kill or a crash leaves `metadata.json` as `prepare_run` wrote it
    (`final_loss` null, no `last_epoch`) and no `training_summary.json`, yet a loadable
    `latest.pth`. Check the epoch it holds with `unattended.last_epoch`.
    `runs/plate_hole_2d_xyr_n166_d3_ep800` is such a run: its checkpoint holds epoch 80.
]

#wip[`unattended.py` (with `prepare_run_like` and `code_vs_parameter`),
  `make_plate_presets.py` (the two triangle presets) and `train_plate_r_only_4h.py` are
  modified and uncommitted; `train_plate_tri.py` and `train_plate_xyr_n166.py` are
  untracked. Edits from 2 October, also uncommitted, make the runs table show a run that
  stopped short as `80/800` (`training.checkpoint_epoch`) and make `from_sheet` note the
  missing skip-connection row.]

== Analysing a run

A low final loss says the decoder reproduces its training shapes. It says nothing about
the question this project asks of a run (@ch:status[Chapter]): did the latent space
learn the parameters that generated the shapes? Three functions answer that. All of
them match codes to the rows of `params.csv` by shape name in split order
(`unattended.split_names`), because the latent index is the split order.

#[
  #set text(size: 9.5pt)
  #set par(justify: false)
  #kv(
    columns: (auto, 1fr),
    [*Function*], [*What it computes and writes*],
    [`code_vs_parameter` \ #f("structsept/app/unattended.py", 525)],
    [For one parameter column: per component Pearson, Spearman and whether the code is
      monotonic. For $d > 1$ also the variance share on the first principal axis (PC1),
      the correlation along it, and the $R^2$ of a linear fit (with intercept) of the
      parameter on all components. Writes `code_vs_<p>.csv` and `.png`.],
    [`compare_codes_with_params` \ #f("experiments/train_plate_xyr_4h.py", 261)],
    [Correlation tables code $times$ parameter, an affine fit of $(x_c, y_c, r)$ on the
      codes ($R^2$, RMSE, coefficients) and the reverse fit, and the variance share of
      every principal axis. Writes `code_vs_params.csv`, `.json`, `.png`.],
    [`latent_coverage` \ #f("experiments/train_plate_xyr_n166.py", 160)],
    [Per latent axis, the widest gap between neighbouring trained codes as a share of
      the axis range, against `viz.GAP_FRACTION` = 0.12, the share at which the GUI
      shades a gap as untrained. Writes `latent_coverage.json`.],
  )
]

*Judge with an affine fit over all components, never component by component.* The loss
does not care how the latent axes are oriented: the decoder's first layer is linear in
$lambda$ and absorbs a rotation, and the regulariser $norm(lambda)$ is the same in any
rotated basis. So no single component has to track a parameter. In
`plate2d_r_only_d2_8x256_4h` the two components correlate with $r$ at +0.904 and
$-0.933$ and neither is monotonic, yet 84.8 % of the variance lies on one axis and the
position along it has Spearman +1.000 with $r$. In `plate_tri_2d_hw_d2_4x64` the
components correlate with $h$ at only 0.675 and 0.744, but the linear fit recovers $h$
with $R^2$ = 1.000 and $w$ with 0.996.

The range of the trained codes matters later too: it is the box the design variables
may move in. For `plate2d_r_only_d1_8x256_4h` it is $-0.17$ to $+0.22$, not the
[0.15, 0.75] the library's tests use for the shipped decoders.

== Trainer traps

#f("DeepSDFStruct/DeepSDFStruct/deep_sdf/training.py", 299) is where `train_deep_sdf`
starts. Reading it line by line (23 September) turned up the traps below. Each row
was checked against the library code, with lines relative to
`DeepSDFStruct/DeepSDFStruct/deep_sdf/`; the right column says what this repository
does about it.

#[
  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (1.35fr, auto, 1fr),
    [*Trap*], [*Library line*], [*What the repo does*],
    [The decoder is built before the seed is set, so its initial weights depend on
      whatever used the torch RNG before.], [`training.py` 409, 415],
    [`training.train` seeds first.],
    [Gradient clipping is hardcoded to max-norm 1.0; `GradientClipNorm` is read, then
      overwritten.], [`training.py` 341, 637], [not offered; listed in `FIXED`],
    [`clampedL1` clamps at its own $plus.minus 0.1$, and the trainer clamps both sides
      at $plus.minus delta$: the band learned is $min(delta, 0.1)$.],
    [`nn_utils.py` 18, \ `training.py` 581, 607], [`validate` note; use `L1` for a
      wider band],
    [Each shape draws `SamplesPerScene / 2` rows inside and outside; an odd count
      leaves the batch one sample short of its codes and crashes.],
    [`data.py` 176, \ `training.py` 585], [`validate` error],
    [`latest.pth` is written only when `epoch % LogFrequency == 0`; the default 10
      leaves a 5-epoch run with nothing to load.], [`training.py` 684],
    [`auto` picks 10, 5, 2 or 1; `prepare_run` clamps it],
    [`drop_last=True` whenever the batch fits: `n_shapes % ScenesPerBatch` shapes sit
      out each epoch (134 at 4: two).], [`training.py` 450],
    [`validate` note],
    [`CodeRegularization` (bool) is never read. The penalty is $lambda_"reg"$ times the
      mean code norm (not squared), ramped in over 100 epochs.], [`training.py` 622],
    [ramp listed in `FIXED`; the bool is written but inert],
    [A Ctrl-C handler calling `sys.exit(0)` is installed; installing it off the main
      thread raises.], [`training.py` 371, 391],
    [GUI: `signals_off()`; scripts: the deadline uses it],
    [The tqdm bar writes to stderr, which is `None` under `pythonw`.],
    [`training.py` 555], [`TQDM_DISABLE`; the launcher redirects stderr
      (@ch:gui[Chapter])],
    [`training_summary.json` is written only after the last epoch, so a stopped run
      keeps `final_loss` null.], [`training.py` 696], [check `last_epoch` instead],
    [`continue_from` reloads decoder and optimizer, but assigns the loaded codes to `_`:
      they restart from random under a trained decoder.], [`training.py` 517],
    [not exposed; a dead run is retrained with `--force`],
  )
]

`experiments/IDEIAS.md` (23 September) marks the hardcoded clipping as worth
reporting upstream; the seed order is arguably another candidate.
@app:gotchas[Appendix] collects these traps with the rest of the project's.
