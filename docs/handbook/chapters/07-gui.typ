#import "../template.typ": *

= The GUI <ch:gui>

In the paper the optimizer never looks at a picture: MMA moves the latent control
points $hat(lambda)$, the code meshes and simulates, and the loop repeats. A latent
value of $0.6$ is not a radius or a thickness, though; it is a coordinate in a
learned space. The GUI makes it visible. You load a trained decoder $f_theta$, set
the control points by hand and watch the geometry change while you drag. It also
puts the offline half of the pipeline (dataset, training) behind buttons. The GUI
is *not* part of the optimization loop: the slice and field evaluations run under
`torch.no_grad`, and the extracted surface goes to trimesh (NumPy), so the "no
`.detach()`" rule of `CLAUDE.md` does not apply here. All code lives in `structsept/app/`.

== Two windows and how to start them

#kv(
  columns: (auto, auto, 1fr),
  [*Window*], [*Module*], [*What it is for*],
  [Lattice explorer], [`structsept.app.main`],
  [Tabs *Explore* (a 3-D unit-cell decoder tiled into a lattice), *Explore 2-D*
    (planar decoders) and *Train* (train a decoder, manage `runs/`).],
  [SDF maker], [`structsept.app.sdf_maker`],
  [A folder of meshes becomes an `SdfSamples` dataset; also validates datasets.],
)

```bash
uv run python -m structsept.app.main                 # explorer
uv run python -m structsept.app.sdf_maker            # SDF maker
uv run python -m structsept.app.launcher explorer    # what the desktop shortcut runs
```

The SDF maker used to be the explorer's first tab. It became its own window because
a dataset is built once in a while, and the explorer assumes that valid datasets
already exist. On Windows, `install_shortcuts.bat` puts both windows as shortcuts
into `Desktop\NN\` (section on the desktop launcher below).

#figure(
  image("../../figures/gui_explore.png", width: 90%),
  placement: auto,
  caption: [The Explore tab with the shipped `ChiAndCross` decoder ($d = 2$, 120
    codes): a $3 times 3 times 2$ control net on $2 times 2 times 2$ cells, i.e. 36
    design variables. Left: one layer of the control net, two values set by hand.
    Centre: the $f_theta$ slice as material/void, control points as dots. Right: the
    unit cell of the selected control point and the latent coverage strip.],
) <fig:gui-explore>

#wip[The screenshot predates the Explore 2-D tab and the _Edit..._ button of the
  decoder picker. The GUI is committed on branch `gui-train-tab` (`6c07cfc`), not yet
  merged into `main`. Uncommitted at the time of writing: the pickers and the runs
  table are being taught to flag a run whose checkpoint stops short of its planned
  epochs ("trained here, epoch 80 of 800" via `models.entry_tag`).]

== Architecture: one dict, one queue, no classes

`build_app(root=None)` (#f("structsept/app/main.py", 53)) builds the window without
starting `mainloop`, so both `main()` and the desktop launcher can call it. The
order of its steps matters:

+ *Agg backend first* (#f("structsept/app/main.py", 31)). DeepSDFStruct calls
  `pyplot` from worker threads (the trainer's loss curve, the mesher's debug
  figures). An interactive backend would then create Tk widgets off the main thread
  and kill the app. The app's own plots are hand-built `FigureCanvasTkAgg`
  canvases, which work with any backend.
+ *Theme before widgets.* `theme.apply_theme` works through `option_add`, which
  only reaches widgets created after the call.
+ *One shared state dict.* `st = runtime.new_state(root)` starts as
  `{"root", "queue", "busy"}`. Each tab keeps its widgets, Tk variables and data in
  it under its own key prefix: `ex_` (Explore), `e2_` (Explore 2-D), `tr_` (Train).
  The tabs are modules, not classes, and every function takes `st` first.
  `root.app_state = st` lets tests and a debugger reach all of it.
+ *Tabs.* Each tab module's `build(st, frame, ...)` fills one `ttk.Notebook` page
  and registers a refresh callback in `st["run_listeners"]`.
+ *Pump and shutdown.* `root.after(100, ...)` starts the queue pump. Closing the
  window runs `_shutdown` (#f("structsept/app/main.py", 124)), which cancels
  `poll_job` and every pending `after` whose key starts with `ex_job_`, `e2_job_` or
  `tr_job_`; otherwise Tk fires them into a dead interpreter. A new debounced job
  only needs a key with one of these prefixes.

=== The threading model

#figure(
  diagram(
    spacing: (9mm, 5mm), edge-stroke: 0.7pt, mark-scale: 70%,
    node(
      enclose: ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)),
      stroke: (paint: c-muted, thickness: 0.6pt, dash: "dashed"),
      corner-radius: 6pt, inset: 9pt, snap: false,
    ),
    node(
      enclose: ((3, 0), (3, 0)),
      stroke: (paint: c-muted, thickness: 0.6pt, dash: "dashed"),
      corner-radius: 6pt, inset: 9pt, snap: false,
    ),
    nui((0, 0), [click _Load_, \ _Extract_, _Train_]),
    nsym((1, 0), [`run_worker`: refuse \ if `busy`, else set \ it and lock widgets]),
    nsym((3, 0), [`work(log)` \ (training inside \ `signals_off()`)]),
    ndat((2, 1), [`st["queue"]`]),
    nsym((1, 1), [`poll` \ every 100 ms]),
    nui((0, 1), [widgets \ (logs, canvases)]),
    nsym((1, 2), [`_worker_done`: \ `busy = False`, \ unlock, `on_done()`]),
    nsym((0, 2), [redraw \ (SDF evaluated here)]),
    nui((0, 3), [slider event]),
    nsym((1, 3), [`reschedule` \ 120 / 400 ms]),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (3, 0), "-|>", lbl[`Thread.start`]),
    edge((3, 0), (2, 1), "-|>", lbl[`log`, `ui(fn)`], label-side: right),
    edge((2, 1), (1, 1), "-|>", lbl[drain], label-pos: 0.75),
    edge((1, 1), (0, 1), "-|>"),
    edge((1, 1), (1, 2), "-|>", lbl[last item], label-side: left),
    edge((0, 3), (1, 3), "-|>"),
    edge((1, 3), (0, 2), "-|>"),
    edge((0, 2), (0, 1), "-|>"),
  ),
  caption: [The threading model of `structsept/app/runtime.py`. Left box: the Tk main
    thread, the only one that touches widgets. Right box: a daemon worker. Slow
    one-shot jobs run on the worker and reach the interface only through the queue;
    interactive redraws stay on the Tk thread, throttled by a debounce.],
) <fig:gui-threads>

There is one rule: *only the Tk main thread touches widgets.* Tk is not
thread-safe, and a widget changed from another thread crashes the app or corrupts
it silently. `runtime.py` exists to keep that rule:

- *Workers* run the slow one-shot jobs: loading a decoder, extracting a surface,
  training, and the SDF maker's _Generate_ and _Validate_. `run_worker`
  (#f("structsept/app/runtime.py", 65)) refuses with a "Busy" dialog if
  `st["busy"]` is set. Otherwise it sets the flag, disables the button and the
  widget subtrees passed as `lock`, and runs `work(log)` on a daemon thread. In a
  `finally`, the worker queues `_worker_done`, which clears `busy`, re-enables the
  widgets and calls `on_done`: the place where a job releases what it held.
- *The queue is the only way back.* `log(line)` and `runtime.ui(st, fn)` put items
  on `st["queue"]`. A worker hands over its result as a closure: the Explore loader
  queues `_install_model`, which then runs on the Tk thread. `poll`
  (#f("structsept/app/runtime.py", 39)) drains the queue every 100 ms. A callback
  that raises does not stop the pump; the error is printed as `INTERNAL ERROR` in
  the Explore tab's log, whichever tab raised it.
- *Redraws are debounced, not threaded.* `reschedule`
  (#f("structsept/app/runtime.py", 126)) cancels the pending `after` stored under a
  key and arms a new one, so during a drag only the last event does any work.
- *`signals_off()`* (#f("structsept/app/runtime.py", 112)) swaps `signal.signal`
  for a no-op while training runs: the DeepSDF trainer installs a SIGINT handler,
  and installing one off the main thread raises.

#watch[
  - Each window has *one* `busy` flag. While the Train tab trains (possibly for
    hours), _Load_ in both explorers and _Extract surface_ only answer "Busy".
    Sliders on a decoder that is already loaded keep working. To explore while
    training, open a second window.
  - The docstrings of `main.py` and `runtime.py` say "the Tk loop only draws". Not
    quite: the slice, the latent field, the unit cell and the $32^3$ volume grid are
    evaluated on the Tk thread, the volume grid alone in about 0.2 s. Keep new
    per-drag work cheap, or put it in the slow redraw tier.
]

== Explore: one optimizer iteration by hand

The Explore tab runs the online steps of the paper with you in place of MMA. Steps
1--2 (sampling, training) are the Train tab, the SDF maker and `datagen`
(@ch:training[Chapter], @ch:datagen[Chapter]).

#kv(
  columns: (auto, 1fr, 1.25fr),
  [*Paper step*], [*In an optimization run*], [*In the Explore tab*],
  [3 network inputs], [spline field $lambda(bold(x))$ from $hat(lambda)$, tiling
    $T(bold(x))$], [`models.build_lattice`, same classes],
  [4 mesh], [FlexiCubes, FFD, tetrahedra],
  [_Extract surface_: the FlexiCubes surface only],
  [5 FEM], [compliance $J$], [absent (see @ch:fem[Chapter])],
  [6 update], [MMA moves $hat(lambda)$],
  [*you* move $hat(lambda)$; a tile shows the volume fraction $V(hat(lambda))$],
)

=== What each panel answers

The layout follows the questions you ask while exploring, in that order (the
docstring of `tab_explore.py` writes the order down):

#[
  #set text(size: 9.5pt)
  #kv(
    columns: (auto, 1fr, auto),
    [*Panel*], [*Question it answers*], [*Fed by*],
    [Header: picker], [Which decoder, and is it usable? Shows $d$ and the number of
      codes; after _Load_, warns when all stored codes are identical.],
    [`list_models`],
    [Header: readout], [How many design variables, how many cells? The two are
      independent: _18 x d=2 = 36 design variables · 2 x 2 x 1 knot spans ·
      2 x 2 x 2 = 8 cells_.], [`_update_derived`],
    [Left: control net], [Which knob moves which region? One $n_x times n_y$ layer
      laid out in space, $j = 0$ at the bottom; red outline = outside the trained
      range.], [`ControlPointGrid`],
    [Centre: $f_theta$ slice], [What does the shape look like now? A $z$-slice of the
      capped lattice, only the sign of $phi$, control points as markers.],
    [`eval_sdf_slice`],
    [Centre: $lambda(bold(x))$], [How does the latent field vary? Up to three
      components, on a fixed colour scale (the trained range).],
    [`eval_latent_slice`],
    [Centre: tiles], [How much material is there? Does this slice have a zero level
      set at all?], [`volume_fraction`],
    [Right: unit cell], [What does this latent value mean? $f_theta$ at the selected
      control point on the bare cube $[-1, 1]^3$.], [`eval_cell_slice`],
    [Right: coverage], [Is the design still supported by training data? Codes as
      ticks (a histogram past 400 codes), the widest gap shaded, the distance to
      the nearest code.],
    [`LatentNeighbors`],
  )
]

=== The data model and Load

The design vector is a numpy array `st["ex_cps"]` of shape $(n_"cp", d)$ with
$n_"cp" = n_x n_y n_z$. Control point $(i, j, k)$ is row $i + n_x (j + n_y k)$
(`models.flat_index`, #f("structsept/app/models.py", 307)), because splinepy stores
the net with $x$ running fastest; a test checks this against splinepy. The latent
spline is clamped and of degree 1, so control point $(i, j, k)$ sits at
$(i slash (n_x - 1), j slash (n_y - 1), k slash (n_z - 1))$ in the unit cube, where
its marker is drawn. The spinboxes allow 2--6 control points and 1--8 cells per axis.

Changing the decoder, the net or the tiling only shows "settings changed - press
Load to apply". _Load_ (`_load_model`, #f("structsept/app/tab_explore.py", 532))
runs on a worker: it loads the weights with `get_model(ref, checkpoint="latest")`,
puts the mean trained code on every control point, builds the lattice and runs a
small warm-up evaluation (the first one builds lazy kernels and would otherwise
freeze the window). The slider spans the trained min/max of the active component
plus 10 % of the span on either side. `build_lattice` (#f("structsept/app/models.py",
201)) is paper step 3 almost word for word (classes in @ch:library[Chapter]; no
bounds are passed, so the domain is $[0, 1]^3$):

```python
spline = build_parameter_spline([1, 1, 1], [n - 1 for n in n_ctrl], d)
spline.control_points = control_points          # lambda-hat, shape (n_cp, d)
parametrization = SplineParametrization(spline, device=model.device)
lattice = LatticeSDFStruct(
    tiling=tiling,
    microtile=SDFfromDeepSDF(model),
    parametrization=parametrization,
)
sdf = CappedBorderSDF(lattice, None)            # flat, closed faces at the domain border
```

=== From slider to picture

#figure(
  diagram(
    spacing: (16mm, 6mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nui((0, 0), [slider drag]),
    nsym((1, 0), [`_on_scale` \ repaint grid, drop surface]),
    ndat((2, 0), [`ex_cps` \ $(n_"cp", d)$, numpy]),
    nsym((1, 1), [`_schedule_redraw`]),
    nsym((0, 2), [`_redraw_geometry`]),
    nsym((2, 2), [`_redraw_context`]),
    nui((0, 3), [$f_theta$ slice, \ $phi$ range, zero set]),
    nui((2, 3), [$lambda(bold(x))$ panels, unit cell, \ coverage, volume fraction]),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (2, 0), "-|>", lbl[writes\ one entry]),
    edge((1, 0), (1, 1), "-|>"),
    edge((1, 1), (0, 2), "-|>", lbl[120 ms quiet], label-side: right),
    edge((1, 1), (2, 2), "-|>", lbl[400 ms quiet], label-side: left),
    edge((0, 2), (0, 3), "-|>"),
    edge((2, 2), (2, 3), "-|>"),
  ),
  caption: [A slider event in the Explore tab: it writes one entry of the design
    vector, then two debounced tiers redraw. All of it runs on the Tk thread.],
) <fig:gui-slider>

`_on_scale` (#f("structsept/app/tab_explore.py", 768)) writes the value into
`ex_cps[row, component]`, repaints the grid and drops any extracted surface, which
no longer matches the design. `_schedule_redraw` then arms two timers. *120 ms*
after the last event, `_redraw_geometry` pushes `ex_cps` into the live torch spline
(`set_param`, no rebuild) and evaluates the capped lattice on the slice plane. *400
ms* after it, `_redraw_context` redraws the latent panels, the unit cell, the
coverage strip and the volume fraction, plus the status line (values outside the
trained range) and the coverage note (distance from the worst control point to its
nearest trained code, amber above 0.15). The geometry is what you cannot predict,
so it follows the drag; the context costs more and does not change meaning
mid-drag.

Two numbers deserve a note. The *volume fraction* samples $32^3$ cell centres,
because `CappedBorderSDF` forces $phi >= 0$ on the domain faces and a node grid puts
many samples there (27 % at resolution 20). The *nearest-code distance* exists
because a per-component min/max box cannot see holes: the 20 codes of `RoundCross`
span $[-1, 1]$ with a 0.4-wide empty gap, and their mean, the starting design,
falls right into it. The box test passes; the distance shows the problem.

=== Extract surface, Save design and the shared-SDF lock

_Extract surface_ (#f("structsept/app/tab_explore.py", 1038)) runs
`models.surface_mesh` on a worker: `create_3D_mesh` with `mesh_type="surface"` and
`differentiate=False`, converted to `trimesh`. This is paper step 4 *without* FFD
and *without* tetrahedra, so no FEM and no MMA can follow. The marching resolution is
`N_base` $times$ tiling $+ 1$ per axis (`N_base` defaults to 12). _Export STL..._
writes `.stl`, `.obj` or `.ply` in unit-cube coordinates; _Open 3D window_ opens a
pyvista view that blocks the interface until closed.

_Save design..._ (#f("structsept/app/tab_explore.py", 1139)) writes what the tab
actually produces: $hat(lambda)$ and its grid, as JSON with the keys `decoder`,
`source`, `n_ctrl`, `tiling`, `latent_dim`, `control_points` and
`control_point_order`. That is the starting design of an optimization run; the STL
is only a picture of one evaluation of it.

#watch[*Never let two things evaluate the same lattice SDF at once.* Each call of
  `LatticeSDFStruct` writes one latent vector per query point into its microtile, so
  overlapping calls fail with "Latent vector shape mismatch". `_extract_surface`
  therefore sets `st["ex_sdf_busy"]` (both redraw functions return at once while it
  is set), cancels the pending redraw timers and locks the design and view panels via
  `run_worker(lock=...)`. When the mesher is done, `_worker_done` unlocks the panels
  and `_sdf_released` (its `on_done`) clears the flag and re-arms the redraw. Disabling
  widgets alone is not enough: an already scheduled redraw would still fire. For the
  same reason the unit-cell preview has its own `SDFfromDeepSDF` wrapper
  (`models.unit_cell_sdf`) around the same weights.]

== Explore 2-D

A planar decoder $f_theta (lambda, x, y)$, such as those trained on the `datagen`
plate families (@ch:datagen[Chapter]), describes one *whole shape* per latent
vector. There is no lattice, no control net and no tiling: the design variables are
the components of $lambda$ itself, one slider each, starting at the mean code with
the same range rule (#f("structsept/app/tab_explore2d.py", 345)). _Load_ refuses a decoder whose `geom_dimension` is not 2. The field is a
node grid over $[-1, 1]^2$ evaluated by `decode_2d` (#f("structsept/app/models.py",
484)), which builds the decoder input by hand (the code repeated for every point,
then $(x, y)$) and calls `model._decoder` directly. The same 120/400 ms tiers drive
the material/void field with its area fraction, one coverage strip per component
and the nearest-code distance. There is no surface extraction and no _Save design_.

== How a run reaches the explorers

`models.list_models` (#f("structsept/app/models.py", 128)) lists the 7 shipped
`PretrainedModels` decoders plus every `runs/<name>/` that holds both `specs.json`
and `ModelParameters/latest.pth`, and keeps those whose `geom_dimension` (in the
`NetworkSpecs` of `specs.json`, default 3) matches: 3 for Explore, 2 for Explore 2-D.
A run without a checkpoint (an untrained preset, a run stopped before its first
save) does not appear at all. A run that stopped early (deadline, crash) is listed
like a finished one; only the uncommitted `models.entry_tag` marks it (box at the top
of this chapter).

The pickers re-read the disk only on the app's own events: when the Train tab
finishes, renames or deletes a run, `run_editor.runs_changed` calls every function
in `st["run_listeners"]` (both pickers and the runs table). Runs written by another
process, such as a headless `experiments/train_*.py` script, appear after
_Refresh_. Two windows can also collide on the automatic run name
`<dataset>_d<d>_<YYYYmmdd_HHMM>`, which has minute resolution: two trainings of one
dataset and $d$ started in the same minute get the same folder (the second sees an
overwrite prompt only if the first has already written its `specs.json`).
Double-clicking a row of the Train tab's runs table (or _Open_) calls
`_open_in_explore` (#f("structsept/app/tab_train.py", 486)): it selects the run in
whichever explorer lists it and switches tab, but does *not* load it.

The Train tab, including how it trains on a worker and polls `Logs.pth`, is covered
in @ch:training[Chapter]. That poll follows only the run started in this window; a
headless run shows its progress only in its log.

#tip[Today all 11 loadable runs in `runs/` are planar, so the Explore picker lists
  only the shipped decoders and every local run opens in Explore 2-D. `Primitives2D`
  is a 3-D decoder despite its name and appears on Explore.]

== The SDF maker

The SDF maker is step 1 of the offline pipeline for arbitrary meshes. You pick a
mesh folder, an extension (default `stl`), dataset and class names, the samples per
geometry (default 50 000) and whether to add surface samples.

- _Generate dataset_ (#f("structsept/app/sdf_maker.py", 171)) reads the meshes and
  asks before going on if any is not watertight: the SDF sign comes from a winding
  number, so an open mesh gives a silently wrong dataset. On a worker,
  `datasets.make_dataset` (#f("structsept/app/datasets.py", 168)) then removes stale
  files and runs the library's `SDFSampler`, which scales each mesh into
  $[-1, 1]^3$ and writes `data/SdfSamples/<ds>/<class>/*.npz` and
  `data/splits/<ds>.json`.
- _Validate_ checks an existing dataset for missing inside samples, NaNs, points
  outside the $[-1, 1]^d$ box (tolerance 0.2), an unnormalized $phi$, a skewed inside/outside ratio and
  files that mix 2-D and 3-D rows, and plots a $phi$ histogram.

It uses the same `runtime` plumbing with its own `st` and Tk root. Every dataset in
`data/` today was made by `datagen`, not by the SDF maker (@ch:datagen[Chapter]);
formats are in @app:formats[Appendix].

== The desktop launcher

A shortcut runs `.venv\Scripts\pythonw.exe -m structsept.app.launcher explorer` (or
`sdf_maker`) in the repository root. `structsept/app/launcher.py` exists only so the
app can be double-clicked, and three facts shape it:

#[
  #set par(justify: false)
  - *`pythonw` has no console*: `sys.stdout` and `sys.stderr` are `None`, but tqdm
    and the mesh loader write to stderr, so the Train tab and the SDF maker would
    crash as soon as they start. `attach_log` (#f("structsept/app/launcher.py", 158)) sends both streams to
    `outputs/logs/<target>.log` (append mode, rotated past 2 MB), sets
    `TQDM_DISABLE=1` and points `faulthandler` there, all *before* DeepSDFStruct is
    imported, because a `logging.StreamHandler` captures `sys.stderr` when created.
  - *The imports take 15--30 s*, and a shortcut that shows nothing gets clicked again.
    `run` (#f("structsept/app/launcher.py", 529)) shows a splash within a second and
    imports the app on a worker thread that reports through a queue polled every
    100 ms (the app's own pattern), then calls `build_app(root=root)` to build the
    window into the root the splash already owns.
  - *Shortcuts hold absolute paths*, and editable installs keep only the code current.
    `env_hash` (#f("structsept/app/launcher.py", 185)) hashes `pyproject.toml`,
    `uv.lock` and `DeepSDFStruct/pyproject.toml`; if the hash differs from
    `.venv/.structsept-env-hash`, the launcher runs `uv sync`. It skips the sync while
    another structsept window is open, because Windows cannot replace a loaded `.pyd`
    and a half-finished sync breaks the environment. `InstanceLock`
    (#f("structsept/app/launcher.py", 292)) detects open windows: each holds an OS
    lock on one of 16 bytes of `.venv/.structsept-instances`.
]

`install_shortcuts.bat` sets this up on a fresh clone: it checks for `uv`, fetches
the submodule if it is empty, runs `uv sync` and then `launcher --install`, which
writes the `.lnk` files through PowerShell's `WScript.Shell` (no `pywin32` needed)
and records the hash. Run it again after moving the folder or re-creating `.venv`;
the full setup is in @ch:workflows[Chapter].

#watch[A window started from a shortcut shows its output nowhere: tracebacks and the
  trainer output of GUI training runs go to `outputs/logs/explorer.log`. A terminal
  launch redirects nothing. If a dialog at start-up reports an `ImportError`, close
  every structsept window and run `install_shortcuts.bat` (or `uv sync`).]

== Known limits and tests

- `Primitives`, `PrimitivesCL16` and `PrimitivesCL08` store all-zero latent codes:
  the slider range falls back to $plus.minus 1$ and the coverage panel says nothing.
- The SDF maker reads the mesh folder on the Tk thread and freezes on large folders.
- The 3-D preview is static, Explore 2-D has no surface or export, and nothing in
  the repository reads a saved design JSON yet.

The GUI tests drive the real widgets. `_hide` (#f("tests/test_app_explore.py", 26))
makes the window transparent and frameless and parks it at $(-6000, -6000)$ instead
of withdrawing it, because a withdrawn window has no device context and the
matplotlib canvases would never get a size. The tests need a display, but nothing
appears on it.

#[
  #set text(size: 9.5pt)
  #kv(
    columns: (auto, 1fr),
    [*File*], [*What it covers*],
    [`test_app_explore.py`], [One test per defect that shipped once, on `ChiAndCross`
      with a $4 times 2 times 3$ net: flat index vs splinepy, layer clamp, surface
      dropped after an edit, mesher holding the SDF alone, unbiased volume fraction.],
    [`test_app_explore2d.py`], [Planar path end to end: a 6-shape `datagen` set, a
      2-epoch run through the Train tab's code, the run landing on Explore 2-D.],
    [`test_app_runs.py`], [Rename, notes, delete, the lock on the run being trained.],
    [`test_app_launcher.py`], [22 tests; every launch scenario runs in a fresh
      interpreter, one under a real `pythonw.exe` without standard handles.],
  )
]

```bash
uv run pytest tests/test_app_explore.py tests/test_app_explore2d.py -v
uv run pytest tests/test_app_runs.py tests/test_app_launcher.py -v
```

#tip[Four rules when you change the app. (1) Never touch a widget from a worker;
  queue a closure with `runtime.ui`. (2) Keep state under your tab's prefix and name
  debounced jobs `<prefix>_job_<name>`, so that `_shutdown` cancels them. (3) Code
  that evaluates the lattice SDF must respect `ex_sdf_busy`. (4) Keep torch out of
  `viz.py` (field evaluation belongs in `models.py`); it imports no torch and is
  testable without the heavy stack.]
