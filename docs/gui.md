# The GUI — what each panel is, and why it is there

Two windows, deliberately separate:

```bash
uv run python -m structsept.app.main        # explorer: Explore, Explore 2-D, Train
uv run python -m structsept.app.sdf_maker   # dataset builder, run once in a while
```

Or from the desktop: double-click `install_shortcuts.bat` once and both windows
appear as shortcuts in `Desktop\NN` (see *Running it from the desktop* below).

The explorer assumes valid SDF datasets already exist on disk. Building them is
a different job with a different rhythm, so it lives in its own window and kept
its old, unstyled layout — only its text is English now. Parametric datasets
(the plate with a hole) come from the `datagen` package instead:
`uv run python -m datagen.make_plate_hole --dim 2`, or with `--radius-only`
for the family whose hole stays at the plate centre and only grows.

![Explore tab](figures/gui_explore.png)

---

## The idea the interface has to carry

From `paper_context.md` (Kofler et al. 2025): the unit-cell geometry is a
trained DeepSDF decoder `f_θ`. Its latent vector is allowed to vary in space as
a B-spline field `λ(x) = Σᵢ φᵢ(x) λ̂ᵢ`, and the spline control points `λ̂ᵢ` **are
the design variables** of the optimization. A transformation function `T(x)`
tiles the cell over the domain. Compliance is minimized subject to a volume
constraint; MMA moves `λ̂`.

So the explorer is a hand-driven version of one optimizer iteration. Every
panel answers one question in that chain:

| Question | Where it is answered |
|---|---|
| Which `f_θ` do I have, and is it usable? | header: `d`, number of trained codes, source |
| How many design variables, how many cells, are the two independent? | header: derived readout |
| Which knob moves which region? | left: the control net, laid out in space |
| What shape is this *now*? | centre: `f_θ` slice |
| How much material is there? | centre: volume fraction |
| What does this latent number *mean*? | right: the bare unit cell |
| Is the design still supported by training? | right: latent coverage |

---

## Panel by panel

**Header — decoder and grid.** The derived line spells out the decoupling the
paper makes a point of: `18 × d=2 = 36 design variables · 2×2×1 knot spans ·
2×2×2 = 8 cells`. The number of control points is independent of the number of
unit cells (Eq. 21); two identical-looking spinbox triples hid that. The header
also warns when a decoder's stored latent codes are all identical — the
`Primitives*` nets ship exact zeros, which used to give every slider ±0.0001 of
travel with no explanation.

**Left — design variables.** The control net is drawn as an `nx × ny` grid for
one layer `k` and one component `λⱼ`, with row 0 at the bottom so the grid
matches the slice beside it. Screen position = domain position: cell `(i,j)`
sits at `(i/(nx−1), j/(ny−1))`. The flat row index into `λ̂` is
`i + nx·(j + ny·k)`, and the caption states it, so the array and the picture
agree. A cell outlined in red is outside the trained range.

This replaces a flat list of `cp0 λ₁` sliders. That list threw away the index's
spatial meaning and did not scale: at the spinbox maxima it is 216 control
points × `d` rows, built synchronously. The grid shows at most 36 cells whatever
`d` and `nz` are, and one slider edits the selected one.

**Centre — geometry.** Three pages: the `f_θ` slice, the `λ(x)` field, and the
extracted surface. Under them, the slice height `z`, the preview resolution, and
three numbers: volume fraction, the `φ` range on this slice, and whether a zero
level set exists at all. Volume fraction is `V(λ̂)`, the paper's constraint — the
one scalar that ties this toy to the optimization it feeds. It is a 32³
midpoint-rule estimate on **cell centres**: a node grid would put 27 % of its
samples on the domain faces, where the border cap forces `φ ≥ 0`, and report
every lattice as about a third emptier than it is. Even at 32³ the thin struts
are only just resolved, so read it to two decimals, not three. "Zero level set:
absent" is the honest early warning for the empty-mesh case.

The control points of the active layer are drawn on the slice. That makes the
B-spline's **local support** checkable rather than something to take on faith:
move one marker and only its neighbourhood of the lattice responds.

**Right — what the numbers mean.** The *unit cell* panel shows
`f_θ(λ̂_selected, ·)` on the bare cube `[−1,1]³`, before tiling. This matters
because a latent value is not a physical parameter: for the trained
`RoundCross` the codes were learned, not prescribed, so `λ = 0.6` is not a
radius. The picture is the only honest interpretation of the number.

The *latent coverage* strip shows, for the active component, every trained code
as a tick, the min/max band shaded, the widest untrained gap in amber, and the
current `λ̂` values as markers. Below it: the distance from the worst control
point to the nearest trained code.

---

## What was removed, and why

**The latent point cloud (`λ₁` vs `λ₂` scatter) is gone.** Of the seven shipped
decoders only one has `d = 2`, so for six of them the panel rendered a
placeholder string. Even at `d = 2` it was a cloud of grey dots that answered no
question anyone had, and its convex hull actively lied: it shaded untrained
interior as supported.

The coverage strip is its honest replacement — same data, one dimension at a
time, readable for any `d`, and it shows the gaps the hull hid. On `RoundCross`
it shows a 0.40-wide hole in the middle of the trained range, with the app's
default design (the mean of the codes, 0.115) sitting inside it. That is why the
default lattice comes out thin. The old status line reported
zero values out of range, because the mean *is* inside the bounding box.

---

## Rendering decisions worth not undoing

**The SDF slice is material/void, not a diverging colour map.** The old panel
used `seismic` rescaled to `max|φ|` every frame. Two problems: `φ` saturates at
+1 over most of the domain, so the material got about a tenth of the ramp and
rendered as a pale smudge — worse the thinner the struts; and the normalization
moved *while* the geometry moved, so colour and shape changed together. The
magnitude of `φ` away from the surface is an artefact of the SDF construction
anyway, and the network is only accurate near the zero set. What the panel has
to answer during a drag is `sign(φ)`. The transition is anti-aliased over ~1
screen pixel, with the width derived from the measured `|∇φ|` (3.9 for
`AnalyticRoundCross`, 0.85 for `ChiAndCross` — `T(x)` rescales the gradient, so
it cannot be assumed to be 1).

**The `λ(x)` panels use a fixed normalization per component** — the trained
range, stated in each panel title. Autoscaled per frame, the starting design (a
constant field with ~1e-8 of float round-off) rendered that round-off as
full-contrast signal. Now a constant field says `constant −0.070`. Scales are
per component, not shared: DeepSDF latent dimensions are not commensurable.

**Two redraw tiers.** Geometry follows the drag at 120 ms; the latent field, the
unit cell, the coverage strip and the volume fraction wait 400 ms for it to
stop. Measured: `f_θ` eval+draw ≈ 38 ms, the `λ(x)` panel ≈ 89 ms rebuilt and
~20 ms updated in place, the volume grid ≈ 100 ms. Panels are updated in place
and the constrained layout is solved once and pinned, re-solved only on a
debounced resize.

---

## Explore 2-D tab

The same tab as Explore, for a decoder trained on **2-D samples** `(x, y, φ)` —
the `datagen` plate with a hole. Such a decoder is `f_θ(λ, x, y)`: one latent
vector is one whole shape, so there is no lattice and no control net. The design
variables are the components of `λ` itself, one slider each, and the tab shows
how `λ₁`, `λ₂`, ... change the shape. Explore lists only 3-D decoders; planar
ones appear here.

| Panel | Same as on Explore |
|---|---|
| **Design variables** (left) | one slider per component of `λ`, starting at the mean of the trained codes; Reset to mean |
| **Geometry** (centre) | `f_θ(λ, ·)` on `[-1, 1]²` as material/void, with material fraction, `f_θ` range and zero level set |
| **Latent coverage** (right) | one strip per component: where the trained codes are, where `λ` is, the widest untrained gap, and the distance to the nearest trained code |

---

## Train tab

Dataset picker with a readiness badge that states the shapes-per-dimension
problem *before* the run instead of logging it afterwards (`d ≥ 2` with fewer
than 40 shapes: the paper used 120). Architecture and budget, a live loss curve
read from the run's own `Logs.pth` checkpoint on a 1.5 s Tk timer — the worker
thread is inside the trainer for the whole run and cannot report anything — and
a table of past runs; double-click opens one in the Explore tab, or in Explore
2-D for a planar run.

**The runs table is newest first.** The date is the `metadata.json` timestamp
— written when a run finished, or when a preset was generated. A run without
one (started by hand, or still training) is dated by its `specs.json` and
shown with a `~`. Clicking a heading sorts by that column; the same heading
again flips the direction, and the sorted column carries a `▾`/`▴`. The
"Start from" list in the hyperparameter window uses the same order.

**Runs can be edited after the fact.** Under the table: *Open*, *Edit...*,
*Load hyperparameters*, *Delete...* and *Refresh*; the same four actions are
on the right-click menu, and F2 / Delete act on the selected row. *Edit...*
opens a small window (`structsept/app/run_editor.py`) with the run's name and
its notes. The name is the directory under `runs/`, so renaming it renames
the decoder everywhere, and the model pickers and this table re-read the
disk at once. The notes are the `Description` the trainer already keeps in
`specs.json` - it only carries the string along, so a finished run stays
loadable and a waiting preset still trains the same - and they are the last
column of the table. *Load hyperparameters* is "Start from" without the
search: it opens the hyperparameter window with that run's settings loaded.
*Delete...* asks first, refuses anything without a `specs.json`, and cannot
be undone. The run being trained right now is locked against all three. The
table takes the wider half of the tab; the two text columns (run, notes) grow
with the window, the numbers keep their width.

The same *Edit...* sits on the decoder picker of both Explore tabs, for a
decoder "trained here" - that is a run directory. It stays grey on a shipped
decoder: those are the library's files, not runs. The pickers are as wide as
their longest label (`widgets.combo_width`), so a run name is no longer
clipped in the drop-down.

**The dataset decides the decoder's input.** Each dataset in the picker says
`2-D` or `3-D` — from `datagen`'s `dataset.json` when there is one, from the
width of the stored rows otherwise. A 2-D set trains `f_θ(λ, x, y)`
(`geom_dimension` 2 in `specs.json`), a 3-D one the usual unit cell. It is not a
setting: a mismatch between the specs and the rows kills the trainer on its
first batch with an `IndexError`, so it is read from the data, never typed.

The Decoder card keeps the four values changed most often — `d`, hidden layers,
width, epochs — as spinboxes. Under them, one line says which of the *other*
hyperparameters differ from the defaults ("Changed: decoder lr 0.001 · loss
huber"), or, in red, why the current set cannot be trained. Train refuses such
a set with the reason instead of crashing a few minutes into the run.

### The hyperparameter window

![Hyperparameter window](figures/gui_hyperparams.png)

"All hyperparameters..." opens a window with **every `specs.json` key the
DeepSDF trainer reads**, grouped as the trainer uses them: architecture, latent
codes, loss, the two learning-rate schedules, sampling and batches, budget and
checkpoints. Each row carries a sentence on what the value does to `f_θ` and
the `specs.json` key it lands in. The schema behind it is
`structsept/app/hyperparams.py`: one `Field` per value, and the window, the card
summary and `training.write_specs` are all built from it.

- **It edits a draft.** Nothing reaches the card until Apply, and Apply stays
  grey while anything is an error. The four card values are the same values in
  both places. The window holds the input grab, so the card cannot change
  underneath it.
- **Checks run as you type.** Errors are combinations the library crashes on,
  each confirmed against the real code: an odd `SamplesPerScene`, a skip
  connection at layer 0 or at `layers + 1`, `width ≤ d + 3` with a skip, a
  `LogFrequency` longer than the run. Warnings are legal but almost certainly
  unintended (a batch larger than the dataset, `tanh` output with `δ ≥ 1`).
  Notes explain an effect that is easy to misread, such as the default step
  schedule never decaying in a 200-epoch run. Clicking a line jumps to the
  field.
- **The learning-rate card is a table.** The trainer's Adam has two parameter
  groups, decoder weights and latent codes, and gives each its own schedule
  (`Step`, `Warmup` or `Constant`). A row the chosen type does not use shows a
  dash, and disappears when neither column needs it.
- **"Start from"** loads the settings of a local run or of a shipped decoder
  (`ChiAndCross`, `Primitives2D`, …) into the draft. Whatever could not be
  carried over — another architecture, non-uniform layer widths — is said in
  the status line, not dropped silently.
- **"Import sheet..."** loads the supervisor's Excel template
  (`docs/hyperparameters/NN_Training_Hyperparameters_Template_Clean.xlsx`:
  one hyperparameter per row, its value in column B). The template does not
  speak the trainer's language one to one, so `hyperparams.from_sheet` does
  the arithmetic and says so in the status line: *Points per training step*
  is the whole batch (samples per shape = points ÷ *Geometries per batch*,
  rounded down to an even number), *Latent initialization variance* is per
  component while the trainer draws codes from N(0, σ²/d) (σ = √(variance·d)),
  and the decay factor and interval go to both learning-rate schedules as
  Step. Rows that are not settings — number of geometries, samples per
  geometry, sampling strategy, activation, optimizer — are checked against
  what the app does and reported when they disagree; blank rows keep their
  defaults; a row the template does not have is named and ignored. The file
  is read with `structsept/app/xlsx.py`, a standard-library reader of the
  cells of one worksheet, rather than adding `openpyxl` to the environment.
- **The window remembers where its values came from.** Load or Import, then
  Apply: the card says "From run: X (loaded 14:02)", and the window, opened
  again later, has that source selected in "Start from" and says the same in
  its status line — plus "edited since: epochs, width" for whatever was
  changed by hand afterwards, on the card or in the window. The comparison
  is against the values as loaded (`hyperparams.origin_summary`), so a set
  that *started* as a run's settings is never presented as that run's
  settings. "Reset to defaults" forgets the source.
- **The `specs.json` page** renders the file the next run will write, from the
  draft. It answers "what exactly will the trainer see?" without starting a
  run.
- **"Not editable here"** lists what the window cannot change and why. Some
  values the app pins: the CPU, no data-loader processes, the
  `deep_sdf_decoder` architecture. Others the trainer hardcodes whatever the
  file says: gradient clipping at 1.0 (`GradientClipNorm` is read, then
  overwritten) and the 100-epoch ramp of the code regularization.

With every value at its default, the window writes exactly the `specs.json` the
app wrote before it existed. The test suite pins that.

---

## Running it from the desktop

`install_shortcuts.bat` at the repo root (double-click, or
`uv run python -m structsept.app.launcher --install`) writes two shortcuts into
`Desktop\NN`: *Lattice explorer* and *SDF maker*. They are generated per machine
and never committed: a `.lnk` stores absolute paths, and every clone's `.venv`
lives somewhere else. Running the installer again is always safe.

Both shortcuts run `structsept/app/launcher.py` under `pythonw.exe`. The
launcher exists because a shortcut differs from a terminal in three ways:

- **No console.** Under `pythonw`, `sys.stderr` is `None`, and the library's
  tqdm bars (`deep_sdf/training.py`, `sampling.py`) write to it
  unconditionally, so training from a shortcut crashed on the first epoch with
  `'NoneType' object has no attribute 'write'`. The launcher gives the process
  `outputs/logs/<window>.log` as its stdout and stderr and disables tqdm — the
  Train tab reads progress from `Logs.pth` anyway. Whatever a terminal would
  have shown (tracebacks, Tk callback errors, a segfault report) is in that
  file. A terminal run is left untouched.
- **15–30 s of nothing.** torch, DeepSDFStruct and matplotlib take that long to
  import here, and a shortcut that shows nothing gets double-clicked again. The
  splash appears within a second and counts the seconds; the import runs on a
  worker thread, and `build_app(root=...)` then builds the app into the root
  the splash already owns. A second `tk.Tk()` would not do: the SDF maker's
  master-less `StringVar`s bind to whichever interpreter is the default.
- **Staying current.** Code changes need nothing — both packages are editable
  installs, so a shortcut always runs the files in the clone. Dependencies are
  the one gap, so before importing, the launcher hashes `pyproject.toml`,
  `uv.lock` and `DeepSDFStruct/pyproject.toml` and runs `uv sync` (with a
  splash message) when they differ from the last successful sync. It will not
  update while another structsept window is open: Windows cannot replace a
  loaded DLL, and a sync that stops halfway is worse than a stale environment.
  Offline, without `uv` reachable, or with another window open it says so in
  the log (`[launcher] …`) and starts anyway; a package that is then missing
  becomes a dialog naming the package and the fix, not a silent exit.

For someone new to the project the whole setup is: install `uv`,
`git clone --recursive …`, double-click `install_shortcuts.bat`. The batch file
fetches the submodule if the clone forgot `--recursive`, runs `uv sync` (the
first time downloads torch and takes minutes) and writes the shortcuts.

---

## Two windows at once

Nothing stops a second explorer next to the first: each window is its own
process with its own Tk interpreter and state - `busy` is per window, so one
can explore while the other trains - the shortcut's instance lock has room
for 16, and the launcher's log is opened in append mode. What the windows
share is the disk, `runs/` above all, and that is where one can get in the
other's way:

- **A run finished in one window shows in the other after *Refresh*.** The
  runs table and the model pickers re-read the disk on the app's own events
  (its own run ending, a rename, a delete), not on a timer.
- **The same run name in both is a collision.** Train checks for an existing
  `specs.json` when it starts, and the automatic name is minute-resolved
  (`<dataset>_d<d>_<yyyymmdd_hhmm>`), so two windows started on the same
  dataset within a minute would write into one directory. Type a name.
- **Renaming or deleting a run the other window is training** is refused by
  Windows while a file of it is open, and would otherwise break that run at
  its next checkpoint: `run_editor.training_now` only sees this window's run.
- **Two trainings share the CPU.** torch uses every core, so the epoch time
  of both goes up; the 4 h plate scripts were run one after the other for
  that reason.

---

## Tests

`tests/test_app_launcher.py` covers the launcher without showing a window: the
console-less stream redirect (with a real tqdm bar under `None` streams), the
dependency hash and the sync guards, the splash-to-app handover, the
missing-package dialog, and — on Windows — that the written `.lnk` files
resolve to this clone's `pythonw.exe` and icons. Each launch scenario runs in
a fresh interpreter, one of them under `pythonw.exe` with nothing attached,
which is the shortcut's situation for real. That is also why they are not
in-process: several Tk interpreters created and destroyed inside one pytest
process, with pytest's fd capture swapping the std handles between them,
failed to read `init.tcl` every third run.

`tests/test_app_explore.py` drives the real widgets off-screen. Each test is a
defect that shipped once: the context column freezing on a resolution change,
an unclamped layer spinbox indexing past the design vector, a stale mesh
surviving a design edit, two threads on one SDF, and the volume fraction's node
-grid bias.

`tests/test_app_hyperparams.py` covers the hyperparameter window from the
schema up. Every set the validator accepts builds a working `DeepSDFDecoder`,
and every architecture the decoder rejects is refused by the validator. One
four-epoch run on two synthetic spheres checks that the learning rates the
trainer logged, the checkpoints it kept and the layer shapes it saved are the
values typed into the window. The window itself is driven off-screen.

`tests/test_app_explore2d.py` runs the planar path end to end on a 6-shape
`datagen` plate set: the dataset is recognised as 2-D, a two-epoch throwaway run
is trained by the Train tab's own code, lands on Explore 2-D and not on Explore,
and its sliders are driven off-screen.
`tests/test_app_runs.py` covers editing runs after the fact: the file
operations (`check_run_name`, `rename_run`, `write_description`,
`delete_run`) on temporary directories, and the editor driven off-screen from
the runs table and from the Explore 2-D picker - a rename that lands in every
list, a taken name refused, the run being trained locked, a delete that asks
first, *Load hyperparameters* reaching the window.

`tests/test_datagen.py` covers the generator itself — the field against a
brute-force distance, the file contract, overwrite safety. Run them with the
rest:

```bash
uv run pytest tests/ -q
```

---

## Known limits

- The trained-range test is a per-component bounding box. For `d ≥ 2` that box
  is not the trained set; the nearest-code distance is there because of it.
- `Primitives*` decoders carry their shape information outside the stored latent
  codes, so the coverage panel is uninformative for them. The header says so.
- The 3-D preview is a flat-shaded, depth-sorted `Poly3DCollection`. It is
  static: "Open 3D window" still gives the real pyvista view, and that one
  still blocks the interface until closed — VTK needs the main thread.
- The extracted surface belongs to the design it came from. Moving any control
  point drops it, and Export STL goes grey until you extract again.
- Extraction and the interface share one SDF object, and `LatticeSDFStruct`
  writes a latent vector per query point into its microtile, so whoever
  queries second gets a shape mismatch. The design and view panels are
  disabled and pending redraws are cancelled while the mesher runs.
- The SDF maker reads its mesh folder on the Tk thread, so it freezes while
  loading a large folder. Pre-existing, and kept as-is with the rest of it.
- Explore 2-D shows the field, not a mesh: there is no surface extraction or
  STL export for planar decoders yet.
- Runs are listed from the app's own events, not from a timer: a run that
  another window finished, renamed or deleted shows up after *Refresh*.
