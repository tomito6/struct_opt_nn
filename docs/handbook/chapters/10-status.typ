#import "../template.typ": *

= Where the research stands <ch:status>

This chapter is a snapshot taken on *2 October 2026*. Its source is the lab notebook
`experiments/IDEIAS.md` (written in Portuguese), checked against the run folders in
`runs/` and the logs in `outputs/logs/`. Those folders are gitignored, so the notebook is
the only versioned record of what was found. If this chapter and the notebook disagree,
the notebook is newer. Write your own findings there as well.

== The research question

The paper treats the latent space as a design space. MMA moves the spline control points
$hat(lambda)_i$ inside box bounds, and the library template uses $[0.15, 0.75]$ for every
one of them (#f("DeepSDFStruct/tests/test_structural_optimization.py", 65)). Those bounds
make sense only if the box contains shapes the decoder was trained on. A DeepSDF
auto-decoder does not guarantee that. It has no encoder. Each training shape gets a free
code, initialized as noise and optimized together with the weights, so the network
invents its own coordinates. Before building an optimization on self-trained decoders,
this repo therefore asks:

#align(center, block(width: 85%, {
  set par(justify: false)
  emph[Does the latent space that an auto-decoder learns on its own recover the
    parameters that generated the shapes, and how many latent dimensions does that take?]
}))

Why the answer matters for the optimization:

- *Reading a design back.* If the codes are an affine relabelling of the parameters, an
  optimized code can be mapped back to $(x_c, y_c, r)$ with the fitted map, and a
  constraint written in parameter space stays linear in $lambda$. If the codes are
  curved, folded, or have lost a parameter, neither is true.
- *The admissible set is not a box, even in parameter space.* A hole must keep a margin
  to the plate edge, so the largest radius shrinks as the centre moves outwards: the
  feasible set in $(x_c, y_c, r)$ is a pyramid (module docstring of
  #f("datagen/plate_hole_params.py", 1)). `datagen` therefore also stores each shape as a
  point $(u_x, u_y, t)$ of the unit cube, where every point maps to an admissible
  shape. Freezing the codes to these unit-cube coordinates during training would keep the
  design space a box all the way to MMA. `HoleParameters.latent_codes`
  (#f("datagen/plate_hole_params.py", 688)) and `TriParameters.latent_codes`
  (#f("datagen/plate_tri_params.py", 573)) compute such codes. *No training path uses
  them yet.*
- *Gaps.* A box drawn around the learned codes also contains regions with no training
  code in them, where the decoder extrapolates (rule 4 below).

That leaves two routes: learn the codes and check that they are a clean relabelling of
the parameters, or prescribe them. All experiments so far take the first route. Every
dataset writes `params.csv`, whose row $i$ belongs to latent code $i$. After training, `code_vs_parameter` (#f("structsept/app/unattended.py", 525))
and `compare_codes_with_params` (#f("experiments/train_plate_xyr_4h.py", 261)) compute
per-component correlations, the share of variance on each principal axis, and a linear
or affine fit of each parameter on all code components. The verdict is the $R^2$ of that
fit. How to run these checks is in @ch:training[Chapter]. The test bed is 2-D plates with
exact SDFs (@ch:datagen[Chapter]). A plate of constant thickness with a through-hole is
an extrusion of the 2-D shape, so the 2-D field holds all the information.

== Timeline

The project ran in two phases (@fig:status-timeline). Until about 21 September the work
was learning the library and building the pieces of the online pipeline (geometry, FEM
up to $K$, the point-cloud fit). From 23 September on, it has been the latent-space
question. All training runs on the laptop's CPU (`torch 2.13.0+cpu`; the GPU is not
used), which is why the long runs take hours and are chained overnight
(@ch:training[Chapter]).

#figure(
  diagram(
    spacing: (4mm, 7mm), edge-stroke: 0.7pt, mark-scale: 70%,
    nmod((0, 0), [*31 Aug*\ repo, submodule,\ `CLAUDE.md`]),
    nmod((1, 0), [*early Sep*\ library tour;\ plate, $K$ matrix]),
    nstep((2, 0), [*13 Sep*\ point cloud fit:\ IoU 99.5 %]),
    nmod((3, 0), [*18 Sep*\ `structsept/` +\ `experiments/`]),
    nmod((4, 0), [*21--23 Sep*\ GUI, trainer\ audit, `datagen`]),
    nstep((0, 1), [*23 Sep*\ $d = 2$, 25 plates;\ two hole families]),
    nstep((1, 1), [*27--29 Sep*\ $r$ only: $d = 1$,\ $d = 2$ (8×256)]),
    nstep((2, 1), [*28--29 Sep*\ $d = 2$ drops $r$;\ $d = 3$ keeps it]),
    nstep((3, 1), [*30 Sep*\ coverage gaps,\ n166 (dies)]),
    nstep((4, 1), [*1 Oct*\ triangles:\ $h$, $w$ recovered]),
    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (2, 0), "-|>"),
    edge((2, 0), (3, 0), "-|>"),
    edge((3, 0), (4, 0), "-|>"),
    edge((4, 0), (4, 0.5), (0, 0.5), (0, 1), "-|>"),
    edge((0, 1), (1, 1), "-|>"),
    edge((1, 1), (2, 1), "-|>"),
    edge((2, 1), (3, 1), "-|>"),
    edge((3, 1), (4, 1), "-|>"),
  ),
  caption: [Milestones from the first commit to the triangle results. Blue: code and
    tooling; green: experiments with a measured result. Top row: learning the library;
    bottom row: the latent-space question. Commit `6c07cfc` (29 Sep) is the last one;
    everything from 30 Sep on is uncommitted.],
) <fig:status-timeline>

== Families, datasets and runs

Two plate families exist, each in a one-parameter and a multi-parameter variant
(@ch:datagen[Chapter]). The table lists every run that produced a result or was meant
to. Numbers come from `code_vs_*.csv/json`, `metadata.json` and the run logs; $R^2$ is
the fit of the parameter on all code components. Presets in `runs/preset_*` hold only a
`specs.json` and a `metadata.json`, no checkpoints, and are left out.

#[
  #set text(size: 8.6pt)
  #set par(justify: false)
  #show raw: set text(size: 7.6pt)
  #kv(
    columns: (auto, auto, 1fr),
    [*Dataset (shapes)*], [*Run*; $d$, decoder, epochs], [*Headline result*],
    table.cell(colspan: 3)[_Circular hole, free:_ $x_c$, $y_c$, $r$],
    [`plate_hole_2d_n25` (25)], [`First_test`\ $d = 2$, 4×64, 250],
    [smoke test: $R^2$ $x_c$ 0.07, $y_c$ 0.87, $r$ 0.50; keeps $y_c$, drops $x_c$],
    [`plate_hole_2d_n134` (134)], [`plate_hole_2d_n134_d2_20260928_1338`\ $d = 2$, 8×256, 600],
    [$R^2$ $x_c$ 0.759, $y_c$ 0.794, *$r$ 0.009*: keeps the centre, drops the radius],
    [`plate_hole_2d_xyr` (134)], [`plate2d_xyr_d3_8x256_4h`\ $d = 3$, 8×256, 300],
    [$R^2$ *$x_c$ 0.988*, *$r$ 0.857*, $y_c$ 0.741; principal axes 49/28/23 %],
    [same], [`plate_hole_2d_xyr_d3_20260930_0124`\ $d = 3$, 8×256, 800],
    [complete, loss 0.0058; codes not yet compared with the parameters],
    [`plate_hole_2d_xyr_n166` (166)], [`plate_hole_2d_xyr_n166_d3_ep800`\ $d = 3$, 8×256, 800],
    [*died at epoch 80*; no result],
    table.cell(colspan: 3)[_Circular hole, centred:_ $r$ only],
    [`plate_hole_2d_r_only` (40)], [`plate2d_r_only_d1_8x256_4h`\ $d = 1$, 8×256, 600],
    [code linear in $r$: Pearson −1.000, monotonic],
    [same], [`plate2d_r_only_d2_8x256_4h`\ $d = 2$, 8×256, 600],
    [codes on a line (84.8 % of variance on PC1, Spearman 1.000 along it); the other
      direction is frozen initial noise],
    table.cell(colspan: 3)[_Four triangular holes:_ height $h$, then $h$ and width $w$],
    [`plate_tri_2d_h` (40)], [`plate_tri_2d_h_d1_4x64`\ $d = 1$, 4×64, 1500],
    [code linear in $h$: Pearson −1.000, monotonic (20.6 min)],
    [`plate_tri_2d_hw` (133)], [`plate_tri_2d_hw_d2_4x64`\ $d = 2$, 4×64, 1500],
    [$R^2$ *$h$ 1.000*, *$w$ 0.996*; no single component is monotonic (49.9 min)],
  )
]

`plate_hole_2d_n134` and `plate_hole_2d_xyr` are byte-identical (same seed); only the
name differs. `First_test` has no analysis file; its $R^2$ values were recomputed from
its saved codes with the same affine fit. The small 4×64 preset recipe was enough for the
triangle families; the analysed hole runs after `First_test` used the 8×256 recipe of the
unattended scripts.

== Rules of thumb learned so far

+ *With $d$ equal to the number of generating parameters, the decoder recovers them up
  to an arbitrary affine basis.* The centred hole and the triangle height give a
  straight line ($d = 1$, Pearson −1.000 both). The triangles with $(h, w)$ give
  $R^2 = 1.000$ and $0.996$, yet single components correlate only 0.47 to 0.88 with
  either parameter: the codes fill the $(h, w)$ feasible set (a rectangle with one corner
  cut off, @ch:datagen[Chapter]), rotated and sheared.
  Consequence: judge by a fit on all components, never component by component. In the
  Explore 2-D tab, no single slider is "height" or "radius".
+ *With $d$ smaller than the number of parameters, the decoder keeps what dominates the
  loss.* On 134 free-hole plates with $d = 2$ it kept the centre and dropped the radius
  ($R^2 = 0.009$). Moving the hole changes the SDF over the whole plate; changing the
  radius changes only a ring around the hole. The small first run (25 plates, 4×64,
  250 epochs) instead kept $y_c$ and dropped $x_c$: which parameter survives also
  depends on the data and the training budget.
+ *A latent direction the decoder does not need keeps its random initialization.* In the
  centred-hole run with $d = 2$, the coordinate across the line of codes correlates
  0.96 between epoch 1 and epoch 600 and not at all with $r$; its spread only fell from
  0.092 to 0.053. The gradient along that direction is about zero, and only the weak code
  regularization pulls it in. For an optimizer this would be a design variable the
  geometry may not respond to (still to be measured, see below).
+ *Gaps in the training data become gaps in the latent space.* The Explore tabs shade the
  widest stretch of an axis without a trained code when it exceeds 12 % of the range
  (`GAP_FRACTION`, #f("structsept/app/viz.py", 28)). In the $d = 3$ hole runs the first
  axis, which mostly carries the radius, has a 29 % gap: only 8 of 134 shapes have
  $r > 0.25$, because a large hole needs a central position _and_ $t$ near 1, a corner of
  the unit cube that a 128-point Sobol draw barely reaches. Not every gap is missing
  data: a 23 % gap on the third axis sits next to a corner shape whose nearest neighbour
  is close in parameter space; there the latent space is stretched.

== Open threads and next steps

+ *Rerun the n166 training* (the run died, see below) and check that the radius gap
  closes. The dataset adds 32 holes with $r in [0.25, 0.45]$: 40 shapes above
  $r = 0.25$ instead of 8, and the largest jump between neighbouring radii falls from
  0.108 to 0.017. The script copies the reference run's `specs.json` key for key and
  writes `latent_coverage.json` (`latent_coverage`,
  #f("experiments/train_plate_xyr_n166.py", 160)). The expectation is that the first-axis
  gap disappears and the third-axis gap stays. Budget more than a night: the first launch
  ran 80 epochs in 83 min with the laptop in use (62 s per epoch), and the script's
  docstring now says to plan for 14 h for the 800 epochs. Resuming from epoch 80 is not an
  option: the library trainer accepts `continue_from`
  (#f("DeepSDFStruct/DeepSDFStruct/deep_sdf/training.py", 299)) but discards the latent
  codes it loads (line 517), so the codes would restart from noise, and
  `structsept.app.training.train` does not pass it on anyway
  (#f("structsept/app/training.py", 157)). Relaunch with `--force`.
+ *Find out why $y_c$ is recovered worst at $d = 3$.* The error sits in the lower half
  (RMSE 0.164 for $y_c < 0.45$ against 0.051 above); the worst shapes are holes near the
  bottom edge that the fit places near the middle. The plate is symmetric in $x$ and $y$
  and $x_c$ comes out at 0.988, so the notebook's hypothesis is a training accident, not
  geometry. Three cheap checks: compare the codes of the complete 800-epoch run, look in
  Explore 2-D whether the hole moves down to the bottom edge, and train a second seed.
+ *Measure the dead direction* of the centred-hole $d = 2$ decoder: sweep $plus.minus 0.1$
  across the line of codes from a middle code and see whether the SDF changes.
+ *Four independent triangle heights.* The SDF code already takes one height per
  triangle (`triangles_in_frame`, #f("datagen/plate_tri_sdf.py", 78)); what is missing
  is a sampler in `plate_tri_params.py` and the extra `params.csv` columns.
+ *Prescribed codes.* A training path that freezes the codes to `latent_codes` (the
  second route above), so that MMA gets box bounds by construction.
+ *Towards an optimization loop.* `structsept/fem.py` stops at $K$, and the 2-D plate
  decoders do not plug into `LatticeSDFStruct`; @ch:fem[Chapter] lists what is missing.
+ *Small items.* Two GUI windows can collide on automatic run names (minute resolution)
  and do not refresh each other's run lists; a seconds-plus-PID suffix and a refresh timer
  are about 30 lines (@ch:gui[Chapter]). Two cheap trainer experiments are queued: a
  decoder learning-rate sweep from `1e-4` to `3e-3` with a step decay about every 50
  epochs, and `clampedL1` with $delta = 0.1$ against `L1` with $delta = 0.3$. The early
  queue at the top of `IDEIAS.md` (radius sweep against the analytic volume, a hole
  through the border as a `CappedBorderSDF` test, several holes, other hole shapes, a
  graded lattice with `SplineParametrization`) is also still open.

#block(breakable: false)[
  The first two threads need no new code. For the second, `<run>` is
  `plate_hole_2d_xyr_d3_20260930_0124`:

  ```bash
  # n166 from scratch; add --after outputs/logs/<run>.pid to chain it after another run
  uv run python experiments/train_plate_xyr_n166.py --force
  # code-vs-parameter check of an existing d = 3 run, no training
  uv run python experiments/train_plate_xyr_4h.py --check-only --run <run>
  ```
]

== Broken or incomplete

+ *The n166 run died at epoch 80.* `runs/plate_hole_2d_xyr_n166_d3_ep800` was started
  on 30 Sep for 800 epochs. Its `latest.pth` was last written the same evening, at
  epoch 80. The log ends with the start-up lines and shows no error, and the
  notebook records that the machine did not reboot; the cause is unknown.
  `metadata.json` has `"final_loss": null` and there is no `latent_coverage.json`, but
  the GUI still lists the run as loadable. Do not read its codes as a result.
+ *The "ep800" run stopped at epoch 680.* `runs/plate_hole_2d_xyr_d3_ep800`, started
  from the GUI, has `latest.pth` at epoch 680 and no `800.pth`, `metadata.json` or
  `training_summary.json`. The complete 800-epoch run with an identical `specs.json` is
  `plate_hole_2d_xyr_d3_20260930_0124`. `train_plate_xyr_n166.py` still takes ep800 as
  its default `--like`; that is harmless: its `specs.json` is identical, and its epoch-680
  codes, which the script also reads for the reference coverage, show the same 29 / 5 /
  23 % gaps as the complete run. Three more GUI runs are leftovers:
  `plate_hole_2d_xyr_d3_20260929_0759` (epoch-1 checkpoint only), `Test2` (500 epochs on
  `plate_hole_2d_n25`, never analysed) and `Test_2` (no checkpoints).
+ *Uncommitted work.* The triangle results and the n166 dataset rest on uncommitted code,
  and `gui-train-tab` is not merged (@ch:repo[Chapter], Git state).

#watch[*A test would fail as written (not run).*
  `test_the_xyr_d3_sheet_is_the_same_recipe_sized_for_134_shapes`
  (#f("tests/test_app_hyperparams.py", 543)) expects 300 epochs, but `from_sheet` reads
  600 from `docs/hyperparameters/NN_Training_Hyperparameters_plate2d_xyr_d3_8x256_4h.xlsx`
  (the decay interval, 75, agrees). The run the sheet documents trained 300 epochs, so
  the sheet is the likely error; see the open questions below.]

== Open questions for the supervisor

Recorded in the notebook and not yet answered:

- What plate thickness fits the real HiWi case? The 0.1 m in `structsept.plate_with_hole`
  was a guess.
- Is the hole a requirement of the problem or an exercise?
- Point-cloud input: is the target part already porous, so that fitting the latent field
  to the cloud is right (this is what `experiments/pointcloud_to_lattice.py` does), or is
  it solid, so that the lattice should only fill it (a trim, which leaves the latents
  free for MMA)? Orienting the normals of a raw cloud properly is also not implemented.

#block(breakable: false)[
  Not recorded in the notebook, but the next phase depends on them:

  - Which route into the optimization: learned codes read back through the fitted affine
    map, or prescribed unit-cube codes?
  - How do the plate-level 2-D decoders connect to the paper's setting, where the decoder
    describes a 3-D unit cell that `LatticeSDFStruct` tiles?
]

From a check of the training recipes against the paper (2 October 2026); recorded here,
not decided:

#[
  #set par(justify: false)
  - *Decay interval 75.* The `xyr_d3` sheet halves the learning rate every 75 epochs,
    where the two `r_only` sheets use 150. The 75 went into the three 800-epoch runs
    (`plate_hole_2d_xyr_d3_ep800`, `plate_hole_2d_xyr_d3_20260930_0124`, the `n166` run),
    so from epoch 750 the rate is 1/1024 of its start. In the complete run the loss falls
    only from 0.0060 at epoch 450 to 0.0058 at epoch 800.
  - *Points per step.* The sheet row _Points per training step_ is divided by the batch
    (16~384~/~4 = 4096 per shape), whereas the shipped `round_cross` specs use the
    paper's 16~000 per shape (`SamplesPerScene` 16000).
  - *Regularization ramp.* The paper's Eq. 9 prints the code-norm weight as
    $sigma_0 min(1, 1 slash n_"epoch")$, a decay; the library ramps it up,
    $min(1, "epoch" slash 100)$, at
    #f("DeepSDFStruct/DeepSDFStruct/deep_sdf/training.py", 622) (also noted in
    `docs/paper_context.md`).
]

== Upstream bug candidates

#block(breakable: false)[
  Found while reading or running the library, not yet reported to `DeepSDFStruct`. The
  hardcoded gradient clip and the late seed (trap table in @ch:training[Chapter]) are
  candidates too; the other trainer traps there are documented behaviour rather than bugs.

  #set text(size: 9pt)
  #set par(justify: false)
  #kv(
    columns: (1.25fr, 1fr),
    [*Problem and location*], [*Effect and workaround*],
    [`create_3D_mesh(mesh_type="volume")` leaves cavities: a solid cube keeps only about
      70 % of its volume at any resolution.\ #f("DeepSDFStruct/DeepSDFStruct/mesh.py", 669)],
    [Mesh the surface and call `tetrahedralize_surface`. The library's optimization test
      uses the lossy path (line 77 of the test).],
    [`plot_reconstruction_loss` smooths over 41 steps and, when given a CSV path, fails
      with fewer than 40.\
      #f("DeepSDFStruct/DeepSDFStruct/deep_sdf/plotting.py", 121)],
    [`ValueError` after short reconstructions that write `loss_csv_path`; run at least 40
      steps or drop the CSV.],
  )
]
