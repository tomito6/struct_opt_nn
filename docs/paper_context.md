# Method context — the paper behind this project

Distilled from: Kofler M., Giritsch M., Elgeti S., *"Structural optimization of
lattice structures using deep neural networks as geometry representation"*,
Graphical Models 142 (2025) 101307, doi:10.1016/j.gmod.2025.101307.
Institute of Lightweight Design and Structural Biomechanics, TU Wien.

The full PDF is at `../Struct_Opt_Neural_Networks.pdf`, one level above the VS Code
workspace. This file exists so the method is readable from inside the workspace.
It is a summary — for equations, figures and the appendix, open the PDF.

## The core idea

Classical lattice optimization either assumes scale separation and periodicity
(homogenization) or reduces cells to beams/shells. Both restrict the geometry.

Here the unit-cell geometry is encoded in the **latent space of a DeepSDF network**
instead. That gives a continuous, low-dimensional design space restricted to
geometries the network was trained on — unlike SIMP-style topology optimization,
which can produce anything. The latent vector is then allowed to **vary continuously
in space**, which grades the lattice. No scale separation and no periodicity are
assumed: a **full-scale FEM** runs at every optimization step.

What makes this optimizable at all is that every stage is differentiable —
including mesh extraction, via FlexiCubes.

## Notation → code

| Paper | Meaning | In the code |
|---|---|---|
| `f_θ` | trained DeepSDF decoder | `SDFfromDeepSDF(get_model(...))` |
| `λ(x)` | latent vector as a field over space | `SplineParametrization` |
| `λ̂` | spline control points = **the design variables** | `next(parametrization.parameters())` |
| `T(x)` | transformation / stacking function | inside `LatticeSDFStruct` |
| `t_x, t_y, t_z` | number of cells per direction | `tiling=(tx, ty, tz)` |
| `Ω`, `Γ` | interior (SDF < 0) and surface (SDF = 0) | — |
| `Ω̄` | domain after freeform deformation | `TorchSpline` deformation |
| `J` | compliance (objective) | computed from the `torch-fem` solve |
| `V` | volume (constraint) | from the tet mesh |

## The transformation function

Eq. (18): `T(x) := 4·| t_x·x/2 − ⌊t_x·x/2 + 1/2⌋ | − 1`

(The floor argument is `t_x·x/2 + 1/2`, i.e. `t_x·x/2` rounded to the nearest
integer — that is what `lattice_structure.transform` implements. It makes `T` a
triangle wave: one cell rises from −1 to 1, the next falls back. With `t_x·x + 1/2`
inside the floor it would be a sawtooth, which does not mirror.)

One **global, continuous** function, applied per coordinate, mapping the unit cube to
the network's input range `[−1, 1]`. This is a deliberate departure from Chabra et al.
(Deep Local Shapes), who use one discontinuous transformation per cell plus a
weighting function.

Why it matters: it mirrors at cell interfaces, so the SDF stays continuous even for
**asymmetric** unit cells — which is the normal case for network-represented
geometry. Continuity holds only if both the transformation function and the latent
field are continuous. Its range is independent of the number of cells.

The latent field uses **B-splines with local support** (Eq. 21):
`λ(x) = Σ_i φ_i(x) λ̂_i`. Local support means a control point changes its region
without disturbing the rest of the design domain. The number of spline control points
is independent of the number of unit cells.

## The loop

Offline (once):

1. **Training data** — geometry scaled into `(−1, 1)`, points sampled on the domain,
   signed distance to the surface computed. 500 000 points per training shape,
   16 000 drawn per training step, batches of 10 geometries. The 16 000 is **per
   shape** (`SamplesPerScene` in the shipped specs), so one optimizer step sees
   160 000 points. Sampling is uniform — unlike Park et al., near-surface points
   are *not* prioritized; the authors found uniform sufficient for unit-cell
   complexity.
2. **Training** — minimize the difference between network output and sampled
   distances. Latent vectors initialized from `N(0, 0.01)`.

### Where the reference code differs from the printed paper

Checked on 2026-10-02 against `DeepSDFStruct/DeepSDFStruct/deep_sdf/training.py`, the decoder and
the shipped `round_cross` / `chi_and_cross` specs. The loss (Eq. 7–8, δ = 0.1), Adam
with 5e-4 / 1e-3 and the step schedule of Eq. 10 (first halving at epoch 500) are
the same in both. These are not:

| | Paper | Reference code (what every run here inherits) |
|---|---|---|
| Regularizer, Eq. 6 | `σ · ‖λ‖²` per shape | `σ · mean(‖λ‖)`, the norm **not** squared |
| Its weight, Eq. 9 | `σ₀ · min(1, 1/n_epoch)`, decays | `σ₀ · min(1, epoch/100)`, ramps **up**; hardcoded |
| Latent init | variance 0.01 | `CodeInitStdDev` default 1.0 → variance `1/d`; the shipped codes fill `[−1, 1]` |
| Code bound | not mentioned | `CodeBound` 1.0 as `Embedding(max_norm)` |
| Gradient clipping | not mentioned | max-norm 1.0 on the decoder, hardcoded |
| Architecture | 6 × 128, ReLU, dropout 0.2 | the same, plus a skip connection at layer 2 and weight norm on every layer |
| Per-step sampling | 16 000 uniform points | half from inside, half from outside, whatever the volume fraction |

The regularizer is about 0.3 % of the loss in this project's runs, so its form and
ramp direction change nothing measurable. The latent init does matter for the
**latent range**: runs here use the paper's variance (`σ = 0.1·√d`) and end with
code norms of 0.16–0.59, not the `[−1, 1]` of the shipped decoders — bounds for the
optimization have to come from each run's own codes.

Online (every iteration):

3. **Network inputs** — rectilinear grid → transformation function; latent vector
   interpolated from the spline control points. Both defined on the unit cube.
4. **Mesh generation** — FlexiCubes (Shen et al., differentiable extension of dual
   marching cubes) → surface mesh; then freeform deformation to reach non-rectangular
   domains; then tetrahedralization. Plain marching cubes was rejected for producing
   sliver triangles that are bad for FEM.
5. **Forward simulation** — FEM for the physical response.
6. **Update** — MMA (Method of Moving Asymptotes) updates `λ̂` using gradients from
   adjoint sensitivity analysis plus autodiff. The **FFD control points are not design
   variables** in this work, though the method extends to them.

## The optimization problem

```
min_λ̂  J(λ̂)                    compliance,  J = ∫ σ(u) : ε(u) dx
s.t.   V(λ̂) ≤ V_target          volume
       c(u) = 0 on Ω̄            equilibrium
```

The gradient follows the chain `λ̂ ↦ λ ↦ f_θ ↦ Ω ↦ Ω̄ ↦ J` — spline interpolation,
network, FlexiCubes, FFD, FEM. Every link must stay differentiable; that is the
constraint that shapes the whole codebase.

## Test cases (useful as reference configurations)

**Test case 1 — cantilever beam.** Training geometry: three perpendicular cylinders,
radius varied 0.1 → 0.75, 20 shapes, **latent dimension 1** (network input = 4
neurons: 3 coordinates + 1 latent). Domain `b = (2, 1, 1)`, left side fully clamped,
uniform unit pressure on the top surface. Compliance minimization, volume constraint
0.5 (a quarter of the domain). Lattice `6 × 3 × 3` cells, linear B-spline with
`9 × 2 × 5` control points, resolution 20 cubes per cell per direction ≈ **200 000
tetrahedra**. One iteration ≈ 1 min 50 s on an Intel Xeon Gold 6326; 40 iterations
≈ 80 min. FEM + adjoint sensitivity is ≈ 1 min 30 s of that — the bottleneck.

**Test case 2 — hinge-like structure.** Training on spline-based geometries rather
than analytic SDFs: a cross cell (thickness 0.2 → 0.3, 20 shapes) and a chi-like
drilled cell (5 parameters; β varied 0 → π/6 and w from −0.2 → 0.4, 10 steps each,
100 shapes) — 120 geometries total, **latent dimension 2**. Objective: maximize
stiffness for load case 1 while keeping load case 2's compliance below a target.
Mean reconstruction error ≈ 0.0156 (~0.8%).

## Known limitations (from the conclusion)

- Sharp edges are not reconstructed accurately; struts come out slightly thicker than
  in the training geometries.
- Higher-dimensional latent spaces (needed for more diverse cells) are expected to
  bring nonconvex design spaces.
- Manufacturing constraints such as minimum feature size are enforced **only** through
  the choice of training data — there is no explicit constraint on the latent space.
- Manufacturing imperfections are not modeled, so predicted behavior will differ from
  fabricated parts.
- Open directions named by the authors: splines as the transformation function,
  immersed boundary methods instead of the body-fitted mesh, feature-size constraints
  on the latent space or as post-processing.
