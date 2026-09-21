# Context maintenance log

One entry per pass. See `docs/context-maintenance.md` for the procedure.
An entry dated today means the pass already ran today.

## 2026-08-31
- Initial setup. Read the repo, the `DeepSDFStruct` submodule and the reference paper
  (`../Struct_Opt_Neural_Networks.pdf`).
- Created: `CLAUDE.md`, `docs/paper_context.md`, `docs/context-maintenance.md`,
  `.claude/commands/update-context.md`.
- Open: the project description covers what the codebase *is*, not what the current
  HiWi task is. Add a "Current task" section once that is known.

## 2026-09-18
- Restructured the repo. `playground/` and `scripts/` mixed library code with runnable
  scripts, so reusable modules were being imported *out of* `playground/` — including
  by `docs/stiffness_theory/make_figures.py`, via a relative `sys.path.insert` that
  only worked from the repo root.
- New layout: `structsept/` (library, importable) and `experiments/` (runnable, never
  imported). `oficina/` became `structsept/app/`; its `data/` and `runs/` moved to the
  repo root. Extracted `structsept/fem.py` (215 lines) out of
  `plate_with_hole_stiffness.py`, which is now a 175-line driver.
- `pyproject.toml` now installs this folder editable (`package = true`), so all four
  `sys.path.insert` hacks are gone and every script runs from any directory.
- Created `docs/structure.md`: the dependency rule, a "where does my file go" table,
  the annotated tree, and the run commands. `CLAUDE.md` links to it and is at 144 lines.
- Verified: 7 entry points start, `pytest tests/` passes, and
  `plate_with_hole_stiffness.py --solid` runs end to end (rigid-body residual 3e-16).
- Open: `np.py` and `round_cross_00000.npz` are still loose scratch files at the root,
  and nothing is committed yet — the repo is still at its single initial commit.
