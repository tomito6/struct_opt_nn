# Keeping CLAUDE.md current

The procedure for the daily context-maintenance pass. Run manually with
`/update-context` in VS Code, or automatically by the scheduled task.

**The goal is a `CLAUDE.md` that stays small and true — not one that grows.**
A file that accumulates is worse than one that never changed: past ~200 lines it eats
context every session and adherence drops. Every pass should be prepared to *delete*.

---

## Step 0 — Is there anything to do?

Read `docs/context-log.md`. If it already has an entry dated today, stop immediately
and say so. This makes the pass safe to run many times a day.

## Step 1 — Gather evidence

Only from the repo. Never from memory, never from guesswork.

```bash
git log --oneline --since="<date of last log entry>"
git status --short
git diff --stat HEAD~5..HEAD          # if there are commits
ls -lt --time-style=+%Y-%m-%d $(git ls-files) | head -30   # recently touched
```

Also look for:

- New or renamed top-level files and directories.
- Changes to `pyproject.toml` or `uv.lock` — new dependencies, changed Python version.
- New test files, or tests that were deleted.
- A new commit in the `DeepSDFStruct/` submodule pointer (`git diff` shows it as a
  changed subproject hash) — the library API may have moved.
- New documents in the parent folder `../` (papers, notes from the supervisor).

## Step 2 — Decide what actually belongs

Ask of each candidate: **would a new person need this on day one, in every session?**

Belongs in `CLAUDE.md`:

- A command that changed, or a new one that gets run often.
- A new module, script or directory that is now part of the normal workflow.
- A convention that was decided and will hold (naming, units, where outputs go).
- A trap that cost real time and will cost it again.
- A dependency or environment change.

Does **not** belong:

- Anything derivable by reading the code — file listings, function signatures,
  architecture that the module docstrings already state.
- One-off debugging, a bug that got fixed, the state of today's experiment.
- Results, numbers, run outputs. Those go in a lab notebook, not here.
- Anything already covered by `DeepSDFStruct/AGENT_INSTRUCTIONS.md` (imported) or
  `docs/paper_context.md`.
- Speculation about what will be done next.

Method details from the paper go to `docs/paper_context.md`, not `CLAUDE.md`.

## Step 3 — Curate, don't append

Before adding anything, re-read `CLAUDE.md` in full and remove:

- Statements that are no longer true.
- Entries that contradict each other — conflicting instructions make Claude pick
  arbitrarily, which is worse than having neither.
- Anything that has become obvious from the code since it was written.

Then integrate the new material **into the existing sections**. Do not bolt on a
"Updates" or "Recent changes" section — that is how these files rot.

Budget: **CLAUDE.md stays under 150 lines.** If a change would push it over, something
else comes out, or the material moves to `docs/` or to a path-scoped rule under
`.claude/rules/` (which loads only when Claude touches matching files).

If nothing earns a change, change nothing. That is a normal outcome.

## Step 4 — Log

Append one entry to `docs/context-log.md`, newest at the bottom:

```markdown
## 2026-09-01
- Evidence: 3 commits, new `runs/` directory, torch bumped to 2.9.
- Changed: added `runs/` to the layout; replaced the old smoke-test command.
- Removed: the note about untracked pyproject.toml (committed in a1b2c3d).
```

Write the entry even when nothing changed — that is what stops the pass from
running again the same day.

## Hard rules

- **Never run `uv`, `python` or `pytest` against this folder from a Linux shell over
  the device bridge.** The `.venv` here is a Windows environment; running from Linux
  can recreate and destroy it. Verification of that kind belongs in a VS Code terminal
  on the machine itself.
- **Never edit anything under `DeepSDFStruct/`** — third-party submodule.
- **Never invent a fact.** If a command, path or convention cannot be verified in the
  repo, it does not go in the file.
- **Never rewrite `CLAUDE.md` wholesale.** Targeted edits only, so the diff stays
  reviewable.
- Do not commit. Leave the changes in the working tree for review.
