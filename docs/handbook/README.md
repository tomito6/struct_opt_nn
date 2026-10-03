# Handbook

Documentation of the code and the workflow for someone new to the project.

| File | What it is |
|---|---|
| `handbook.pdf` | The handbook: the method, the code part by part, recipes, research status, reference appendix |
| `board.html` | The same material as an interactive Miro-style board (boxes, arrows, guided tour). Open it in a browser; no server needed |
| `handbook.typ`, `chapters/*.typ`, `template.typ` | Typst sources of the PDF |
| `build.py` | Builds `handbook.pdf` |
| `_check.py` | Compiles one chapter on its own (syntax + layout check) |
| `_check_board.js` | Checks that `board.html` parses and that its cards, arrows and file links are consistent |

## Rebuilding

From the repo root. Typst runs in a throwaway environment, so nothing is added to
`pyproject.toml`:

```bash
uv run --no-project --with typst python docs/handbook/build.py
```

Add `--png 110` to also render every page to `_pages/` for a visual check. The first build
downloads the `fletcher` diagram package into the Typst cache.

After editing the board:

```bash
node docs/handbook/_check_board.js
```

## Updating

- The board's content is the `BOARD` block inside `board.html`, written with four helpers:
  `F` (frame), `N` (card), `E` (arrow), `T` (tour step). The comment above the block
  explains their arguments.
- File references in both documents carry line numbers. They were checked against the code on
  2 October 2026 (branch `gui-train-tab` plus uncommitted work) and drift as the code changes.
- `_build/` and `_pages/` are scratch output and gitignored.
