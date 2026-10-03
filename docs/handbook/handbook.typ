// Struct_Sept handbook: the code and the workflow, for someone new to the project.
// Build from the repo root (nothing is installed into the project environment):
//   uv run --no-project --with typst python docs/handbook/build.py
// Chapters live in chapters/, the look and the macros in template.typ.
// Companion: board.html, an interactive map of the same material.

#import "template.typ": *

#show: handbook.with(
  title: [Struct_Sept handbook],
  subtitle: [Lattice structures with a neural network as geometry: \ how the code is organised and how the work flows],
  date: [2 October 2026],
  state: [branch `gui-train-tab` at commit `6c07cfc`, plus the uncommitted work on disk (triangle family, 166-shape hole dataset); submodule `DeepSDFStruct` at `bbe9881`],
  cover: [
    #set text(size: 11pt)
    #set par(justify: false)
    This handbook is for someone joining the project. It explains what the code does, why it is
    organised the way it is, and how a day of work actually runs: build a dataset, train a decoder,
    look at what it learned, and push a geometry through to a finite element model.

    #v(0.4cm)
    #grid(columns: (1fr, 1fr), column-gutter: 14pt,
      block(fill: rgb("#eef3f9"), inset: 10pt, radius: 3pt, width: 100%)[
        *If you are new*, start with Chapter 1. Chapters 2–4 are background (method,
        repository, library), Chapters 5–8 follow the code (data, training, GUI, FEM),
        Chapter 9 has the recipes and Chapter 10 the state of the research.
      ],
      block(fill: rgb("#edf7f0"), inset: 10pt, radius: 3pt, width: 100%)[
        *Open `board.html`* (same folder) in a browser for the visual version: a
        Miro-style board with boxes and arrows, a guided tour, and links that open
        VS Code at the right line.
      ],
    )
  ],
)

#[
  #show outline.entry.where(level: 1): it => { v(0.5em); strong(it) }
  // Appendix subsections are left out so the contents fit on two pages.
  #outline(
    title: [Contents],
    depth: 2,
    indent: auto,
    target: heading
      .where(level: 1, outlined: true)
      .or(heading.where(level: 2, outlined: true).before(<app:glossary>)),
  )
]

#include "chapters/01-idea.typ"
#include "chapters/02-method.typ"
#include "chapters/03-repo.typ"
#include "chapters/04-library.typ"
#include "chapters/05-datagen.typ"
#include "chapters/06-training.typ"
#include "chapters/07-gui.typ"
#include "chapters/08-fem.typ"
#include "chapters/09-workflows.typ"
#include "chapters/10-status.typ"

#show: appendix
#include "chapters/11-reference.typ"
