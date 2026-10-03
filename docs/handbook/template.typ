// Shared look and macros for the handbook. Imported by handbook.typ and every chapter.
// Build (from the repo root, nothing is installed into the project env):
//   uv run --no-project --with typst python docs/handbook/build.py

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

// ---------------------------------------------------------------- palette
#let c-accent  = rgb("#2f5f9e")
#let c-concept = rgb("#7a4ddb")
#let c-module  = rgb("#2f6fdf")
#let c-symbol  = rgb("#0e9384")
#let c-data    = rgb("#d97b00")
#let c-ui      = rgb("#d1437a")
#let c-ext     = rgb("#7d838c")
#let c-step    = rgb("#2e9a4e")
#let c-grad    = rgb("#d6336c")
#let c-ink     = rgb("#1d2126")
#let c-muted   = rgb("#5d6570")

// ---------------------------------------------------------------- boxes
#let _box(label, color, fill, body) = block(
  fill: fill, stroke: (left: 2.4pt + color), inset: (x: 10pt, y: 7pt), radius: 2pt,
  width: 100%, above: 0.9em, below: 0.9em, breakable: true,
)[#set par(justify: false)
  #text(weight: "bold", fill: color, size: 0.92em)[#label] #h(0.5em) #body]

/// Ties a piece of theory to the file and line that implements it.
#let incode(body) = _box("In the code", c-accent, rgb("#eef3f9"), body)
/// Something that has bitten someone before.
#let watch(body) = _box("Watch out", rgb("#c2410c"), rgb("#fdf2e8"), body)
/// A practical shortcut.
#let tip(body) = _box("Tip", c-step, rgb("#edf7f0"), body)
/// Work that exists on disk but is not committed / not finished.
#let wip(body) = _box("In progress", c-concept, rgb("#f3effc"), body)

/// File reference: #f("structsept/fem.py", 42) -> structsept/fem.py:42
#let f(path, ..line) = {
  let l = line.pos()
  let s = if l.len() > 0 { path + ":" + str(l.at(0)) } else { path }
  box(text(font: ("Cascadia Code", "DejaVu Sans Mono"), size: 0.86em, fill: rgb("#28466e"), s))
}

// Shell commands: write them as ```bash (or ```powershell for Windows-only
// commands) fenced raw blocks; the show rule in handbook() turns those into
// dark terminal boxes.

/// Key / value table with a light header.
#let kv(columns: (auto, 1fr), ..cells) = table(
  columns: columns, stroke: none, inset: (x: 5pt, y: 4pt),
  fill: (_, y) => if y == 0 { rgb("#e9eef5") } else if calc.odd(y) { rgb("#fafbfc") } else { none },
  table.hline(y: 1, stroke: 0.5pt + rgb("#b9c4d3")),
  ..cells,
)

// ---------------------------------------------------------------- diagram nodes
#let _n(color, fill: none) = (stroke: 0.9pt + color, fill: if fill == none { color.lighten(90%) } else { fill }, corner-radius: 4pt, inset: 7pt)
#let nmod(pos, body, ..a) = node(pos, text(size: 8.6pt, body), .._n(c-module), ..a)
#let nsym(pos, body, ..a) = node(pos, text(size: 8.6pt, body), .._n(c-symbol), ..a)
#let ndat(pos, body, ..a) = node(pos, text(size: 8.6pt, body), .._n(c-data), ..a)
#let ncon(pos, body, ..a) = node(pos, text(size: 8.6pt, body), .._n(c-concept), ..a)
#let nui(pos, body, ..a)  = node(pos, text(size: 8.6pt, body), .._n(c-ui), ..a)
#let next(pos, body, ..a) = node(pos, text(size: 8.6pt, body), stroke: (paint: c-ext, thickness: 0.9pt, dash: "dashed"), fill: c-ext.lighten(92%), corner-radius: 4pt, inset: 7pt, ..a)
#let nstep(pos, body, ..a) = node(pos, text(size: 8.6pt, body), .._n(c-step), ..a)
#let lbl(s) = text(size: 7.4pt, fill: c-muted, s)

// ---------------------------------------------------------------- document setup
#let in-appendix = state("in-appendix", false)
/// Switch to appendix numbering (A, B, ...) for every heading that follows.
#let appendix(body) = {
  in-appendix.update(true)
  counter(heading).update(0)
  set heading(numbering: "A.1")
  body
}

#let handbook(title: "", subtitle: "", date: "", state: "", cover: none, body) = {
  set document(title: title, author: "Struct_Sept")
  set page(
    paper: "a4", margin: (x: 2.1cm, top: 2.3cm, bottom: 2.2cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 2 {
        set text(size: 8pt, fill: c-muted)
        let pg = here().page()
        let on-page = query(heading.where(level: 1)).filter(h => h.location().page() == pg)
        let before = query(selector(heading.where(level: 1)).before(here()))
        let cur = if on-page.len() > 0 { on-page.first().body } else if before.len() > 0 { before.last().body } else { [] }
        grid(columns: (1fr, auto), [Struct_Sept handbook], cur)
        v(-6pt)
        line(length: 100%, stroke: 0.4pt + rgb("#d0d5dc"))
      }
    },
    footer: context {
      if counter(page).get().first() > 1 {
        set text(size: 8.5pt, fill: c-muted)
        align(center, counter(page).display("1"))
      }
    },
  )
  set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em, spacing: 0.95em)
  set heading(numbering: "1.1")
  set math.equation(numbering: "(1)")
  set list(indent: 0.6em, spacing: 0.55em)
  set enum(indent: 0.6em, spacing: 0.55em)
  show raw: set text(font: ("Cascadia Code", "DejaVu Sans Mono"), size: 8.4pt)
  show raw.where(block: true): it => {
    if it.lang in ("bash", "sh", "console", "powershell", "ps1") {
      block(fill: rgb("#1f252d"), inset: (x: 10pt, y: 8pt), radius: 3pt, width: 100%, above: 0.8em, below: 0.8em, {
        set text(fill: rgb("#e8edf2"), font: ("Cascadia Code", "DejaVu Sans Mono"), size: 8.4pt)
        set par(justify: false, leading: 0.5em)
        it.text.split("\n").map(l => {
          if l.trim().starts-with("#") { text(fill: rgb("#8fa1b3"), l) } else { l }
        }).join(linebreak())
      })
    } else {
      block(fill: luma(247), stroke: 0.5pt + luma(205), inset: 8pt, radius: 3pt, width: 100%, it)
    }
  }
  show raw.where(block: false): it => box(fill: luma(242), inset: (x: 2pt), outset: (y: 2pt), radius: 2pt, it)
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(0.4em)
    block(below: 1.1em)[
      #if it.numbering != none {
        text(size: 11pt, fill: c-accent, weight: "bold", context {
          if in-appendix.get() [Appendix ] else [Chapter ]
          counter(heading).display(it.numbering)
        })
        linebreak()
      }
      #text(size: 20pt, weight: "bold", it.body)
      #v(-4pt)
      #line(length: 100%, stroke: 1pt + c-accent)
    ]
  }
  show heading.where(level: 2): it => { v(0.5em); text(size: 13pt, it); v(0.15em) }
  show heading.where(level: 3): it => { v(0.3em); text(size: 11pt, it); v(0.05em) }
  show link: it => text(fill: c-accent, it)
  show figure.caption: set text(size: 9pt)
  set table(stroke: 0.4pt + rgb("#c8ced6"))

  // ---- title page
  page(numbering: none, header: none, footer: none, margin: (x: 2.4cm, y: 2.6cm))[
    #v(3.2cm)
    #text(size: 9.5pt, fill: c-accent, weight: "bold", tracking: 0.06em)[TU WIEN · INSTITUTE OF LIGHTWEIGHT DESIGN AND STRUCTURAL BIOMECHANICS]
    #v(0.6cm)
    #text(size: 30pt, weight: "bold", title)
    #v(0.2cm)
    #text(size: 14pt, fill: c-muted, subtitle)
    #v(0.8cm)
    #line(length: 100%, stroke: 1.2pt + c-accent)
    #v(0.5cm)
    #cover
    #v(1fr)
    #set text(size: 9.5pt, fill: c-muted)
    #grid(columns: (auto, 1fr), column-gutter: 12pt, row-gutter: 6pt,
      [*Written*], date,
      [*Code state*], state,
      [*Companion*], [`docs/handbook/board.html` — the interactive board (open it in a browser)],
      [*Reference paper*], [Kofler, Giritsch, Elgeti, _Structural optimization of lattice structures using deep neural networks as geometry representation_, Graphical Models 142 (2025) 101307],
    )
  ]
  body
}
