"""Compile ONE handbook chapter on its own, to check syntax and layout.

The chapter is wrapped in a throwaway harness (``_build/check_<name>.typ``)
that applies the template and adds stub headings for every label the other
chapters define, so cross-references like ``@ch:datagen[Chapter]`` resolve::

    uv run --no-project --with typst python docs/handbook/_check.py chapters/05-datagen.typ
    uv run --no-project --with typst python docs/handbook/_check.py chapters/05-datagen.typ --png 110

Prints errors and warnings; ``--png`` also writes the chapter's pages to
``_build/<name>/page-NN.png`` so the layout can be looked at.
"""

import argparse
import pathlib
import re
import sys

import typst

HERE = pathlib.Path(__file__).resolve().parent
LABELS = [
    "ch:idea", "ch:method", "ch:repo", "ch:library", "ch:datagen", "ch:training",
    "ch:gui", "ch:fem", "ch:workflows", "ch:status",
    "app:glossary", "app:formats", "app:commands", "app:gotchas",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chapter", help="path relative to docs/handbook, e.g. chapters/05-datagen.typ")
    parser.add_argument("--png", type=int, default=0, metavar="PPI")
    args = parser.parse_args()

    chapter = (HERE / args.chapter).resolve()
    if not chapter.exists():
        print(f"no such chapter: {chapter}", file=sys.stderr)
        return 2
    rel = chapter.relative_to(HERE).as_posix()
    own = set(re.findall(r"<((?:ch|app|sec|fig|tab):[\w-]+)>", chapter.read_text(encoding="utf8")))
    stubs = [lab for lab in LABELS if lab not in own]

    build = HERE / "_build"
    build.mkdir(exist_ok=True)
    name = chapter.stem
    harness = build / f"check_{name}.typ"
    harness.write_text(
        '#import "../template.typ": *\n'
        '#show: handbook.with(title: [Chapter check], subtitle: [' + name + '], date: [], state: [])\n'
        f'#include "../{rel}"\n'
        "#pagebreak()\n"
        "// stubs for labels defined in other chapters\n"
        + "".join(f"#heading(level: 2)[stub {lab}] <{lab}>\n" for lab in stubs),
        encoding="utf8",
    )
    try:
        _, warnings = typst.compile_with_warnings(str(harness), output=str(build / f"{name}.pdf"), root=str(HERE.parent))
    except typst.TypstError as err:
        print("TYPST ERROR:\n" + str(err), file=sys.stderr)
        return 1
    for w in warnings:
        print(f"warning: {w}")
    pdf = build / f"{name}.pdf"
    print(f"OK: {pdf.relative_to(HERE)}")
    if args.png:
        out = build / name
        out.mkdir(exist_ok=True)
        for old in out.glob("page-*.png"):
            old.unlink()
        typst.compile(str(harness), output=str(out / "page-{0p}.png"), format="png", ppi=args.png, root=str(HERE.parent))
        pages = sorted(out.glob("page-*.png"))
        # page 1 is the harness title page, the last page holds the stubs
        print(f"{len(pages)} pages in {out.relative_to(HERE)} (page 1 = harness cover, last = stubs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
