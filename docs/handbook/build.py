"""Build the handbook PDF from its Typst sources.

The handbook (``docs/handbook/handbook.typ`` plus ``chapters/*.typ``) is
compiled with the ``typst`` Python package in a throwaway environment, so
nothing is added to the project's own dependencies::

    uv run --no-project --with typst python docs/handbook/build.py
    uv run --no-project --with typst python docs/handbook/build.py --png 150

The first build downloads the ``fletcher`` diagram package from Typst
Universe into the Typst cache; later builds are offline.

``--png PPI`` additionally renders every page to ``_pages/page-NN.png`` (gitignored
by the handbook's own ``.gitignore``), which is how the layout was checked.
"""

import argparse
import pathlib
import sys

import typst

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--png", type=int, default=0, metavar="PPI", help="also render pages to PNG at this resolution")
    args = parser.parse_args()

    src = HERE / "handbook.typ"
    out = HERE / "handbook.pdf"
    _, warnings = typst.compile_with_warnings(str(src), output=str(out), root=str(HERE.parent))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {out.relative_to(HERE.parent.parent)}")

    if args.png:
        pages = HERE / "_pages"
        pages.mkdir(exist_ok=True)
        for old in pages.glob("page-*.png"):
            old.unlink()
        typst.compile(str(src), output=str(pages / "page-{0p}.png"), format="png", ppi=args.png, root=str(HERE.parent))
        print(f"wrote {len(list(pages.glob('page-*.png')))} pages to {pages.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
