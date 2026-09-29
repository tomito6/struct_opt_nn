"""Read the cells of an Excel workbook with the standard library alone.

The hyperparameter window imports the supervisor's sheet
(``docs/hyperparameters/*.xlsx``: one hyperparameter per row, its value in
column B). That needs the cell values of one worksheet and nothing else - no
styles, formulas, merged ranges or charts - so this module reads them straight
out of the ``.xlsx`` zip instead of adding ``openpyxl`` to an environment
that already resolves slowly.

What is covered: shared strings (with rich-text runs), inline strings,
numbers, booleans, and the cached result of a formula. Dates come back as the
serial number Excel stores, and an error cell (``#DIV/0!``) as ``None``.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
SUFFIXES = (".xlsx", ".xlsm")


def read_rows(path, sheet=None) -> list[list]:
    """The cells of one worksheet as a list of rows, each a list of values.

    Parameters
    ----------
    path : path-like
        A ``.xlsx`` or ``.xlsm`` file.
    sheet : str, optional
        Worksheet name; the first sheet of the workbook when not given.

    Returns
    -------
    list of list
        Row ``i`` (0-based) is the sheet's row ``i + 1``; column ``j`` is
        column ``j + 1`` (``A`` = 0). Empty cells are ``None`` and every row
        is padded to the width of the widest row, so ``rows[5][1]`` is cell
        ``B6`` whether or not anything is written before it. Text is ``str``,
        numbers are ``int`` when whole and ``float`` otherwise, booleans are
        ``bool``.

    Raises
    ------
    ValueError
        When the file is not a workbook, or ``sheet`` is not in it.
    """
    path = Path(path)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path.name} is not an Excel workbook: {exc}") from None
    with archive:
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names:
            raise ValueError(
                f"{path.name} is not an Excel workbook (no xl/workbook.xml)"
            )
        target = _sheet_part(archive, sheet)
        if target not in names:
            raise ValueError(f"{path.name}: worksheet part {target} is missing")
        shared = _shared_strings(archive) if "xl/sharedStrings.xml" in names else []
        cells = _cells(ET.fromstring(archive.read(target)), shared)

    if not cells:
        return []
    n_rows = max(r for r, _ in cells) + 1
    n_cols = max(c for _, c in cells) + 1
    rows = [[None] * n_cols for _ in range(n_rows)]
    for (r, c), value in cells.items():
        rows[r][c] = value
    return rows


# --------------------------------------------------------------------------- #
# parts of the package
# --------------------------------------------------------------------------- #


def _sheet_part(archive, sheet) -> str:
    """Zip member holding the requested (or first) worksheet."""
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find("m:sheets", NS)
    entries = [] if sheets is None else list(sheets.findall("m:sheet", NS))
    if not entries:
        raise ValueError("the workbook has no worksheets")
    if sheet is None:
        chosen = entries[0]
    else:
        chosen = next((s for s in entries if s.get("name") == sheet), None)
        if chosen is None:
            available = ", ".join(s.get("name", "?") for s in entries)
            raise ValueError(f"no worksheet named {sheet!r} (has: {available})")
    rel_id = chosen.get(f"{{{NS['r']}}}id")
    rels_name = "xl/_rels/workbook.xml.rels"
    if rel_id and rels_name in archive.namelist():
        rels = ET.fromstring(archive.read(rels_name))
        for rel in rels.findall("rel:Relationship", NS):
            if rel.get("Id") == rel_id:
                target = rel.get("Target", "")
                if target.startswith("/"):
                    return target.lstrip("/")
                return "xl/" + target
    # No relationships part (some writers skip it): fall back to the position.
    index = entries.index(chosen) + 1
    return f"xl/worksheets/sheet{index}.xml"


def _shared_strings(archive) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    tag = f"{{{NS['m']}}}t"
    return [
        "".join(t.text or "" for t in item.iter(tag))
        for item in root.findall("m:si", NS)
    ]


def _cells(sheet_root, shared) -> dict[tuple[int, int], object]:
    """``{(row, col): value}`` for every non-empty cell of a worksheet."""
    out = {}
    tag_t = f"{{{NS['m']}}}t"
    for cell in sheet_root.iter(f"{{{NS['m']}}}c"):
        ref = cell.get("r")
        if not ref:
            continue
        kind = cell.get("t", "n")
        node = cell.find("m:v", NS)
        raw = None if node is None else node.text
        if kind == "s":
            if raw is None:
                continue
            try:
                value = shared[int(raw)]
            except (ValueError, IndexError):
                continue
        elif kind == "inlineStr":
            inline = cell.find("m:is", NS)
            if inline is None:
                continue
            value = "".join(t.text or "" for t in inline.iter(tag_t))
        elif kind == "b":
            value = raw not in (None, "0", "false", "FALSE")
        elif kind == "e":
            continue  # an error cell: nothing usable
        elif kind == "str":
            value = raw if raw is not None else ""
        else:  # "n", or a number stored without a type
            if raw is None:
                continue
            value = _number(raw)
        if value is None:
            continue
        out[_cell_index(ref)] = value
    return out


def _number(text):
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and abs(number) < 2**53:
        return int(number)
    return number


_REF = re.compile(r"^([A-Z]+)([0-9]+)$")


def _cell_index(ref: str) -> tuple[int, int]:
    """``"B6"`` -> ``(5, 1)``: 0-based (row, column)."""
    match = _REF.match(ref.upper())
    if not match:
        raise ValueError(f"not a cell reference: {ref!r}")
    letters, digits = match.groups()
    col = 0
    for ch in letters:
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return int(digits) - 1, col - 1
