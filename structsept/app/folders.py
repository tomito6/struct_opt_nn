"""Which folder each picker of the app lists, remembered between launches.

The decoder pickers of the Explore and Explore 2-D tabs list the training
runs of one folder each (Explore adds the decoders shipped with the library,
whatever the folder); the Train tab's dataset picker lists the datasets of
one data root, and its runs table the runs of one folder, which is also where
a new run is written. By default those are ``runs/`` and ``data/`` at the
repo root; a Browse... button points any of them at another folder - one per
dataset family, say, so the plate-with-hole runs and the triangle runs stop
sharing one long list. What each folder must hold is checked by the tab that
lists it (``models.runs_folder``, ``datasets.data_root_for``); this module
only stores, compares and prints paths.

The choice is stored in one small JSON file at the repo root, gitignored like
``data/`` and ``runs/``: a folder on this laptop means nothing on another one.
A folder inside the repo is stored relative to it, so the setting survives the
repo moving (OneDrive syncing it to another machine under another user name).

A missing, unreadable or half-written file reads as "nothing remembered", and
a remembered folder that no longer exists falls back to the default, so
nothing here can keep the app from starting.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / ".structsept_app.json"

# keys of the remembered folders in SETTINGS_PATH
EXPLORE_RUNS = "explore_runs_dir"
EXPLORE2D_RUNS = "explore2d_runs_dir"
TRAIN_DATA = "train_data_root"
TRAIN_RUNS = "train_runs_dir"


def _read() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def remembered(key, default) -> Path:
    """The folder stored under ``key``, or ``default``.

    ``default`` is also the answer when the stored folder was deleted or
    renamed since: an empty picker pointed at a folder that is gone helps
    nobody.
    """
    value = _read().get(key)
    if isinstance(value, str) and value:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_dir():
            return path
    return Path(default)


def remember(key, folder) -> bool:
    """Store ``folder`` under ``key``. False when the file cannot be written.

    Never raises: losing a remembered folder is a nuisance, an exception in
    the middle of a Browse... click is a broken button.
    """
    data = _read()
    path = Path(folder).resolve()
    try:
        data[key] = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        data[key] = str(path)
    # write-then-rename, so a crash mid-write cannot leave half a file
    tmp = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_PATH)
    except OSError:
        return False
    return True


def display(folder) -> str:
    r"""``folder`` as a picker shows it: ``runs\plate_tri`` inside the repo,
    the full path anywhere else."""
    path = Path(folder)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def same(a, b) -> bool:
    """True when ``a`` and ``b`` name the same folder, however spelled."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return Path(a) == Path(b)
