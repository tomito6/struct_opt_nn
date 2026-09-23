"""Write a DeepSDF training set to disk in the layout the trainer and the GUI read.

This file is the contract between ``datagen`` and the rest of the repo. Nothing
imports across that boundary: the GUI (``structsept.app``) and the trainer
(``DeepSDFStruct``) only ever see the files below, so any generator that writes
them is a valid source of training data.

The layout
----------
::

    <data_root>/
    ├── SdfSamples/<dataset>/
    │   ├── <class>/<instance>.npz   arrays "pos" (phi >= 0) and "neg" (phi < 0),
    │   │                            float32, rows (x, y, [z,] phi)
    │   ├── dataset.json             manifest: how it was made, geom dimension,
    │   │                            normalized frame -- written by datagen only
    │   └── params.csv               the generating parameters of each instance
    └── splits/<dataset>.json        {"<dataset>": {"<class>": [instances...]}}

``SdfSamples/`` and ``splits/`` are fixed names: the trainer resolves
``<DataSource>/SdfSamples/<dataset>/<class>/<instance>.npz`` from the split.
The order of the split is the order of the latent codes after training, so
``params.csv`` row ``i`` and latent code ``i`` describe the same shape.

Why the parameter table matters
-------------------------------
The decoder never sees how a shape was generated; it learns a latent code per
shape on its own. ``params.csv`` is what makes those codes interpretable
afterwards -- plotting ``lambda_1, lambda_2`` coloured by ``x_c``, ``y_c`` or
``r`` shows which of the generating parameters the latent space kept.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable

import numpy as np

SDF_SAMPLES_DIR = "SdfSamples"
SPLITS_DIR = "splits"
MANIFEST_NAME = "dataset.json"
PARAMS_NAME = "params.csv"

# data/ at the repository root, where structsept.app.main looks for datasets.
DEFAULT_DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"

# Files a previous run of datagen may have left in a dataset folder. Only these
# are removed on overwrite; anything else in the folder is left alone.
_OWNED_TOP_LEVEL = (MANIFEST_NAME, PARAMS_NAME, "preview.png", "parameters.png")


def dataset_paths(data_root, dataset_name, class_name) -> dict:
    """Every path a dataset occupies, as a dict of :class:`pathlib.Path`."""
    data_root = pathlib.Path(data_root)
    dataset_dir = data_root / SDF_SAMPLES_DIR / dataset_name
    return {
        "data_root": data_root,
        "dataset_dir": dataset_dir,
        "class_dir": dataset_dir / class_name,
        "split": data_root / SPLITS_DIR / f"{dataset_name}.json",
        "manifest": dataset_dir / MANIFEST_NAME,
        "params": dataset_dir / PARAMS_NAME,
    }


def split_pos_neg(rows) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(n, dim + 1)`` rows by the sign of the last column.

    Same convention as ``DeepSDFStruct.sampling``: ``phi < 0`` is ``neg``
    (inside), everything else ``pos``. The trainer draws half of each batch
    from either array.
    """
    rows = np.asarray(rows, dtype=np.float32)
    inside = rows[:, -1] < 0.0
    return rows[~inside], rows[inside]


def prepare(data_root, dataset_name, class_name, overwrite=False, log=print) -> dict:
    """Create the folders of a dataset, refusing to mix it with an old one.

    Parameters
    ----------
    data_root : str or pathlib.Path
    dataset_name, class_name : str
    overwrite : bool
        If the dataset exists, delete the files datagen owns in it (the
        ``.npz``/``.stl`` of the class folder, manifest, parameter table,
        previews, split) instead of raising.
    log : callable

    Returns
    -------
    dict
        :func:`dataset_paths` of the dataset.

    Raises
    ------
    FileExistsError
        If the dataset already holds samples and ``overwrite`` is False.
    """
    paths = dataset_paths(data_root, dataset_name, class_name)
    dataset_dir, class_dir = paths["dataset_dir"], paths["class_dir"]

    existing = sorted(dataset_dir.glob("*/*.npz")) if dataset_dir.is_dir() else []
    if existing or paths["split"].is_file():
        if not overwrite:
            found = [f"{len(existing)} .npz under {dataset_dir}"] if existing else []
            if paths["split"].is_file():
                found.append(f"split file {paths['split']}")
            raise FileExistsError(
                f"Dataset '{dataset_name}' already exists ({'; '.join(found)}). "
                "Pick another name or pass --overwrite."
            )
        stale = [p for p in existing if p.parent == class_dir]
        stale += sorted(class_dir.glob("*.stl")) if class_dir.is_dir() else []
        stale += [
            dataset_dir / n for n in _OWNED_TOP_LEVEL if (dataset_dir / n).is_file()
        ]
        stale += [paths["split"]] if paths["split"].is_file() else []
        others = [p for p in existing if p.parent != class_dir]
        if others:
            raise FileExistsError(
                f"Dataset '{dataset_name}' holds samples of other classes "
                f"({sorted({p.parent.name for p in others})}); not overwriting it."
            )
        log(f"Overwriting: removing {len(stale)} file(s) of the old dataset")
        for path in stale:
            path.unlink()

    class_dir.mkdir(parents=True, exist_ok=True)
    paths["split"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def write_instance(class_dir, instance_name, rows) -> dict:
    """Write one ``<instance>.npz`` and return a short summary of it."""
    pos, neg = split_pos_neg(rows)
    np.savez(pathlib.Path(class_dir) / f"{instance_name}.npz", pos=pos, neg=neg)
    return {"name": instance_name, "n_pos": len(pos), "n_neg": len(neg)}


def write_split(paths, dataset_name, class_name, instance_names: Iterable[str]):
    """Write ``splits/<dataset>.json``. Returns its path."""
    content = {dataset_name: {class_name: list(instance_names)}}
    with open(paths["split"], "w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=4)
    return paths["split"]


def write_manifest(paths, manifest: dict):
    """Write ``dataset.json``. Returns its path."""
    with open(paths["manifest"], "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return paths["manifest"]


def read_manifest(dataset_dir) -> dict | None:
    """The manifest of a dataset folder, or None if it has none.

    Datasets made by the GUI's mesh sampler carry no manifest; theirs are
    always 3-D.
    """
    path = pathlib.Path(dataset_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
