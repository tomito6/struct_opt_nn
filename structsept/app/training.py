"""Training stage of the structsept app.

Writes the specs.json a run needs, drives the DeepSDFStruct trainer on CPU while
forwarding its log records to the GUI, and stores a metadata.json so that a run
directory later describes itself to the model picker without being reloaded.

What goes into specs.json is described field by field in
``structsept.app.hyperparams``; this module only writes and reads the files.
"""

import json
import logging
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.training import train_deep_sdf

from structsept.app import hyperparams

LIBRARY_LOGGER = "DeepSDFStruct"
ARCH = hyperparams.ARCH


def write_specs(
    run_dir,
    split_path,
    data_source,
    hparams=None,
    geom_dimension=hyperparams.GEOM_DIMENSION,
    **overrides,
):
    """Write <run_dir>/specs.json for a DeepSDF run and return its path.

    Parameters
    ----------
    run_dir : path-like
        Experiment directory. Created if missing; the trainer writes its
        checkpoints next to the specs file.
    split_path : path-like
        Split json listing the training instances.
    data_source : path-like
        Directory holding SdfSamples/<dataset>/<class>/*.npz.
    hparams : dict, optional
        A hyperparameter set as described by ``structsept.app.hyperparams``.
        Missing keys take their defaults, so ``None`` writes the default run.
    geom_dimension : int
        Coordinates per sample of the dataset - ``datasets.geom_dimension``
        of it. 3 builds the usual unit-cell decoder f(λ, x, y, z); 2 a planar
        one, f(λ, x, y).
    **overrides
        Individual hyperparameters on top of ``hparams``, by their
        ``hyperparams.FIELDS`` key - ``latent_dim=2, num_epochs=30``.

    Raises
    ------
    ValueError
        When the set has an error-level problem (``hyperparams.validate``):
        the trainer would crash on it minutes in, or finish with no checkpoint
        to load, so it is refused before anything is written.
    """
    hp = hyperparams.defaults()
    hp.update(hparams or {})
    unknown = sorted(set(overrides) - set(hyperparams.FIELD_BY_KEY))
    if unknown:
        raise TypeError(f"unknown hyperparameter(s): {', '.join(unknown)}")
    hp.update(overrides)
    problems = hyperparams.errors(
        hyperparams.validate(hp, geom_dimension=geom_dimension)
    )
    if problems:
        raise ValueError(" ".join(p.message for p in problems))

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = hyperparams.to_specs(hp, split_path, data_source, geom_dimension)
    path = run_dir / ws.specifications_filename
    path.write_text(json.dumps(specs, indent=4), encoding="utf-8")
    return path


def spec_sources(runs_dir):
    """Every ``specs.json`` a hyperparameter set can be loaded from.

    Local runs first, sorted by name like ``list_runs``, then the decoders
    shipped with DeepSDFStruct - their specs are the reference settings of the
    paper's test cases.

    Returns
    -------
    list of (str, pathlib.Path)
        Display label and specs.json path.
    """
    from DeepSDFStruct.pretrained_models import PRETRAINED_MODELS_DIR, PretrainedModels

    sources = []
    runs_dir = Path(runs_dir)
    if runs_dir.is_dir():
        for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            specs = path / ws.specifications_filename
            if specs.is_file():
                sources.append((f"run: {path.name}", specs))
    for member in PretrainedModels:
        specs = Path(PRETRAINED_MODELS_DIR) / member.value / ws.specifications_filename
        if specs.is_file():
            sources.append((f"shipped: {member.name}", specs))
    return sources


def read_specs(path):
    """The dict stored in a specs.json, or ``{}`` when it cannot be read."""
    return _read_json(path)


class _CallbackHandler(logging.Handler):
    """Logging handler that pushes formatted records into a callable."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def emit(self, record):
        try:
            self._callback(self.format(record))
        except Exception:
            self.handleError(record)


def train(run_dir, data_source, log=print):
    """Run the library trainer on CPU and return {"run_dir", "epochs", "seconds"}.

    Records emitted by the DeepSDFStruct logger are forwarded to ``log`` for the
    duration of the run so a GUI text box can follow the progress.

    The run's ``seed`` is applied here, before the library is called, and not
    only by the library: ``train_deep_sdf`` builds the decoder - drawing its
    initial weights from the global torch RNG - a few lines *before* it seeds.
    In a long-lived app process that RNG has been advanced by every earlier
    run and every decoder the Explore tab built, so without this two runs with
    identical settings start from different weights, and an A/B comparison of
    one hyperparameter silently compares two initialisations as well.
    """
    run_dir = Path(run_dir)
    specs = ws.load_experiment_specifications(run_dir)
    _seed_everything(specs.get("seed", 42))

    handler = _CallbackHandler(log)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    lib_logger = logging.getLogger(LIBRARY_LOGGER)
    previous_level = lib_logger.level
    lib_logger.addHandler(handler)
    if not lib_logger.isEnabledFor(logging.INFO):
        lib_logger.setLevel(logging.INFO)

    start = time.perf_counter()
    try:
        summary = train_deep_sdf(str(run_dir), str(data_source), device="cpu")
    finally:
        lib_logger.removeHandler(handler)
        lib_logger.setLevel(previous_level)
        handler.close()
    seconds = time.perf_counter() - start

    summary = summary or {}
    return {
        "run_dir": str(run_dir),
        "epochs": summary.get("num_epochs", specs.get("NumEpochs")),
        "seconds": seconds,
    }


def _seed_everything(seed):
    """Seed Python, NumPy and torch exactly as the trainer later does."""
    import random

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_metadata(run_dir, **info):
    """Write <run_dir>/metadata.json, defaulting every field from specs.json.

    Anything passed as a keyword overrides the value read from the run itself.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = _read_json(run_dir / ws.specifications_filename)
    network_specs = specs.get("NetworkSpecs", {})

    meta = {
        "timestamp": datetime.now().isoformat(),
        "dataset": None,
        "split_path": specs.get("TrainSplit"),
        "data_source": specs.get("DataSource"),
        "latent_dim": specs.get("CodeLength"),
        "geom_dimension": network_specs.get("geom_dimension", 3),
        "arch": specs.get("NetworkArch"),
        "dims": network_specs.get("dims"),
        "epochs": specs.get("NumEpochs"),
        "final_loss": _final_loss(run_dir),
        "deepsdfstruct_version": _library_version(),
    }
    meta.update(info)

    path = run_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=4), encoding="utf-8")
    return path


def read_metadata(run_dir):
    """Return the metadata dict of a run, or {} when it has none."""
    return _read_json(Path(run_dir) / "metadata.json")


def list_runs(runs_dir):
    """List every run directory under runs_dir as a row for the GUI.

    A directory counts as a run once it holds a specs.json; metadata.json and a
    saved checkpoint only enrich the row.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    rows = []
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        specs = _read_json(path / ws.specifications_filename)
        if not specs:
            continue
        meta = _read_json(path / "metadata.json")
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "latent_dim": meta.get("latent_dim", specs.get("CodeLength")),
                "geom_dimension": (specs.get("NetworkSpecs") or {}).get(
                    "geom_dimension", 3
                ),
                "dataset": meta.get("dataset"),
                "date": meta.get("timestamp"),
                "epochs": meta.get("epochs", specs.get("NumEpochs")),
                "final_loss": meta.get("final_loss", _final_loss(path)),
                "trained": (path / ws.model_params_subdir / "latest.pth").is_file(),
            }
        )
    return rows


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _final_loss(run_dir):
    summary = _read_json(Path(run_dir) / ws.experiment_summary_name)
    return summary.get("loss")


def _library_version():
    try:
        return version("DeepSDFStruct")
    except PackageNotFoundError:
        return None


def read_progress(run_dir):
    """Per-batch loss and last completed epoch of a run, or ``{}``.

    The trainer reports epoch progress only through a tqdm bar, and its
    per-batch ``logger.debug`` line is the wrong thing to parse, so progress is
    read from the checkpoint it already writes: ``Logs.pth``, saved every
    ``LogFrequency`` epochs by ``deep_sdf.training.save_logs``. That makes the
    update cadence known - LogFrequency is 10/5/2/1 unless set in the
    hyperparameter window - and costs one small ``torch.load``.

    Returns ``{"loss": [...], "epoch": int}``; an empty dict while the first
    checkpoint has not been written or if the file is being rewritten as we
    read it.
    """
    import torch

    path = Path(run_dir) / ws.logs_filename
    if not path.is_file():
        return {}
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        # a half-written file during save_logs; the next poll picks it up
        return {}
    if not isinstance(data, dict):
        return {}
    loss = data.get("loss") or []
    try:
        return {"loss": [float(x) for x in loss], "epoch": int(data.get("epoch", 0))}
    except (TypeError, ValueError):
        return {}
