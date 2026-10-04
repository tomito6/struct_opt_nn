"""Training stage of the structsept app.

Writes the specs.json a run needs, drives the DeepSDFStruct trainer on CPU while
forwarding its log records to the GUI, and stores a metadata.json so that a run
directory later describes itself to the model picker without being reloaded.

What goes into specs.json is described field by field in
``structsept.app.hyperparams``; this module only writes and reads the files.
"""

import json
import logging
import shutil
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

    Local runs first, newest first like ``list_runs``, then the decoders
    shipped with DeepSDFStruct - their specs are the reference settings of the
    paper's test cases.

    Returns
    -------
    list of (str, pathlib.Path)
        Display label and specs.json path.
    """
    from DeepSDFStruct.pretrained_models import PRETRAINED_MODELS_DIR, PretrainedModels

    sources = [
        (f"run: {row['name']}", Path(row["path"]) / ws.specifications_filename)
        for row in list_runs(runs_dir)
    ]
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
    """List every run directory under runs_dir as a row for the GUI, newest
    first.

    A directory counts as a run once it holds a specs.json; metadata.json and a
    saved checkpoint only enrich the row.

    ``date`` is the metadata timestamp - written when the run finished, or
    when a preset was generated. A run without metadata (started by hand, or
    still training) is dated by its specs.json instead, and
    ``date_is_estimate`` says so. Both are ISO strings, so they sort as text.

    ``epochs`` is what the run was asked for (``NumEpochs``); ``last_epoch``
    is what its checkpoint holds (:func:`checkpoint_epoch`). They differ for
    a run that was stopped or died, which ``trained`` alone cannot tell from
    a finished one: it only says that a checkpoint exists.

    ``train_seconds`` is how long the trainer took for the epochs it logged
    (:func:`training_seconds`), ``None`` for a run that never started.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    rows = []
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        specs_path = path / ws.specifications_filename
        specs = _read_json(specs_path)
        if not specs:
            continue
        meta = _read_json(path / "metadata.json")
        date = meta.get("timestamp")
        estimated = not isinstance(date, str) or not date
        if estimated:
            date = _file_date(specs_path)
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "latent_dim": meta.get("latent_dim", specs.get("CodeLength")),
                "geom_dimension": (specs.get("NetworkSpecs") or {}).get(
                    "geom_dimension", 3
                ),
                "dataset": meta.get("dataset"),
                "date": date,
                "date_is_estimate": estimated,
                "epochs": meta.get("epochs", specs.get("NumEpochs")),
                "last_epoch": checkpoint_epoch(path),
                "train_seconds": training_seconds(path),
                "final_loss": meta.get("final_loss", _final_loss(path)),
                "description": specs.get("Description") or "",
                "trained": (path / ws.model_params_subdir / "latest.pth").is_file(),
            }
        )
    rows.sort(key=lambda row: row["date"], reverse=True)
    return rows


def checkpoint_epoch(run_dir):
    """Epoch the run's ``latest`` checkpoint was written at, or ``None``.

    ``NumEpochs`` in specs.json is the epoch count that was asked for. A run
    that was stopped at a deadline, or whose process died, holds fewer, and
    its ``latest.pth`` loads all the same - so this is what tells an
    80-epoch decoder from the 800-epoch one its specs describe.

    Read from ``LatentCodes/latest.pth``: the trainer writes it in the same
    ``save_latest`` call as ``ModelParameters/latest.pth`` and with the same
    epoch, and it is a few kilobytes where the decoder is megabytes. That
    matters because the runs table calls this for every row. ``None`` when
    the run has no checkpoint yet or the file cannot be read.
    """
    import torch

    path = Path(run_dir) / ws.latent_codes_subdir / "latest.pth"
    if not path.is_file():
        return None
    try:
        return int(torch.load(path, map_location="cpu", weights_only=True)["epoch"])
    except Exception:
        # half-written while the trainer saves, or not a checkpoint at all
        return None


def epochs_text(epochs, last_epoch) -> str:
    """How far a run got, for a table cell or a label: ``800`` or ``80/800``.

    Only a checkpoint that stops short of the planned count is spelled out;
    a finished run, and one whose checkpoint cannot be read, show the plan.
    """
    if epochs is None:
        return "?" if last_epoch is None else str(last_epoch)
    if isinstance(last_epoch, int) and isinstance(epochs, int) and last_epoch < epochs:
        return f"{last_epoch}/{epochs}"
    return str(epochs)


# Logs.pth path -> ((mtime_ns, size), seconds): a finished run's log never
# changes, and the runs table re-reads every row on each refresh
_TIMING_CACHE: dict = {}


def training_seconds(run_dir):
    """Seconds the trainer spent on a run's logged epochs, or ``None``.

    The sum of the per-epoch ``timing`` list the trainer keeps in
    ``Logs.pth``. That one source covers every run alike, whoever launched
    it - the Train tab, ``unattended.py`` or a script in ``experiments/`` -
    where a ``metadata.json`` field would exist only for the launchers that
    write it. It counts the epoch loops only: data loading before epoch 1 and
    checkpoint writes are left out, which makes it ~0.3 % shorter than the
    wall time the scripts measure.

    ``Logs.pth`` is rewritten with the ``latest`` checkpoint, so a run that
    was stopped or died reports the time of the epochs its checkpoint holds;
    a resumed run carries its log along and reports all sessions together.
    ``None`` when there is no log yet or it cannot be read (half-written
    while the trainer saves it).
    """
    import torch

    path = Path(run_dir) / ws.logs_filename
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _TIMING_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        seconds = float(sum(data.get("timing") or []))
    except Exception:
        return None
    _TIMING_CACHE[str(path)] = (stamp, seconds)
    return seconds


def duration_text(seconds) -> str:
    """A training time for a table cell: ``45 s``, ``20 min``, ``1 h 56 min``.

    ``-`` when unknown. Minutes are rounded, so an hour-long run does not
    pretend to second precision.
    """
    if not isinstance(seconds, (int, float)):
        return "-"
    if seconds < 60:
        return f"{seconds:.0f} s"
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def _file_date(path) -> str:
    """Modification time of a file as an ISO string; empty when unreadable."""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat()
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# run housekeeping: rename, annotate, delete
# --------------------------------------------------------------------------- #

# What Windows refuses in a file name, plus the separators; the same set is
# refused everywhere so a run made on one machine opens on another.
RUN_NAME_FORBIDDEN = frozenset('<>:"/\\|?*')


def check_run_name(runs_dir, name, current=None):
    """Why ``name`` cannot be a run directory under ``runs_dir``, or ``None``.

    ``current`` is the run being renamed: keeping its own name is allowed
    (nothing to do), any other existing directory is a clash. Leading and
    trailing blanks are not part of a name; the caller strips them the same
    way before using it.
    """
    name = str(name).strip()
    if not name:
        return "The run needs a name."
    if name in (".", "..") or name.endswith(".") or name.endswith(" "):
        return f"'{name}' is not a usable directory name."
    if any(c in RUN_NAME_FORBIDDEN or ord(c) < 32 for c in name):
        return 'A run name cannot contain / \\ : * ? " < > |.'
    if name != current and (Path(runs_dir) / name).exists():
        return f"A run called '{name}' already exists."
    return None


def rename_run(runs_dir, old, new):
    """Rename the run directory ``old`` to ``new`` and return the new path.

    Only the directory moves: ``specs.json`` names the dataset by its split
    file, not the run, so nothing inside has to change. Raises ``ValueError``
    for a name :func:`check_run_name` refuses and ``OSError`` when the file
    system does - on Windows that includes a run another process is writing
    to, or a file of it open in a viewer.
    """
    new = str(new).strip()
    problem = check_run_name(runs_dir, new, current=old)
    if problem:
        raise ValueError(problem)
    src = Path(runs_dir) / old
    if not (src / ws.specifications_filename).is_file():
        raise FileNotFoundError(f"'{old}' is not a run directory.")
    if new == old:
        return src
    dst = src.with_name(new)
    src.rename(dst)
    return dst


def delete_run(runs_dir, name):
    """Remove the run directory ``name`` with everything in it.

    Refuses anything that is not a run - no ``specs.json`` - and anything
    outside ``runs_dir``, so a bad name can never take another folder with
    it. Raises ``FileNotFoundError`` / ``ValueError`` before touching
    anything, ``OSError`` when a file cannot be removed.
    """
    runs_dir = Path(runs_dir).resolve()
    path = (runs_dir / str(name)).resolve()
    if path.parent != runs_dir or path == runs_dir:
        raise ValueError(f"'{name}' is not a run under {runs_dir}.")
    if not path.is_dir():
        raise FileNotFoundError(f"'{name}' does not exist.")
    if not (path / ws.specifications_filename).is_file():
        raise ValueError(f"'{name}' has no specs.json, so it is not a run.")
    shutil.rmtree(path)


def read_description(run_dir) -> str:
    """The free-text ``Description`` of a run's specs.json (may be empty)."""
    return str(
        _read_json(Path(run_dir) / ws.specifications_filename).get("Description") or ""
    )


def write_description(run_dir, text):
    """Replace the ``Description`` of a run's specs.json, nothing else.

    The trainer only carries the string along, so a finished run stays
    loadable and a waiting preset still trains the same. Returns the path.
    """
    path = Path(run_dir) / ws.specifications_filename
    specs = _read_json(path)
    if not specs:
        raise FileNotFoundError(f"{path} is missing or not valid JSON.")
    specs["Description"] = str(text)
    path.write_text(json.dumps(specs, indent=4), encoding="utf-8")
    return path


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
