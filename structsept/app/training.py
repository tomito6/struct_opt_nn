"""Training stage of the structsept app.

Writes the specs.json a run needs, drives the DeepSDFStruct trainer on CPU while
forwarding its log records to the GUI, and stores a metadata.json so that a run
directory later describes itself to the model picker without being reloaded.
"""

import json
import logging
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.training import train_deep_sdf

LIBRARY_LOGGER = "DeepSDFStruct"
ARCH = "deep_sdf_decoder"


def write_specs(
    run_dir,
    latent_dim,
    split_path,
    data_source,
    n_layers=6,
    width=128,
    num_epochs=200,
    samples_per_scene=8000,
    scenes_per_batch=10,
    description="",
):
    """Write <run_dir>/specs.json for a DeepSDF run and return its path.

    Parameters
    ----------
    run_dir : path-like
        Experiment directory. Created if missing; the trainer writes its
        checkpoints next to the specs file.
    latent_dim : int
        Latent dimension d, stored as CodeLength.
    split_path : path-like
        Split json listing the training instances.
    data_source : path-like
        Directory holding SdfSamples/<dataset>/<class>/*.npz.
    n_layers, width : int
        Decoder MLP shape, stored as dims = [width] * n_layers.
    num_epochs, samples_per_scene, scenes_per_batch : int
        Training budget. ``LogFrequency`` is derived from ``num_epochs`` so the
        last epoch always lands in ModelParameters/latest.pth.
    description : str
        Free text; a default is generated when empty.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # The decoder prepends the input layer and appends the output layer, so the
    # per-layer dropout/norm indices have to cover n_layers + 2 entries.
    layer_indices = list(range(n_layers + 2))

    # The trainer only writes ModelParameters/latest.pth when
    # epoch % LogFrequency == 0, and its default of 10 silently leaves a short
    # run with no loadable checkpoint at all.
    log_frequency = next(f for f in (10, 5, 2, 1) if num_epochs % f == 0)

    specs = {
        "Description": description
        or f"structsept run: d={latent_dim}, {n_layers}x{width}, {num_epochs} epochs",
        "DataSource": str(data_source),
        "NetworkArch": ARCH,
        # The trainer resolves TrainSplit against DataSource, so an absolute
        # path here is the only form that survives both layouts.
        "TrainSplit": str(split_path),
        "TestSplit": str(split_path),
        "ReconstructionSplit": "",
        "NetworkSpecs": {
            "dims": [width] * n_layers,
            "dropout": layer_indices,
            "dropout_prob": 0.2,
            "norm_layers": layer_indices,
            "latent_in": [2],
            "xyz_in_all": False,
            "use_tanh": False,
            "latent_dropout": False,
            "weight_norm": True,
            "geom_dimension": 3,
        },
        "CodeLength": latent_dim,
        "NumEpochs": num_epochs,
        "LogFrequency": log_frequency,
        "SnapshotFrequency": max(1, num_epochs // 4),
        "AdditionalSnapshots": [1],
        "LearningRateSchedule": [
            {"Type": "Step", "Initial": 0.0005, "Interval": 500, "Factor": 0.5},
            {"Type": "Step", "Initial": 0.001, "Interval": 500, "Factor": 0.5},
        ],
        "SamplesPerScene": samples_per_scene,
        "ScenesPerBatch": scenes_per_batch,
        "DataLoaderThreads": 0,
        "ClampingDistance": 0.1,
        "CodeRegularization": True,
        "CodeRegularizationLambda": 1e-4,
        "CodeBound": 1.0,
    }

    path = run_dir / ws.specifications_filename
    path.write_text(json.dumps(specs, indent=4), encoding="utf-8")
    return path


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
    """
    run_dir = Path(run_dir)
    specs = ws.load_experiment_specifications(run_dir)

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
