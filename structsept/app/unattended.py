"""Run the Train tab's trainer unattended: wait, train, stop at a deadline.

The time-boxed experiment scripts (``experiments/train_plate_r_only_4h.py``,
``experiments/train_plate_xyr_4h.py``) are started once and left alone for
hours. What they need beyond ``structsept.app.training`` is the same every
time, so it lives here rather than in either script:

* a log that goes to a file as well as the console (there is none when the
  script is launched hidden), and a ``.pid`` file to find the job again;
* waiting - until a clock time (:func:`wait_until`) or until another run has
  finished (:func:`wait_for_process`), without letting Windows sleep
  (:class:`KeepAwake`);
* writing a validated ``specs.json`` + ``metadata.json`` for a dataset and a
  hyperparameter set (:func:`prepare_run`), so the run already shows in the
  GUI's run table while it waits;
* training with a hard wall-clock stop (:func:`train_with_deadline`), and
  reading back what is left of a run afterwards (:func:`last_epoch`,
  :func:`load_codes`, :func:`split_names`).

Nothing here knows about a particular dataset or what the learned codes are
compared with afterwards; that is the scripts' business.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import pathlib
import sys
import threading
import time

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


class Tee:
    """Write to several streams at once; used to log to file and console."""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for s in self.streams:
            try:
                s.write(text)
                s.flush()
            except (OSError, ValueError):
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (OSError, ValueError):
                pass


def stamp() -> str:
    """Current local time, ``HH:MM:SS``."""
    return dt.datetime.now().strftime("%H:%M:%S")


def say(*parts):
    """Print with a timestamp; the only logging the scripts do."""
    print(stamp(), *parts, flush=True)


def open_log(log_dir, run: str):
    """Send stdout and stderr to ``<log_dir>/<run>.log`` as well as the console.

    Also writes the process id to ``<log_dir>/<run>.pid`` so a running job can
    be found (``taskkill /PID <pid> /F`` kills it). Returns the open log file;
    the caller keeps it alive for the life of the process.
    """
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / f"{run}.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    (log_dir / f"{run}.pid").write_text(str(os.getpid()), encoding="utf-8")
    return log_file


# --------------------------------------------------------------------------- #
# waiting
# --------------------------------------------------------------------------- #


def parse_start(text: str) -> dt.datetime:
    """Turn a ``--start-at`` argument into a local datetime.

    Accepts ``now``, ``HH:MM`` (today, or tomorrow if already past) and
    ``YYYY-MM-DD HH:MM``. Raises ``argparse.ArgumentTypeError`` otherwise, so
    it can be used directly as an argparse ``type``.
    """
    now = dt.datetime.now()
    if text.strip().lower() == "now":
        return now
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%H:%M"):
        try:
            parsed = dt.datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
        if fmt == "%H:%M":
            parsed = now.replace(
                hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0
            )
            if parsed < now:
                parsed += dt.timedelta(days=1)
        return parsed
    raise argparse.ArgumentTypeError(
        f"{text!r}: use 'now', 'HH:MM' or 'YYYY-MM-DD HH:MM'"
    )


def parse_pid(text: str) -> int:
    """Turn an ``--after`` argument into a process id.

    Accepts a number, or the path of a ``.pid`` file holding one (the file
    :func:`open_log` writes: ``outputs/logs/<run>.pid``). Raises
    ``argparse.ArgumentTypeError`` for anything else.
    """
    text = text.strip()
    path = pathlib.Path(text)
    if path.is_file():
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text}: does not hold a process id")
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: give a process id or a .pid file")


class KeepAwake:
    """Ask Windows not to sleep while this object is alive.

    ``SetThreadExecutionState`` with ``ES_SYSTEM_REQUIRED`` is a per-process
    request, reset the moment the process exits or ``clear`` runs; it changes
    no power setting. Elsewhere than Windows it does nothing.
    """

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        self._kernel32 = getattr(ctypes, "windll", None)
        self._kernel32 = self._kernel32.kernel32 if self._kernel32 else None

    def hold(self):
        if self._kernel32:
            self._kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )

    def clear(self):
        if self._kernel32:
            self._kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)


def wait_until(start: dt.datetime, log=say):
    """Sleep until ``start``, reporting every 30 minutes and near the end."""
    remaining = (start - dt.datetime.now()).total_seconds()
    if remaining <= 0:
        return
    log(f"waiting until {start:%Y-%m-%d %H:%M} ({remaining / 60:.0f} min)")
    last_report = time.monotonic()
    while True:
        remaining = (start - dt.datetime.now()).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(60.0, remaining))
        if time.monotonic() - last_report >= 1800:
            log(f"still waiting, {remaining / 60:.0f} min to go")
            last_report = time.monotonic()


def process_alive(pid: int) -> bool:
    """Whether a process with this id is still running.

    On Windows ``OpenProcess`` fails for a process id nobody holds, and
    ``GetExitCodeProcess`` tells a process that has exited (but whose handles
    are still open somewhere) from one that is running. Elsewhere
    ``os.kill(pid, 0)`` asks the same question.
    """
    if pid <= 0:
        return False
    windll = getattr(ctypes, "windll", None)
    if windll is not None:
        SYNCHRONIZE = 0x00100000
        QUERY_LIMITED = 0x00001000  # PROCESS_QUERY_LIMITED_INFORMATION
        handle = windll.kernel32.OpenProcess(SYNCHRONIZE | QUERY_LIMITED, False, pid)
        if not handle:
            return False
        try:
            STILL_ACTIVE = 259
            code = ctypes.c_ulong()
            if windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process(pid: int, log=say, poll: float = 30.0):
    """Return once the process ``pid`` has exited.

    Meant for chaining runs: ``--after outputs/logs/<run>.pid`` starts this
    training the moment the previous one is done, instead of guessing a clock
    time. The process is checked every ``poll`` seconds; progress is said
    every 30 minutes. A pid nobody owns returns at once. (The system recycles
    process ids eventually, so this trusts that the number still means the
    run it was read for - true for anything launched in the last hours.)
    """
    if not process_alive(pid):
        log(f"process {pid} is not running - nothing to wait for")
        return
    log(f"waiting for process {pid} to finish")
    started = last_report = time.monotonic()
    while process_alive(pid):
        time.sleep(poll)
        if time.monotonic() - last_report >= 1800:
            elapsed = (time.monotonic() - started) / 60
            log(f"still waiting for process {pid} ({elapsed:.0f} min so far)")
            last_report = time.monotonic()
    log(f"process {pid} has finished")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def prepare_run(run_dir, data_root, dataset, hp, force=False, log=say, build_hint=""):
    """Write ``specs.json`` and ``metadata.json``, validated against the dataset.

    Parameters
    ----------
    run_dir : path-like
        The run directory (``runs/<name>``); created if missing.
    data_root : path-like
        Root holding ``SdfSamples/`` and ``splits/``.
    dataset : str
        Dataset name, as ``datasets.list_datasets`` lists it.
    hp : dict
        A complete hyperparameter set (``hyperparams.defaults()`` updated with
        the recipe). ``log_frequency`` and ``snapshot_frequency`` are clamped
        to ``num_epochs`` here, so a 2-epoch test run still ends with a
        ``latest.pth`` to load. The dict is modified in place.
    force : bool
        Train over a run directory that already holds a checkpoint.
    log : callable
        Where validation notes and the summary line go.
    build_hint : str
        Printed when the dataset is missing: the command that makes it.

    Returns
    -------
    dict
        The dataset row from ``datasets.list_datasets`` (``path``, ``split``,
        ``n_instances``, ``geom_dimension``, ...).

    Exits the process (``sys.exit``) when the dataset is missing, the run
    directory holds a checkpoint and ``force`` is off, or the hyperparameter
    set has a validation error.
    """
    from structsept.app import datasets, hyperparams, training

    run_dir = pathlib.Path(run_dir)
    if (run_dir / "ModelParameters" / "latest.pth").is_file() and not force:
        sys.exit(
            f"{run_dir} already holds a trained checkpoint; pick another "
            "--run or pass --force to train over it"
        )

    rows = {d["name"]: d for d in datasets.list_datasets(data_root)}
    row = rows.get(dataset)
    if row is None or not row["split"]:
        hint = f"; build it with\n  {build_hint}" if build_hint else ""
        sys.exit(f"dataset {dataset} not found under {data_root}{hint}")

    epochs = int(hp["num_epochs"])
    hp["log_frequency"] = min(hyperparams.log_frequency(hp), epochs)
    hp["snapshot_frequency"] = min(hyperparams.snapshot_frequency(hp), epochs)
    issues = hyperparams.validate(hp, row["n_instances"], row["geom_dimension"])
    for issue in issues:
        log(f"  [{issue.level}] {issue.key}: {issue.message}")
    if hyperparams.errors(issues):
        sys.exit("hyperparameter set is not trainable")

    training.write_specs(
        run_dir,
        row["split"],
        data_root,
        hparams=hp,
        geom_dimension=row["geom_dimension"],
    )
    training.write_metadata(run_dir, dataset=dataset, preset=False)
    log(
        f"run {run_dir.name}: {row['n_instances']} shapes, geom {row['geom_dimension']}-D, "
        f"{hp['n_layers']}x{hp['width']}, d={hp['latent_dim']}, {epochs} epochs"
    )
    return row


def train_with_deadline(run_dir, data_root, deadline: dt.datetime, log=say):
    """Run the trainer and stop it at ``deadline`` if it is still going.

    The library trainer has no time limit, but it installs a Ctrl-C handler
    that does ``sys.exit(0)``. A watchdog thread raises SIGINT in the main
    thread at the deadline (``_thread.interrupt_main``), that handler runs,
    and the training loop exits cleanly; ``ModelParameters/latest.pth``
    (written every ``LogFrequency`` epochs) and the numbered snapshots are
    what remains.

    Returns ``(summary, stopped)``: ``summary`` is the trainer's result dict
    (``None`` when stopped) and ``stopped`` says whether the watchdog fired.
    A real Ctrl-C is re-raised.
    """
    import _thread

    from structsept.app import training

    fired = threading.Event()

    def watchdog():
        while not fired.is_set():
            remaining = (deadline - dt.datetime.now()).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(30.0, remaining))
        if not fired.is_set():
            fired.set()
            log(f"deadline {deadline:%H:%M} reached - asking the trainer to stop")
            _thread.interrupt_main()

    thread = threading.Thread(target=watchdog, name="deadline", daemon=True)
    thread.start()
    try:
        # The library's own log handler already prints to the (tee'd) console,
        # so the trainer's records are not forwarded a second time.
        summary = training.train(run_dir, data_root, log=lambda s: None)
    except (SystemExit, KeyboardInterrupt):
        if not fired.is_set():
            raise  # a real Ctrl-C or exit, not ours
        return None, True
    finally:
        fired.set()
    return summary, False


# --------------------------------------------------------------------------- #
# reading a run back
# --------------------------------------------------------------------------- #


def last_epoch(run_dir) -> int | None:
    """Epoch stored in ``ModelParameters/latest.pth``, or None."""
    import torch

    path = pathlib.Path(run_dir) / "ModelParameters" / "latest.pth"
    if not path.is_file():
        return None
    try:
        return int(torch.load(path, map_location="cpu", weights_only=True)["epoch"])
    except Exception:
        return None


def load_codes(run_dir):
    """The learned latent codes of a run as a ``(N, d)`` numpy array, or None.

    Read from ``LatentCodes/latest.pth``, which the trainer writes alongside
    ``ModelParameters/latest.pth``; either the codes tensor itself or an
    ``Embedding`` state dict (``{"weight": (N, d)}``) is accepted. Row ``i``
    is the shape at index ``i`` of the split (see :func:`split_names`).
    """
    import torch

    path = pathlib.Path(run_dir) / "LatentCodes" / "latest.pth"
    if not path.is_file():
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    codes = data["latent_codes"]
    if isinstance(codes, dict):
        codes = codes["weight"]
    return codes.detach().cpu().numpy().reshape(len(codes), -1)


def split_names(split_path) -> list[str]:
    """Instance names of a split file, in the order the trainer indexes them.

    The trainer's latent index is the split order, so matching codes to
    ``params.csv`` goes through these names rather than trusting row order.
    """
    split = json.loads(pathlib.Path(split_path).read_text(encoding="utf-8"))
    return [n for ds in split.values() for cls in ds.values() for n in cls]


# --------------------------------------------------------------------------- #
# a run whose recipe is another run's specs.json
# --------------------------------------------------------------------------- #

#: The only ``specs.json`` keys :func:`prepare_run_like` lets a copied run
#: differ in: where the data is, and what the run says about itself.
REPLACED_KEYS = ("DataSource", "TrainSplit", "TestSplit", "Description")


def prepare_run_like(
    run_dir,
    like_dir,
    data_root,
    dataset,
    description,
    epochs=None,
    force=False,
    log=say,
    build_hint="",
):
    """Write the ``specs.json`` of ``like_dir`` for ``dataset`` into ``run_dir``.

    The recipe is not spelled out by the caller: it is the reference run's
    (or preset's) ``specs.json``, copied key for key, so "same recipe" is a
    fact about files rather than a promise. Only :data:`REPLACED_KEYS` are
    then pointed at ``dataset`` and ``description``; the keys that differ are
    printed, and if any other one does the process exits before training.

    Parameters
    ----------
    run_dir, like_dir : path-like
        The run to write and the run whose specs are the recipe.
    data_root : path-like
    dataset : str
        Dataset name, as ``datasets.list_datasets`` lists it. Must have the
        dimension the reference decoder takes.
    description : str
        The new run's ``Description``.
    epochs : int or None
        A smoke test only, never the real run: replaces the epoch count and
        clamps the two save intervals to it, so a 2-epoch run still ends
        with a ``latest.pth``. The recipe check is skipped when set.
    force : bool
        Train over a run directory that already holds a checkpoint.
    log : callable
    build_hint : str
        Printed when the dataset is missing: the command that makes it.

    Returns
    -------
    (dict, list of str)
        The dataset row of ``datasets.list_datasets`` and the specs keys
        that differ from the reference.

    Exits the process (``sys.exit``) when the reference has no specs, the
    dataset is missing or of another dimension, ``run_dir`` already holds a
    checkpoint and ``force`` is off, or the recipe itself would differ.
    """
    from structsept.app import datasets, training

    run_dir, like_dir = pathlib.Path(run_dir), pathlib.Path(like_dir)
    if (run_dir / "ModelParameters" / "latest.pth").is_file() and not force:
        sys.exit(
            f"{run_dir} already holds a trained checkpoint; pick another "
            "--run or pass --force to train over it"
        )
    reference = training.read_specs(like_dir / "specs.json")
    if not reference:
        sys.exit(f"{like_dir} has no readable specs.json to copy the recipe from")
    rows = {d["name"]: d for d in datasets.list_datasets(data_root)}
    row = rows.get(dataset)
    if row is None or not row["split"]:
        hint = f"; build it with\n  {build_hint}" if build_hint else ""
        sys.exit(f"dataset {dataset} not found under {data_root}{hint}")
    geom = (reference.get("NetworkSpecs") or {}).get("geom_dimension", 3)
    if row["geom_dimension"] != geom:
        sys.exit(
            f"{dataset} is {row['geom_dimension']}-D, the decoder of "
            f"{like_dir.name} takes {geom} coordinates"
        )

    specs = json.loads(json.dumps(reference))  # a deep copy
    specs["DataSource"] = str(data_root)
    specs["TrainSplit"] = specs["TestSplit"] = str(row["split"])
    specs["Description"] = description
    if epochs is not None:
        specs["NumEpochs"] = int(epochs)
        for key in ("LogFrequency", "SnapshotFrequency"):
            specs[key] = min(int(specs[key]), int(epochs))

    changed = sorted(
        k for k in set(specs) | set(reference) if specs.get(k) != reference.get(k)
    )
    log(f"specs.json differs from {like_dir.name} in: {', '.join(changed)}")
    if epochs is None and not set(changed) <= set(REPLACED_KEYS):
        sys.exit("refusing to train: the recipe itself would differ")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "specs.json").write_text(json.dumps(specs, indent=4), encoding="utf-8")
    training.write_metadata(run_dir, dataset=dataset, preset=False)
    log(
        f"run {run_dir.name}: {row['n_instances']} shapes, d={specs['CodeLength']}, "
        f"{specs['NumEpochs']} epochs"
    )
    return row, changed


# --------------------------------------------------------------------------- #
# the codes against the generating parameter
# --------------------------------------------------------------------------- #


def code_vs_parameter(run_dir, dataset_dir, split_path, param, label=None, log=say):
    """Compare the learned code of every shape with one column of ``params.csv``.

    The check every one-parameter family ends with: does the number the
    decoder learned per shape track the number the shape was made from?
    Every component of the code is looked at on its own -- Pearson and
    Spearman correlation with the parameter, and whether it is monotonic in
    it. With more than one component the codes are also looked at together:
    the share of their variance along the first principal axis says how
    one-dimensional the set is (1.0 = every code on one straight line, which
    is all a single generating parameter can need), and the correlation of
    the position along that axis with the parameter says whether the line is
    ordered by it.

    Parameters
    ----------
    run_dir : path-like
    dataset_dir : path-like
        Folder holding ``params.csv``.
    split_path : path-like
        The split the run was trained on. The latent index is its order, so
        the names in it are matched against the ``name`` column of
        ``params.csv`` rather than trusting row order.
    param : str
        The column to compare against: ``"r"`` for the hole, ``"h"`` for
        the triangles.
    label : str or None
        Axis label of the parameter; default ``"<param> (design units)"``.
    log : callable

    Returns
    -------
    dict
        The numbers, or ``{}`` when the run holds no codes. Also written into
        the run directory: ``code_vs_<param>.csv`` (name, param, code_0,
        code_1, ...) and ``code_vs_<param>.png``.
    """
    import csv

    import numpy as np

    run_dir = pathlib.Path(run_dir)
    label = label or f"{param} (design units)"
    codes = load_codes(run_dir)
    if codes is None:
        log(f"no latent codes saved, skipping the code-vs-{param} check")
        return {}

    names = split_names(split_path)
    with open(pathlib.Path(dataset_dir) / "params.csv", newline="") as fh:
        table = {row["name"]: float(row[param]) for row in csv.DictReader(fh)}
    p = np.array([table[n] for n in names])
    codes = codes[: len(names)]  # (N, d)
    d = codes.shape[1]
    order = np.argsort(p)
    rank = lambda v: np.argsort(np.argsort(v))

    def against(values):
        """Correlations of one number per shape with the parameter."""
        steps = np.diff(values[order])
        return {
            "pearson": float(np.corrcoef(p, values)[0, 1]),
            "spearman": float(np.corrcoef(rank(p), rank(values))[0, 1]),
            "monotonic": bool(np.all(steps > 0) or np.all(steps < 0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    components = [against(codes[:, k]) for k in range(d)]
    result = {"parameter": param, "latent_dim": d, "components": components}
    if d > 1:
        centred = codes - codes.mean(axis=0)
        # Principal axes by SVD: the squared singular values are the variances
        # of the codes along each axis.
        _, sigma, axes = np.linalg.svd(centred, full_matrices=False)
        total = float(np.sum(sigma**2))
        along = centred @ axes[0]
        result["pc1_variance_share"] = sigma[0] ** 2 / total if total else float("nan")
        result["pc1_axis"] = axes[0].tolist()
        result["pc1"] = against(along)
        # A learned latent basis is arbitrary, so with several generating
        # parameters no single component has to track this one. If the codes
        # are an affine relabelling of the parameters, a linear map of the
        # codes recovers it: R^2 of that least-squares fit, with intercept.
        # A low R^2 says "not linearly readable", not "absent": the d = 2 run
        # on the three-parameter plate has R^2 0.009 for the radius, yet a
        # 3-nearest-neighbour fit in code space recovers it with 0.69 and the
        # decoded hole follows the true radius (Spearman 0.86). The codes
        # hold it on a curved surface.
        design = np.column_stack([codes, np.ones(len(codes))])
        coef, *_ = np.linalg.lstsq(design, p, rcond=None)
        ss_res = float(np.sum((p - design @ coef) ** 2))
        ss_tot = float(np.sum((p - p.mean()) ** 2))
        result["linear_fit_r2"] = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        result["linear_fit_direction"] = coef[:-1].tolist()

    with open(run_dir / f"code_vs_{param}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", param] + [f"code_{k}" for k in range(d)])
        for n, value, row in zip(names, p, codes):
            writer.writerow([n, f"{value:.6f}"] + [f"{c:.6f}" for c in row])

    import matplotlib.pyplot as plt

    def describe(stats):
        return (
            f"Pearson {stats['pearson']:+.3f}, Spearman {stats['spearman']:+.3f}, "
            f"{'monotonic' if stats['monotonic'] else 'NOT monotonic'}"
        )

    if d == 1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(p[order], codes[order, 0], "o-", ms=4)
        ax.set_xlabel(label)
        ax.set_ylabel("learned latent code")
        ax.set_title(f"{run_dir.name}\n{describe(components[0])}")
        ax.grid(True, alpha=0.3)
    else:
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
        for k in range(d):
            left.plot(p[order], codes[order, k], "o-", ms=4, label=f"code {k}")
        left.set_xlabel(label)
        left.set_ylabel("learned latent code, per component")
        left.legend()
        left.grid(True, alpha=0.3)
        left.set_title(
            "\n".join(f"code {k}: {describe(s)}" for k, s in enumerate(components)),
            fontsize=9,
        )
        # The latent plane itself (the first two principal coordinates when
        # d > 2), coloured by the parameter and threaded in its order.
        if d == 2:
            xy, xlabel, ylabel = codes, "code 0", "code 1"
        else:
            xy = centred @ axes[:2].T
            xlabel, ylabel = "principal coordinate 1", "principal coordinate 2"
        right.plot(xy[order, 0], xy[order, 1], "-", color="0.7", lw=1, zorder=1)
        sc = right.scatter(xy[:, 0], xy[:, 1], c=p, cmap="viridis", s=28, zorder=2)
        fig.colorbar(sc, ax=right, label=label)
        right.set_xlabel(xlabel)
        right.set_ylabel(ylabel)
        right.set_aspect("equal", adjustable="datalim")
        right.grid(True, alpha=0.3)
        right.set_title(
            f"{result['pc1_variance_share'] * 100:.1f} % of the variance on one axis\n"
            f"along it: {describe(result['pc1'])}\n"
            f"linear fit of {param} on the codes: R² = {result['linear_fit_r2']:.3f}",
            fontsize=8,
        )
        fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / f"code_vs_{param}.png", dpi=120)
    plt.close(fig)

    for k, stats in enumerate(components):
        log(
            f"code {k} vs {param}: {describe(stats)} "
            f"(from {stats['min']:+.3f} to {stats['max']:+.3f})"
        )
    if d > 1:
        log(
            f"codes together: {result['pc1_variance_share'] * 100:.1f} % of the "
            f"variance along one axis; along it {describe(result['pc1'])}; "
            f"linear fit of {param} on the codes R^2 = {result['linear_fit_r2']:.3f}"
        )
    return result
