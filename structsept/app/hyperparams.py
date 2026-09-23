"""Training hyperparameters of the structsept app: one schema, every consumer.

The DeepSDF trainer (``DeepSDFStruct.deep_sdf.training.train_deep_sdf``) takes
no arguments beyond a run directory: everything it does is read from the
``specs.json`` inside it. This module is the single description of every key of
that file the app lets you change - its type, its admissible range, its
default, the specs.json key it lands in, and one or two sentences on what it
does to the decoder ``f_theta``. Everything else is built from it:

* the hyperparameter window (``hparam_window``) lays out its form from
  :data:`FIELDS` and :data:`GROUPS`,
* the Train tab card summarises the values that differ from the defaults,
* ``training.write_specs`` serialises a hyperparameter dict through
  :func:`to_specs`, and "Load from run" reads one back with :func:`from_specs`.

Adding a hyperparameter is therefore one :class:`Field` entry plus one line in
:func:`to_specs` and :func:`from_specs`. No Tk here: the module imports and
tests without a display.

A hyperparameter set is a plain ``dict`` from :attr:`Field.key` to a typed
value (``int``, ``float``, ``bool``, ``str``, ``list[int]`` or ``None``), so it
can be copied, compared and written to json without ceremony.

What is deliberately *not* here is listed in :data:`FIXED`: values the app
pins (CPU, no loader processes, the ``deep_sdf_decoder`` architecture) and
values the trainer hardcodes whatever specs.json says (gradient clipping at
1.0, the code-regularization ramp). The window shows that list too, so a value
that cannot be changed is at least not a secret.

Where the defaults come from
----------------------------
They reproduce, key for key, the ``specs.json`` the app wrote before this
module existed, which in turn follows the shipped ``RoundCross`` and
``ChiAndCross`` decoders (6 x 128, skip at layer 2, weight norm, dropout 0.2,
clamping 0.1, Adam at 5e-4 / 1e-3 halved every 500 epochs). Only
``SamplesPerScene`` is smaller (8000 against 16000) - a CPU budget.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# The decoder class the app trains. The other architectures in
# DeepSDFStruct.deep_sdf.workspace.ARCHITECTURES take NetworkSpecs keys of
# their own that this form does not describe.
ARCH = "deep_sdf_decoder"

# Coordinates per sample. Not a hyperparameter: the dataset decides it (3 for
# the SDF maker's meshes, 2 for a planar datagen set), so it is passed in next
# to the set rather than stored in it. This is the value when nobody says.
GEOM_DIMENSION = 3

LOSS_FUNCTIONS = ("clampedL1", "leakyClampedL1", "L1", "MSE", "huber")
SCHEDULE_TYPES = ("Step", "Warmup", "Constant")

# ``ClampedL1Loss`` and ``LeakyClampedL1Loss`` are built with their default
# clamp_val of 0.1 by ``nn_utils.get_loss_function``, independent of
# ClampingDistance.
LOSS_OWN_CLAMP = 0.1

# Order in which ``auto`` picks LogFrequency. It has to divide NumEpochs, or
# the last epoch never reaches ModelParameters/latest.pth.
_AUTO_LOG_FREQUENCIES = (10, 5, 2, 1)

ERROR, WARNING, NOTE = "error", "warning", "note"


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Field:
    """One editable hyperparameter.

    Attributes
    ----------
    key : str
        Key in the hyperparameter dict. Never shown to the user.
    label : str
        Row label in the window.
    group : str
        Key of the :data:`GROUPS` entry the row belongs to.
    kind : str
        How the value is typed and parsed:

        ``"int"``, ``"float"``, ``"bool"``, ``"text"``
            the obvious ones;
        ``"choice"``
            one of :attr:`choices`;
        ``"int_list"``
            a list of ints, typed as ``"2, 4"``; empty or ``none`` is ``[]``;
        ``"layers"``
            ``"all"``, ``"none"`` or a list of layer indices;
        ``"opt_int"``, ``"opt_float"``
            a number, or ``None`` when left empty (meaning: :attr:`blank`).
    default : object
        Value of a fresh hyperparameter set.
    help : str
        What the value does, in one or two sentences.
    spec : str
        Where it lands in specs.json, shown under the row.
    low, high : float or None
        Inclusive bounds for numbers and list entries.
    low_open : bool
        Makes :attr:`low` exclusive (a learning rate must be > 0, not >= 0).
    choices : tuple of str
        Admissible values of a ``"choice"`` field.
    blank : str
        What an empty ``opt_*`` entry means, shown as its placeholder.
    card : bool
        The value also has a spinbox on the Train tab card.
    sub : str
        Sub-heading inside the group, for the two learning-rate columns.
    short : str
        Name used in the one-line summary on the card; defaults to the label.
    schedules : tuple of str
        Learning-rate fields only: the schedule types the value applies to.
        The window hides the row for the other types.
    """

    key: str
    label: str
    group: str
    kind: str
    default: object
    help: str
    spec: str
    low: float | None = None
    high: float | None = None
    low_open: bool = False
    choices: tuple = ()
    blank: str = ""
    card: bool = False
    sub: str = ""
    short: str = ""
    schedules: tuple = ()

    @property
    def name(self) -> str:
        return self.short or self.label

    @property
    def full_label(self) -> str:
        """The label with its sub-heading, for text outside the form.

        Inside the learning-rate table the column says which schedule a row
        belongs to; in a log line or a check message nothing does.
        """
        return f"{self.label} ({self.sub.lower()})" if self.sub else self.label


@dataclass(frozen=True)
class Issue:
    """One finding of :func:`validate`: ``level`` is error, warning or note.

    ``key`` names the field the finding is about, or is empty when it concerns
    the combination rather than one value. Errors block training; warnings
    and notes do not.
    """

    level: str
    key: str
    message: str


# (key, title, subtitle, column in the window)
GROUPS = (
    ("arch", "Architecture", "the decoder MLP fθ", 0),
    ("latent", "Latent codes", "one learned vector λ per training shape", 0),
    ("loss", "Loss", "what fθ is fitted to", 0),
    ("lr", "Learning rate", "Adam, one schedule per parameter group", 1),
    ("data", "Sampling and batches", "what one optimizer step sees", 1),
    ("run", "Budget and checkpoints", "how long, and what is kept", 1),
)


def _lr_fields(prefix, sub, who, initial):
    """The six fields of one ``LearningRateSchedule`` entry.

    The trainer builds Adam with two parameter groups - the decoder weights
    first, the latent codes second - and gives each its own schedule, so the
    window shows two identical columns of these.
    """
    index = 0 if prefix == "lr_dec" else 1
    spec = f"LearningRateSchedule[{index}]"
    return (
        Field(
            f"{prefix}_type",
            "Schedule",
            "lr",
            "choice",
            "Step",
            "Step: multiply by a factor every N epochs. Warmup: ramp linearly "
            "to a target, then hold. Constant: never change.",
            f"{spec}.Type",
            choices=SCHEDULE_TYPES,
            sub=sub,
            short=f"{who} schedule",
        ),
        Field(
            f"{prefix}_initial",
            "Learning rate",
            "lr",
            "float",
            initial,
            "The base rate. Step: used until the first step. Warmup: where "
            "the ramp starts (epoch 1 is already one increment up). Constant: "
            "the only rate.",
            f"{spec}.Initial (Value for Constant)",
            low=0.0,
            high=1.0,
            low_open=True,
            sub=sub,
            short=f"{who} lr",
        ),
        Field(
            f"{prefix}_interval",
            "Step every (epochs)",
            "lr",
            "int",
            500,
            "The rate is multiplied by the factor once every this many epochs. "
            "Longer than the run: it never changes.",
            f"{spec}.Interval",
            low=1,
            high=1_000_000,
            sub=sub,
            short=f"{who} lr step",
            schedules=("Step",),
        ),
        Field(
            f"{prefix}_factor",
            "Step factor",
            "lr",
            "float",
            0.5,
            "0.5 halves the rate at every step.",
            f"{spec}.Factor",
            low=0.0,
            high=1.0,
            low_open=True,
            sub=sub,
            short=f"{who} lr factor",
            schedules=("Step",),
        ),
        Field(
            f"{prefix}_final",
            "Warm-up target",
            "lr",
            "float",
            initial,
            "Rate reached at the end of the ramp and kept from then on.",
            f"{spec}.Final",
            low=0.0,
            high=1.0,
            low_open=True,
            sub=sub,
            short=f"{who} warm-up target",
            schedules=("Warmup",),
        ),
        Field(
            f"{prefix}_length",
            "Warm-up length (epochs)",
            "lr",
            "int",
            100,
            "Epochs the linear ramp takes.",
            f"{spec}.Length",
            low=1,
            high=1_000_000,
            sub=sub,
            short=f"{who} warm-up length",
            schedules=("Warmup",),
        ),
    )


FIELDS: tuple[Field, ...] = (
    # -- architecture ------------------------------------------------------- #
    Field(
        "n_layers",
        "Hidden layers",
        "arch",
        "int",
        6,
        "Fully-connected hidden layers. The input layer, which takes (λ, x), "
        "and the scalar output layer come on top, so the linear layers are "
        "numbered 0 to <hidden layers>.",
        "NetworkSpecs.dims (length)",
        low=2,
        high=12,
        card=True,
        short="layers",
    ),
    Field(
        "width",
        "Layer width",
        "arch",
        "int",
        128,
        "Neurons per hidden layer; every hidden layer gets the same width.",
        "NetworkSpecs.dims (value)",
        low=16,
        high=512,
        card=True,
        short="width",
    ),
    Field(
        "latent_in",
        "Skip connection at layers",
        "arch",
        "int_list",
        [2],
        "Layers that receive (λ, x) again, concatenated to their input - the "
        "DeepSDF skip connection. Indices 1 to <hidden layers>; empty for "
        "none. The layer before a skip outputs width − (d + 3) neurons "
        "(d + 2 on a 2-D dataset).",
        "NetworkSpecs.latent_in",
        low=0,
        high=64,
        short="skip layers",
    ),
    Field(
        "xyz_in_all",
        "Feed x to every layer",
        "arch",
        "bool",
        False,
        "Concatenate the query point x to the input of every hidden layer. "
        "Each layer outputs 3 fewer neurons to keep the width.",
        "NetworkSpecs.xyz_in_all",
        short="x in all layers",
    ),
    Field(
        "weight_norm",
        "Weight normalization",
        "arch",
        "bool",
        True,
        "On: weight normalization on the normalized layers. Off: a LayerNorm "
        "after each of them instead.",
        "NetworkSpecs.weight_norm",
        short="weight norm",
    ),
    Field(
        "norm_layers",
        "Normalized layers",
        "arch",
        "layers",
        "all",
        "'all', 'none', or layer indices such as '0, 1, 2'.",
        "NetworkSpecs.norm_layers",
        low=0,
        high=64,
        short="normalized layers",
    ),
    Field(
        "dropout_layers",
        "Dropout layers",
        "arch",
        "layers",
        "all",
        "Layers followed by dropout while training: 'all', 'none' or "
        "indices. The output layer never gets dropout.",
        "NetworkSpecs.dropout",
        low=0,
        high=64,
        short="dropout layers",
    ),
    Field(
        "dropout_prob",
        "Dropout probability",
        "arch",
        "float",
        0.2,
        "Fraction of activations zeroed on the dropout layers, training only.",
        "NetworkSpecs.dropout_prob",
        low=0.0,
        high=0.95,
        short="dropout",
    ),
    Field(
        "use_tanh",
        "tanh on the output",
        "arch",
        "bool",
        False,
        "Squash the predicted distance into (−1, 1). Only sensible while the "
        "band the loss compares stays below 1.",
        "NetworkSpecs.use_tanh",
        short="output tanh",
    ),
    # -- latent codes ------------------------------------------------------- #
    Field(
        "latent_dim",
        "Latent dimension d",
        "latent",
        "int",
        1,
        "Length of λ. It is also the number of design variables per spline "
        "control point. d ≥ 2 needs far more training shapes to fill the "
        "latent space (the paper used 120 for d = 2).",
        "CodeLength",
        low=1,
        high=64,
        card=True,
        short="d",
    ),
    Field(
        "code_init_std",
        "Initial spread σ",
        "latent",
        "float",
        1.0,
        "Codes start as N(0, σ²/d). Larger values spread the shapes apart in "
        "latent space from the first epoch.",
        "CodeInitStdDev",
        low=0.0,
        high=10.0,
        low_open=True,
        short="code init σ",
    ),
    Field(
        "code_bound",
        "Maximum norm",
        "latent",
        "opt_float",
        1.0,
        "Codes are rescaled to at most this length whenever they are looked "
        "up. It bounds the latent range the optimizer later moves in. Empty: "
        "unbounded.",
        "CodeBound",
        low=0.0,
        high=100.0,
        low_open=True,
        blank="unbounded",
        short="code bound",
    ),
    Field(
        "code_reg_lambda",
        "Regularization weight",
        "latent",
        "float",
        1e-4,
        "Weight of the mean code norm in the loss, ramped in over the first "
        "100 epochs. Pulls the codes towards 0; 0 switches it off.",
        "CodeRegularizationLambda",
        low=0.0,
        high=10.0,
        short="code reg.",
    ),
    Field(
        "latent_dropout",
        "Latent dropout",
        "latent",
        "bool",
        False,
        "Zero random components of λ while training (p = 0.2, fixed in the "
        "decoder).",
        "NetworkSpecs.latent_dropout",
        short="latent dropout",
    ),
    # -- loss --------------------------------------------------------------- #
    Field(
        "loss_function",
        "Loss function",
        "loss",
        "choice",
        "clampedL1",
        "clampedL1 is DeepSDF's own and clamps at ±0.1 inside the loss, "
        "whatever δ says. leakyClampedL1 continues past ±0.1 with slope 0.01. "
        "L1, MSE and huber see δ alone.",
        "LossFunction",
        choices=LOSS_FUNCTIONS,
        short="loss",
    ),
    Field(
        "clamping_distance",
        "Clamping distance δ",
        "loss",
        "float",
        0.1,
        "Target and prediction are clamped to ±δ before the loss, so only "
        "the band |φ| < δ around the surface is learned accurately.",
        "ClampingDistance",
        low=0.0,
        high=10.0,
        low_open=True,
        short="δ",
    ),
    # -- learning rate ------------------------------------------------------ #
    *_lr_fields("lr_dec", "Decoder weights", "decoder", 5e-4),
    *_lr_fields("lr_code", "Latent codes", "codes", 1e-3),
    # -- sampling and batches ----------------------------------------------- #
    Field(
        "samples_per_scene",
        "Samples per shape",
        "data",
        "int",
        8000,
        "SDF samples drawn per shape per step, half inside and half outside "
        "the surface. Must be even.",
        "SamplesPerScene",
        low=2,
        high=1_000_000,
        short="samples/shape",
    ),
    Field(
        "scenes_per_batch",
        "Shapes per batch",
        "data",
        "int",
        10,
        "Shapes in one optimizer step. An epoch is shapes // batch steps: "
        "the shapes left over (a random few each epoch) sit that epoch out.",
        "ScenesPerBatch",
        low=1,
        high=100_000,
        short="shapes/batch",
    ),
    # -- budget and checkpoints --------------------------------------------- #
    Field(
        "num_epochs",
        "Epochs",
        "run",
        "int",
        200,
        "Passes over the dataset, minus the partial last batch (see Shapes "
        "per batch).",
        "NumEpochs",
        low=1,
        high=100_000,
        card=True,
        short="epochs",
    ),
    Field(
        "seed",
        "Random seed",
        "run",
        "int",
        42,
        "Seeds Python, NumPy and torch at the start. Same seed, data and "
        "settings: the same network.",
        "seed",
        low=0,
        high=2**31 - 1,
        short="seed",
    ),
    Field(
        "log_frequency",
        "Save latest every (epochs)",
        "run",
        "opt_int",
        None,
        "How often latest.pth and Logs.pth are written - also how often the "
        "loss curve here updates. Auto: 10, 5, 2 or 1, whichever divides the "
        "epochs, so the last epoch is the one saved.",
        "LogFrequency",
        low=1,
        high=100_000,
        blank="auto",
        short="save every",
    ),
    Field(
        "snapshot_frequency",
        "Keep a snapshot every (epochs)",
        "run",
        "opt_int",
        None,
        "Numbered checkpoints <epoch>.pth kept alongside latest. Auto: a "
        "quarter of the epochs.",
        "SnapshotFrequency",
        low=1,
        high=100_000,
        blank="auto",
        short="snapshot every",
    ),
    Field(
        "additional_snapshots",
        "Extra snapshots at epochs",
        "run",
        "int_list",
        [1],
        "Epochs that get a numbered checkpoint on top of the regular ones.",
        "AdditionalSnapshots",
        low=1,
        high=100_000,
        short="extra snapshots",
    ),
    Field(
        "description",
        "Description",
        "run",
        "text",
        "",
        "Free text kept in specs.json. Empty: generated from d, the "
        "architecture and the epochs.",
        "Description",
        short="description",
    ),
)

FIELD_BY_KEY = {f.key: f for f in FIELDS}
CARD_KEYS = tuple(f.key for f in FIELDS if f.card)

# Shown read-only in the window: (what, value, why it is not editable here).
FIXED = (
    (
        "Architecture",
        ARCH,
        "the other DeepSDFStruct decoders need NetworkSpecs keys this form "
        "does not describe",
    ),
    (
        "Geometry dimension",
        "from the dataset",
        "3 for (x, y, z, φ) rows, 2 for (x, y, φ); set by the data, not here",
    ),
    ("Optimizer", "Adam", "hardcoded in the trainer, betas at the torch defaults"),
    (
        "Gradient clipping",
        "max-norm 1.0 on the decoder",
        "hardcoded in deep_sdf/training.py; GradientClipNorm is read, then "
        "overwritten",
    ),
    (
        "Code regularization ramp",
        "weight × min(1, epoch / 100)",
        "hardcoded in the trainer",
    ),
    ("Device", "CPU", "the app trains on CPU"),
    (
        "Loader processes",
        "0 (DataLoaderThreads)",
        "a worker process re-imports the whole app on Windows",
    ),
)


def defaults() -> dict:
    """A fresh hyperparameter dict holding every default."""
    return {f.key: _copy(f.default) for f in FIELDS}


def _copy(value):
    return list(value) if isinstance(value, list) else value


def fields_in(group: str, sub: str | None = None) -> list[Field]:
    """Fields of one group (and one sub-heading, if given), in schema order."""
    return [f for f in FIELDS if f.group == group and (sub is None or f.sub == sub)]


def subgroups(group: str) -> list[str]:
    """Distinct sub-headings of a group, in schema order."""
    seen = []
    for f in fields_in(group):
        if f.sub not in seen:
            seen.append(f.sub)
    return seen


# --------------------------------------------------------------------------- #
# text <-> value
# --------------------------------------------------------------------------- #


def _parse_int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        value = float(text)  # "1e3" is a fine way to type 1000
        if not value.is_integer():
            raise ValueError(f"'{text}' is not a whole number") from None
        return int(value)


def _parse_float(text: str) -> float:
    # A decimal comma is what half the keyboards in this building type.
    return float(text.replace(",", "."))


def _parse_int_list(text: str) -> list[int]:
    parts = [p for p in re.split(r"[,;\s]+", text.strip()) if p]
    if len(parts) == 1 and parts[0].lower() == "none":
        return []
    out = []
    for part in parts:
        try:
            out.append(_parse_int(part))
        except ValueError:
            raise ValueError(f"'{part}' is not a whole number") from None
    return sorted(set(out))


def parse(field: Field, text) -> object:
    """Typed value of ``field`` from what was typed into its entry.

    Raises ``ValueError`` with a message meant for the user when the text is
    not a value of the right type or lies outside the field's range.
    """
    if field.kind == "bool":
        return bool(text)
    raw = str(text).strip()
    if raw == "" and field.kind in ("int", "float", "choice"):
        raise ValueError("required")
    try:
        if field.kind == "text":
            value = raw
        elif field.kind == "choice":
            if raw not in field.choices:
                raise ValueError(f"pick one of {', '.join(field.choices)}")
            value = raw
        elif field.kind == "int":
            value = _parse_int(raw)
        elif field.kind == "float":
            value = _parse_float(raw)
        elif field.kind in ("opt_int", "opt_float"):
            if raw == "" or raw.lower() in ("auto", "none", field.blank.lower()):
                return None
            value = _parse_int(raw) if field.kind == "opt_int" else _parse_float(raw)
        elif field.kind == "int_list":
            value = _parse_int_list(raw)
        elif field.kind == "layers":
            if raw.lower() in ("all", "*"):
                return "all"
            if raw == "" or raw.lower() == "none":
                return "none"
            value = _parse_int_list(raw)
        else:  # pragma: no cover - a typo in FIELDS
            raise ValueError(f"unknown kind {field.kind!r}")
    except ValueError as exc:
        message = str(exc)
        if message.startswith("could not convert") or message.startswith(
            "invalid literal"
        ):
            message = f"'{raw}' is not a number"
        raise ValueError(message) from None
    problem = check(field, value)
    if problem:
        raise ValueError(problem)
    return value


def check(field: Field, value) -> str | None:
    """Range problem of an already-typed value, or ``None`` when it is fine.

    Kept apart from :func:`parse` because values also arrive typed - from the
    card spinboxes and from :func:`from_specs` - and have to be checked
    without a round-trip through text.
    """
    if value is None:
        return None if field.kind in ("opt_int", "opt_float") else "required"
    if field.kind == "choice":
        return None if value in field.choices else f"not one of {field.choices}"
    if field.kind in ("int", "opt_int", "float", "opt_float"):
        numbers = [value]
    elif field.kind == "int_list" or (
        field.kind == "layers" and isinstance(value, list)
    ):
        numbers = list(value)
    else:
        return None
    for number in numbers:
        if isinstance(number, float) and not math.isfinite(number):
            return "must be a finite number"
        if field.low is not None:
            if field.low_open and number <= field.low:
                return f"must be greater than {_fmt_number(field.low)}"
            if number < field.low:
                return f"must be at least {_fmt_number(field.low)}"
        if field.high is not None and number > field.high:
            return f"must be at most {_fmt_number(field.high)}"
    return None


def _fmt_number(value) -> str:
    if isinstance(value, float) and not value.is_integer():
        return repr(value)
    return str(int(value)) if isinstance(value, float) else str(value)


def format_value(field: Field, value) -> str:
    """Text for ``field``'s entry. ``parse(format_value(v)) == v`` for any
    valid value: floats go through ``repr``, which round-trips exactly."""
    if value is None:
        return ""
    if field.kind == "bool":
        return "on" if value else "off"
    if field.kind in ("int_list", "layers") and isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


# --------------------------------------------------------------------------- #
# derived values
# --------------------------------------------------------------------------- #


def auto_log_frequency(num_epochs: int) -> int:
    """LogFrequency that lands the last epoch in ModelParameters/latest.pth.

    The trainer only writes latest.pth when ``epoch % LogFrequency == 0``,
    and its own default of 10 silently leaves a short run with no loadable
    checkpoint at all.
    """
    return next(f for f in _AUTO_LOG_FREQUENCIES if int(num_epochs) % f == 0)


def auto_snapshot_frequency(num_epochs: int) -> int:
    return max(1, int(num_epochs) // 4)


def log_frequency(hp: dict) -> int:
    value = hp.get("log_frequency")
    return auto_log_frequency(hp["num_epochs"]) if value is None else int(value)


def snapshot_frequency(hp: dict) -> int:
    value = hp.get("snapshot_frequency")
    return auto_snapshot_frequency(hp["num_epochs"]) if value is None else int(value)


def layer_list(value, n_layers: int) -> list[int]:
    """Resolve an ``all``/``none``/list layer selection to explicit indices.

    ``all`` covers indices 0 to ``n_layers + 1``: the decoder has linear
    layers 0 to ``n_layers``, and the extra index keeps the output identical
    to the specs the app wrote before, and to the shipped decoders.
    """
    if value == "all":
        return list(range(int(n_layers) + 2))
    if value == "none" or value is None:
        return []
    return sorted(set(int(v) for v in value))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def validate(
    hp: dict, n_shapes: int | None = None, geom_dimension: int = GEOM_DIMENSION
) -> list[Issue]:
    """Everything wrong with a hyperparameter set, worst first.

    Errors are combinations the trainer crashes on, or that would leave a run
    with nothing to load. Warnings are legal but almost certainly not meant.
    Notes explain a value whose effect is easy to misread - most often a
    learning-rate schedule that never steps inside the run.

    Parameters
    ----------
    hp : dict
        A hyperparameter set, typed.
    n_shapes : int, optional
        Shapes in the selected dataset, for the batch-size check.
    geom_dimension : int
        Coordinates per sample of that dataset; the decoder input is
        ``d + geom_dimension`` wide.
    """
    issues: list[Issue] = []

    def add(level, key, message):
        issues.append(Issue(level, key, message))

    for field in FIELDS:
        problem = check(field, hp.get(field.key))
        if problem:
            add(ERROR, field.key, f"{field.full_label}: {problem}.")
    if any(i.level == ERROR for i in issues):
        # the combination checks below assume every value is in range
        return issues

    d = hp["latent_dim"]
    n_layers = hp["n_layers"]
    width = hp["width"]
    epochs = hp["num_epochs"]
    input_width = d + int(geom_dimension)

    # -- architecture: combinations DeepSDFDecoder cannot build or run ------ #
    skips = hp["latent_in"]
    if 0 in skips:
        add(
            ERROR,
            "latent_in",
            "Layer 0 cannot take a skip connection: its input already is "
            "(λ, x), and doubling it breaks the first layer.",
        )
    # DeepSDFDecoder shrinks the layer *before* a skip, testing "layer + 1 in
    # latent_in" for every layer up to the output one. Index n_layers + 1 is
    # therefore not ignored: it shrinks the output layer to 1 − (d + geom)
    # neurons and the constructor fails. Only indices past that are inert.
    if n_layers + 1 in skips:
        add(
            ERROR,
            "latent_in",
            f"Skip layer {n_layers + 1} does not exist (valid: 1 to {n_layers}), "
            "and the decoder does not ignore it: it shrinks the output layer "
            "to a negative width.",
        )
    beyond = [i for i in skips if i > n_layers + 1]
    if beyond:
        add(
            WARNING,
            "latent_in",
            f"Skip layer(s) {beyond} do not exist in a {n_layers}-layer decoder "
            f"(valid: 1 to {n_layers}) and are ignored.",
        )
    if any(1 <= i <= n_layers for i in skips) and width <= input_width:
        add(
            ERROR,
            "width",
            f"A skip connection needs width > d + {int(geom_dimension)} = "
            f"{input_width}: the layer before it outputs width − {input_width} "
            "neurons.",
        )
    if hp["xyz_in_all"] and width <= geom_dimension:
        add(
            ERROR,
            "width",
            f"Feeding x to every layer needs width > {int(geom_dimension)}.",
        )

    for key in ("norm_layers", "dropout_layers"):
        value = hp[key]
        if isinstance(value, list):
            unused = [i for i in value if i > n_layers]
            if unused:
                add(
                    NOTE,
                    key,
                    f"{FIELD_BY_KEY[key].label}: index(es) {unused} past the "
                    f"output layer ({n_layers}) have no effect.",
                )
    dropout_on = layer_list(hp["dropout_layers"], n_layers)
    if hp["dropout_prob"] > 0 and not any(i < n_layers for i in dropout_on):
        add(
            NOTE,
            "dropout_prob",
            "Dropout probability is set but no hidden layer has dropout, so "
            "it has no effect.",
        )
    elif dropout_on and hp["dropout_prob"] == 0:
        add(
            NOTE,
            "dropout_layers",
            "Dropout layers are listed but the probability is 0: no dropout.",
        )

    # -- loss --------------------------------------------------------------- #
    delta = hp["clamping_distance"]
    loss = hp["loss_function"]
    # What the loss compares: clampedL1 cuts at its own ±0.1 whatever δ is;
    # leakyClampedL1 only flattens past it, so the band still reaches δ.
    band = min(delta, LOSS_OWN_CLAMP) if loss == "clampedL1" else delta
    if hp["use_tanh"] and band >= 1.0:
        add(
            WARNING,
            "use_tanh",
            f"tanh keeps the output inside (−1, 1), but the loss compares "
            f"targets up to ±{band:g}: the network cannot reach them.",
        )
    if loss == "clampedL1" and delta > LOSS_OWN_CLAMP:
        add(
            NOTE,
            "clamping_distance",
            f"clampedL1 clamps at ±{LOSS_OWN_CLAMP:g} by itself, so the band "
            f"actually learned is ±{LOSS_OWN_CLAMP:g}, not ±{delta:g}. Pick L1 "
            "to learn the wider band.",
        )
    elif loss == "leakyClampedL1" and delta > LOSS_OWN_CLAMP:
        add(
            NOTE,
            "clamping_distance",
            f"leakyClampedL1 weights errors past ±{LOSS_OWN_CLAMP:g} 100 times "
            f"less (slope 0.01), so the band from ±{LOSS_OWN_CLAMP:g} to "
            f"±{delta:g} is learned only weakly. L1 weights it fully.",
        )

    # -- sampling ----------------------------------------------------------- #
    if hp["samples_per_scene"] % 2:
        add(
            ERROR,
            "samples_per_scene",
            "Samples per shape must be even: the loader draws half inside, "
            "half outside, and an odd count leaves each batch one sample short "
            "of its latent codes (the trainer crashes on the mismatch).",
        )
    batch = hp["scenes_per_batch"]
    if n_shapes is not None and n_shapes > 0 and batch > n_shapes:
        add(
            WARNING,
            "scenes_per_batch",
            f"The dataset has {n_shapes} shape(s), so every epoch is a single "
            f"batch of {n_shapes}, not {batch}.",
        )
    elif n_shapes and n_shapes % batch:
        # the trainer's DataLoader runs with drop_last=True whenever the batch
        # fits into the dataset
        left = n_shapes % batch
        add(
            NOTE,
            "scenes_per_batch",
            f"{left} of the {n_shapes} shapes sit out every epoch: the trainer "
            f"drops the partial last batch. A batch size that divides "
            f"{n_shapes} uses them all.",
        )

    # -- checkpoints -------------------------------------------------------- #
    save_every = log_frequency(hp)
    if save_every > epochs:
        add(
            ERROR,
            "log_frequency",
            f"latest.pth is written every {save_every} epochs, but the run has "
            f"only {epochs}: it would end with no checkpoint to load.",
        )
    elif epochs % save_every:
        last = epochs - epochs % save_every
        add(
            WARNING,
            "log_frequency",
            f"{epochs} is not a multiple of {save_every}: latest.pth - the one "
            f"the Explore tab opens - will hold epoch {last}, not {epochs}.",
        )
    if snapshot_frequency(hp) > epochs:
        add(
            NOTE,
            "snapshot_frequency",
            f"Snapshots every {snapshot_frequency(hp)} epochs: none is reached "
            f"in {epochs}.",
        )
    late = [e for e in hp["additional_snapshots"] if e > epochs]
    if late:
        add(
            NOTE, "additional_snapshots", f"Extra snapshot(s) {late} are never reached."
        )

    # -- learning-rate schedules -------------------------------------------- #
    for prefix, who in (("lr_dec", "Decoder"), ("lr_code", "Latent-code")):
        kind = hp[f"{prefix}_type"]
        if kind == "Step" and hp[f"{prefix}_interval"] > epochs:
            add(
                NOTE,
                f"{prefix}_interval",
                f"{who} learning rate steps every {hp[f'{prefix}_interval']} "
                f"epochs, so it stays at {hp[f'{prefix}_initial']:g} for the "
                f"whole {epochs}-epoch run.",
            )
        if kind == "Warmup" and hp[f"{prefix}_length"] > epochs:
            add(
                NOTE,
                f"{prefix}_length",
                f"{who} warm-up takes {hp[f'{prefix}_length']} epochs and never "
                f"finishes in {epochs}.",
            )

    order = {ERROR: 0, WARNING: 1, NOTE: 2}
    return sorted(issues, key=lambda i: order[i.level])


def errors(issues) -> list[Issue]:
    return [i for i in issues if i.level == ERROR]


# --------------------------------------------------------------------------- #
# specs.json
# --------------------------------------------------------------------------- #


def _schedule(hp: dict, prefix: str) -> dict:
    kind = hp[f"{prefix}_type"]
    if kind == "Step":
        return {
            "Type": "Step",
            "Initial": hp[f"{prefix}_initial"],
            "Interval": hp[f"{prefix}_interval"],
            "Factor": hp[f"{prefix}_factor"],
        }
    if kind == "Warmup":
        return {
            "Type": "Warmup",
            "Initial": hp[f"{prefix}_initial"],
            "Final": hp[f"{prefix}_final"],
            "Length": hp[f"{prefix}_length"],
        }
    return {"Type": "Constant", "Value": hp[f"{prefix}_initial"]}


def default_description(hp: dict) -> str:
    return (
        f"structsept run: d={hp['latent_dim']}, {hp['n_layers']}x{hp['width']}, "
        f"{hp['num_epochs']} epochs"
    )


def to_specs(
    hp: dict, split_path, data_source, geom_dimension: int = GEOM_DIMENSION
) -> dict:
    """The ``specs.json`` content for a hyperparameter set.

    Parameters
    ----------
    hp : dict
        A hyperparameter set. Missing keys take their defaults, so a partial
        dict such as ``{"latent_dim": 2}`` is fine.
    split_path : path-like
        Split json listing the training instances. Written absolute: the
        trainer resolves TrainSplit against DataSource, and an absolute path
        is the only form that survives both layouts.
    data_source : path-like
        Directory holding ``SdfSamples/<dataset>/<class>/*.npz``.
    geom_dimension : int
        Coordinates per sample of the dataset (2 or 3). A mismatch with the
        stored rows is not caught by the trainer until its first batch, where
        it dies with an IndexError - so it comes from the data, never a guess.
    """
    full = defaults()
    full.update(hp)
    hp = full
    n_layers = hp["n_layers"]
    return {
        "Description": hp["description"] or default_description(hp),
        "DataSource": str(data_source),
        "NetworkArch": ARCH,
        "TrainSplit": str(split_path),
        "TestSplit": str(split_path),
        "ReconstructionSplit": "",
        "NetworkSpecs": {
            "dims": [hp["width"]] * n_layers,
            "dropout": layer_list(hp["dropout_layers"], n_layers),
            "dropout_prob": hp["dropout_prob"],
            "norm_layers": layer_list(hp["norm_layers"], n_layers),
            "latent_in": list(hp["latent_in"]),
            "xyz_in_all": hp["xyz_in_all"],
            "use_tanh": hp["use_tanh"],
            "latent_dropout": hp["latent_dropout"],
            "weight_norm": hp["weight_norm"],
            "geom_dimension": int(geom_dimension),
        },
        "CodeLength": hp["latent_dim"],
        "NumEpochs": hp["num_epochs"],
        "LogFrequency": log_frequency(hp),
        "SnapshotFrequency": snapshot_frequency(hp),
        "AdditionalSnapshots": list(hp["additional_snapshots"]),
        "LearningRateSchedule": [_schedule(hp, "lr_dec"), _schedule(hp, "lr_code")],
        "SamplesPerScene": hp["samples_per_scene"],
        "ScenesPerBatch": hp["scenes_per_batch"],
        # Worker processes re-import the app on Windows; see FIXED.
        "DataLoaderThreads": 0,
        "ClampingDistance": hp["clamping_distance"],
        "LossFunction": hp["loss_function"],
        # Not read by the trainer - the lambda alone decides - but kept so the
        # file still says what it did.
        "CodeRegularization": hp["code_reg_lambda"] > 0,
        "CodeRegularizationLambda": hp["code_reg_lambda"],
        "CodeBound": hp["code_bound"],
        "CodeInitStdDev": hp["code_init_std"],
        "seed": hp["seed"],
    }


def _layers_from_specs(value, n_layers) -> object:
    indices = sorted(set(int(v) for v in value or []))
    if not indices:
        return "none"
    if set(range(n_layers + 1)) <= set(indices):
        return "all"
    return indices


def from_specs(specs: dict) -> tuple[dict, list[str]]:
    """Read a hyperparameter set back out of a ``specs.json`` dict.

    Used by "Load from" to start from an earlier run or a shipped decoder.
    Every key is read on its own, so one odd value costs that value, not the
    whole load; whatever could not be carried over is described in the
    returned notes. The description is not carried over - it belongs to the
    run it came from.

    Returns
    -------
    (dict, list of str)
        The hyperparameter set, defaults where the specs had nothing usable,
        and one note per value that was changed or dropped on the way in.
    """
    hp = defaults()
    notes: list[str] = []

    arch = specs.get("NetworkArch")
    if arch is not None and arch != ARCH:
        notes.append(
            f"Trained with '{arch}'; the app trains {ARCH}, so only the settings "
            "the two share were taken."
        )

    def take(key, value, convert):
        field = FIELD_BY_KEY[key]
        try:
            typed = convert(value)
        except (TypeError, ValueError):
            notes.append(
                f"{field.full_label}: {value!r} is not usable, kept the default."
            )
            return
        problem = check(field, typed)
        if problem:
            notes.append(
                f"{field.full_label}: {format_value(field, typed)} {problem}; kept the "
                "default."
            )
            return
        hp[key] = typed

    network = specs.get("NetworkSpecs") or {}
    dims = network.get("dims")
    if isinstance(dims, list) and dims:
        take("n_layers", len(dims), int)
        if len(set(dims)) > 1:
            notes.append(
                f"Layer widths {dims} are not uniform; using the widest, "
                f"{max(dims)}, for every layer."
            )
        take("width", max(dims), int)
    n_layers = hp["n_layers"]
    if "latent_in" in network:
        take("latent_in", network["latent_in"], lambda v: sorted(set(map(int, v))))
    for key, spec_key in (
        ("norm_layers", "norm_layers"),
        ("dropout_layers", "dropout"),
    ):
        if spec_key in network:
            take(key, network[spec_key], lambda v: _layers_from_specs(v, n_layers))
    for key in ("xyz_in_all", "weight_norm", "use_tanh", "latent_dropout"):
        if key in network:
            take(key, network[key], bool)
    if "dropout_prob" in network:
        take("dropout_prob", network["dropout_prob"], float)
    if network.get("geom_dimension", GEOM_DIMENSION) != GEOM_DIMENSION:
        notes.append(
            f"These specs were for {network['geom_dimension']}-D samples. The "
            "geometry dimension is not carried over: the dataset you train on "
            "sets it."
        )

    code_length = specs.get("CodeLength")
    if isinstance(code_length, list):
        notes.append(
            f"CodeLength {code_length} is a hierarchical code; kept d = "
            f"{hp['latent_dim']}."
        )
    elif code_length is not None:
        take("latent_dim", code_length, int)

    simple = (
        ("NumEpochs", "num_epochs", int),
        ("SamplesPerScene", "samples_per_scene", int),
        ("ScenesPerBatch", "scenes_per_batch", int),
        ("ClampingDistance", "clamping_distance", float),
        ("CodeRegularizationLambda", "code_reg_lambda", float),
        ("CodeInitStdDev", "code_init_std", float),
        ("seed", "seed", int),
        ("LossFunction", "loss_function", str),
    )
    for spec_key, key, convert in simple:
        if spec_key in specs:
            take(key, specs[spec_key], convert)
    if "CodeBound" in specs:
        take(
            "code_bound", specs["CodeBound"], lambda v: None if v is None else float(v)
        )
    if "AdditionalSnapshots" in specs:
        take(
            "additional_snapshots",
            specs["AdditionalSnapshots"],
            lambda v: sorted(set(map(int, v))),
        )

    # Frequencies equal to what "auto" would pick are loaded as auto, so they
    # keep following the epochs when those are edited afterwards.
    epochs = hp["num_epochs"]
    if "LogFrequency" in specs:
        take("log_frequency", specs["LogFrequency"], int)
        if hp["log_frequency"] == auto_log_frequency(epochs):
            hp["log_frequency"] = None
    if "SnapshotFrequency" in specs:
        take("snapshot_frequency", specs["SnapshotFrequency"], int)
        if hp["snapshot_frequency"] == auto_snapshot_frequency(epochs):
            hp["snapshot_frequency"] = None

    schedules = specs.get("LearningRateSchedule")
    if isinstance(schedules, list):
        if len(schedules) < 2:
            notes.append(
                f"{len(schedules)} learning-rate schedule(s) instead of 2 "
                "(decoder, codes); the missing one keeps its default."
            )
        for prefix, entry in zip(("lr_dec", "lr_code"), schedules):
            _schedule_from_specs(hp, prefix, entry, take, notes)

    return hp, notes


def _schedule_from_specs(hp, prefix, entry, take, notes):
    if not isinstance(entry, dict) or entry.get("Type") not in SCHEDULE_TYPES:
        kind = entry.get("Type") if isinstance(entry, dict) else entry
        notes.append(f"Learning-rate schedule {kind!r} is not supported; kept Step.")
        return
    kind = entry["Type"]
    hp[f"{prefix}_type"] = kind
    if kind == "Constant":
        if "Value" in entry:
            take(f"{prefix}_initial", entry["Value"], float)
        return
    if "Initial" in entry:
        take(f"{prefix}_initial", entry["Initial"], float)
    if kind == "Step":
        if "Interval" in entry:
            take(f"{prefix}_interval", entry["Interval"], int)
        if "Factor" in entry:
            take(f"{prefix}_factor", entry["Factor"], float)
    else:
        if "Final" in entry:
            take(f"{prefix}_final", entry["Final"], float)
        if "Length" in entry:
            take(f"{prefix}_length", entry["Length"], int)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #


def in_use(field: Field, hp: dict) -> bool:
    """Whether a field takes part in the current configuration at all.

    A Step interval is irrelevant under a Warmup schedule: a changed value
    there is not worth reporting as a change, and the window hides the row.
    """
    if not field.schedules:
        return True
    prefix = field.key.rsplit("_", 1)[0]
    return hp.get(f"{prefix}_type") in field.schedules


def changed(hp: dict, include_card: bool = False) -> list[Field]:
    """Fields whose value differs from the default, in schema order."""
    base = defaults()
    return [
        f
        for f in FIELDS
        if (include_card or not f.card)
        and in_use(f, hp)
        and hp.get(f.key) != base[f.key]
    ]


def summary(hp: dict, limit: int = 4) -> str:
    """One line for the Train tab card: what differs from the defaults."""
    diff = changed(hp)
    if not diff:
        return "Every other hyperparameter is at its default."
    parts = [f"{f.name} {_short_value(f, hp[f.key])}" for f in diff]
    text = " · ".join(parts[:limit])
    if len(parts) > limit:
        text += f" · +{len(parts) - limit} more"
    return f"Changed: {text}"


def _short_value(field: Field, value) -> str:
    text = format_value(field, value) or field.blank
    if field.kind == "text" and len(text) > 24:
        text = text[:23] + "…"
    return text


def describe(hp: dict) -> list[str]:
    """One line per non-default value, including the card ones, for the log."""
    return [
        f"  {f.full_label}: {format_value(f, hp[f.key]) or f.blank}"
        for f in changed(hp, include_card=True)
    ]
