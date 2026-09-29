"""Tests for the training hyperparameters: schema, specs.json, trainer, window.

Four layers, cheapest first:

* the schema on its own - text round-trips, validation, specs.json in and out;
* the specs it writes against the real ``DeepSDFDecoder`` and learning-rate
  schedules, so "the validator accepts it" means "the library can build it";
* one tiny training run on a synthetic two-sphere dataset, which proves the
  values in the window are the values the trainer actually uses - the learning
  rate it logged, the checkpoints it kept, the layer shapes it saved;
* the Tk window itself, driven off-screen like ``test_app_explore``.

The defaults test pins the specs.json the app wrote before the window existed:
opening a hyperparameter screen must not quietly change what "Train" does.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from structsept.app import hyperparams

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _hp(**overrides):
    hp = hyperparams.defaults()
    hp.update(overrides)
    return hp


def _error_keys(hp, n_shapes=None):
    return {i.key for i in hyperparams.errors(hyperparams.validate(hp, n_shapes))}


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def test_defaults_reproduce_the_old_specs():
    """Byte-for-byte what ``training.write_specs`` wrote before this module,
    plus three keys the trainer already defaulted to the same values."""
    specs = hyperparams.to_specs(hyperparams.defaults(), "SPLIT", "DATA")
    layers = list(range(8))
    assert specs == {
        "Description": "structsept run: d=1, 6x128, 200 epochs",
        "DataSource": "DATA",
        "NetworkArch": "deep_sdf_decoder",
        "TrainSplit": "SPLIT",
        "TestSplit": "SPLIT",
        "ReconstructionSplit": "",
        "NetworkSpecs": {
            "dims": [128] * 6,
            "dropout": layers,
            "dropout_prob": 0.2,
            "norm_layers": layers,
            "latent_in": [2],
            "xyz_in_all": False,
            "use_tanh": False,
            "latent_dropout": False,
            "weight_norm": True,
            "geom_dimension": 3,
        },
        "CodeLength": 1,
        "NumEpochs": 200,
        "LogFrequency": 10,
        "SnapshotFrequency": 50,
        "AdditionalSnapshots": [1],
        "LearningRateSchedule": [
            {"Type": "Step", "Initial": 0.0005, "Interval": 500, "Factor": 0.5},
            {"Type": "Step", "Initial": 0.001, "Interval": 500, "Factor": 0.5},
        ],
        "SamplesPerScene": 8000,
        "ScenesPerBatch": 10,
        "DataLoaderThreads": 0,
        "ClampingDistance": 0.1,
        "CodeRegularization": True,
        "CodeRegularizationLambda": 1e-4,
        "CodeBound": 1.0,
        # new, and equal to the trainer's own fallbacks
        "LossFunction": "clampedL1",
        "CodeInitStdDev": 1.0,
        "seed": 42,
    }


def test_defaults_are_valid():
    assert hyperparams.errors(hyperparams.validate(hyperparams.defaults())) == []


@pytest.mark.parametrize("field", hyperparams.FIELDS, ids=lambda f: f.key)
def test_every_default_round_trips_through_its_entry(field):
    text = hyperparams.format_value(field, field.default)
    if field.kind == "bool":
        return
    assert hyperparams.parse(field, text) == field.default


@pytest.mark.parametrize(
    "key, text, expected",
    [
        ("lr_dec_initial", "5e-4", 0.0005),
        ("lr_dec_initial", "0,001", 0.001),  # decimal comma
        ("samples_per_scene", "1e3", 1000),
        ("samples_per_scene", "16_384", 16384),
        ("latent_in", "4, 2 2", [2, 4]),
        ("latent_in", "", []),
        ("latent_in", "none", []),
        ("norm_layers", "ALL", "all"),
        ("norm_layers", "", "none"),
        ("dropout_layers", "3;1", [1, 3]),
        ("code_bound", "", None),
        ("log_frequency", "auto", None),
        ("log_frequency", "5", 5),
    ],
)
def test_parse_accepts(key, text, expected):
    assert hyperparams.parse(hyperparams.FIELD_BY_KEY[key], text) == expected


@pytest.mark.parametrize(
    "key, text",
    [
        ("lr_dec_initial", "0"),  # open lower bound
        ("lr_dec_initial", "-1e-3"),
        ("lr_dec_initial", "nan"),
        ("lr_dec_initial", "inf"),
        ("lr_dec_initial", "fast"),
        ("lr_dec_initial", ""),
        ("samples_per_scene", "12.5"),
        ("n_layers", "13"),
        ("dropout_prob", "0.99"),
        ("latent_in", "2, x"),
        ("loss_function", "hinge"),
        ("log_frequency", "0"),
    ],
)
def test_parse_rejects(key, text):
    with pytest.raises(ValueError):
        hyperparams.parse(hyperparams.FIELD_BY_KEY[key], text)


MODIFIED = dict(
    latent_dim=3,
    n_layers=4,
    width=96,
    latent_in=[1, 3],
    xyz_in_all=True,
    weight_norm=False,
    norm_layers=[0, 2],
    dropout_layers="none",
    dropout_prob=0.0,
    use_tanh=True,
    code_init_std=0.5,
    code_bound=None,
    code_reg_lambda=0.0,
    latent_dropout=True,
    loss_function="huber",
    clamping_distance=0.05,
    lr_dec_type="Warmup",
    lr_dec_initial=1e-4,
    lr_dec_final=2e-3,
    lr_dec_length=40,
    lr_code_type="Constant",
    lr_code_initial=3e-3,
    samples_per_scene=4096,
    scenes_per_batch=4,
    num_epochs=120,
    seed=7,
    log_frequency=3,
    snapshot_frequency=40,
    additional_snapshots=[1, 5],
)


def test_specs_round_trip():
    hp = _hp(**MODIFIED)
    back, notes = hyperparams.from_specs(hyperparams.to_specs(hp, "S", "D"))
    assert notes == []
    # schedule values the chosen types do not use are not in specs.json
    for key in ("lr_dec_interval", "lr_dec_factor", "lr_code_interval"):
        back[key] = hp[key]
    for key in ("lr_code_factor", "lr_code_final", "lr_code_length"):
        back[key] = hp[key]
    assert back == hp


def test_automatic_frequencies_stay_automatic():
    """A frequency equal to what auto would pick loads as auto, so it keeps
    following the epochs when those are changed afterwards."""
    back, _ = hyperparams.from_specs(hyperparams.to_specs(_hp(num_epochs=30), "S", "D"))
    assert back["log_frequency"] is None and back["snapshot_frequency"] is None


def test_shipped_decoders_load():
    from DeepSDFStruct.pretrained_models import PRETRAINED_MODELS_DIR, PretrainedModels

    loaded = {}
    for member in PretrainedModels:
        path = f"{PRETRAINED_MODELS_DIR}/{member.value}/specs.json"
        with open(path, encoding="utf-8") as f:
            hp, notes = hyperparams.from_specs(json.load(f))
        assert hyperparams.errors(hyperparams.validate(hp)) == [], member.name
        loaded[member.name] = (hp, notes)

    hp, _ = loaded["ChiAndCross"]
    assert (hp["latent_dim"], hp["samples_per_scene"], hp["num_epochs"]) == (
        2,
        16384,
        1000,
    )
    assert hp["norm_layers"] == "all" and hp["latent_in"] == [2]
    hp, _ = loaded["Primitives2D"]
    assert hp["norm_layers"] == [0, 1, 2, 3] and hp["dropout_layers"] == "none"
    assert hp["use_tanh"] and hp["clamping_distance"] == 1.0
    _, notes = loaded["AnalyticRoundCross"]
    assert any("analytic_round_cross" in n for n in notes)


def test_from_specs_keeps_going_past_a_bad_value():
    specs = hyperparams.to_specs(_hp(num_epochs=40), "S", "D")
    specs["SamplesPerScene"] = "many"
    specs["ScenesPerBatch"] = -3
    specs["LearningRateSchedule"] = [{"Type": "Cosine"}]
    hp, notes = hyperparams.from_specs(specs)
    assert hp["num_epochs"] == 40
    assert hp["samples_per_scene"] == 8000 and hp["scenes_per_batch"] == 10
    assert len(notes) == 4  # two bad values, one unknown schedule, one missing


@pytest.mark.parametrize(
    "overrides, key",
    [
        (dict(latent_in=[0]), "latent_in"),
        (dict(latent_in=[7]), "latent_in"),  # n_layers + 1: not ignored
        (dict(latent_dim=13, width=16), "width"),
        (dict(latent_dim=20, width=16), "width"),
        (dict(samples_per_scene=4001), "samples_per_scene"),
        (dict(num_epochs=8, log_frequency=10), "log_frequency"),
        (dict(num_epochs=3), None),  # auto picks 1: fine
        (dict(latent_dim=20, width=16, latent_in=[]), None),  # no skip: fine
    ],
)
def test_validation_errors(overrides, key):
    keys = _error_keys(_hp(**overrides))
    assert keys == ({key} if key else set())


def test_validation_warnings_and_notes():
    def levels(hp, n_shapes=None):
        return {(i.level, i.key) for i in hyperparams.validate(hp, n_shapes)}

    assert ("warning", "scenes_per_batch") in levels(_hp(), n_shapes=4)
    assert ("warning", "log_frequency") in levels(_hp(num_epochs=25, log_frequency=10))
    tanh = dict(use_tanh=True, clamping_distance=1.0)
    assert ("warning", "use_tanh") in levels(_hp(loss_function="L1", **tanh))
    # clampedL1 compares only +-0.1, which tanh reaches: the shipped
    # Primitives2D set (clampedL1, delta 1, tanh) must not be flagged
    assert ("warning", "use_tanh") not in levels(_hp(**tanh))
    assert ("warning", "latent_in") in levels(_hp(latent_in=[9]))
    assert ("note", "lr_dec_interval") in levels(_hp())
    assert ("note", "lr_dec_interval") not in levels(_hp(num_epochs=1000))
    assert ("note", "clamping_distance") in levels(_hp(clamping_distance=0.5))
    assert ("note", "clamping_distance") not in levels(
        _hp(clamping_distance=0.5, loss_function="L1")
    )


def test_schedule_notes_match_the_trainer_at_the_boundary():
    """Epochs run 1..N and Step uses epoch // interval, so an interval equal
    to the run length does step once, on the last epoch; a warm-up as long as
    the run does finish."""

    def keys(hp):
        return {i.key for i in hyperparams.validate(hp)}

    assert "lr_dec_interval" not in keys(_hp(num_epochs=500))
    assert "lr_dec_interval" in keys(_hp(num_epochs=499))
    warm = dict(lr_dec_type="Warmup", lr_dec_length=100)
    assert "lr_dec_length" not in keys(_hp(num_epochs=100, **warm))
    assert "lr_dec_length" in keys(_hp(num_epochs=99, **warm))


def test_loss_notes_tell_the_two_clamped_losses_apart():
    def note(loss):
        hp = _hp(loss_function=loss, clamping_distance=0.5)
        found = [i for i in hyperparams.validate(hp) if i.key == "clamping_distance"]
        return found[0].message if found else None

    assert "actually learned is ±0.1" in note("clampedL1")
    assert "100 times" in note("leakyClampedL1")
    assert note("L1") is None


def test_partial_last_batch_is_noted():
    """The trainer's DataLoader drops the partial last batch whenever the
    batch fits in the dataset; the shapes in it sit that epoch out."""

    def found(n):
        return [
            i.message
            for i in hyperparams.validate(_hp(), n_shapes=n)
            if i.key == "scenes_per_batch"
        ]

    assert "5 of the 15 shapes" in found(15)[0]
    assert found(20) == []
    assert "single batch" in found(4)[0]


def test_summary_names_only_what_changed():
    assert hyperparams.summary(hyperparams.defaults()).startswith("Every other")
    text = hyperparams.summary(_hp(lr_dec_initial=1e-3, num_epochs=30))
    assert "decoder lr 0.001" in text
    assert "epochs" not in text  # a card value, visible on the card already
    # a Step interval does not matter under Warmup, so it is not a change
    assert "step" not in hyperparams.summary(
        _hp(lr_dec_type="Warmup", lr_dec_interval=7)
    )


def test_write_specs_refuses_an_error(tmp_path):
    from structsept.app import training

    with pytest.raises(ValueError, match="even"):
        training.write_specs(tmp_path / "run", "S", "D", samples_per_scene=11)
    assert not (tmp_path / "run" / "specs.json").exists()
    with pytest.raises(TypeError, match="learning_rate"):
        training.write_specs(tmp_path / "run", "S", "D", learning_rate=1e-3)


def test_preview_json_is_the_same_content():
    from structsept.app.hparam_window import compact_json

    specs = hyperparams.to_specs(_hp(**MODIFIED), "S", "D")
    assert json.loads(compact_json(specs)) == specs


# --------------------------------------------------------------------------- #
# the supervisor's sheet
# --------------------------------------------------------------------------- #

SHEET_DIR = Path(__file__).resolve().parents[1] / "docs" / "hyperparameters"
FILLED_SHEET = SHEET_DIR / "NN_Training_Hyperparameters_plate2d_r_only_d1_8x256_4h.xlsx"

# The template's rows, in its order, with the values of the 8x256 run.
SHEET_ROWS = [
    ("Initial NN Training — Hyperparameter Template", None),
    (None, None),
    ("Hyperparameter", "Value"),
    ("Training data", None),
    ("Number of training geometries", 40),
    ("Samples per geometry", 50000),
    ("Points per training step", 16384),
    ("Geometries per batch", 4),
    ("Sampling strategy", "Random window per step"),
    ("Latent space", None),
    ("Latent dimension", 1),
    ("Latent initialization mean", 0),
    ("Latent initialization variance", 0.01),
    ("Initial latent regularization", 0.0001),
    ("Network architecture", None),
    ("Hidden layers", 8),
    ("Neurons per hidden layer", 256),
    ("Activation function", "ReLU"),
    ("Dropout", 0.2),
    ("Training", None),
    ("Epochs", 600),
    ("Optimizer", "Adam"),
    ("Learning rate — network weights", 0.0005),
    ("Learning rate — latent vectors", 0.001),
    ("Learning-rate decay factor", 0.5),
    ("Learning-rate decay interval", 150),
    ("Loss", None),
    ("Loss function", "Clamped L1"),
    ("Clamp value", 0.1),
]
# what those rows mean in hyperparams keys (experiments/train_plate_r_only_4h.py)
SHEET_EXPECTED = dict(
    n_layers=8,
    width=256,
    dropout_prob=0.2,
    latent_dim=1,
    code_init_std=0.1,
    code_reg_lambda=1e-4,
    loss_function="clampedL1",
    clamping_distance=0.1,
    samples_per_scene=4096,
    scenes_per_batch=4,
    num_epochs=600,
    lr_dec_type="Step",
    lr_dec_initial=5e-4,
    lr_dec_interval=150,
    lr_dec_factor=0.5,
    lr_code_type="Step",
    lr_code_initial=1e-3,
    lr_code_interval=150,
    lr_code_factor=0.5,
)


def _write_xlsx(path, rows, shared=False):
    """A minimal workbook: one sheet, inline or shared strings, numbers,
    booleans - written by hand so the reader is tested against the file
    format, not against a library that reads what it wrote."""
    import zipfile
    from xml.sax.saxutils import escape

    strings = []

    def cell(ref, value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value!r}</v></c>'
        if shared:
            strings.append(value)
            return f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'

    body = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(cell(f"{chr(65 + c)}{r}", v) for c, v in enumerate(row))
        body.append(f'<row r="{r}">{cells}</row>')
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    sheet = (
        f'<worksheet xmlns="{main}"><sheetData>{"".join(body)}</sheetData></worksheet>'
    )
    workbook = (
        f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>'
        '<sheet name="Hyperparameters" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
        'relationships"><Relationship Id="rId1" Type="x" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        if shared:
            items = "".join(
                f"<si><r><t>{escape(s[:1])}</t></r><r><t>{escape(s[1:])}</t></r></si>"
                for s in strings
            )
            z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{main}">{items}</sst>')
    return path


@pytest.mark.parametrize("shared", [False, True], ids=["inline", "shared"])
def test_xlsx_reader(tmp_path, shared):
    from structsept.app import xlsx

    rows = [("Hyperparameter", "Value", "Notes"), ("Epochs", 600, None)]
    rows += [("Dropout", 0.2, True), ("Loss function", "Clamped L1", None)]
    path = _write_xlsx(tmp_path / "t.xlsx", rows, shared=shared)
    got = xlsx.read_rows(path)
    assert got[0] == ["Hyperparameter", "Value", "Notes"]
    assert got[1] == ["Epochs", 600, None] and isinstance(got[1][1], int)
    assert got[2] == ["Dropout", 0.2, True]
    assert got[3][1] == "Clamped L1"  # rich-text runs joined
    with pytest.raises(ValueError, match="worksheet named"):
        xlsx.read_rows(path, sheet="Other")
    (tmp_path / "not.xlsx").write_text("hello")
    with pytest.raises(ValueError, match="not an Excel workbook"):
        xlsx.read_rows(tmp_path / "not.xlsx")


def _assert_sheet_values(hp):
    for key, want in SHEET_EXPECTED.items():
        got = hp[key]
        assert got == (pytest.approx(want) if isinstance(want, float) else want), key


def test_from_sheet_reproduces_the_4h_run(tmp_path):
    hp, notes = hyperparams.from_sheet(SHEET_ROWS, n_shapes=40)
    _assert_sheet_values(hp)
    assert hyperparams.errors(hyperparams.validate(hp, 40, 2)) == []
    # every key the sheet does not have keeps its default
    for field in hyperparams.FIELDS:
        if field.key not in SHEET_EXPECTED:
            assert hp[field.key] == field.default, field.key
    text = " ".join(notes)
    assert "4096 = 16384 points per step / 4" in text
    assert "sqrt(0.01 · d=1) = 0.1" in text
    assert "not a row" not in text.lower()
    # the same rows through a real file
    path = _write_xlsx(tmp_path / "run.xlsx", SHEET_ROWS)
    from structsept.app import xlsx

    again, _ = hyperparams.from_sheet(xlsx.read_rows(path), n_shapes=40)
    assert again == hp


@pytest.mark.skipif(not FILLED_SHEET.is_file(), reason="sheet not in docs/")
def test_the_shipped_sheet_matches_the_experiment():
    """The sheet filled in for runs/plate2d_r_only_d1_8x256_4h says what the
    script that trained it says."""
    from structsept.app import xlsx

    hp, _ = hyperparams.from_sheet(xlsx.read_rows(FILLED_SHEET), n_shapes=40)
    _assert_sheet_values(hp)


FILLED_SHEET_D2 = FILLED_SHEET.with_name(
    "NN_Training_Hyperparameters_plate2d_r_only_d2_8x256_4h.xlsx"
)


@pytest.mark.skipif(not FILLED_SHEET_D2.is_file(), reason="sheet not in docs/")
def test_the_d2_sheet_is_the_same_recipe_with_a_two_component_code():
    """The d=2 copy of the sheet (runs/plate2d_r_only_d2_8x256_4h) changes
    only 'Latent dimension'; the variance stays per component, so the
    trainer's sigma becomes sqrt(0.01 * 2)."""
    from structsept.app import xlsx

    hp, notes = hyperparams.from_sheet(xlsx.read_rows(FILLED_SHEET_D2), n_shapes=40)
    expected = dict(SHEET_EXPECTED, latent_dim=2, code_init_std=0.02**0.5)
    for key, want in expected.items():
        got = hp[key]
        assert got == (pytest.approx(want) if isinstance(want, float) else want), key
    assert "sqrt(0.01 · d=2)" in " ".join(notes)
    assert hyperparams.errors(hyperparams.validate(hp, 40, 2)) == []


FILLED_SHEET_XYR_D3 = FILLED_SHEET.with_name(
    "NN_Training_Hyperparameters_plate2d_xyr_d3_8x256_4h.xlsx"
)


@pytest.mark.skipif(not FILLED_SHEET_XYR_D3.is_file(), reason="sheet not in docs/")
def test_the_xyr_d3_sheet_is_the_same_recipe_sized_for_134_shapes():
    """The sheet for runs/plate2d_xyr_d3_8x256_4h (experiments/
    train_plate_xyr_4h.py) changes four cells of the d=2 one: 134 geometries,
    a three-component code, and 300 epochs with the decay every 75 - the
    same recipe, sized for 3.35 x the shapes in the same 4 hours."""
    from structsept.app import xlsx

    hp, notes = hyperparams.from_sheet(
        xlsx.read_rows(FILLED_SHEET_XYR_D3), n_shapes=134
    )
    expected = dict(
        SHEET_EXPECTED,
        latent_dim=3,
        code_init_std=0.03**0.5,
        num_epochs=300,
        lr_dec_interval=75,
        lr_code_interval=75,
    )
    for key, want in expected.items():
        got = hp[key]
        assert got == (pytest.approx(want) if isinstance(want, float) else want), key
    assert "sqrt(0.01 · d=3)" in " ".join(notes)
    issues = hyperparams.validate(hp, 134, 2)
    assert hyperparams.errors(issues) == []
    # 134 is not a multiple of 4: the trainer drops two shapes per epoch, and
    # the set says so without refusing.
    assert any(
        i.key == "scenes_per_batch" and "2 of the 134" in i.message for i in issues
    )


def test_from_sheet_reports_what_it_cannot_take():
    rows = [
        ("Hyperparameter", "Value"),
        ("Number of training geometries", 40),
        ("Points per training step", 1000),
        ("Geometries per batch", 3),
        ("Latent dimension", 2),
        ("Latent initialization variance", 0.02),
        ("Latent initialization mean", 0.5),
        ("Activation function", "GELU"),
        ("Optimizer", "SGD"),
        ("Epochs", "many"),
        ("Hidden layers", 40),
        ("Loss function", "hinge"),
        ("Learning-rate decay factor", 0.25),
        ("Weight decay", 1e-5),
        ("Dropout", None),
    ]
    hp, notes = hyperparams.from_sheet(rows, n_shapes=25)
    text = "\n".join(notes)
    assert hp["samples_per_scene"] == 332 and "4 points dropped" in text
    assert hp["code_init_std"] == pytest.approx((0.02 * 2) ** 0.5)
    assert hp["num_epochs"] == 200 and "'many'" in text
    assert hp["n_layers"] == 6 and "at most 12" in text
    assert hp["loss_function"] == "clampedL1" and "'hinge'" in text
    assert hp["lr_dec_factor"] == hp["lr_code_factor"] == 0.25
    assert "for 40 training geometries; the selected dataset has 25" in text
    for bad in ("0.5 cannot be set", "'GELU'", "'SGD'", "'Weight decay' (row 14)"):
        assert bad in text, bad
    assert "Blank in the sheet, kept the defaults: 'Dropout'" in text
    # no header row, blank template: every value stays at its default
    blank = [(name, None) for name, _ in SHEET_ROWS[3:]]
    hp, notes = hyperparams.from_sheet(blank)
    assert hp == hyperparams.defaults()
    assert notes == [
        "21 of the sheet's rows are blank; those values keep the defaults."
    ]


def test_origin_summary_says_what_was_edited():
    loaded = _hp(num_epochs=600, width=256)
    origin = {"label": "run: big", "hp": loaded, "when": "14:02"}
    assert hyperparams.origin_summary(None, loaded) == ""
    assert hyperparams.origin_summary(origin, dict(loaded)) == (
        "From run: big (loaded 14:02)."
    )
    edited = dict(loaded, num_epochs=50, lr_dec_interval=7, lr_dec_type="Warmup")
    text = hyperparams.origin_summary(origin, edited)
    assert text.startswith("From run: big (loaded 14:02); edited since: ")
    assert "epochs" in text and "decoder schedule" in text
    assert "lr step" not in text  # a hidden Step row is not an edit


def test_runs_are_listed_newest_first(tmp_path):
    from structsept.app import training

    for name, stamp in (("b", "2026-09-23T09:00:00"), ("a", "2026-09-28T02:46:04")):
        run = tmp_path / name
        run.mkdir()
        (run / "specs.json").write_text(json.dumps({"CodeLength": 1}))
        (run / "metadata.json").write_text(json.dumps({"timestamp": stamp}))
    fresh = tmp_path / "c"  # started by hand, no metadata yet
    fresh.mkdir()
    (fresh / "specs.json").write_text(json.dumps({"CodeLength": 2}))
    (tmp_path / "not_a_run").mkdir()

    rows = training.list_runs(tmp_path)
    assert [r["name"] for r in rows] == ["c", "a", "b"]
    assert rows[0]["date_is_estimate"] and rows[0]["date"] > rows[1]["date"]
    assert not rows[1]["date_is_estimate"]
    labels = [label for label, _ in training.spec_sources(tmp_path)]
    assert labels[:3] == ["run: c", "run: a", "run: b"]
    assert labels[3].startswith("shipped: ")


# --------------------------------------------------------------------------- #
# against the library
# --------------------------------------------------------------------------- #


ARCHITECTURES = [
    {},
    dict(xyz_in_all=True),
    dict(weight_norm=False, norm_layers=[0, 2]),
    dict(weight_norm=False, norm_layers="none"),
    dict(latent_in=[1, 3], n_layers=4),
    dict(latent_in=[4], n_layers=4),
    dict(latent_in=[], dropout_layers="none"),
    dict(use_tanh=True, latent_dropout=True),
    dict(latent_dim=8, width=16, xyz_in_all=True),
    MODIFIED,
]


@pytest.mark.parametrize("overrides", ARCHITECTURES)
def test_specs_build_a_working_decoder(overrides):
    import torch

    import DeepSDFStruct.deep_sdf.workspace as ws

    hp = _hp(**overrides)
    assert _error_keys(hp) == set()
    specs = hyperparams.to_specs(hp, "S", "D")
    decoder = ws.init_decoder(specs, "cpu", False)
    decoder.train()
    out = decoder(torch.randn(7, hp["latent_dim"] + 3))
    assert out.shape == (7, 1)


def test_validation_catches_every_decoder_the_library_rejects():
    """No false negatives: whatever DeepSDFDecoder cannot build or run, the
    window refuses before a run starts."""
    import itertools

    import torch

    import DeepSDFStruct.deep_sdf.workspace as ws

    for d, width, n_layers, skips, xyz in itertools.product(
        (1, 5, 13, 14), (16, 17), (2, 3), ([], [0], [1], [2], [3], [1, 2]), (0, 1)
    ):
        hp = _hp(
            latent_dim=d,
            width=width,
            n_layers=n_layers,
            latent_in=skips,
            xyz_in_all=bool(xyz),
        )
        try:
            decoder = ws.init_decoder(hyperparams.to_specs(hp, "S", "D"), "cpu", False)
            decoder(torch.randn(3, d + 3))
            broken = False
        except Exception:
            broken = True
        if broken:
            assert _error_keys(hp), hp


@pytest.mark.parametrize(
    "prefix_values, epochs, expected",
    [
        (
            dict(type="Step", initial=1e-3, interval=2, factor=0.5),
            5,
            [1e-3, 5e-4, 5e-4, 2.5e-4, 2.5e-4],
        ),
        (
            dict(type="Warmup", initial=1e-4, final=5e-4, length=4),
            5,
            [2e-4, 3e-4, 4e-4, 5e-4, 5e-4],
        ),
        (dict(type="Constant", initial=3e-3), 3, [3e-3] * 3),
    ],
)
def test_schedules_mean_what_the_window_says(prefix_values, epochs, expected):
    from DeepSDFStruct.deep_sdf.training import get_learning_rate_schedules

    hp = _hp(**{f"lr_dec_{k}": v for k, v in prefix_values.items()})
    schedule = get_learning_rate_schedules(hyperparams.to_specs(hp, "S", "D"))[0]
    got = [schedule.get_learning_rate(e) for e in range(1, epochs + 1)]
    assert got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# one real training run
# --------------------------------------------------------------------------- #


def _sphere_dataset(root, radii=(0.4, 0.6), n=600, seed=0):
    """Two analytic spheres as an SdfSamples dataset, plus its split."""
    rng = np.random.default_rng(seed)
    names = []
    folder = root / "SdfSamples" / "tiny" / "spheres"
    folder.mkdir(parents=True)
    for index, radius in enumerate(radii):
        pts = rng.uniform(-1, 1, size=(4 * n, 3))
        phi = np.linalg.norm(pts, axis=1) - radius
        rows = np.column_stack([pts, phi]).astype(np.float32)
        name = f"s{index}"
        np.savez(folder / f"{name}.npz", pos=rows[phi > 0][:n], neg=rows[phi <= 0][:n])
        names.append(name)
    split = root / "splits" / "tiny.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"tiny": {"spheres": names}}), encoding="utf-8")
    return split


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    return root, _sphere_dataset(root)


def test_hyperparameters_reach_the_trainer(tiny_data, tmp_path):
    import matplotlib

    matplotlib.use("Agg")  # the trainer plots its loss through pyplot
    import torch

    from structsept.app import training

    data_root, split = tiny_data
    hp = _hp(
        latent_dim=2,
        n_layers=2,
        width=24,
        latent_in=[1],
        num_epochs=4,
        samples_per_scene=64,
        scenes_per_batch=2,
        loss_function="huber",
        lr_dec_type="Warmup",
        lr_dec_initial=1e-3,
        lr_dec_final=3e-3,
        lr_dec_length=2,
        lr_code_type="Constant",
        lr_code_initial=5e-3,
        log_frequency=2,
        snapshot_frequency=3,
        additional_snapshots=[1],
        seed=3,
    )
    run_dir = tmp_path / "run"
    training.write_specs(run_dir, split, data_root, hp)
    lines = []
    info = training.train(run_dir, data_root, log=lines.append)
    assert info["epochs"] == 4

    logs = torch.load(run_dir / "Logs.pth", weights_only=False)
    assert logs["epoch"] == 4
    assert np.allclose(
        logs["learning_rate"],
        [[2e-3, 5e-3], [3e-3, 5e-3], [3e-3, 5e-3], [3e-3, 5e-3]],
    )
    saved = sorted(p.name for p in (run_dir / "ModelParameters").iterdir())
    assert saved == ["1.pth", "3.pth", "latest.pth"]
    state = torch.load(run_dir / "ModelParameters" / "latest.pth", weights_only=False)[
        "model_state_dict"
    ]
    assert state["lin0.parametrizations.weight.original1"].shape == (24 - 5, 5)
    codes = torch.load(run_dir / "LatentCodes" / "latest.pth", weights_only=False)
    assert codes["latent_codes"]["weight"].shape == (2, 2)
    assert any("Setting random seed to 3" in line for line in lines)


def test_same_seed_gives_the_same_network(tiny_data, tmp_path):
    """train_deep_sdf builds the decoder before it seeds, so without
    training.train seeding first the initial weights depend on whatever used
    the torch RNG earlier in the process - here, a stray torch.rand."""
    import matplotlib

    matplotlib.use("Agg")
    import torch

    from structsept.app import training

    data_root, split = tiny_data
    hp = _hp(n_layers=2, width=24, num_epochs=2, samples_per_scene=64, seed=11)
    weights = []
    for name in ("a", "b"):
        torch.rand(17)  # anything else in the app that drew random numbers
        run_dir = tmp_path / name
        training.write_specs(run_dir, split, data_root, hp)
        training.train(run_dir, data_root, log=lambda line: None)
        weights.append(
            torch.load(run_dir / "ModelParameters" / "latest.pth", weights_only=False)[
                "model_state_dict"
            ]
        )
    assert weights[0].keys() == weights[1].keys()
    for key in weights[0]:
        assert torch.equal(weights[0][key], weights[1][key]), key


def test_odd_sample_count_does_crash_the_trainer(tiny_data, tmp_path):
    """The reason validate() calls an odd SamplesPerScene an error rather
    than a warning. Written around the validator, straight to specs.json."""
    import matplotlib

    matplotlib.use("Agg")

    import DeepSDFStruct.deep_sdf.workspace as ws

    from structsept.app import training

    data_root, split = tiny_data
    hp = _hp(n_layers=2, width=24, num_epochs=1, samples_per_scene=63)
    run_dir = tmp_path / "odd"
    run_dir.mkdir()
    specs = hyperparams.to_specs(hp, split, data_root)
    (run_dir / ws.specifications_filename).write_text(json.dumps(specs))
    with pytest.raises(RuntimeError):
        training.train(run_dir, data_root, log=lambda line: None)


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #

tk = pytest.importorskip("tkinter")


def _hide(window):
    for step in (
        lambda: window.attributes("-alpha", 0.0),
        lambda: window.overrideredirect(True),
        lambda: window.geometry("+{}+{}".format(-6000, -6000)),
    ):
        try:
            step()
        except Exception:
            pass
    window.update_idletasks()


def _pump(root, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.02)


@pytest.fixture(scope="module")
def app():
    from structsept.app import main

    try:
        root = main.build_app()
    except tk.TclError as exc:  # pragma: no cover - headless or broken Tcl
        pytest.skip(f"Tk unavailable: {exc}")
    _hide(root)
    root.update()
    root.app_state["notebook"].select(root.app_state["tab_frames"]["train"])
    yield root
    main._shutdown(root.app_state, root)


@pytest.fixture
def window(app):
    from structsept.app import tab_train

    st = app.app_state
    win = tab_train.open_hparams(st)
    _hide(win.top)
    _pump(app, 0.3)
    yield win
    win.close()
    _pump(app, 0.2)
    st["tr_hparams"] = hyperparams.defaults()
    for key, var_name in tab_train.CARD_VARS.items():
        st[var_name].set(hyperparams.FIELD_BY_KEY[key].default)


def test_window_opens_on_the_card_values(app):
    from structsept.app import tab_train

    st = app.app_state
    st["tr_latent_dim"].set(2)
    st["tr_epochs"].set(30)
    win = tab_train.open_hparams(st)
    _hide(win.top)
    try:
        assert win.vars["latent_dim"].get() == "2"
        assert win.vars["num_epochs"].get() == "30"
        assert tab_train.open_hparams(st) is win  # one window, raised again
    finally:
        win.close()
        st["tr_latent_dim"].set(1)
        st["tr_epochs"].set(200)
    _pump(app, 0.2)
    assert st.get("tr_hp_window") is None
    assert st.get("tr_job_hparams") is None


def test_apply_reaches_the_card(app, window):
    from structsept.app import tab_train

    st = app.app_state
    window.vars["lr_dec_initial"].set("1e-3")
    window.vars["width"].set("64")
    window.vars["loss_function"].set("huber")
    assert window.apply() is True
    _pump(app, 0.3)
    assert not window.alive()
    assert st["tr_hparams"]["lr_dec_initial"] == 0.001
    assert st["tr_width"].get() == 64
    assert tab_train.current_hparams(st)["loss_function"] == "huber"
    assert "decoder lr 0.001" in st["tr_hp_summary"].get()
    assert "Learning rate (decoder weights)" in st["tr_log"].get("1.0", "end")


def test_apply_is_refused_on_an_error(app, window):
    st = app.app_state
    window.vars["samples_per_scene"].set("4001")
    window.vars["seed"].set("forty-two")
    _pump(app, 0.4)
    assert str(window.btn_apply.cget("state")) == "disabled"
    assert str(window.inputs["seed"].cget("style")) == "Invalid.TEntry"
    assert str(window.inputs["samples_per_scene"].cget("style")) == "Invalid.TEntry"
    assert window.apply() is False
    assert window.alive()
    assert st["tr_hparams"]["samples_per_scene"] == 8000
    window.vars["samples_per_scene"].set("4000")
    window.vars["seed"].set("42")
    _pump(app, 0.4)
    assert str(window.btn_apply.cget("state")) == "normal"


def test_schedule_rows_follow_the_type(app, window):
    def shown(key):
        return window.inputs[key].winfo_manager() == "grid"

    assert shown("lr_dec_interval") and not shown("lr_dec_final")
    window.vars["lr_code_type"].set("Warmup")
    _pump(app, 0.4)
    assert shown("lr_dec_interval") and not shown("lr_code_interval")
    assert shown("lr_code_final") and not shown("lr_dec_final")
    window.vars["lr_dec_type"].set("Constant")
    window.vars["lr_code_type"].set("Constant")
    _pump(app, 0.4)
    for key in ("interval", "factor", "final", "length"):
        assert not shown(f"lr_dec_{key}") and not shown(f"lr_code_{key}")
    assert shown("lr_dec_initial") and shown("lr_code_initial")


def test_load_from_a_shipped_decoder(app, window):
    label = next(s for s in window.sources if s.endswith("ChiAndCross"))
    window.source_combo.set(label)
    window.load_selected()
    _pump(app, 0.3)
    assert window.vars["latent_dim"].get() == "2"
    assert window.vars["samples_per_scene"].get() == "16384"
    assert window.vars["num_epochs"].get() == "1000"
    assert "ChiAndCross" in window.status.get()
    window.reset()
    assert window.vars["latent_dim"].get() == "1"


def test_changed_rows_are_marked(app, window):
    row = window.rows["dropout_prob"]
    assert "•" not in str(row["label"].cget("text"))
    window.vars["dropout_prob"].set("0.1")
    _pump(app, 0.4)
    assert "•" in str(row["label"].cget("text"))
    assert str(row["label"].cget("style")) == "Card.Accent.TLabel"


def test_card_flags_an_invalid_combination(app):
    from structsept.app import tab_train

    st = app.app_state
    st["tr_latent_dim"].set(20)
    st["tr_width"].set(16)
    _pump(app, 0.2)
    try:
        assert str(st["tr_hp_summary_label"].cget("style")) == "Card.Danger.TLabel"
        # d + 3 on a 3-D dataset, d + 2 on a 2-D one: whatever is in data/
        geom = tab_train._geom(tab_train._selected_dataset(st))
        assert f"d + {geom}" in st["tr_hp_summary"].get()
    finally:
        st["tr_latent_dim"].set(1)
        st["tr_width"].set(128)
    _pump(app, 0.2)
    assert str(st["tr_hp_summary_label"].cget("style")) == "Card.Subtle.TLabel"


def test_retyping_a_card_spinbox_is_not_undone(app):
    """Selecting "200" and typing "50" first empties the field. The live
    summary runs from that very trace; if it wrote the last good value back,
    the typed digits would land in front of it: "50200"."""
    from structsept.app import widgets

    st = app.app_state
    epochs = st["tr_epochs"]
    try:
        epochs.set("")
        _pump(app, 0.2)
        assert widgets.peek_int(epochs) is None  # still empty, not 200
        assert "not a whole number" in st["tr_hp_summary"].get()
        assert str(st["tr_hp_summary_label"].cget("style")) == "Card.Danger.TLabel"
        epochs.set("50")
        _pump(app, 0.2)
        assert epochs.get() == 50
        assert str(st["tr_hp_summary_label"].cget("style")) == "Card.Subtle.TLabel"
    finally:
        epochs.set(200)
    _pump(app, 0.2)


def test_a_typo_in_a_hidden_schedule_row_does_not_block_apply(app, window):
    st = app.app_state
    window.vars["lr_dec_type"].set("Warmup")
    window.vars["lr_dec_length"].set("0")
    _pump(app, 0.4)
    assert str(window.btn_apply.cget("state")) == "disabled"
    window.vars["lr_dec_type"].set("Step")  # the Warm-up rows disappear
    _pump(app, 0.4)
    assert str(window.btn_apply.cget("state")) == "normal"
    assert window.apply() is True
    assert st["tr_hparams"]["lr_dec_length"] == 100  # the value it opened with


def test_a_malformed_specs_file_is_reported(app, window, tmp_path):
    bad = tmp_path / "specs.json"
    bad.write_text(json.dumps({"NetworkSpecs": "six layers"}), encoding="utf-8")
    window.sources["run: broken"] = bad
    window.source_combo.set("run: broken")
    window.load_selected()
    _pump(app, 0.2)
    assert "Could not use" in window.status.get()
    assert app.app_state.get("callback_errors") is None


def test_import_sheet_into_the_window(app, window, tmp_path, monkeypatch):
    from structsept.app import hparam_window

    st = app.app_state
    path = _write_xlsx(tmp_path / "sheet.xlsx", SHEET_ROWS)
    monkeypatch.setattr(
        hparam_window.filedialog, "askopenfilename", lambda **kw: str(path)
    )
    window.import_sheet()
    _pump(app, 0.4)
    assert window.vars["width"].get() == "256"
    assert window.vars["samples_per_scene"].get() == "4096"
    assert window.vars["code_init_std"].get() == "0.1"
    assert window.status.get().startswith("Imported sheet.xlsx")
    assert window.origin["label"] == "sheet: sheet.xlsx"
    assert window.apply() is True
    _pump(app, 0.3)
    try:
        assert st["tr_hparams"]["num_epochs"] == 600 and st["tr_epochs"].get() == 600
        assert st["tr_hp_summary"].get().startswith("From sheet: sheet.xlsx")
        assert "loaded from sheet: sheet.xlsx" in st["tr_log"].get("1.0", "end")
        assert st["tr_sheet_dir"] == tmp_path  # the next dialog opens here
        # a file that is not a workbook is reported, not raised
        bad = tmp_path / "bad.xlsx"
        bad.write_text("nope")
        monkeypatch.setattr(
            hparam_window.filedialog, "askopenfilename", lambda **kw: str(bad)
        )
        again = tab_train_open(st)
        try:
            again.import_sheet()
            assert "Could not read bad.xlsx" in again.status.get()
        finally:
            again.close()
    finally:
        st["tr_hp_origin"] = None
    _pump(app, 0.2)


def tab_train_open(st):
    from structsept.app import tab_train

    win = tab_train.open_hparams(st)
    _hide(win.top)
    return win


def test_the_window_remembers_where_its_values_came_from(app, window):
    """Load, apply, close, open again: the source is still named, and so is
    whatever was edited by hand in between."""
    st = app.app_state
    label = next(s for s in window.sources if s.endswith("ChiAndCross"))
    window.source_combo.set(label)
    window.load_selected()
    assert window.apply() is True
    _pump(app, 0.3)
    try:
        assert st["tr_hp_origin"]["label"] == label
        assert st["tr_hp_summary"].get().startswith(f"From {label} (loaded ")
        assert "edited since" not in st["tr_hp_summary"].get()

        again = tab_train_open(st)
        try:
            assert again.source_combo.get() == label
            assert again.status.get().startswith(f"From {label}")
            assert "edited since" not in again.status.get()
            again.vars["num_epochs"].set("30")  # one edit by hand, applied
            assert again.apply() is True
        finally:
            again.close()
        _pump(app, 0.3)
        assert st["tr_hp_origin"]["label"] == label  # still that source
        assert "edited since: epochs" in st["tr_hp_summary"].get()
        st["tr_width"].set(64)  # a card edit counts too
        _pump(app, 0.2)
        assert "edited since: width, epochs" in st["tr_hp_summary"].get()

        third = tab_train_open(st)
        try:
            assert "edited since: width, epochs" in third.status.get()
            third.reset()
            assert third.source_combo.get() == "" and third.origin is None
            assert third.apply() is True
        finally:
            third.close()
        _pump(app, 0.3)
        assert st["tr_hp_origin"] is None
        assert not st["tr_hp_summary"].get().startswith("From ")
    finally:
        st["tr_hp_origin"] = None
        st["tr_width"].set(128)
    _pump(app, 0.2)


def test_runs_table_sorts_by_column(app, tmp_path):
    from structsept.app import tab_train

    st = app.app_state
    for name, stamp, d in (
        ("old", "2026-09-01T10:00:00", 3),
        ("new", "2026-09-28T10:00:00", 1),
    ):
        run = tmp_path / name
        run.mkdir()
        (run / "specs.json").write_text(json.dumps({"CodeLength": d}))
        (run / "metadata.json").write_text(json.dumps({"timestamp": stamp}))
    old_dir = st["tr_runs_dir"]
    st["tr_runs_dir"] = tmp_path
    try:

        def shown():
            tree = st["tr_tree"]
            return [tree.item(i, "text") for i in tree.get_children()]

        tab_train.refresh_runs(st)
        assert shown() == ["new", "old"]
        assert "▾" in str(st["tr_tree"].heading("date", "text"))
        tab_train.sort_runs(st, "latent_dim")
        assert shown() == ["new", "old"]  # d = 1 before d = 3
        tab_train.sort_runs(st, "latent_dim")
        assert shown() == ["old", "new"]
        assert "▾" in str(st["tr_tree"].heading("d", "text"))
        assert "▾" not in str(st["tr_tree"].heading("date", "text"))
        tab_train.sort_runs(st, "name")
        assert shown() == ["new", "old"]
    finally:
        st["tr_runs_dir"] = old_dir
        st["tr_runs_sort"] = ("date", True)
        tab_train.refresh_runs(st)


def test_train_refuses_an_invalid_set(app, tiny_data, monkeypatch):
    from structsept.app import tab_train

    st = app.app_state
    data_root, _ = tiny_data
    shown = []
    monkeypatch.setattr(
        tab_train.messagebox, "showerror", lambda *a, **k: shown.append(a)
    )
    old_root = st["tr_data_root"]
    st["tr_data_root"] = data_root
    try:
        tab_train.refresh_datasets(st)
        assert st["tr_combo"].get().startswith("tiny")
        st["tr_hparams"] = _hp(samples_per_scene=4001)
        tab_train._start_training(st)
        assert shown and "cannot be trained" in shown[0][1]
        assert st["busy"] is False
    finally:
        st["tr_hparams"] = hyperparams.defaults()
        st["tr_data_root"] = old_root
        tab_train.refresh_datasets(st)
