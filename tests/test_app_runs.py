"""Editing runs after the fact: rename, notes, delete - and the lists that show them.

A run is a directory under ``runs/``; the only things worth changing about
it afterwards are that directory's name and the ``Description`` in its
``specs.json``. The file operations in ``structsept.app.training`` are
tested on temporary directories. The GUI parts drive the real Train tab and
both Explore tabs hidden off-screen, on a temporary runs directory swapped
into the app's state, so the repo's ``runs/`` is never touched.
"""

from __future__ import annotations

import json
import time

import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_run(
    runs,
    name,
    d=2,
    epochs=10,
    stamp="2026-09-20T10:00:00",
    description="",
    trained=False,
    geom=2,
):
    """A run directory exactly as the Train tab writes one, minus the training.

    ``trained`` fakes the checkpoint file the model pickers look for, so the
    run is listed on Explore 2-D (``geom`` 2) without ever loading it.
    """
    from structsept.app import hyperparams, training

    run = runs / name
    hp = hyperparams.defaults() | {
        "latent_dim": d,
        "n_layers": 2,
        "width": 24,
        "latent_in": [1],
        "num_epochs": epochs,
        "description": description,
    }
    training.write_specs(run, runs / "split.json", runs, hp, geom_dimension=geom)
    training.write_metadata(run, dataset="toy", timestamp=stamp)
    if trained:
        (run / "ModelParameters").mkdir()
        (run / "ModelParameters" / "latest.pth").write_bytes(b"not a checkpoint")
    return run


# --------------------------------------------------------------------------- #
# the file operations
# --------------------------------------------------------------------------- #


def test_check_run_name(tmp_path):
    from structsept.app import training

    (tmp_path / "taken").mkdir()
    for bad in ("", "   ", "a/b", "a\\b", "a:b", "..", ".", "ends.", "tab\there"):
        assert training.check_run_name(tmp_path, bad), repr(bad)
    assert "already exists" in training.check_run_name(tmp_path, "taken")
    assert training.check_run_name(tmp_path, "taken", current="taken") is None
    assert training.check_run_name(tmp_path, " fresh ") is None


def test_rename_run_moves_the_directory(tmp_path):
    from structsept.app import training

    run = _make_run(tmp_path, "old")
    new = training.rename_run(tmp_path, "old", " new ")
    assert new == tmp_path / "new"
    assert (new / "specs.json").is_file() and not run.exists()
    assert training.rename_run(tmp_path, "new", "new") == new  # nothing to do
    _make_run(tmp_path, "other")
    with pytest.raises(ValueError):
        training.rename_run(tmp_path, "new", "other")
    with pytest.raises(ValueError):
        training.rename_run(tmp_path, "new", "a/b")
    with pytest.raises(FileNotFoundError):
        training.rename_run(tmp_path, "missing", "x")
    (tmp_path / "plain").mkdir()  # a directory without specs.json is not a run
    with pytest.raises(FileNotFoundError):
        training.rename_run(tmp_path, "plain", "x")


def test_write_description_touches_nothing_else(tmp_path):
    from structsept.app import training

    run = _make_run(tmp_path, "r", description="first")
    before = json.loads((run / "specs.json").read_text(encoding="utf-8"))
    assert before["Description"] == "first"
    training.write_description(run, "what this run is for")
    after = json.loads((run / "specs.json").read_text(encoding="utf-8"))
    assert after.pop("Description") == "what this run is for"
    before.pop("Description")
    assert after == before
    assert training.read_description(run) == "what this run is for"
    (row,) = training.list_runs(tmp_path)
    assert row["description"] == "what this run is for"
    with pytest.raises(FileNotFoundError):
        training.write_description(tmp_path / "none", "x")


def test_delete_run_refuses_what_is_not_a_run(tmp_path):
    from structsept.app import training

    run = _make_run(tmp_path, "r")
    (tmp_path / "plain").mkdir()
    with pytest.raises(ValueError):
        training.delete_run(tmp_path, "plain")
    with pytest.raises(FileNotFoundError):
        training.delete_run(tmp_path, "missing")
    with pytest.raises(ValueError):
        training.delete_run(tmp_path, "..")
    with pytest.raises(ValueError):
        training.delete_run(tmp_path, ".")
    assert (tmp_path / "plain").is_dir() and tmp_path.is_dir()
    training.delete_run(tmp_path, "r")
    assert not run.exists()


def test_describe_states_what_the_run_is(tmp_path):
    from structsept.app import run_editor

    _make_run(tmp_path, "r", d=3, epochs=10, trained=True)
    text = run_editor.describe(tmp_path, "r")
    for part in ("d=3", "2x24", "10 epochs", "checkpoint saved", "2026-09-20 10:00"):
        assert part in text, text
    _make_run(tmp_path, "waiting")
    assert "no checkpoint yet" in run_editor.describe(tmp_path, "waiting")


def test_a_run_that_stopped_short_says_how_far_it_got(tmp_path):
    """A checkpoint at epoch 3 of 10 loads like a finished one, so the table,
    the editor and the model pickers have to say it is not: the n166 run sat
    in all three as '800 epochs, checkpoint saved' while holding epoch 80."""
    import torch

    from structsept.app import models, run_editor, training

    def checkpoint(run, epoch):
        (run / "LatentCodes").mkdir()
        torch.save(
            {"epoch": epoch, "latent_codes": {"weight": torch.zeros(4, 3)}},
            run / "LatentCodes" / "latest.pth",
        )

    checkpoint(_make_run(tmp_path, "short", d=3, epochs=10, trained=True), 3)
    checkpoint(_make_run(tmp_path, "full", d=3, epochs=10, trained=True), 10)
    _make_run(tmp_path, "unreadable", d=3, epochs=10, trained=True)
    rows = {r["name"]: r for r in training.list_runs(tmp_path)}
    assert rows["short"]["epochs"] == 10 and rows["short"]["last_epoch"] == 3
    assert rows["full"]["last_epoch"] == 10
    assert rows["unreadable"]["last_epoch"] is None

    assert training.epochs_text(10, 3) == "3/10"
    assert training.epochs_text(10, 10) == training.epochs_text(10, None) == "10"
    assert training.epochs_text(None, None) == "?"

    assert "epoch 3 of 10" in run_editor.describe(tmp_path, "short")
    assert "10 epochs" in run_editor.describe(tmp_path, "full")
    assert "10 epochs" in run_editor.describe(tmp_path, "unreadable")

    tags = {e.name: models.entry_tag(e) for e in models.list_models(tmp_path)}
    assert tags["short"] == "trained here, epoch 3 of 10"
    assert tags["full"] == tags["unreadable"] == "trained here"
    assert tags["ChiAndCross"] == "pretrained"


def test_combo_width_fits_the_longest_label():
    from structsept.app import widgets

    assert widgets.combo_width([]) == 44
    assert widgets.combo_width(["short"]) == 44
    assert widgets.combo_width(["x" * 70, "y"]) == 72
    assert widgets.combo_width(["x" * 500]) == 96


# --------------------------------------------------------------------------- #
# the app
# --------------------------------------------------------------------------- #


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
    import matplotlib

    matplotlib.use("Agg")
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
def runs(app, tmp_path):
    """Two runs in a temporary directory, shown by every list of the app."""
    from structsept.app import run_editor

    st = app.app_state
    old = st["tr_runs_dir"], st["ex_runs_dir"], st["e2_runs_dir"]
    _make_run(
        tmp_path, "alpha", d=3, stamp="2026-09-01T10:00:00", description="the first one"
    )
    _make_run(tmp_path, "beta", d=2, stamp="2026-09-28T10:00:00", trained=True)
    st["tr_runs_dir"] = st["ex_runs_dir"] = st["e2_runs_dir"] = tmp_path
    run_editor.runs_changed(st)
    _pump(app, 0.2)
    yield tmp_path
    editor = st.get("run_editor")
    if editor is not None and editor.alive():
        editor.close()
    st["tr_runs_dir"], st["ex_runs_dir"], st["e2_runs_dir"] = old
    st["tr_runs_sort"] = ("date", True)
    run_editor.runs_changed(st)
    _pump(app, 0.2)


def _shown(st):
    tree = st["tr_tree"]
    return [tree.item(i, "text") for i in tree.get_children()]


def _select(st, name):
    from structsept.app import tab_train

    assert tab_train._select_row(st, name), name


def test_table_shows_notes_and_keeps_the_selection(app, runs):
    from structsept.app import tab_train

    st = app.app_state
    assert _shown(st) == ["beta", "alpha"]
    tree = st["tr_tree"]
    assert tree.item(tree.get_children()[1], "values")[-1] == "the first one"
    assert str(tree.heading("notes", "text")).startswith("notes")
    _select(st, "alpha")
    tab_train.refresh_runs(st)
    assert tab_train.selected_run(st)["name"] == "alpha"
    assert set(st["tr_run_buttons"]) == {"open", "edit", "hparams", "delete"}


def test_the_table_says_how_long_a_run_trained(app, runs):
    """The time column sums the per-epoch timings the trainer logs, the same
    way for every run whoever launched it, and sorts as a number."""
    import torch

    from structsept.app import tab_train, training

    def logs(name, timing):
        torch.save(
            {"epoch": len(timing), "loss": [0.1] * len(timing), "timing": timing},
            runs / name / "Logs.pth",
        )

    logs("alpha", [1800.0, 1800.0, 2160.0])  # 1 h 36 min
    logs("beta", [20.0, 25.0])
    st = app.app_state
    tab_train.refresh_runs(st)
    rows = {r["name"]: r for r in st["tr_runs"]}
    assert rows["alpha"]["train_seconds"] == 5760.0
    assert rows["beta"]["train_seconds"] == 45.0

    tree = st["tr_tree"]
    column = tree["columns"].index("time")
    cells = {
        tree.item(i, "text"): tree.item(i, "values")[column]
        for i in tree.get_children()
    }
    assert cells == {"alpha": "1 h 36 min", "beta": "45 s"}
    tab_train.sort_runs(st, "train_seconds")
    assert _shown(st) == ["beta", "alpha"]

    logs("beta", [20.0, 25.0, 1200.0])  # rewritten on disk: the cache follows
    assert training.training_seconds(runs / "beta") == 1245.0
    (runs / "beta" / "Logs.pth").write_bytes(b"half written")
    assert training.training_seconds(runs / "beta") is None
    assert training.training_seconds(runs / "nothing") is None

    assert training.duration_text(None) == "-"
    assert training.duration_text(59) == "59 s"
    assert training.duration_text(1200) == "20 min"
    assert training.duration_text(6982) == "1 h 56 min"


def test_edit_renames_and_annotates(app, runs):
    from structsept.app import tab_train

    st = app.app_state
    _select(st, "alpha")
    editor = tab_train.edit_run(st)
    _hide(editor.top)
    _pump(app, 0.2)
    assert tab_train.edit_run(st) is editor  # one window, raised again
    assert editor.name_var.get() == "alpha"
    assert editor.notes.get("1.0", "end-1c") == "the first one"

    editor.name_var.set("beta")  # taken
    assert editor.save() is False
    assert "already exists" in editor.status.get()
    assert editor.alive()

    editor.name_var.set("gamma")
    editor.notes.delete("1.0", "end")
    editor.notes.insert("1.0", "renamed in the test")
    assert editor.save() is True
    _pump(app, 0.2)
    assert not editor.alive() and st.get("run_editor") is None
    assert (runs / "gamma" / "specs.json").is_file() and not (runs / "alpha").exists()
    specs = json.loads((runs / "gamma" / "specs.json").read_text(encoding="utf-8"))
    assert specs["Description"] == "renamed in the test"
    assert _shown(st) == ["beta", "gamma"]
    assert tab_train.selected_run(st)["name"] == "gamma"  # re-selected as renamed
    assert "renamed to 'gamma'" in st["tr_log"].get("1.0", "end")


def test_edit_from_the_explore2d_picker(app, runs):
    from structsept.app import tab_explore, tab_explore2d

    st = app.app_state
    assert tab_explore2d.select_model(st, "beta")
    _pump(app, 0.1)
    assert not st["e2_btn_edit"].instate(["disabled"])
    # the shipped decoders on Explore are the library's: read-only
    tab_explore.refresh_models(st)
    first_label, first_entry = next(iter(st["ex_models"].items()))
    assert first_entry.source == "pretrained"
    st["ex_combo_model"].set(first_label)
    _pump(app, 0.1)
    assert st["ex_btn_edit"].instate(["disabled"])
    assert tab_explore._edit_model(st) is None

    editor = tab_explore2d._edit_model(st)
    _hide(editor.top)
    _pump(app, 0.2)
    editor.name_var.set("beta_renamed")
    assert editor.save() is True
    _pump(app, 0.2)
    assert (runs / "beta_renamed").is_dir() and not (runs / "beta").exists()
    # the picker follows the run under its new name, and the Train tab saw it
    assert st["e2_models"][st["e2_combo_model"].get()].name == "beta_renamed"
    assert _shown(st) == ["beta_renamed", "alpha"]


def test_the_picker_is_as_wide_as_its_longest_label(app, runs):
    from structsept.app import widgets

    st = app.app_state
    labels = list(st["e2_models"])
    assert any("beta" in label for label in labels)
    assert int(st["e2_combo_model"].cget("width")) == widgets.combo_width(labels)


def test_delete_asks_first_then_removes(app, runs, monkeypatch):
    from structsept.app import tab_train

    st = app.app_state
    asked = []
    monkeypatch.setattr(
        tab_train.messagebox, "askyesno", lambda *a, **k: asked.append(a) or False
    )
    _select(st, "alpha")
    assert tab_train.delete_run(st) is False
    assert (runs / "alpha").is_dir() and len(asked) == 1
    assert "alpha" in str(asked[0])
    monkeypatch.setattr(tab_train.messagebox, "askyesno", lambda *a, **k: True)
    assert tab_train.delete_run(st) is True
    assert not (runs / "alpha").exists()
    assert _shown(st) == ["beta"]
    assert "'alpha' deleted" in st["tr_log"].get("1.0", "end")


def test_the_run_being_trained_is_locked(app, runs, monkeypatch):
    from structsept.app import run_editor, tab_train

    st = app.app_state
    shown = []
    monkeypatch.setattr(
        run_editor.messagebox, "showinfo", lambda *a, **k: shown.append(a)
    )
    monkeypatch.setattr(
        tab_train.messagebox, "showinfo", lambda *a, **k: shown.append(a)
    )
    monkeypatch.setattr(tab_train.messagebox, "askyesno", lambda *a, **k: True)
    st["busy"] = True
    st["tr_watch_dir"] = runs / "beta"
    try:
        _select(st, "beta")
        assert tab_train.edit_run(st) is None
        assert tab_train.delete_run(st) is False
        assert (runs / "beta").is_dir()
        assert len(shown) == 2 and all("trained" in str(a) for a in shown)
        # the other run is free
        _select(st, "alpha")
        editor = tab_train.edit_run(st)
        assert editor is not None
        _hide(editor.top)
        editor.close()
    finally:
        st["busy"] = False
        st["tr_watch_dir"] = None
    _pump(app, 0.2)


def test_load_hyperparameters_from_the_table(app, runs):
    from structsept.app import hyperparams, tab_train

    st = app.app_state
    _select(st, "alpha")  # d = 3, 2 x 24 in its specs
    window = tab_train.load_run_hparams(st)
    _hide(window.top)
    _pump(app, 0.3)
    try:
        assert window.source_combo.get() == "run: alpha"
        assert window.vars["latent_dim"].get() == "3"
        assert window.vars["width"].get() == "24"
    finally:
        window.close()
        _pump(app, 0.2)
        st["tr_hparams"] = hyperparams.defaults()
        for key, var_name in tab_train.CARD_VARS.items():
            st[var_name].set(hyperparams.FIELD_BY_KEY[key].default)
