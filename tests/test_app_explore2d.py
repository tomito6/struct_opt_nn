"""Planar (2-D) decoders from end to end: dataset, Train tab, Explore 2-D tab.

One tiny ``datagen`` plate dataset (6 shapes, a few hundred samples each) is
trained for two epochs *through the Train tab's own code path* - the run is a
throwaway, its only job is to exist - and then loaded into the Explore 2-D tab,
where the controls are driven the way a user would. Everything lives under
pytest's temporary directories; the repo's data/ and runs/ are never touched.

The GUI parts drive real Tk widgets, hidden off-screen like the other app
tests, and skip when no display is available.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

N_SHAPES = 6


def _hide(root):
    for step in (
        lambda: root.attributes("-alpha", 0.0),
        lambda: root.overrideredirect(True),
        lambda: root.geometry("+{}+{}".format(-6000, -6000)),
    ):
        try:
            step()
        except Exception:
            pass
    root.update_idletasks()


def _pump(root, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.02)


def _wait_idle(root, timeout=300.0):
    st = root.app_state
    end = time.perf_counter() + timeout
    while st["busy"] and time.perf_counter() < end:
        root.update()
        time.sleep(0.02)
    _pump(root, 0.5)


def _tiny_hparams(**extra):
    from structsept.app import hyperparams

    return (
        hyperparams.defaults()
        | {
            "latent_dim": 2,
            "n_layers": 2,
            "width": 24,
            "latent_in": [1],
            "num_epochs": 2,
            "samples_per_scene": 64,
            "scenes_per_batch": 3,
        }
        | extra
    )


@pytest.fixture(scope="module")
def plate_data(tmp_path_factory):
    """A 6-shape 2-D plate dataset written by datagen, as the user would."""
    from datagen import make_plate_hole

    root = tmp_path_factory.mktemp("data2d")
    make_plate_hole.main(
        [
            "--dim",
            "2",
            "--n",
            str(N_SHAPES),
            "--no-extremes",
            "--method",
            "lhs",
            "--n-uniform",
            "300",
            "--n-band",
            "300",
            "--data-root",
            str(root),
        ]
    )
    return root


# --------------------------------------------------------------------------- #
# dataset audit and specs, no Tk
# --------------------------------------------------------------------------- #


def test_the_dataset_says_it_is_2d(plate_data):
    from structsept.app import datasets

    (row,) = datasets.list_datasets(plate_data)
    assert row["geom_dimension"] == 2
    assert row["n_instances"] == N_SHAPES

    report = datasets.validate_dataset(row["path"], log=lambda *_: None)
    assert report["problems"] == []
    assert report["geom_dimension"] == 2
    assert all(len(i["xyz_min"]) == 2 for i in report["instances"])


def test_row_width_decides_without_a_manifest(plate_data, tmp_path):
    """A 2-D set without datagen's manifest is still recognised, and a set
    mixing 2-D and 3-D files is refused rather than read as either."""
    import shutil

    from structsept.app import datasets

    copy = tmp_path / "copy"
    shutil.copytree(plate_data / "SdfSamples" / "plate_hole_2d", copy)
    (copy / "dataset.json").unlink()
    assert datasets.geom_dimension(copy) == 2

    rows = np.zeros((8, 4), dtype=np.float32)
    rows[:4, 3], rows[4:, 3] = 0.1, -0.1
    np.savez(copy / "plate" / "odd_one.npz", pos=rows[:4], neg=rows[4:])
    report = datasets.validate_dataset(copy, log=lambda *_: None)
    assert any("disagree on the geometry dimension" in p for p in report["problems"])


def test_specs_carry_the_dataset_dimension():
    from structsept.app import hyperparams

    specs = hyperparams.to_specs(_tiny_hparams(), "S", "D", geom_dimension=2)
    assert specs["NetworkSpecs"]["geom_dimension"] == 2
    # the width checks count d + 2 inputs, not d + 3: width 16 is too narrow
    # for a skip with d = 14 in 2-D, while d = 13 fits in 2-D (15) but not in
    # 3-D (16)
    hp = _tiny_hparams(latent_dim=14, width=16, latent_in=[1])
    messages = [i.message for i in hyperparams.validate(hp, geom_dimension=2)]
    assert any("d + 2 = 16" in m for m in messages)
    hp = _tiny_hparams(latent_dim=13, width=16, latent_in=[1])
    assert not hyperparams.errors(hyperparams.validate(hp, geom_dimension=2))
    assert hyperparams.errors(hyperparams.validate(hp, geom_dimension=3))


def test_width_checks_match_the_2d_decoder():
    """Twin of the 3-D test in test_app_hyperparams: every combination the
    planar decoder cannot build or run is refused before a run starts."""
    import itertools

    import torch

    import DeepSDFStruct.deep_sdf.workspace as ws
    from structsept.app import hyperparams

    for d, width, skips, xyz in itertools.product(
        (1, 13, 14, 15), (16, 17), ([], [1], [2], [1, 2]), (0, 1)
    ):
        hp = _tiny_hparams(
            latent_dim=d, width=width, latent_in=skips, xyz_in_all=bool(xyz)
        )
        try:
            specs = hyperparams.to_specs(hp, "S", "D", geom_dimension=2)
            decoder = ws.init_decoder(specs, "cpu", False)
            decoder(torch.randn(3, d + 2))
            broken = False
        except Exception:
            broken = True
        errors = hyperparams.errors(hyperparams.validate(hp, geom_dimension=2))
        if broken:
            assert errors, hp


# --------------------------------------------------------------------------- #
# the app
# --------------------------------------------------------------------------- #


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
    yield root
    main._shutdown(root.app_state, root)


@pytest.fixture(scope="module")
def trained_run(app, plate_data, tmp_path_factory):
    """Train through the Train tab exactly as a click on Train would."""
    from structsept.app import tab_train

    st = app.app_state
    runs = tmp_path_factory.mktemp("runs")
    old = st["tr_data_root"], st["tr_runs_dir"], st["e2_runs_dir"], st["ex_runs_dir"]
    st["tr_data_root"], st["tr_runs_dir"] = plate_data, runs
    st["e2_runs_dir"] = st["ex_runs_dir"] = runs
    try:
        tab_train.refresh_datasets(st)
        label = st["tr_combo"].get()
        assert "2-D" in label, label
        hp = _tiny_hparams()
        st["tr_hparams"] = dict(hp)
        for key, var in tab_train.CARD_VARS.items():
            st[var].set(hp[key])
        st["tr_run_name"].set("plate2d_throwaway")
        tab_train._update_readiness(st)
        readiness = st["tr_readiness"].get()
        tab_train._start_training(st)
        _wait_idle(app)
        yield runs / "plate2d_throwaway", readiness
    finally:
        from structsept.app import hyperparams

        st["tr_hparams"] = hyperparams.defaults()
        for key, var in tab_train.CARD_VARS.items():
            st[var].set(st["tr_hparams"][key])
        st["tr_data_root"], st["tr_runs_dir"], st["e2_runs_dir"], st["ex_runs_dir"] = (
            old
        )


def test_train_tab_trains_a_planar_decoder(trained_run):
    import torch

    run_dir, readiness = trained_run
    assert "planar decoder" in readiness
    specs = json.loads((run_dir / "specs.json").read_text(encoding="utf-8"))
    assert specs["NetworkSpecs"]["geom_dimension"] == 2
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["geom_dimension"] == 2
    codes = torch.load(run_dir / "LatentCodes" / "latest.pth", weights_only=False)
    assert tuple(codes["latent_codes"]["weight"].shape) == (N_SHAPES, 2)


def test_the_run_goes_to_the_right_explorer(app, trained_run):
    from structsept.app import tab_explore, tab_explore2d

    st = app.app_state
    run_dir, _ = trained_run
    tab_explore.refresh_models(st)
    tab_explore2d.refresh_models(st)
    assert not tab_explore.select_model(st, run_dir.name)
    assert tab_explore2d.select_model(st, run_dir.name)


@pytest.fixture(scope="module")
def loaded(app, trained_run):
    from structsept.app import tab_explore2d

    st = app.app_state
    run_dir, _ = trained_run
    st["e2_runs_dir"] = run_dir.parent
    tab_explore2d.refresh_models(st)
    assert tab_explore2d.select_model(st, run_dir.name)
    tab_explore2d._load_model(st)
    _wait_idle(app)
    assert "e2_model" in st, "the planar decoder did not load"
    return app


def test_explore2d_loads_one_slider_per_component(loaded):
    st = loaded.app_state
    assert st["e2_trained"].shape == (N_SHAPES, 2)
    assert len(st["e2_sliders"]) == 2
    assert np.allclose(st["e2_latent"], st["e2_trained"].mean(axis=0))
    assert st["e2_metrics"]["area"][0].get() != "--"
    assert st["e2_coverage_note"].get().startswith("nearest trained code")
    assert st.get("callback_errors") is None


def test_sliders_move_lambda_and_redraw(loaded):
    from structsept.app import tab_explore2d

    st = loaded.app_state
    lo, hi = st["e2_slider_range"]
    before = st["e2_metrics"]["phi"][0].get()
    tab_explore2d._on_scale(st, 0, float(lo[0]))
    tab_explore2d._on_scale(st, 1, float(hi[1]))
    _pump(loaded, 0.8)
    assert st["e2_latent"][0] == pytest.approx(lo[0])
    assert st["e2_latent"][1] == pytest.approx(hi[1])
    assert st["e2_sliders"][1][1].cget("text") == f"{hi[1]:+.3f}"
    assert st["e2_metrics"]["phi"][0].get() != before
    # both sliders sit outside the trained range: the status says so
    assert "outside the trained range" in st["e2_status"].get()

    tab_explore2d._reset_to_mean(st)
    _pump(loaded, 0.8)
    assert np.allclose(st["e2_latent"], st["e2_trained"].mean(axis=0))
    assert "outside" not in st["e2_status"].get()
    assert st.get("callback_errors") is None


def test_the_field_is_the_decoder_at_lambda(loaded):
    """What the centre panel shows is f_theta at the current code, on the
    [-1, 1]^2 grid the image is placed on."""
    from structsept.app import models

    st = loaded.app_state
    model, latent = st["e2_model"], st["e2_latent"]
    field = models.eval_field_2d(model, latent, res=9)
    corner = models.decode_2d(model, latent, np.array([[-1.0, -1.0], [1.0, 1.0]]))
    assert field[0, 0] == pytest.approx(corner[0], abs=1e-6)
    assert field[-1, -1] == pytest.approx(corner[1], abs=1e-6)
