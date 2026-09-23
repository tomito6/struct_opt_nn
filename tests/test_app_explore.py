"""Regression tests for the Explore tab.

These drive the real Tk widgets - there is no way to test a GUI's wiring
without them - so the window is made transparent and parked off-screen before
the first ``update()``. It never appears on anyone's display. The whole module
skips when no display is available.

Every test here corresponds to a defect that shipped once: a crash that froze
the context column, an unclamped spinbox that indexed past the design vector,
a stale mesh that Export STL would have written, and two threads evaluating
the same SDF.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _hide(root):
    """Off-screen and fully transparent, but still mapped.

    A withdrawn window has no device context, and the embedded matplotlib
    canvases never get a size, so the panels under test would not lay out.
    """
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
    _pump(root, 1.0)


@pytest.fixture(scope="module")
def app():
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
def loaded(app):
    """ChiAndCross on a deliberately non-cubic control net."""
    from structsept.app import tab_explore

    st = app.app_state
    for label, entry in st["ex_models"].items():
        if entry.name == "ChiAndCross":
            st["ex_combo_model"].set(label)
    for var, value in zip(st["ex_n_ctrl"], (4, 2, 3)):
        var.set(value)
    for var, value in zip(st["ex_tiling"], (3, 1, 2)):
        var.set(value)
    tab_explore._load_model(st)
    _wait_idle(app)
    assert "ex_cps" in st, "the decoder did not load"
    return app


def test_loads_a_non_cubic_control_net(loaded):
    st = loaded.app_state
    assert st["ex_cps"].shape == (4 * 2 * 3, 2)
    assert st["ex_n_ctrl_loaded"] == (4, 2, 3)


def test_flat_index_matches_the_spline(loaded):
    """The grid's row index has to be the spline's row index, or the editor
    writes to a control point other than the one on screen."""
    from DeepSDFStruct.geom_reconstruction import build_parameter_spline
    from structsept.app import models

    n_ctrl = loaded.app_state["ex_n_ctrl_loaded"]
    spline = build_parameter_spline([1, 1, 1], [n - 1 for n in n_ctrl], 1)
    spline.control_points = np.arange(int(np.prod(n_ctrl)), dtype=float).reshape(-1, 1)
    for k in range(n_ctrl[2]):
        for j in range(n_ctrl[1]):
            for i in range(n_ctrl[0]):
                position = models.control_point_position(i, j, k, n_ctrl)
                value = spline.evaluate(np.array([position]))[0, 0]
                assert value == pytest.approx(models.flat_index(i, j, k, n_ctrl))


def test_resolution_change_keeps_the_context_column_alive(loaded):
    """A constant latent panel carries a text annotation; rebuilding the panel
    at a new resolution used to call remove() on an artist ax.clear() had
    already detached, which killed every later context redraw."""
    from structsept.app import tab_explore

    st = loaded.app_state
    st["ex_slice_res"].set("160")
    tab_explore._schedule_redraw(st)
    _pump(loaded, 2.5)
    tab_explore._on_cell(st, 1, 1)
    st["ex_scale"].set(0.31)
    _pump(loaded, 2.0)

    assert st.get("callback_errors") is None
    assert st["ex_metrics"]["volume"][0].get() != "--"
    st["ex_slice_res"].set("96")
    tab_explore._schedule_redraw(st)
    _pump(loaded, 2.0)


@pytest.mark.parametrize("typed, expected", [(7, 2), (-1, 0), (1, 1)])
def test_layer_spinbox_is_clamped(loaded, typed, expected):
    """ttk.Spinbox does not clamp its textvariable; an out-of-range layer used
    to raise IndexError on a click, or wrap to a layer the user cannot see."""
    from structsept.app import tab_explore

    st = loaded.app_state
    st["ex_layer"].set(typed)
    _pump(loaded, 0.8)
    assert tab_explore._layer(st) == expected
    st["ex_grid"]._click(1, 0)
    tab_explore._on_scale(st, 0.2)
    _pump(loaded, 0.3)
    assert st.get("callback_errors") is None
    st["ex_layer"].set(0)
    _pump(loaded, 0.5)


def test_surface_is_invalidated_when_the_design_moves(loaded):
    from structsept.app import tab_explore

    st = loaded.app_state
    tab_explore._extract_surface(st)
    _wait_idle(loaded)
    assert st.get("ex_mesh") is not None

    tab_explore._on_scale(st, -0.4)
    _pump(loaded, 0.3)
    assert st.get("ex_mesh") is None
    assert str(st["ex_btn_export"].cget("state")) == "disabled"


def test_mesher_holds_the_sdf_alone(loaded):
    """LatticeSDFStruct writes one latent vector per query point into its
    microtile, so a redraw during extraction raises a shape mismatch."""
    from structsept.app import tab_explore

    st = loaded.app_state
    tab_explore._extract_surface(st)
    loaded.update()
    assert st["ex_sdf_busy"] is True
    assert st["ex_grid"].enabled is False

    st["ex_z_scale"].set(0.4)
    tab_explore._redraw_geometry(st)  # must be a no-op while the worker runs

    _wait_idle(loaded)
    assert st["ex_sdf_busy"] is False
    assert st["ex_grid"].enabled is True
    assert st.get("callback_errors") is None


def test_changing_the_decoder_marks_the_configuration_stale(loaded):
    st = loaded.app_state
    other = next(
        label for label, e in st["ex_models"].items() if e.name != "ChiAndCross"
    )
    st["ex_combo_model"].set(other)
    _pump(loaded, 0.8)
    assert st["ex_stale"].get()


def test_volume_fraction_is_unbiased():
    """Cell centres, not grid nodes: a node grid puts 27% of its samples on the
    domain faces, where CappedBorderSDF forces phi >= 0."""
    import torch
    from DeepSDFStruct.sdf_primitives import SphereSDF

    from structsept.app import models

    class UnitCube(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def get_device(self):
            return "cpu"

        def get_dtype(self):
            return torch.float32

        def _get_domain_bounds(self):
            return torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        def __call__(self, points):
            return self.inner(points)

    sphere = UnitCube(SphereSDF(center=torch.tensor([0.5, 0.5, 0.5]), radius=0.35))
    exact = 4.0 / 3.0 * np.pi * 0.35**3
    assert models.volume_fraction(sphere, None, res=40) == pytest.approx(
        exact, abs=0.01
    )


def test_latent_components_always_include_the_active_one():
    from structsept.app import tab_explore

    for d in (1, 2, 3, 4, 8, 16, 32):
        for active in range(d):
            shown = tab_explore._latent_components(active, d)
            assert active in shown
            assert len(shown) == min(3, d)
            assert shown == sorted(shown)
            assert all(0 <= c < d for c in shown)
