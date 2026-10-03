"""Explore tab: drive f_theta(lambda(x), x) by hand.

This is the online half of the paper's method made interactive. The user picks
a trained decoder f_theta, chooses how many latent control points and how many
unit cells the domain gets, and then moves the control points - the design
variables an MMA run would optimise - while the geometry redraws.

The panel layout follows the questions in that order:

    which decoder, and is it usable      -> header, with d, code count and a
                                            warning when the codes are empty
    how many design variables, how many
    cells, and are the two independent   -> header, derived readout
    which knob moves which region        -> left, the control net laid out in
                                            space, markers on the slice
    what shape is this now               -> centre, f_theta as material/void
    how much material is there           -> centre, volume fraction
    what does this latent value mean     -> right, the bare unit cell
    is the design still supported        -> right, latent coverage + the
                                            distance to the nearest trained code
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from structsept.app import models, run_editor, runtime, theme, viz, widgets

DEBOUNCE_MS = 120
SETTLE_MS = 400
CELL_RES = 56
# 32**3 cell centres: at 20 the midpoint rule cannot resolve thin struts
# (measured 0.22 at res 24 against 0.29 at res 96 on the same design), and
# past 32 the 0.25 s already spent on the Tk thread starts to show.
VOLUME_RES = 32
DEFAULT_SLICE_RES = 96
SLICE_CHOICES = ("64", "96", "128", "160")
DEFAULT_N_BASE = 12


def _sub(n):
    return viz._sub(n)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def build(st, parent, runs_dir):
    """Populate the Explore tab. ``parent`` is an empty padded frame."""
    st["ex_runs_dir"] = Path(runs_dir)
    # a run renamed or deleted anywhere in the app is re-read here too
    st.setdefault("run_listeners", []).append(lambda: refresh_models(st))
    st["ex_n_ctrl"] = [tk.IntVar(value=v) for v in (3, 3, 2)]
    st["ex_tiling"] = [tk.IntVar(value=v) for v in (2, 2, 2)]
    st["ex_component"] = tk.StringVar(value="")
    st["ex_layer"] = tk.IntVar(value=0)
    st["ex_slice_res"] = tk.StringVar(value=str(DEFAULT_SLICE_RES))
    st["ex_n_base"] = tk.IntVar(value=DEFAULT_N_BASE)
    st["ex_selected"] = (0, 0)
    st["ex_muted"] = False
    st["ex_mesh"] = None

    palette = st["palette"]

    _build_header(st, parent, palette)
    # the footer claims its height before the paned window takes the rest;
    # packed the other way round, pack squeezes the log and the action bar
    # off the bottom of the tab
    _build_footer(st, parent, palette)

    paned = ttk.PanedWindow(parent, orient="horizontal")
    paned.pack(side="top", fill="both", expand=True, pady=(8, 0))
    st["ex_paned"] = paned

    left = ttk.Frame(paned)
    centre = ttk.Frame(paned)
    right = ttk.Frame(paned)
    paned.add(left, weight=0)
    paned.add(centre, weight=3)
    paned.add(right, weight=1)

    _build_design(st, left, palette)
    _build_geometry(st, centre, palette)
    _build_context(st, right, palette)

    # sashpos is silently ignored while the pane is unmapped
    paned.bind("<Map>", lambda e: _place_sashes(st), add="+")

    refresh_models(st)
    _update_derived(st)
    return parent


def _place_sashes(st):
    if st.get("ex_sashes_placed"):
        return
    paned = st["ex_paned"]
    try:
        total = paned.winfo_width()
        if total < 200:
            return
        paned.sashpos(0, int(total * 0.25))
        paned.sashpos(1, int(total * 0.755))
        st["ex_sashes_placed"] = True
    except tk.TclError:
        pass


def _build_header(st, parent, palette):
    outer, body = widgets.card(parent, palette)
    outer.pack(fill="x")

    row1 = ttk.Frame(body, style="Card.TFrame")
    row1.pack(fill="x")
    ttk.Label(row1, text="Decoder", style="Card.TLabel").pack(side="left")
    # a textvariable rather than <<ComboboxSelected>>: the event only fires for
    # a user pick, so a decoder set from the Train tab would not mark the
    # configuration stale
    st["ex_model_choice"] = tk.StringVar(value="")
    st["ex_combo_model"] = ttk.Combobox(
        row1, state="readonly", width=40, textvariable=st["ex_model_choice"]
    )
    st["ex_combo_model"].pack(side="left", padx=(8, 4))
    st["ex_model_choice"].trace_add("write", lambda *_: _schedule_derived(st))
    ttk.Button(
        row1,
        text="Refresh",
        style="CardGhost.TButton",
        command=lambda: refresh_models(st),
    ).pack(side="left")
    # a decoder "trained here" is a run directory: rename it or note what it
    # is for; the shipped decoders are the library's and stay read-only
    st["ex_btn_edit"] = ttk.Button(
        row1,
        text="Edit...",
        style="CardGhost.TButton",
        command=lambda: _edit_model(st),
    )
    st["ex_btn_edit"].pack(side="left", padx=(4, 0))
    st["ex_model_choice"].trace_add("write", lambda *_: _update_edit_button(st))
    st["ex_btn_load"] = ttk.Button(
        row1, text="Load", style="Accent.TButton", command=lambda: _load_model(st)
    )
    st["ex_btn_load"].pack(side="right")
    st["ex_model_info"] = tk.StringVar(value="no decoder loaded")
    st["ex_model_info_label"] = ttk.Label(
        row1, textvariable=st["ex_model_info"], style="Card.Subtle.TLabel"
    )
    st["ex_model_info_label"].pack(side="left", padx=(12, 0))

    ttk.Separator(body).pack(fill="x", pady=8)

    row2 = ttk.Frame(body, style="Card.TFrame")
    row2.pack(fill="x")
    ttk.Label(row2, text="Latent control points", style="Card.TLabel").pack(side="left")
    for axis, var in zip("xyz", st["ex_n_ctrl"]):
        ttk.Label(row2, text=axis, style="Card.Subtle.TLabel").pack(
            side="left", padx=(6, 1)
        )
        widgets.spinbox(row2, var, 2, 6, width=3).pack(side="left")
        var.trace_add("write", lambda *_: _schedule_derived(st))

    ttk.Label(row2, text="     Unit cells", style="Card.TLabel").pack(side="left")
    for axis, var in zip("xyz", st["ex_tiling"]):
        ttk.Label(row2, text=axis, style="Card.Subtle.TLabel").pack(
            side="left", padx=(6, 1)
        )
        widgets.spinbox(row2, var, 1, 8, width=3).pack(side="left")
        var.trace_add("write", lambda *_: _schedule_derived(st))

    st["ex_derived"] = tk.StringVar(value="")
    ttk.Label(row2, textvariable=st["ex_derived"], style="Card.Subtle.TLabel").pack(
        side="left", padx=(16, 0)
    )
    st["ex_stale"] = tk.StringVar(value="")
    ttk.Label(row2, textvariable=st["ex_stale"], style="Card.Warning.TLabel").pack(
        side="right"
    )


def _build_design(st, parent, palette):
    outer, body = widgets.card(
        parent, palette, "Design variables", "the spline control net"
    )
    outer.pack(fill="both", expand=True)

    picker = ttk.Frame(body, style="Card.TFrame")
    picker.pack(fill="x")
    ttk.Label(picker, text="Component", style="Card.TLabel").pack(side="left")
    st["ex_combo_component"] = ttk.Combobox(
        picker, state="readonly", width=6, textvariable=st["ex_component"]
    )
    st["ex_combo_component"].pack(side="left", padx=(6, 12))
    st["ex_combo_component"].bind("<<ComboboxSelected>>", lambda e: _on_component(st))
    ttk.Label(picker, text="Layer k", style="Card.TLabel").pack(side="left")
    st["ex_spin_layer"] = widgets.spinbox(
        picker, st["ex_layer"], 0, 0, width=3, command=lambda: _on_layer(st)
    )
    st["ex_spin_layer"].pack(side="left", padx=(6, 0))
    # command= only fires for the arrows; typing goes through the variable
    st["ex_layer"].trace_add(
        "write",
        lambda *_: runtime.reschedule(st, "ex_job_layer", 250, lambda: _on_layer(st)),
    )

    st["ex_layer_info"] = tk.StringVar(value="")
    ttk.Label(
        body,
        textvariable=st["ex_layer_info"],
        style="Card.Subtle.TLabel",
        wraplength=300,
        justify="left",
    ).pack(fill="x", pady=(6, 2))

    st["ex_grid"] = widgets.ControlPointGrid(
        body, palette, lambda i, j: _on_cell(st, i, j)
    )
    # fill="x" only: the cells size to their content instead of stretching to
    # whatever height the pane happens to have
    st["ex_grid"].frame.pack(fill="x", pady=(2, 6))

    ttk.Separator(body).pack(fill="x", pady=(2, 8))

    st["ex_cell_info"] = tk.StringVar(value="load a decoder to start")
    ttk.Label(
        body,
        textvariable=st["ex_cell_info"],
        style="Card.Subtle.TLabel",
        wraplength=300,
        justify="left",
    ).pack(fill="x")

    editor = ttk.Frame(body, style="Card.TFrame")
    editor.pack(fill="x", pady=(4, 0))
    st["ex_value_label"] = ttk.Label(
        editor, text="+0.000", style="Card.Value.TLabel", width=7
    )
    st["ex_value_label"].pack(side="right")
    st["ex_scale"] = ttk.Scale(
        editor, from_=0.0, to=1.0, command=lambda v: _on_scale(st, v)
    )
    st["ex_scale"].pack(side="left", fill="x", expand=True)

    buttons = ttk.Frame(body, style="Card.TFrame")
    buttons.pack(fill="x", pady=(6, 0))
    ttk.Button(
        buttons,
        text="Set whole part",
        style="CardGhost.TButton",
        command=lambda: _set_all(st),
    ).pack(side="left")
    ttk.Button(
        buttons,
        text="Reset to mean",
        style="CardGhost.TButton",
        command=lambda: _reset_design(st),
    ).pack(side="left")

    st["ex_design_panel"] = body


def _build_geometry(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Geometry")
    outer.pack(fill="both", expand=True)

    # The fixed rows are packed against the bottom first: pack hands the
    # expanding widget everything left over, so a notebook packed first would
    # squeeze the z slider and the metrics off the card entirely.
    strip = ttk.Frame(body, style="Card.TFrame")
    strip.pack(side="bottom", fill="x", pady=(8, 0))
    slider = ttk.Frame(body, style="Card.TFrame")
    slider.pack(side="bottom", fill="x", pady=(8, 0))

    notebook = ttk.Notebook(body)
    notebook.pack(side="top", fill="both", expand=True)
    st["ex_inner_nb"] = notebook

    page_phi = ttk.Frame(notebook, style="Card.TFrame", padding=2)
    page_lat = ttk.Frame(notebook, style="Card.TFrame", padding=2)
    page_mesh = ttk.Frame(notebook, style="Card.TFrame", padding=2)
    notebook.add(page_phi, text="fθ slice")
    notebook.add(page_lat, text="λ(x) field")
    notebook.add(page_mesh, text="Surface")

    dpi = st["dpi"]
    st["ex_fig_phi"], st["ex_canvas_phi"] = widgets.figure_canvas(
        page_phi, palette, (3.4, 2.8), dpi
    )
    st["ex_fig_lat"], st["ex_canvas_lat"] = widgets.figure_canvas(
        page_lat, palette, (3.4, 2.8), dpi
    )
    st["ex_fig_mesh"], st["ex_canvas_mesh"] = widgets.figure_canvas(
        page_mesh, palette, (3.4, 2.8), dpi
    )
    st["ex_ax_phi"] = None
    st["ex_axes_lat"] = []

    ttk.Label(slider, text="Slice z", style="Card.TLabel").pack(side="left")
    st["ex_z_scale"] = ttk.Scale(
        slider, from_=0.0, to=1.0, value=0.5, command=lambda v: _on_z(st, v)
    )
    st["ex_z_scale"].pack(side="left", fill="x", expand=True, padx=8)
    st["ex_z_text"] = tk.StringVar(value="0.50")
    ttk.Label(
        slider, textvariable=st["ex_z_text"], style="Card.Value.TLabel", width=5
    ).pack(side="left")
    ttk.Label(slider, text="   Preview res", style="Card.Subtle.TLabel").pack(
        side="left"
    )
    res = ttk.Combobox(
        slider,
        state="readonly",
        width=4,
        values=SLICE_CHOICES,
        textvariable=st["ex_slice_res"],
    )
    res.pack(side="left", padx=(6, 0))
    # both tiers: the lambda panels are sampled at the same resolution, so
    # redrawing only the geometry leaves them at the old one
    res.bind("<<ComboboxSelected>>", lambda e: _schedule_redraw(st))

    st["ex_metrics"] = {}
    for key, label in (
        ("volume", f"Volume fraction ({VOLUME_RES}³ estimate)"),
        ("phi", "fθ range on this slice"),
        ("zero", "Zero level set"),
    ):
        tile, var, value = widgets.metric(strip, palette, label)
        tile.pack(side="left", padx=(0, 22))
        st["ex_metrics"][key] = (var, value)

    st["ex_view_panel"] = body


def _build_context(st, parent, palette):
    dpi = st["dpi"]

    top_outer, top = widgets.card(
        parent, palette, "Unit cell", "fθ at the selected control point"
    )
    st["ex_cell_caption"] = tk.StringVar(value="")
    ttk.Label(
        top,
        textvariable=st["ex_cell_caption"],
        style="Card.Subtle.TLabel",
        wraplength=250,
        justify="left",
    ).pack(side="bottom", fill="x", pady=(4, 0))
    st["ex_fig_cell"], st["ex_canvas_cell"] = widgets.figure_canvas(
        top, palette, (1.9, 1.9), dpi
    )

    bottom_outer, bottom = widgets.card(
        parent, palette, "Latent coverage", "a box, not the trained set"
    )
    # the coverage strip claims its height first; packed after the unit cell
    # card, the expanding card above leaves it a 44 px sliver and constrained
    # layout gives up on it
    bottom_outer.pack(side="bottom", fill="x", pady=(8, 0))
    top_outer.pack(side="top", fill="both", expand=True)
    st["ex_coverage_note"] = tk.StringVar(value="")
    st["ex_coverage_label"] = ttk.Label(
        bottom,
        textvariable=st["ex_coverage_note"],
        style="Card.Subtle.TLabel",
        wraplength=250,
        justify="left",
    )
    st["ex_coverage_label"].pack(side="bottom", fill="x", pady=(4, 0))
    st["ex_fig_cover"], st["ex_canvas_cover"] = widgets.figure_canvas(
        bottom, palette, (1.9, 1.15), dpi
    )


def _build_footer(st, parent, palette):
    outer, body = widgets.card(parent, palette)
    outer.pack(side="bottom", fill="x", pady=(8, 0))

    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(fill="x")
    ttk.Label(row, text="Surface resolution N_base", style="Card.TLabel").pack(
        side="left"
    )
    spin = widgets.spinbox(row, st["ex_n_base"], 4, 40, width=4)
    spin.pack(side="left", padx=(8, 8))
    st["ex_n_base"].trace_add("write", lambda *_: _schedule_derived(st))
    st["ex_mesh_info"] = tk.StringVar(value="")
    ttk.Label(row, textvariable=st["ex_mesh_info"], style="Card.Subtle.TLabel").pack(
        side="left"
    )

    st["ex_btn_export"] = ttk.Button(
        row,
        text="Export STL...",
        style="CardGhost.TButton",
        command=lambda: _export_stl(st),
        state="disabled",
    )
    st["ex_btn_export"].pack(side="right")
    ttk.Button(
        row,
        text="Save design...",
        style="CardGhost.TButton",
        command=lambda: _save_design(st),
    ).pack(side="right", padx=(0, 4))
    st["ex_btn_window"] = ttk.Button(
        row,
        text="Open 3D window (blocks)",
        style="CardGhost.TButton",
        command=lambda: _open_3d_window(st),
        state="disabled",
    )
    st["ex_btn_window"].pack(side="right", padx=(0, 4))
    st["ex_btn_mesh"] = ttk.Button(
        row,
        text="Extract surface",
        style="Accent.TButton",
        command=lambda: _extract_surface(st),
    )
    st["ex_btn_mesh"].pack(side="right", padx=(0, 8))

    st["ex_status"] = tk.StringVar(value="Pick a decoder and press Load.")
    st["ex_status_label"] = ttk.Label(
        body, textvariable=st["ex_status"], style="Card.Subtle.TLabel"
    )
    st["ex_status_label"].pack(fill="x", pady=(6, 4))

    frame, st["ex_log"] = widgets.log_box(body, palette, height=3)
    frame.pack(fill="x")
    st["error_log"] = st["ex_log"]


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #


def refresh_models(st):
    # Only unit-cell decoders: a planar (2-D) one cannot be tiled into a
    # lattice, and lives on the Explore 2-D tab instead.
    entries = models.list_models(st["ex_runs_dir"], geom_dimension=3)
    labels = {}
    for entry in entries:
        d = "?" if entry.latent_dim is None else entry.latent_dim
        tag = models.entry_tag(entry)
        labels[f"{entry.name} - d={d}, {entry.n_latents} shapes ({tag})"] = entry
    st["ex_models"] = labels
    combo = st["ex_combo_model"]
    # as wide as the longest label: a fixed width clipped the run names
    combo.configure(values=list(labels), width=widgets.combo_width(labels))
    if labels and combo.get() not in labels:
        combo.set(next(iter(labels)))
    _update_edit_button(st)


def select_model(st, name):
    """Point the picker at a run by name; used by the Train tab."""
    for label, entry in st.get("ex_models", {}).items():
        if entry.name == name:
            st["ex_combo_model"].set(label)
            return True
    return False


def _update_edit_button(st):
    """Edit... is for local runs only; a shipped decoder has no directory here."""
    entry = st.get("ex_models", {}).get(st["ex_model_choice"].get())
    editable = entry is not None and entry.source == "run"
    st["ex_btn_edit"].configure(state="normal" if editable else "disabled")


def _edit_model(st):
    """Rename the picked local run or change its notes (``run_editor``)."""
    entry = st.get("ex_models", {}).get(st["ex_combo_model"].get())
    if entry is None or entry.source != "run":
        return None
    return run_editor.open_editor(
        st, st["ex_runs_dir"], entry.name, on_done=lambda name: select_model(st, name)
    )


# --------------------------------------------------------------------------- #
# derived readouts
# --------------------------------------------------------------------------- #


def _schedule_derived(st):
    # a trace fires on every keystroke, including the empty string while the
    # user retypes a number
    runtime.reschedule(st, "ex_job_derived", 250, lambda: _update_derived(st))


def _update_derived(st):
    st["ex_job_derived"] = None
    n_ctrl = [widgets.read_int(v, 2) for v in st["ex_n_ctrl"]]
    tiling = [widgets.read_int(v, 2) for v in st["ex_tiling"]]
    n_cp = int(np.prod(n_ctrl))
    d = st["ex_cps"].shape[1] if "ex_cps" in st else None
    dof = (
        f"{n_cp} x d={d} = {n_cp * d} design variables"
        if d
        else f"{n_cp} control points"
    )
    spans = " x ".join(str(n - 1) for n in n_ctrl)
    cells = " x ".join(str(t) for t in tiling)
    st["ex_derived"].set(
        f"{dof}  ·  {spans} knot spans  ·  {cells} = {int(np.prod(tiling))} cells"
    )

    n_base = widgets.read_int(st["ex_n_base"], DEFAULT_N_BASE)
    grid = [n_base * t + 1 for t in tiling]
    st["ex_mesh_info"].set("→ " + " x ".join(str(g) for g in grid) + " marching cells")

    loaded = st.get("ex_loaded_config")
    if loaded is not None:
        changed = loaded != (st["ex_combo_model"].get(), tuple(n_ctrl), tuple(tiling))
        st["ex_stale"].set("settings changed - press Load to apply" if changed else "")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _load_model(st):
    entry = st.get("ex_models", {}).get(st["ex_combo_model"].get())
    if entry is None:
        messagebox.showerror("No decoder", "Pick a decoder from the list.")
        return
    n_ctrl = tuple(widgets.read_int(v, 2) for v in st["ex_n_ctrl"])
    tiling = tuple(widgets.read_int(v, 2) for v in st["ex_tiling"])

    def work(log):
        log(f"Loading {entry.name}...")
        model = models.load_model(entry)
        trained = models.trained_latents(model)
        control_points = models.default_control_points(model, n_ctrl)
        sdf, param, lattice = models.build_lattice(
            model, tiling, n_ctrl, control_points
        )
        bounds = np.asarray(lattice._get_domain_bounds().detach().cpu(), dtype=float)
        cell = models.unit_cell_sdf(model)
        # the first evaluation builds the lazy kernels; do it here, not on the
        # Tk loop, or the interface freezes for seconds on the first redraw
        models.eval_sdf_slice(sdf, float(bounds[:, 2].mean()), res=32)
        models.eval_cell_slice(cell, trained.mean(axis=0), res=32)
        log(
            f"d={trained.shape[1]}, {len(trained)} trained codes, "
            f"{len(control_points)} control points, tiling {list(tiling)}"
        )
        runtime.ui(
            st,
            lambda: _install_model(
                st,
                entry,
                trained,
                control_points,
                sdf,
                param,
                cell,
                bounds,
                n_ctrl,
                tiling,
            ),
        )

    runtime.run_worker(st, st["ex_btn_load"], st["ex_log"], work)


def _install_model(
    st, entry, trained, control_points, sdf, param, cell, bounds, n_ctrl, tiling
):
    st["ex_entry"] = entry
    st["ex_trained"] = trained
    st["ex_neighbors"] = models.LatentNeighbors(trained)
    st["ex_cps"] = np.asarray(control_points, dtype=float)
    st["ex_sdf"] = sdf
    st["ex_param"] = param
    st["ex_cell_sdf"] = cell
    st["ex_bounds"] = bounds
    st["ex_n_ctrl_loaded"] = tuple(int(n) for n in n_ctrl)
    st["ex_tiling_loaded"] = tuple(int(t) for t in tiling)
    st["ex_ax_phi"] = None
    st["ex_axes_lat"] = []
    st["ex_mesh"] = None
    st["ex_btn_export"].configure(state="disabled")
    st["ex_btn_window"].configure(state="disabled")
    st["ex_loaded_config"] = (st["ex_combo_model"].get(), tuple(n_ctrl), tuple(tiling))
    st["ex_stale"].set("")

    d = st["ex_cps"].shape[1]
    lo, hi = trained.min(axis=0), trained.max(axis=0)
    st["ex_trained_range"] = (lo, hi)
    # A decoder whose stored codes are all identical (the Primitives nets ship
    # exact zeros) would otherwise give every slider 1e-4 of travel.
    degenerate = bool(np.all(hi - lo < 1e-9))
    margin = np.where(hi - lo < 1e-9, 1.0, 0.1 * np.maximum(hi - lo, 1e-9))
    st["ex_slider_range"] = (lo - margin, hi + margin)

    info = (
        f"d={d}  ·  {len(trained)} trained codes  ·  "
        f"{'local run' if entry.source == 'run' else 'pretrained'}"
    )
    if degenerate:
        info += "  ·  stored codes are all identical: latent range unknown"
        widgets.append(
            st["ex_log"],
            f"WARNING: every stored latent code of {entry.name} is the same value. "
            "The decoder carries its shape information elsewhere, so the sliders "
            "below have no trained range to respect and the coverage panel is "
            "uninformative.",
        )
    st["ex_model_info"].set(info)
    st["ex_model_info_label"].configure(
        style="Card.Warning.TLabel" if degenerate else "Card.Subtle.TLabel"
    )

    st["ex_combo_component"].configure(values=[f"λ{_sub(j + 1)}" for j in range(d)])
    st["ex_component"].set(f"λ{_sub(1)}")

    nx, ny, nz = st["ex_n_ctrl_loaded"]
    st["ex_spin_layer"].configure(from_=0, to=max(nz - 1, 0))
    st["ex_layer_applied"] = 0
    st["ex_layer"].set(0)
    st["ex_grid"].build(nx, ny)
    st["ex_selected"] = (0, 0)
    st["ex_grid"].select(0, 0)

    st["ex_z_scale"].configure(from_=float(bounds[0, 2]), to=float(bounds[1, 2]))
    _set_z(st, float(bounds[:, 2].mean()))

    _refresh_grid(st)
    _sync_scale(st)
    _update_derived(st)
    _redraw_geometry(st)
    _redraw_context(st)


# --------------------------------------------------------------------------- #
# design-variable editing
# --------------------------------------------------------------------------- #


def _component(st):
    text = st["ex_component"].get()
    digits = "".join(str("₀₁₂₃₄₅₆₇₈₉".index(ch)) for ch in text if ch in "₀₁₂₃₄₅₆₇₈₉")
    index = max(int(digits) - 1, 0) if digits else 0
    return min(index, st["ex_cps"].shape[1] - 1) if "ex_cps" in st else index


def _layer(st):
    """The active control-point layer, clamped to the loaded grid.

    A ttk.Spinbox does not clamp its textvariable: typing 7 into a 0..2 box
    leaves 7 in the IntVar. Every caller goes through here, so an out-of-range
    number can no longer index past the design vector (IndexError on a click)
    or wrap negatively (silently editing a layer the user is not looking at).
    """
    nz = st.get("ex_n_ctrl_loaded", (1, 1, 1))[2]
    return min(max(widgets.read_int(st["ex_layer"], 0), 0), max(nz - 1, 0))


def _selected_flat(st):
    i, j = st["ex_selected"]
    return models.flat_index(i, j, _layer(st), st["ex_n_ctrl_loaded"])


def _refresh_grid(st):
    """Repaint the cell matrix for the current component and layer."""
    if "ex_cps" not in st:
        return
    comp = _component(st)
    nx, ny, nz = st["ex_n_ctrl_loaded"]
    k = _layer(st)
    values = [
        [
            st["ex_cps"][models.flat_index(i, j, k, st["ex_n_ctrl_loaded"]), comp]
            for i in range(nx)
        ]
        for j in range(ny)
    ]
    lo, hi = st["ex_slider_range"]
    tlo, thi = st["ex_trained_range"]
    st["ex_grid"].update_values(values, lo[comp], hi[comp], tlo[comp], thi[comp])

    _, _, z = models.control_point_position(0, 0, k, st["ex_n_ctrl_loaded"])
    st["ex_layer_info"].set(
        f"layer k = {k} (of 0..{nz - 1}) at z = {z:.2f}" f"  ·  +x right, +y up"
    )
    _describe_selection(st)


def _describe_selection(st):
    i, j = st["ex_selected"]
    k = _layer(st)
    comp = _component(st)
    flat = _selected_flat(st)
    x, y, z = models.control_point_position(i, j, k, st["ex_n_ctrl_loaded"])
    value = float(st["ex_cps"][flat, comp])
    st["ex_cell_info"].set(
        f"control point (i,j,k)=({i},{j},{k})  ·  row {flat}  ·  "
        f"domain ({x:.2f}, {y:.2f}, {z:.2f})"
    )
    st["ex_value_label"].configure(text=f"{value:+.3f}")


def _sync_scale(st):
    """Point the editor slider at the selected control point without a cascade."""
    comp = _component(st)
    lo, hi = st["ex_slider_range"]
    value = float(st["ex_cps"][_selected_flat(st), comp])
    st["ex_muted"] = True
    try:
        st["ex_scale"].configure(from_=float(lo[comp]), to=float(hi[comp]))
        st["ex_scale"].set(value)
    finally:
        st["ex_muted"] = False
    st["ex_value_label"].configure(text=f"{value:+.3f}")


def _on_component(st):
    if "ex_cps" not in st:
        return
    _refresh_grid(st)
    _sync_scale(st)
    _redraw_context(st)


def _on_layer(st):
    if "ex_cps" not in st:
        return
    k = _layer(st)
    # write the clamp back, so the box cannot keep showing a layer that does
    # not exist
    if widgets.read_int(st["ex_layer"], 0) != k:
        st["ex_layer"].set(k)
    if st.get("ex_layer_applied") == k:
        # the variable was written without the layer actually changing - the
        # clamp above, or the reset on load. Moving the slice then would drag
        # the view to a control-point plane nobody asked for; on load that is
        # z = 0, the capped domain face, where there is no geometry at all.
        return
    st["ex_layer_applied"] = k
    # one-directional on purpose: choosing a layer moves the slice to its
    # plane, but dragging the slice does not snap the layer back
    _set_z(st, models.control_point_position(0, 0, k, st["ex_n_ctrl_loaded"])[2])
    _refresh_grid(st)
    _sync_scale(st)
    _schedule_redraw(st)


def _on_cell(st, i, j):
    if "ex_cps" not in st:
        return
    st["ex_selected"] = (i, j)
    _describe_selection(st)
    _sync_scale(st)
    _redraw_context(st)


def _on_scale(st, raw):
    if st.get("ex_muted") or "ex_cps" not in st:
        return
    value = float(raw)
    comp = _component(st)
    st["ex_cps"][_selected_flat(st), comp] = value
    st["ex_value_label"].configure(text=f"{value:+.3f}")
    _refresh_grid(st)
    _invalidate_mesh(st)
    _schedule_redraw(st)


def _invalidate_mesh(st):
    """Drop a surface that no longer belongs to the current design.

    Otherwise Export STL keeps writing the mesh from before the last edit, and
    the Surface page keeps showing it - both silently.
    """
    if st.get("ex_mesh") is None:
        return
    st["ex_mesh"] = None
    st["ex_btn_export"].configure(state="disabled")
    st["ex_btn_window"].configure(state="disabled")
    ax = st.get("ex_ax_mesh")
    if ax is not None and ax in st["ex_fig_mesh"].axes:
        viz.draw_mesh_preview(ax, None, st["palette"])
        st["ex_canvas_mesh"].draw_idle()


def _set_all(st):
    """Make the latent field constant in the active component.

    The uniform lattice is where every design starts, and reaching it by
    dragging every control point to the same spot is busywork. Writing the
    array directly also avoids the widget cascade a per-point ``set()`` would
    trigger.
    """
    if "ex_cps" not in st:
        return
    comp = _component(st)
    st["ex_cps"][:, comp] = float(st["ex_scale"].get())
    _refresh_grid(st)
    _invalidate_mesh(st)
    _schedule_redraw(st)


def _reset_design(st):
    if "ex_cps" not in st or "ex_trained" not in st:
        return
    mean = st["ex_trained"].mean(axis=0)
    st["ex_cps"][:, :] = mean[None, :]
    _refresh_grid(st)
    _sync_scale(st)
    _invalidate_mesh(st)
    _schedule_redraw(st)


def _set_z(st, value):
    st["ex_muted"] = True
    try:
        st["ex_z_scale"].set(float(value))
    finally:
        st["ex_muted"] = False
    st["ex_z_text"].set(f"{float(value):.2f}")


def _on_z(st, raw):
    st["ex_z_text"].set(f"{float(raw):.2f}")
    if st.get("ex_muted"):
        return
    _schedule_redraw(st)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def _schedule_redraw(st):
    """Two tiers, priced by what a redraw actually costs.

    The geometry is what the user cannot predict, so it follows the drag at
    ~120 ms. The latent field, the unit cell and the volume fraction are
    context: each rebuilds panels or samples a 3-D grid, and none of them
    answers a question that changes mid-drag, so they wait for the drag to
    stop.
    """
    runtime.reschedule(st, "ex_job_geom", DEBOUNCE_MS, lambda: _redraw_geometry(st))
    runtime.reschedule(st, "ex_job_ctx", SETTLE_MS, lambda: _redraw_context(st))


def _cancel(st, key):
    """Retire a pending debounce. Called by the redraw it would have run."""
    job = st.get(key)
    st[key] = None
    if job is not None:
        try:
            st["root"].after_cancel(job)
        except tk.TclError:
            pass


def _redraw_geometry(st):
    _cancel(st, "ex_job_geom")
    if "ex_sdf" not in st or st.get("ex_sdf_busy"):
        return

    models.set_control_points(st["ex_param"], st["ex_cps"])
    z = float(st["ex_z_scale"].get())
    res = widgets.read_int(st["ex_slice_res"], DEFAULT_SLICE_RES)
    phi = models.eval_sdf_slice(st["ex_sdf"], z, res=res, bounds=st["ex_bounds"])

    fig = st["ex_fig_phi"]
    if st.get("ex_ax_phi") not in fig.axes:
        st["ex_ax_phi"] = viz.single_axes(fig)
        st["ex_layout_frozen"] = False
    ax = st["ex_ax_phi"]
    viz.draw_sdf_slice(ax, phi, st["ex_bounds"])
    _overlay_control_points(st, ax)
    if not st.get("ex_layout_frozen"):
        widgets.freeze_layout(fig)
        st["ex_layout_frozen"] = True
    st["ex_canvas_phi"].draw_idle()

    lo, hi = float(np.nanmin(phi)), float(np.nanmax(phi))
    st["ex_metrics"]["phi"][0].set(f"{lo:+.3f} .. {hi:+.3f}")
    crosses = lo < 0.0 < hi
    st["ex_metrics"]["zero"][0].set("present" if crosses else "absent")
    st["ex_metrics"]["zero"][1].configure(
        style="Card.Value.TLabel" if crosses else "Card.Danger.TLabel"
    )


def _overlay_control_points(st, ax):
    """Markers for the layer being edited, in the slice's own coordinates."""
    nx, ny, _ = st["ex_n_ctrl_loaded"]
    k = _layer(st)
    comp = _component(st)
    positions, values = [], []
    for j in range(ny):
        for i in range(nx):
            x, y, _ = models.control_point_position(i, j, k, st["ex_n_ctrl_loaded"])
            positions.append((x, y))
            values.append(
                st["ex_cps"][models.flat_index(i, j, k, st["ex_n_ctrl_loaded"]), comp]
            )
    si, sj = st["ex_selected"]
    lo, hi = st["ex_slider_range"]
    viz.draw_control_points(
        ax, positions, values, lo[comp], hi[comp], selected=sj * nx + si
    )


def _redraw_context(st):
    _cancel(st, "ex_job_ctx")
    if "ex_param" not in st or st.get("ex_sdf_busy"):
        return
    comp = _component(st)
    d = st["ex_cps"].shape[1]

    # -- lambda(x) panels, in place: rebuilding them costs an order of
    # magnitude more than pushing new data into the existing images
    z = float(st["ex_z_scale"].get())
    res = widgets.read_int(st["ex_slice_res"], DEFAULT_SLICE_RES)
    latent = models.eval_latent_slice(
        st["ex_param"], z, res=res, bounds=st["ex_bounds"]
    )
    shown = _latent_components(comp, d)
    axes = st.get("ex_axes_lat") or []
    if len(axes) != len(shown) or any(ax not in st["ex_fig_lat"].axes for ax in axes):
        axes = viz.slice_axes(st["ex_fig_lat"], len(shown))
        st["ex_axes_lat"] = axes
        st["ex_lat_frozen"] = False
    tlo, thi = st["ex_trained_range"]
    ranges = [(float(tlo[j]), float(thi[j])) for j in range(d)]
    viz.draw_latent_slices(axes, latent, st["ex_bounds"], ranges, shown)
    if not st.get("ex_lat_frozen"):
        widgets.freeze_layout(st["ex_fig_lat"])
        st["ex_lat_frozen"] = True
    st["ex_canvas_lat"].draw_idle()

    # -- the bare unit cell at the selected control point
    latent_vec = st["ex_cps"][_selected_flat(st)]
    cell = models.eval_cell_slice(st["ex_cell_sdf"], latent_vec, z=0.0, res=CELL_RES)
    ax_cell = st.get("ex_ax_cell")
    if ax_cell not in st["ex_fig_cell"].axes:
        ax_cell = viz.single_axes(st["ex_fig_cell"])
        st["ex_ax_cell"] = ax_cell
        st["ex_cell_frozen"] = False
    viz.draw_sdf_slice(ax_cell, cell, [[-1.0, -1.0], [1.0, 1.0]])
    if not st.get("ex_cell_frozen"):
        widgets.freeze_layout(st["ex_fig_cell"])
        st["ex_cell_frozen"] = True
    st["ex_canvas_cell"].draw_idle()
    st["ex_cell_caption"].set(
        "λ = ["
        + ", ".join(f"{v:+.2f}" for v in latent_vec[:4])
        + (", ..." if len(latent_vec) > 4 else "")
        + "]   mid-plane of [-1,1]³"
    )

    # -- coverage of the active component
    ax_cov = st.get("ex_ax_cover")
    if ax_cov not in st["ex_fig_cover"].axes:
        ax_cov = viz.single_axes(st["ex_fig_cover"])
        st["ex_ax_cover"] = ax_cov
    viz.draw_latent_coverage(
        ax_cov, st["ex_trained"], st["ex_cps"][:, comp], comp, st["palette"]
    )
    st["ex_canvas_cover"].draw_idle()

    # -- volume fraction and the validity verdict
    volume = models.volume_fraction(st["ex_sdf"], st["ex_bounds"], res=VOLUME_RES)
    st["ex_metrics"]["volume"][0].set(f"{volume:.2f}")
    _update_status(st)


def _latent_components(active, d):
    """Which lambda panels to show, in increasing component order.

    Up to three: the active component always, plus its immediate neighbours,
    clamped to the ends. A fixed first three would hide the component the user
    is editing as soon as ``d`` is large.
    """
    if d <= 3:
        return list(range(d))
    start = min(max(active - 1, 0), d - 3)
    return list(range(start, start + 3))


def _update_status(st):
    tlo, thi = st["ex_trained_range"]
    cps = st["ex_cps"]
    outside = int(np.sum((cps < tlo) | (cps > thi)))
    nearest = st["ex_neighbors"].worst_distance(cps)
    comp = _component(st)
    gap, glo, ghi = models.largest_gap(st["ex_trained"][:, comp])
    # the strip shades a gap only once it is worth calling untrained; use the
    # same test in words, or the caption announces a gap with nothing drawn
    span = max(float(thi[comp]) - float(tlo[comp]), 1e-12)
    wide = gap > viz.GAP_FRACTION * span

    parts = [f"{st['ex_entry'].name}: {cps.shape[0]} control points, d={cps.shape[1]}"]
    if outside:
        parts.append(
            f"{outside} value(s) outside the trained range - the decoder learned "
            "nothing there, so the geometry above may be meaningless"
        )
    st["ex_status"].set("   |   ".join(parts))
    st["ex_status_label"].configure(
        style="Card.Danger.TLabel" if outside else "Card.Subtle.TLabel"
    )

    # Kept to two short lines on purpose: the caveat that the shaded band is a
    # bounding box and not the trained set is what the amber gap stripe and
    # this distance are for, so it does not need restating every frame.
    note = f"nearest trained code: {nearest:.3f} away (worst control point)"
    if wide:
        note += f"\nwidest untrained gap here: {glo:+.2f} .. {ghi:+.2f}"
    st["ex_coverage_note"].set(note)
    st["ex_coverage_label"].configure(
        style="Card.Warning.TLabel" if nearest > 0.15 else "Card.Subtle.TLabel"
    )


# --------------------------------------------------------------------------- #
# surface
# --------------------------------------------------------------------------- #


def _extract_surface(st):
    if "ex_sdf" not in st:
        messagebox.showerror("No decoder", "Load a decoder first.")
        return
    models.set_control_points(st["ex_param"], st["ex_cps"])
    sdf = st["ex_sdf"]
    n_base = widgets.read_int(st["ex_n_base"], DEFAULT_N_BASE)

    # The mesher and any redraw evaluate the *same* SDF object, and
    # LatticeSDFStruct writes one latent vector per query point into its
    # microtile on every call: whoever queries second gets "Latent vector
    # shape mismatch". Disabling widgets is not enough on its own - a redraw
    # already scheduled would still fire - so the flag blocks the redraw
    # functions themselves and the pending timers are cancelled here.
    st["ex_sdf_busy"] = True
    st["ex_grid"].enabled = False
    _cancel(st, "ex_job_geom")
    _cancel(st, "ex_job_ctx")

    def work(log):
        log(f"Extracting the surface with N_base={n_base}...")
        mesh = models.surface_mesh(sdf, n_base)
        log(f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        runtime.ui(st, lambda: _show_surface(st, mesh))

    runtime.run_worker(
        st,
        st["ex_btn_mesh"],
        st["ex_log"],
        work,
        lock=(st["ex_design_panel"], st["ex_view_panel"]),
        on_done=lambda: _sdf_released(st),
    )


def _sdf_released(st):
    """Hand the SDF back to the interface once the mesher is done with it."""
    st["ex_sdf_busy"] = False
    st["ex_grid"].enabled = True
    _schedule_redraw(st)


def _show_surface(st, mesh):
    st["ex_mesh"] = mesh
    empty = mesh is None or len(mesh.faces) == 0
    state = "disabled" if empty else "normal"
    st["ex_btn_export"].configure(state=state)
    st["ex_btn_window"].configure(state=state)

    ax = st.get("ex_ax_mesh")
    if ax not in st["ex_fig_mesh"].axes:
        st["ex_fig_mesh"].clear()
        ax = st["ex_fig_mesh"].add_subplot(111, projection="3d")
        st["ex_ax_mesh"] = ax
    viz.draw_mesh_preview(ax, mesh, st["palette"])
    st["ex_canvas_mesh"].draw_idle()
    st["ex_inner_nb"].select(2)

    if empty:
        widgets.append(
            st["ex_log"],
            "The surface came out empty. In order of likelihood: N_base is too "
            "coarse for these struts; the latent values sit where the decoder "
            "was never trained; or the decoder itself is undertrained. Check "
            "the fθ slice - if the 'zero level set' tile says absent, "
            "there is nothing to extract.",
        )


def _open_3d_window(st):
    mesh = st.get("ex_mesh")
    if mesh is None:
        return
    widgets.append(
        st["ex_log"], "Opening the 3D window; it blocks the interface until closed."
    )
    # flush the warning before VTK takes over the event loop
    st["root"].update_idletasks()
    if not viz.show_mesh(mesh, title="lattice"):
        widgets.append(st["ex_log"], "Could not open the 3D window (VTK unavailable).")


def _export_stl(st):
    mesh = st.get("ex_mesh")
    if mesh is None:
        return
    path = filedialog.asksaveasfilename(
        title="Export the surface",
        defaultextension=".stl",
        filetypes=[("STL", "*.stl"), ("OBJ", "*.obj"), ("PLY", "*.ply")],
    )
    if not path:
        return
    try:
        mesh.export(path)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Export failed", str(exc))
        return
    widgets.append(st["ex_log"], f"Surface written to {path}")


def _save_design(st):
    """Write the design vector, which is what this tab actually produces.

    lambda-hat plus the grid it lives on is the starting point an optimisation
    run needs; the STL is only a picture of one evaluation of it.
    """
    if "ex_cps" not in st:
        messagebox.showinfo("No design", "Load a decoder first.")
        return
    path = filedialog.asksaveasfilename(
        title="Save the design variables",
        defaultextension=".json",
        filetypes=[("JSON", "*.json")],
    )
    if not path:
        return
    payload = {
        "decoder": st["ex_entry"].name,
        "source": st["ex_entry"].source,
        "n_ctrl": list(st["ex_n_ctrl_loaded"]),
        "tiling": list(st["ex_tiling_loaded"]),
        "latent_dim": int(st["ex_cps"].shape[1]),
        "control_points": st["ex_cps"].tolist(),
        "control_point_order": "row index = i + nx * (j + ny * k)",
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    widgets.append(st["ex_log"], f"Design written to {path}")
