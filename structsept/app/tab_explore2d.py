"""Explore 2-D tab: what a planar decoder learned, and what its codes kept.

A decoder trained on 2-D samples, such as the ``datagen`` plate with a hole, is
``f_theta(lambda, x, y)``: one latent vector describes one whole shape in the
plane. There is no lattice, no tiling and no latent field over a domain, so
the Explore tab's machinery does not apply. What does apply is the question the
planar experiment was set up to answer: the shapes were generated from a few
parameters (``x_c, y_c, r`` for the plate), the decoder never saw them and had
to invent its own codes - how much of those parameters survived?

The panels follow that question:

    which decoder                          -> header
    where are the trained codes, coloured
    by what generated each shape, and
    where am I                             -> left, the latent map; click or
                                              drag on it to move lambda
    what shape is this lambda              -> centre, f_theta as material/void,
                                              the nearest training shape's true
                                              boundary on top
    which parameters did the codes keep    -> right, one panel per parameter
                                              with a nearest-neighbour R^2, and
                                              the per-shape fit error

The parameters come from the ``params.csv`` next to the dataset, matched to the
codes through the latent index the trainer recorded. A run trained on a dataset
without that table still loads; only the colouring is missing.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from structsept.app import models, runtime, viz, widgets

DEBOUNCE_MS = 60
DEFAULT_RES = 128
RES_CHOICES = ("96", "128", "192", "256")
NO_COLOUR = "none"
ERROR_COLOUR = "fit error"
# A lambda farther from every trained code than this many typical
# code-to-code spacings is extrapolation, and is flagged as such.
FAR_FACTOR = 2.0


def _sub(n):
    return viz._sub(n)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def build(st, parent, runs_dir):
    """Populate the Explore 2-D tab. ``parent`` is an empty padded frame."""
    st["e2_runs_dir"] = Path(runs_dir)
    st["e2_colour"] = tk.StringVar(value=NO_COLOUR)
    st["e2_axes_choice"] = tk.StringVar(value="")
    st["e2_component"] = tk.StringVar(value="")
    st["e2_res"] = tk.StringVar(value=str(DEFAULT_RES))
    st["e2_show_truth"] = tk.BooleanVar(value=True)
    st["e2_muted"] = False

    palette = st["palette"]
    _build_header(st, parent, palette)
    _build_footer(st, parent, palette)

    paned = ttk.PanedWindow(parent, orient="horizontal")
    paned.pack(side="top", fill="both", expand=True, pady=(8, 0))
    st["e2_paned"] = paned
    left, centre, right = ttk.Frame(paned), ttk.Frame(paned), ttk.Frame(paned)
    paned.add(left, weight=2)
    paned.add(centre, weight=3)
    paned.add(right, weight=2)

    _build_latent(st, left, palette)
    _build_shape(st, centre, palette)
    _build_params(st, right, palette)

    paned.bind("<Map>", lambda e: _place_sashes(st), add="+")
    refresh_models(st)
    return parent


def _place_sashes(st):
    if st.get("e2_sashes_placed"):
        return
    paned = st["e2_paned"]
    try:
        total = paned.winfo_width()
        if total < 200:
            return
        paned.sashpos(0, int(total * 0.31))
        paned.sashpos(1, int(total * 0.68))
        st["e2_sashes_placed"] = True
    except tk.TclError:
        pass


def _build_header(st, parent, palette):
    outer, body = widgets.card(parent, palette)
    outer.pack(fill="x")
    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(fill="x")
    ttk.Label(row, text="Planar decoder", style="Card.TLabel").pack(side="left")
    st["e2_combo_model"] = ttk.Combobox(row, state="readonly", width=44)
    st["e2_combo_model"].pack(side="left", padx=(8, 4))
    ttk.Button(
        row,
        text="Refresh",
        style="CardGhost.TButton",
        command=lambda: refresh_models(st),
    ).pack(side="left")
    st["e2_btn_load"] = ttk.Button(
        row, text="Load", style="Accent.TButton", command=lambda: _load_model(st)
    )
    st["e2_btn_load"].pack(side="right")
    st["e2_model_info"] = tk.StringVar(value="no decoder loaded")
    st["e2_model_info_label"] = ttk.Label(
        row, textvariable=st["e2_model_info"], style="Card.Subtle.TLabel"
    )
    st["e2_model_info_label"].pack(side="left", padx=(12, 0))


def _build_latent(st, parent, palette):
    outer, body = widgets.card(
        parent, palette, "Latent space", "trained codes · click or drag to move λ"
    )
    outer.pack(fill="both", expand=True)

    # fixed rows first, against the bottom: the expanding canvas takes the rest
    buttons = ttk.Frame(body, style="Card.TFrame")
    buttons.pack(side="bottom", fill="x", pady=(6, 0))
    ttk.Button(
        buttons,
        text="Snap to nearest shape",
        style="CardGhost.TButton",
        command=lambda: _snap_to_nearest(st),
    ).pack(side="left")
    ttk.Button(
        buttons,
        text="Reset to mean",
        style="CardGhost.TButton",
        command=lambda: _reset_to_mean(st),
    ).pack(side="left", padx=(4, 0))

    editor = ttk.Frame(body, style="Card.TFrame")
    editor.pack(side="bottom", fill="x", pady=(6, 0))
    st["e2_combo_component"] = ttk.Combobox(
        editor, state="readonly", width=5, textvariable=st["e2_component"]
    )
    st["e2_combo_component"].pack(side="left")
    st["e2_combo_component"].bind("<<ComboboxSelected>>", lambda e: _sync_scale(st))
    st["e2_value_label"] = ttk.Label(
        editor, text="+0.000", style="Card.Value.TLabel", width=7
    )
    st["e2_value_label"].pack(side="right")
    st["e2_scale"] = ttk.Scale(
        editor, from_=0.0, to=1.0, command=lambda v: _on_scale(st, v)
    )
    st["e2_scale"].pack(side="left", fill="x", expand=True, padx=(8, 8))

    st["e2_map_caption"] = tk.StringVar(value="")
    ttk.Label(
        body,
        textvariable=st["e2_map_caption"],
        style="Card.Subtle.TLabel",
        wraplength=330,
        justify="left",
    ).pack(side="bottom", fill="x", pady=(4, 0))

    pickers = ttk.Frame(body, style="Card.TFrame")
    pickers.pack(side="top", fill="x")
    ttk.Label(pickers, text="Colour by", style="Card.TLabel").pack(side="left")
    st["e2_combo_colour"] = ttk.Combobox(
        pickers, state="readonly", width=11, textvariable=st["e2_colour"]
    )
    st["e2_combo_colour"].pack(side="left", padx=(6, 12))
    st["e2_combo_colour"].bind("<<ComboboxSelected>>", lambda e: _rebuild_map(st))
    ttk.Label(pickers, text="Axes", style="Card.TLabel").pack(side="left")
    st["e2_combo_axes"] = ttk.Combobox(
        pickers, state="readonly", width=9, textvariable=st["e2_axes_choice"]
    )
    st["e2_combo_axes"].pack(side="left", padx=(6, 0))
    st["e2_combo_axes"].bind("<<ComboboxSelected>>", lambda e: _on_axes(st))

    st["e2_fig_map"], st["e2_canvas_map"] = widgets.figure_canvas(
        body, palette, (3.2, 3.0), st["dpi"]
    )
    st["e2_ax_map"] = None
    canvas = st["e2_canvas_map"]
    canvas.mpl_connect("button_press_event", lambda e: _on_map_mouse(st, e))
    canvas.mpl_connect("motion_notify_event", lambda e: _on_map_mouse(st, e))


def _build_shape(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Shape", "fθ(λ, x, y) on [-1, 1]²")
    outer.pack(fill="both", expand=True)

    strip = ttk.Frame(body, style="Card.TFrame")
    strip.pack(side="bottom", fill="x", pady=(8, 0))
    st["e2_metrics"] = {}
    for key, label in (
        ("nearest", "Nearest trained shape"),
        ("distance", "Latent distance to it"),
        ("fit", "Its fit error |Δφ|"),
        ("material", "Material share of the box"),
    ):
        tile, var, value = widgets.metric(strip, palette, label)
        tile.pack(side="left", padx=(0, 20))
        st["e2_metrics"][key] = (var, value)

    st["e2_shape_caption"] = tk.StringVar(value="")
    ttk.Label(
        body,
        textvariable=st["e2_shape_caption"],
        style="Card.Subtle.TLabel",
        wraplength=520,
        justify="left",
    ).pack(side="bottom", fill="x", pady=(6, 0))

    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(side="bottom", fill="x", pady=(6, 0))
    ttk.Checkbutton(
        row,
        text="Overlay the nearest training shape's true boundary",
        variable=st["e2_show_truth"],
        style="Card.TCheckbutton",
        command=lambda: _schedule_redraw(st),
    ).pack(side="left")
    res = ttk.Combobox(
        row, state="readonly", width=4, values=RES_CHOICES, textvariable=st["e2_res"]
    )
    res.pack(side="right")
    res.bind("<<ComboboxSelected>>", lambda e: _schedule_redraw(st))
    ttk.Label(row, text="Preview res", style="Card.Subtle.TLabel").pack(
        side="right", padx=(0, 6)
    )

    st["e2_fig_field"], st["e2_canvas_field"] = widgets.figure_canvas(
        body, palette, (3.6, 3.4), st["dpi"]
    )
    st["e2_ax_field"] = None


def _build_params(st, parent, palette):
    outer, body = widgets.card(
        parent, palette, "What the codes kept", "one panel per generating parameter"
    )
    outer.pack(fill="both", expand=True)
    st["e2_params_note"] = tk.StringVar(
        value="R²: how well a shape's nearest codes predict its parameter - "
        "near 1 kept, near 0 or below lost."
    )
    st["e2_params_note_label"] = ttk.Label(
        body,
        textvariable=st["e2_params_note"],
        style="Card.Subtle.TLabel",
        wraplength=330,
        justify="left",
    )
    st["e2_params_note_label"].pack(side="bottom", fill="x", pady=(6, 0))
    st["e2_fig_params"], st["e2_canvas_params"] = widgets.figure_canvas(
        body, palette, (3.2, 3.2), st["dpi"]
    )


def _build_footer(st, parent, palette):
    outer, body = widgets.card(parent, palette)
    outer.pack(side="bottom", fill="x", pady=(8, 0))
    st["e2_status"] = tk.StringVar(
        value="Train a decoder on a 2-D dataset (Train tab), then Load it here."
    )
    st["e2_status_label"] = ttk.Label(
        body, textvariable=st["e2_status"], style="Card.Subtle.TLabel"
    )
    st["e2_status_label"].pack(fill="x", pady=(0, 4))
    frame, st["e2_log"] = widgets.log_box(body, palette, height=3)
    frame.pack(fill="x")


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #


def refresh_models(st):
    entries = models.list_models(st["e2_runs_dir"], geom_dimension=2)
    labels = {}
    for entry in entries:
        d = "?" if entry.latent_dim is None else entry.latent_dim
        labels[f"{entry.name} - d={d}, {entry.n_latents} shapes"] = entry
    st["e2_models"] = labels
    combo = st["e2_combo_model"]
    combo.configure(values=list(labels))
    if labels and combo.get() not in labels:
        combo.set(next(iter(labels)))
    if not labels and "e2_model" not in st:
        st["e2_model_info"].set(
            "no planar decoder yet - train one on a 2-D dataset "
            "(uv run python -m datagen.make_plate_hole --dim 2)"
        )


def select_model(st, name):
    """Point the picker at a run by name; used by the Train tab."""
    for label, entry in st.get("e2_models", {}).items():
        if entry.name == name:
            st["e2_combo_model"].set(label)
            return True
    return False


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _load_model(st):
    entry = st.get("e2_models", {}).get(st["e2_combo_model"].get())
    if entry is None:
        messagebox.showerror("No decoder", "Pick a planar decoder from the list.")
        return

    def work(log):
        log(f"Loading {entry.name}...")
        model = models.load_model(entry)
        if models.geom_dimension(model) != 2:
            raise ValueError(
                f"{entry.name} takes {models.geom_dimension(model)}-D points; "
                "open it on the Explore tab."
            )
        trained = models.trained_latents(model)
        shapes = models.run_shapes(entry.ref)
        if shapes["names"] and len(shapes["names"]) != len(trained):
            log(
                f"WARNING: {len(trained)} codes but {len(shapes['names'])} "
                "training shapes on record; the parameters cannot be matched "
                "to the codes and are not shown."
            )
            shapes["params"], shapes["param_names"] = {}, []
        specs = models._read_json(Path(entry.ref) / "specs.json")
        clamp = float(specs.get("ClampingDistance", 0.1))
        errors = models.reconstruction_errors(
            model, trained, shapes["npz"], clamp=clamp
        )
        scores = {
            name: models.neighbour_r2(trained, shapes["params"][name])
            for name in shapes["param_names"]
        }
        # the first forward pass builds lazy kernels; pay for it here, not on
        # the Tk loop
        models.eval_field_2d(model, trained.mean(axis=0), res=32)
        log(
            f"d={trained.shape[1]}, {len(trained)} trained codes"
            + (
                f", parameters {', '.join(shapes['param_names'])}"
                if shapes["param_names"]
                else ", no parameter table"
            )
        )
        if np.isfinite(errors).any():
            log(
                f"fit error |Δφ| per shape: median {np.nanmedian(errors):.4f}, "
                f"worst {np.nanmax(errors):.4f}"
            )
        for name, score in scores.items():
            log(f"R² {name}: {score:.3f}")
        runtime.ui(
            st,
            lambda: _install_model(st, entry, model, trained, shapes, errors, scores),
        )

    runtime.run_worker(st, st["e2_btn_load"], st["e2_log"], work)


def _install_model(st, entry, model, trained, shapes, errors, scores):
    d = trained.shape[1]
    st["e2_entry"] = entry
    st["e2_model"] = model
    st["e2_trained"] = trained
    st["e2_neighbors"] = models.LatentNeighbors(trained)
    st["e2_shapes"] = shapes
    st["e2_errors"] = errors
    st["e2_scores"] = scores
    st["e2_truth_cache"] = {}
    st["e2_latent"] = trained.mean(axis=0).copy()
    st["e2_ax_field"] = None

    lo, hi = trained.min(axis=0), trained.max(axis=0)
    margin = np.where(hi - lo < 1e-9, 1.0, 0.15 * np.maximum(hi - lo, 1e-9))
    st["e2_slider_range"] = (lo - margin, hi + margin)
    st["e2_spacing"] = _typical_spacing(trained)

    comps = [f"λ{_sub(j + 1)}" for j in range(d)]
    st["e2_combo_component"].configure(values=comps)
    st["e2_component"].set(comps[0])

    pairs = {}
    if d == 1:
        pairs[comps[0]] = (0, 0)
    else:
        for a in range(d):
            for b in range(a + 1, d):
                pairs[f"{comps[a]} · {comps[b]}"] = (a, b)
    st["e2_axes_map"] = pairs
    st["e2_combo_axes"].configure(values=list(pairs))
    st["e2_axes_choice"].set(next(iter(pairs)))
    widgets.set_enabled(st["e2_combo_axes"], d > 2)

    colours = [NO_COLOUR] + list(shapes["param_names"])
    if np.isfinite(errors).any():
        colours.append(ERROR_COLOUR)
    st["e2_combo_colour"].configure(values=colours)
    st["e2_colour"].set(colours[1] if len(colours) > 1 else NO_COLOUR)

    st["e2_model_info"].set(
        f"d={d}  ·  {len(trained)} trained shapes  ·  "
        + (
            f"dataset {shapes['dataset_dir'].name}"
            if shapes["dataset_dir"] is not None
            else "dataset unknown"
        )
    )
    st["e2_map_caption"].set(
        "Each dot is one training shape. Red cross: the λ drawn in the centre; "
        "amber ring: the trained shape closest to it."
    )

    _sync_scale(st)
    _rebuild_map(st)
    _draw_params(st)
    _redraw_field(st)


def _typical_spacing(trained) -> float:
    """Median distance from a trained code to its nearest neighbour."""
    if len(trained) < 2:
        return float("nan")
    try:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(trained).query(trained, k=2)
        return float(np.median(dists[:, 1]))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# moving lambda
# --------------------------------------------------------------------------- #


def _component(st) -> int:
    text = st["e2_component"].get()
    digits = "".join(str("₀₁₂₃₄₅₆₇₈₉".index(ch)) for ch in text if ch in "₀₁₂₃₄₅₆₇₈₉")
    index = max(int(digits) - 1, 0) if digits else 0
    return min(index, st["e2_trained"].shape[1] - 1) if "e2_trained" in st else index


def _comps(st) -> tuple[int, int]:
    return st.get("e2_axes_map", {}).get(st["e2_axes_choice"].get(), (0, 1))


def _sync_scale(st):
    """Put the slider on the active component without firing its callback."""
    if "e2_latent" not in st:
        return
    comp = _component(st)
    lo, hi = st["e2_slider_range"]
    value = float(st["e2_latent"][comp])
    st["e2_muted"] = True
    try:
        st["e2_scale"].configure(from_=float(lo[comp]), to=float(hi[comp]))
        st["e2_scale"].set(value)
    finally:
        st["e2_muted"] = False
    st["e2_value_label"].configure(text=f"{value:+.3f}")


def _on_scale(st, raw):
    if st.get("e2_muted") or "e2_latent" not in st:
        return
    value = float(raw)
    st["e2_latent"][_component(st)] = value
    st["e2_value_label"].configure(text=f"{value:+.3f}")
    _moved(st)


def _on_map_mouse(st, event):
    """Click, or drag with the left button held, to set λ on the map axes."""
    if "e2_latent" not in st or event.inaxes is None:
        return
    if event.inaxes is not st.get("e2_ax_map"):
        return
    if event.name == "motion_notify_event" and event.button != 1:
        return
    if event.xdata is None:
        return
    a, b = _comps(st)
    st["e2_latent"][a] = float(event.xdata)
    if st["e2_trained"].shape[1] > 1 and event.ydata is not None:
        st["e2_latent"][b] = float(event.ydata)
    _sync_scale(st)
    _moved(st)


def _snap_to_nearest(st):
    if "e2_latent" not in st:
        return
    index, _ = st["e2_neighbors"].nearest(st["e2_latent"])
    st["e2_latent"] = st["e2_trained"][index].copy()
    _sync_scale(st)
    _moved(st)


def _reset_to_mean(st):
    if "e2_latent" not in st:
        return
    st["e2_latent"] = st["e2_trained"].mean(axis=0).copy()
    _sync_scale(st)
    _moved(st)


def _on_axes(st):
    """Another pair of components: both the map and the panels follow."""
    if "e2_trained" not in st:
        return
    _rebuild_map(st)
    _draw_params(st)


def _moved(st):
    """λ changed: the marker follows at once, the field on a short debounce."""
    _redraw_marker(st)
    _schedule_redraw(st)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def _schedule_redraw(st):
    runtime.reschedule(st, "e2_job_field", DEBOUNCE_MS, lambda: _redraw_field(st))


def _cancel(st, key):
    job = st.get(key)
    st[key] = None
    if job is not None:
        try:
            st["root"].after_cancel(job)
        except tk.TclError:
            pass


def _colour(st):
    """The values the map is coloured by: ``(values, label, is_error)``."""
    choice = st["e2_colour"].get()
    if choice == ERROR_COLOUR:
        return st["e2_errors"], "fit error |Δφ|", True
    params = st["e2_shapes"]["params"]
    if choice in params:
        return params[choice], choice, False
    return None, None, False


def _rebuild_map(st):
    if "e2_trained" not in st:
        return
    values, label, is_error = _colour(st)
    if st["e2_trained"].shape[1] == 1 and values is None:
        # a one-component map needs something on its y axis
        values, label = np.zeros(len(st["e2_trained"])), None
    fig = st["e2_fig_map"]
    st["e2_ax_map"] = viz.draw_latent_map(
        fig, st["e2_trained"], _comps(st), values, label, error=is_error
    )
    widgets.freeze_layout(fig)
    _redraw_marker(st)


def _redraw_marker(st):
    ax = st.get("e2_ax_map")
    if ax is None or "e2_latent" not in st:
        return
    a, b = _comps(st)
    latent, trained = st["e2_latent"], st["e2_trained"]
    index, _ = st["e2_neighbors"].nearest(latent)
    if trained.shape[1] == 1:
        values, _, _ = _colour(st)
        y = 0.0 if values is None else float(values[index])
        viz.draw_latent_marker(ax, float(latent[a]), None, (trained[index, a], y))
    else:
        viz.draw_latent_marker(
            ax,
            float(latent[a]),
            float(latent[b]),
            (trained[index, a], trained[index, b]),
        )
    st["e2_canvas_map"].draw_idle()


def _draw_params(st):
    shapes = st["e2_shapes"]
    names = list(shapes["param_names"])
    columns = [(name, shapes["params"][name]) for name in names]
    scores = [st["e2_scores"].get(name, float("nan")) for name in names]
    if columns and np.isfinite(st["e2_errors"]).any():
        columns.append(("error |Δφ|", st["e2_errors"]))
        scores.append(float("nan"))
    viz.draw_parameter_panels(
        st["e2_fig_params"], st["e2_trained"], _comps(st), columns, scores
    )
    widgets.freeze_layout(st["e2_fig_params"])
    st["e2_canvas_params"].draw_idle()

    kept = [f"{n} {s:.2f}" for n, s in st["e2_scores"].items() if np.isfinite(s)]
    if kept:
        st["e2_params_note"].set(
            "R² (nearest codes predicting the parameter; 1 kept, ≤ 0 lost): "
            + "  ·  ".join(kept)
        )


def _redraw_field(st):
    _cancel(st, "e2_job_field")
    if "e2_model" not in st:
        return
    res = widgets.read_int(st["e2_res"], DEFAULT_RES)
    latent = st["e2_latent"]
    field = models.eval_field_2d(st["e2_model"], latent, res=res)

    fig = st["e2_fig_field"]
    if st.get("e2_ax_field") not in fig.axes:
        st["e2_ax_field"] = viz.single_axes(fig)
        st["e2_field_frozen"] = False
    ax = st["e2_ax_field"]
    viz.draw_sdf_slice(ax, field, models.PLANE_BOUNDS)

    index, distance = st["e2_neighbors"].nearest(latent)
    shapes = st["e2_shapes"]
    truth = np.zeros((0, 2))
    if st["e2_show_truth"].get() and 0 <= index < len(shapes["npz"]):
        cache = st["e2_truth_cache"]
        if index not in cache:
            cache[index] = models.boundary_samples(shapes["npz"][index])
        truth = cache[index]
    viz.draw_boundary_points(ax, truth)

    if not st.get("e2_field_frozen"):
        widgets.freeze_layout(fig)
        st["e2_field_frozen"] = True
    st["e2_canvas_field"].draw_idle()

    _update_readouts(st, field, index, distance)


def _update_readouts(st, field, index, distance):
    shapes = st["e2_shapes"]
    metrics = st["e2_metrics"]
    name = shapes["names"][index] if 0 <= index < len(shapes["names"]) else f"#{index}"
    metrics["nearest"][0].set(f"#{index}")
    metrics["distance"][0].set(f"{distance:.3f}")
    error = st["e2_errors"][index] if 0 <= index < len(st["e2_errors"]) else np.nan
    metrics["fit"][0].set("--" if not np.isfinite(error) else f"{error:.4f}")
    metrics["material"][0].set(f"{float(np.mean(field < 0.0)):.2f}")

    params = shapes["params"]
    described = [
        f"{p} {params[p][index]:.3f}"
        for p in shapes["param_names"]
        if np.isfinite(params[p][index])
    ]
    st["e2_shape_caption"].set(
        f"nearest training shape: {name}"
        + (f"   ({',  '.join(described)})" if described else "")
        + "\namber dots: its stored samples within 0.02 of its true surface"
    )

    spacing = st.get("e2_spacing", float("nan"))
    far = np.isfinite(spacing) and distance > FAR_FACTOR * spacing
    latent = st["e2_latent"]
    text = "λ = [" + ", ".join(f"{v:+.3f}" for v in latent[:6])
    text += ", ...]" if len(latent) > 6 else "]"
    if far:
        text += (
            f"   |   {distance / spacing:.1f}x the typical spacing from the nearest "
            "trained code: the decoder is extrapolating here"
        )
    st["e2_status"].set(text)
    st["e2_status_label"].configure(
        style="Card.Warning.TLabel" if far else "Card.Subtle.TLabel"
    )
    metrics["distance"][1].configure(
        style="Card.Danger.TLabel" if far else "Card.Value.TLabel"
    )
