"""Explore 2-D tab: drive a planar decoder f_theta(lambda, x, y) by hand.

The 2-D counterpart of the Explore tab, laid out the same way. A decoder trained
on 2-D samples (``datagen --dim 2``) describes one whole shape per latent
vector, so there is no lattice and no control net: the design variables are
the components of lambda itself, one slider each. Moving them shows how
lambda_1, lambda_2, ... change the shape.

    which decoder                      -> header
    the design variables               -> left, one slider per component of λ
    what shape is this now             -> centre, f_theta as material/void
    how much material is there         -> centre, area fraction
    is λ still supported               -> right, latent coverage per component
                                          and the distance to the nearest
                                          trained code
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from structsept.app import models, runtime, viz, widgets

DEBOUNCE_MS = 120
SETTLE_MS = 400
DEFAULT_RES = 128
RES_CHOICES = ("96", "128", "160", "192")


def _sub(n):
    return viz._sub(n)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def build(st, parent, runs_dir):
    """Populate the Explore 2-D tab. ``parent`` is an empty padded frame."""
    st["e2_runs_dir"] = Path(runs_dir)
    st["e2_res"] = tk.StringVar(value=str(DEFAULT_RES))
    st["e2_muted"] = False
    st["e2_sliders"] = []

    palette = st["palette"]
    _build_header(st, parent, palette)
    _build_footer(st, parent, palette)

    paned = ttk.PanedWindow(parent, orient="horizontal")
    paned.pack(side="top", fill="both", expand=True, pady=(8, 0))
    st["e2_paned"] = paned
    left, centre, right = ttk.Frame(paned), ttk.Frame(paned), ttk.Frame(paned)
    paned.add(left, weight=0)
    paned.add(centre, weight=3)
    paned.add(right, weight=1)

    _build_design(st, left, palette)
    _build_geometry(st, centre, palette)
    _build_context(st, right, palette)

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
        paned.sashpos(0, int(total * 0.25))
        paned.sashpos(1, int(total * 0.755))
        st["e2_sashes_placed"] = True
    except tk.TclError:
        pass


def _build_header(st, parent, palette):
    outer, body = widgets.card(parent, palette)
    outer.pack(fill="x")
    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(fill="x")
    ttk.Label(row, text="Decoder", style="Card.TLabel").pack(side="left")
    st["e2_model_choice"] = tk.StringVar(value="")
    st["e2_combo_model"] = ttk.Combobox(
        row, state="readonly", width=40, textvariable=st["e2_model_choice"]
    )
    st["e2_combo_model"].pack(side="left", padx=(8, 4))
    st["e2_model_choice"].trace_add("write", lambda *_: _update_stale(st))
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
    st["e2_stale"] = tk.StringVar(value="")
    ttk.Label(row, textvariable=st["e2_stale"], style="Card.Warning.TLabel").pack(
        side="right", padx=(0, 12)
    )
    st["e2_model_info"] = tk.StringVar(value="no decoder loaded")
    st["e2_model_info_label"] = ttk.Label(
        row, textvariable=st["e2_model_info"], style="Card.Subtle.TLabel"
    )
    st["e2_model_info_label"].pack(side="left", padx=(12, 0))


def _build_design(st, parent, palette):
    outer, body = widgets.card(
        parent, palette, "Design variables", "the latent vector λ"
    )
    outer.pack(fill="both", expand=True)

    st["e2_slider_frame"] = ttk.Frame(body, style="Card.TFrame")
    st["e2_slider_frame"].pack(fill="x")

    ttk.Separator(body).pack(fill="x", pady=8)
    st["e2_design_info"] = tk.StringVar(value="load a decoder to start")
    ttk.Label(
        body,
        textvariable=st["e2_design_info"],
        style="Card.Subtle.TLabel",
        wraplength=300,
        justify="left",
    ).pack(fill="x")

    buttons = ttk.Frame(body, style="Card.TFrame")
    buttons.pack(fill="x", pady=(6, 0))
    ttk.Button(
        buttons,
        text="Reset to mean",
        style="CardGhost.TButton",
        command=lambda: _reset_to_mean(st),
    ).pack(side="left")


def _build_geometry(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Geometry")
    outer.pack(fill="both", expand=True)

    # fixed rows against the bottom first, so the canvas cannot squeeze them out
    strip = ttk.Frame(body, style="Card.TFrame")
    strip.pack(side="bottom", fill="x", pady=(8, 0))
    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(side="bottom", fill="x", pady=(8, 0))

    st["e2_fig_phi"], st["e2_canvas_phi"] = widgets.figure_canvas(
        body, palette, (3.4, 2.8), st["dpi"]
    )
    st["e2_ax_phi"] = None

    ttk.Label(row, text="fθ(λ, x, y) on [-1, 1]²", style="Card.Subtle.TLabel").pack(
        side="left"
    )
    res = ttk.Combobox(
        row, state="readonly", width=4, values=RES_CHOICES, textvariable=st["e2_res"]
    )
    res.pack(side="right")
    res.bind("<<ComboboxSelected>>", lambda e: _schedule_redraw(st))
    ttk.Label(row, text="Preview res", style="Card.Subtle.TLabel").pack(
        side="right", padx=(0, 6)
    )

    st["e2_metrics"] = {}
    for key, label in (
        ("area", "Material fraction (area)"),
        ("phi", "fθ range"),
        ("zero", "Zero level set"),
    ):
        tile, var, value = widgets.metric(strip, palette, label)
        tile.pack(side="left", padx=(0, 22))
        st["e2_metrics"][key] = (var, value)


def _build_context(st, parent, palette):
    outer, body = widgets.card(
        parent, palette, "Latent coverage", "a box, not the trained set"
    )
    outer.pack(fill="both", expand=True)
    st["e2_coverage_note"] = tk.StringVar(value="")
    st["e2_coverage_label"] = ttk.Label(
        body,
        textvariable=st["e2_coverage_note"],
        style="Card.Subtle.TLabel",
        wraplength=250,
        justify="left",
    )
    st["e2_coverage_label"].pack(side="bottom", fill="x", pady=(4, 0))
    st["e2_fig_cover"], st["e2_canvas_cover"] = widgets.figure_canvas(
        body, palette, (1.9, 2.4), st["dpi"]
    )
    st["e2_axes_cover"] = []


def _build_footer(st, parent, palette):
    outer, body = widgets.card(parent, palette)
    outer.pack(side="bottom", fill="x", pady=(8, 0))
    st["e2_status"] = tk.StringVar(
        value="Pick a decoder trained on a 2-D dataset and press Load."
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
        labels[f"{entry.name} - d={d}, {entry.n_latents} shapes (trained here)"] = entry
    st["e2_models"] = labels
    combo = st["e2_combo_model"]
    combo.configure(values=list(labels))
    if labels and combo.get() not in labels:
        combo.set(next(iter(labels)))


def select_model(st, name):
    """Point the picker at a run by name; used by the Train tab."""
    for label, entry in st.get("e2_models", {}).items():
        if entry.name == name:
            st["e2_combo_model"].set(label)
            return True
    return False


def _update_stale(st):
    loaded = st.get("e2_loaded_choice")
    if loaded is not None:
        changed = st["e2_model_choice"].get() != loaded
        st["e2_stale"].set("decoder changed - press Load to apply" if changed else "")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _load_model(st):
    entry = st.get("e2_models", {}).get(st["e2_combo_model"].get())
    if entry is None:
        messagebox.showerror("No decoder", "Pick a decoder from the list.")
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
        # the first forward pass builds lazy kernels; pay for it here, not on
        # the Tk loop
        models.eval_field_2d(model, trained.mean(axis=0), res=32)
        log(f"d={trained.shape[1]}, {len(trained)} trained codes")
        runtime.ui(st, lambda: _install_model(st, entry, model, trained))

    runtime.run_worker(st, st["e2_btn_load"], st["e2_log"], work)


def _install_model(st, entry, model, trained):
    d = trained.shape[1]
    st["e2_entry"] = entry
    st["e2_model"] = model
    st["e2_trained"] = trained
    st["e2_neighbors"] = models.LatentNeighbors(trained)
    st["e2_latent"] = trained.mean(axis=0).copy()
    st["e2_ax_phi"] = None
    st["e2_axes_cover"] = []
    st["e2_loaded_choice"] = st["e2_model_choice"].get()
    st["e2_stale"].set("")

    lo, hi = trained.min(axis=0), trained.max(axis=0)
    st["e2_trained_range"] = (lo, hi)
    margin = np.where(hi - lo < 1e-9, 1.0, 0.1 * np.maximum(hi - lo, 1e-9))
    st["e2_slider_range"] = (lo - margin, hi + margin)

    st["e2_model_info"].set(f"d={d}  ·  {len(trained)} trained codes  ·  local run")
    st["e2_design_info"].set(
        "trained range:  "
        + "   ".join(
            f"λ{_sub(j + 1)} [{lo[j]:+.2f}, {hi[j]:+.2f}]" for j in range(min(d, 4))
        )
        + ("  ..." if d > 4 else "")
    )
    _build_sliders(st, d)
    _redraw_geometry(st)
    _redraw_context(st)


def _build_sliders(st, d):
    """One row per latent component: name, slider, value."""
    frame = st["e2_slider_frame"]
    for child in frame.winfo_children():
        child.destroy()
    lo, hi = st["e2_slider_range"]
    sliders = []
    for j in range(d):
        row = ttk.Frame(frame, style="Card.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=f"λ{_sub(j + 1)}", style="Card.TLabel", width=3).pack(
            side="left"
        )
        value = ttk.Label(row, text="+0.000", style="Card.Value.TLabel", width=7)
        value.pack(side="right")
        scale = ttk.Scale(
            row,
            from_=float(lo[j]),
            to=float(hi[j]),
            command=lambda v, j=j: _on_scale(st, j, v),
        )
        scale.pack(side="left", fill="x", expand=True, padx=(6, 6))
        sliders.append((scale, value))
    st["e2_sliders"] = sliders
    _sync_sliders(st)


# --------------------------------------------------------------------------- #
# design-variable editing
# --------------------------------------------------------------------------- #


def _sync_sliders(st):
    """Put every slider on the current λ without firing its callback."""
    st["e2_muted"] = True
    try:
        for j, (scale, value) in enumerate(st["e2_sliders"]):
            scale.set(float(st["e2_latent"][j]))
            value.configure(text=f"{st['e2_latent'][j]:+.3f}")
    finally:
        st["e2_muted"] = False


def _on_scale(st, j, raw):
    if st.get("e2_muted") or "e2_latent" not in st:
        return
    value = float(raw)
    st["e2_latent"][j] = value
    st["e2_sliders"][j][1].configure(text=f"{value:+.3f}")
    _schedule_redraw(st)


def _reset_to_mean(st):
    if "e2_trained" not in st:
        return
    st["e2_latent"] = st["e2_trained"].mean(axis=0).copy()
    _sync_sliders(st)
    _schedule_redraw(st)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def _schedule_redraw(st):
    """Geometry follows the drag; the coverage strips wait for it to stop."""
    runtime.reschedule(st, "e2_job_geom", DEBOUNCE_MS, lambda: _redraw_geometry(st))
    runtime.reschedule(st, "e2_job_ctx", SETTLE_MS, lambda: _redraw_context(st))


def _cancel(st, key):
    job = st.get(key)
    st[key] = None
    if job is not None:
        try:
            st["root"].after_cancel(job)
        except tk.TclError:
            pass


def _redraw_geometry(st):
    _cancel(st, "e2_job_geom")
    if "e2_model" not in st:
        return
    res = widgets.read_int(st["e2_res"], DEFAULT_RES)
    phi = models.eval_field_2d(st["e2_model"], st["e2_latent"], res=res)

    fig = st["e2_fig_phi"]
    if st.get("e2_ax_phi") not in fig.axes:
        st["e2_ax_phi"] = viz.single_axes(fig)
        st["e2_layout_frozen"] = False
    viz.draw_sdf_slice(st["e2_ax_phi"], phi, models.PLANE_BOUNDS)
    if not st.get("e2_layout_frozen"):
        widgets.freeze_layout(fig)
        st["e2_layout_frozen"] = True
    st["e2_canvas_phi"].draw_idle()

    lo, hi = float(np.nanmin(phi)), float(np.nanmax(phi))
    st["e2_metrics"]["area"][0].set(f"{float(np.mean(phi < 0.0)):.2f}")
    st["e2_metrics"]["phi"][0].set(f"{lo:+.3f} .. {hi:+.3f}")
    crosses = lo < 0.0 < hi
    st["e2_metrics"]["zero"][0].set("present" if crosses else "absent")
    st["e2_metrics"]["zero"][1].configure(
        style="Card.Value.TLabel" if crosses else "Card.Danger.TLabel"
    )


def _redraw_context(st):
    _cancel(st, "e2_job_ctx")
    if "e2_trained" not in st:
        return
    trained, latent = st["e2_trained"], st["e2_latent"]
    d = trained.shape[1]
    fig = st["e2_fig_cover"]
    axes = st.get("e2_axes_cover") or []
    if len(axes) != d or any(ax not in fig.axes for ax in axes):
        fig.clear()
        fig.set_layout_engine("constrained")
        axes = [fig.add_subplot(d, 1, j + 1) for j in range(d)]
        st["e2_axes_cover"] = axes
    for j, ax in enumerate(axes):
        viz.draw_latent_coverage(ax, trained, [latent[j]], j, st["palette"])
        ax.set_ylabel(f"λ{_sub(j + 1)}", fontsize=8, rotation=0, labelpad=10)
    st["e2_canvas_cover"].draw_idle()
    _update_status(st)


def _update_status(st):
    trained, latent = st["e2_trained"], st["e2_latent"]
    tlo, thi = st["e2_trained_range"]
    outside = int(np.sum((latent < tlo) | (latent > thi)))
    nearest = st["e2_neighbors"].worst_distance(latent[None, :])

    parts = [
        f"{st['e2_entry'].name}: λ = ["
        + ", ".join(f"{v:+.3f}" for v in latent[:6])
        + (", ...]" if len(latent) > 6 else "]")
    ]
    if outside:
        parts.append(
            f"{outside} value(s) outside the trained range - the decoder learned "
            "nothing there, so the geometry above may be meaningless"
        )
    st["e2_status"].set("   |   ".join(parts))
    st["e2_status_label"].configure(
        style="Card.Danger.TLabel" if outside else "Card.Subtle.TLabel"
    )

    note = f"nearest trained code: {nearest:.3f} away"
    for j in range(trained.shape[1]):
        gap, glo, ghi = models.largest_gap(trained[:, j])
        span = max(float(thi[j]) - float(tlo[j]), 1e-12)
        if gap > viz.GAP_FRACTION * span:
            note += f"\nwidest untrained gap λ{_sub(j + 1)}: {glo:+.2f} .. {ghi:+.2f}"
    st["e2_coverage_note"].set(note)
    st["e2_coverage_label"].configure(
        style="Card.Warning.TLabel" if nearest > 0.15 else "Card.Subtle.TLabel"
    )
