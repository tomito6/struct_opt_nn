"""Standalone SDF maker: a folder of meshes in, an SdfSamples dataset out.

This used to be the first tab of the main app. It was split off because the
explorer assumes the datasets it needs already exist and are valid, and
because this step is run once in a while rather than every session. The
layout, widgets and flow are deliberately unchanged from that tab - only the
text is English now. It keeps its own ``tk.Tk`` root, so it also keeps the
platform's native ttk look rather than the main app's theme.

Step 1 of the offline pipeline: sample points around each geometry, compute
the signed distance, write ``<data>/SdfSamples/<dataset>/<class>/*.npz`` plus
the split json that the trainer reads.

    uv run python -m structsept.app.sdf_maker
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

# The heavy libraries draw with pyplot from worker threads; on an interactive
# backend that builds Tk widgets off the main loop and kills the whole app.
matplotlib.use("Agg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from structsept.app import datasets, runtime, viz  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


# --------------------------------------------------------------------------- #
# small widget helpers
# --------------------------------------------------------------------------- #


def _log_box(parent, height=12):
    """Read-only scrolling text widget."""
    frame = ttk.Frame(parent)
    text = tk.Text(frame, height=height, wrap="word", state="disabled")
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    return frame, text


def _append(text, line):
    text.configure(state="normal")
    text.insert("end", str(line) + "\n")
    text.see("end")
    text.configure(state="disabled")


def _figure_canvas(parent, figsize):
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, canvas


def _spinbox(parent, var, low, high, step=1, width=7):
    return ttk.Spinbox(
        parent, textvariable=var, from_=low, to=high, increment=step, width=width
    )


def _read_int(var, fallback):
    """Spinboxes accept typed text, so a value can be empty or garbage."""
    try:
        return int(var.get())
    except (tk.TclError, ValueError):
        var.set(fallback)
        return fallback


def _refresh_datasets(st):
    rows = datasets.list_datasets(DATA_ROOT)
    st["datasets"] = {f"{r['name']} ({r['n_instances']} shapes)": r for r in rows}
    values = list(st["datasets"])
    combo = st["combo_data"]
    combo.configure(values=values)
    if values and combo.get() not in st["datasets"]:
        combo.set(values[0])


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #


def _build(st, parent):
    st["folder"] = tk.StringVar()
    st["ext"] = tk.StringVar(value="stl")
    st["ds_name"] = tk.StringVar(value="my_dataset")
    st["class_name"] = tk.StringVar(value="cells")
    st["n_samples"] = tk.IntVar(value=50000)
    st["surface_samples"] = tk.BooleanVar(value=True)

    form = ttk.LabelFrame(parent, text="Generate dataset", padding=8)
    form.pack(fill="x")
    form.columnconfigure(1, weight=1)

    ttk.Label(form, text="Mesh folder:").grid(row=0, column=0, sticky="w")
    ttk.Entry(form, textvariable=st["folder"]).grid(
        row=0, column=1, sticky="ew", padx=4
    )
    ttk.Button(form, text="Browse...", command=lambda: _pick_folder(st)).grid(
        row=0, column=2
    )

    line = ttk.Frame(form)
    line.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
    ttk.Label(line, text="Extension:").pack(side="left")
    ttk.Entry(line, textvariable=st["ext"], width=6).pack(side="left", padx=(4, 12))
    ttk.Label(line, text="Dataset:").pack(side="left")
    ttk.Entry(line, textvariable=st["ds_name"], width=18).pack(
        side="left", padx=(4, 12)
    )
    ttk.Label(line, text="Class:").pack(side="left")
    ttk.Entry(line, textvariable=st["class_name"], width=18).pack(
        side="left", padx=(4, 12)
    )
    ttk.Label(line, text="Samples/geometry:").pack(side="left")
    _spinbox(line, st["n_samples"], 1000, 500000, 1000, width=9).pack(
        side="left", padx=(4, 12)
    )
    ttk.Checkbutton(line, text="surface samples", variable=st["surface_samples"]).pack(
        side="left"
    )

    st["btn_generate"] = ttk.Button(
        form, text="Generate dataset", command=lambda: _generate_dataset(st)
    )
    st["btn_generate"].grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

    check = ttk.LabelFrame(parent, text="Check dataset", padding=8)
    check.pack(fill="x", pady=(8, 0))
    ttk.Label(check, text="Dataset:").pack(side="left")
    st["combo_data"] = ttk.Combobox(check, state="readonly", width=40)
    st["combo_data"].pack(side="left", padx=4)
    ttk.Button(check, text="Refresh", command=lambda: _refresh_datasets(st)).pack(
        side="left", padx=4
    )
    st["btn_validate"] = ttk.Button(
        check, text="Validate", command=lambda: _validate(st)
    )
    st["btn_validate"].pack(side="left", padx=4)

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True, pady=(8, 0))
    frame, st["log_data"] = _log_box(body)
    frame.pack(side="left", fill="both", expand=True)
    right = ttk.Frame(body, width=460)
    right.pack(side="right", fill="both")
    st["fig_hist"], st["canvas_hist"] = _figure_canvas(right, (4.6, 3.2))


def _pick_folder(st):
    folder = filedialog.askdirectory(title="Folder with the meshes")
    if folder:
        st["folder"].set(folder)


def _generate_dataset(st):
    folder = st["folder"].get().strip()
    if not folder or not Path(folder).is_dir():
        messagebox.showerror("Invalid folder", "Pick a folder that exists.")
        return
    name = st["ds_name"].get().strip()
    class_name = st["class_name"].get().strip()
    if not name or not class_name:
        messagebox.showerror("Missing names", "Fill in the dataset and the class.")
        return

    ext = st["ext"].get().strip() or "stl"
    meshes = datasets.load_meshes(folder, ext, log=lambda t: _append(st["log_data"], t))
    if not meshes:
        messagebox.showerror("No meshes", f"No readable .{ext} file in that folder.")
        return

    open_meshes = [m for m in meshes if not m.is_watertight]
    if open_meshes:
        _append(
            st["log_data"],
            f"WARNING: {len(open_meshes)} of {len(meshes)} meshes are not watertight.",
        )
        ok = messagebox.askyesno(
            "Open meshes",
            f"{len(open_meshes)} of {len(meshes)} meshes are not watertight.\n\n"
            "The sign of the SDF comes from a winding number: on an open mesh "
            "inside/outside comes out wrong and the dataset is silently "
            "broken.\n\n"
            "Generate anyway?",
        )
        if not ok:
            _append(st["log_data"], "Cancelled. Fix the meshes first.")
            return

    n_samples = _read_int(st["n_samples"], 50000)
    surface = bool(st["surface_samples"].get())

    def work(log):
        info = datasets.make_dataset(
            meshes,
            DATA_ROOT,
            name,
            class_name,
            n_samples=n_samples,
            add_surface_samples=surface,
            log=log,
        )
        log(f"Split: {info['split_path']}")
        runtime.ui(st, lambda: _refresh_datasets(st))

    runtime.run_worker(st, st["btn_generate"], st["log_data"], work)


def _validate(st):
    row = st["datasets"].get(st["combo_data"].get())
    if row is None:
        messagebox.showerror("No dataset", "Pick a dataset from the list.")
        return

    def work(log):
        report = datasets.validate_dataset(row["path"], log=log)
        runtime.ui(st, lambda: _show_report(st, report))

    runtime.run_worker(st, st["btn_validate"], st["log_data"], work)


def _show_report(st, report):
    ax = viz.slice_axes(st["fig_hist"], 1)[0]
    viz.draw_phi_histogram(ax, report["phi_all"])
    st["canvas_hist"].draw_idle()
    problems = report["problems"]
    if problems:
        messagebox.showwarning(
            "Dataset has problems",
            "\n".join(problems[:10])
            + ("\n..." if len(problems) > 10 else "")
            + "\n\nTraining on this produces a meaningless decoder.",
        )


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #


def build_app(root=None):
    """Build the Tk root without starting the loop.

    ``root`` is the launcher's already-created (withdrawn, empty) interpreter;
    left out, a fresh ``tk.Tk()`` is made.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if root is None:
        root = tk.Tk()
    root.title("SDF maker - meshes to an SdfSamples dataset")
    root.geometry("1100x720")

    st = runtime.new_state(root)
    root.app_state = st

    body = ttk.Frame(root, padding=8)
    body.pack(fill="both", expand=True)
    _build(st, body)

    _refresh_datasets(st)
    _append(st["log_data"], f"Datasets in {DATA_ROOT}")

    root.protocol("WM_DELETE_WINDOW", lambda: _shutdown(st, root))
    root.after(runtime.POLL_MS, lambda: runtime.poll(st))
    return root


def _shutdown(st, root):
    """Stop the queue pump before the interpreter goes away."""
    st["closing"] = True
    job = st.get("poll_job")
    if job is not None:
        try:
            root.after_cancel(job)
        except tk.TclError:
            pass
    root.destroy()


def main():
    build_app().mainloop()


if __name__ == "__main__":
    main()
