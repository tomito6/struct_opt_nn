"""GUI shell of the structsept app.

Three tabs over the online half of the pipeline: train a DeepSDF decoder on a
dataset that already exists, then explore ``f_theta(lambda(x), x)`` by moving
the latent B-spline control points - the design variables of the optimization
the paper describes. A decoder trained on planar (2-D) samples describes a
whole shape rather than a unit cell, and gets its own explorer, Explore 2-D.

Building the dataset itself lives in its own window,
``structsept.app.sdf_maker``: it is a once-in-a-while job, and everything here
assumes the samples on disk are already valid.

Training and meshing run on worker threads; the Tk loop only draws.

    uv run python -m structsept.app.main
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

# DeepSDFStruct draws with pyplot from inside worker threads - the trainer for
# its loss curve, the mesher for its debug figures. On an interactive backend
# that builds Tk widgets off the main loop and kills the whole app. Our own
# canvases are FigureCanvasTkAgg instances built by hand, so they do not care
# what the pyplot backend is.
matplotlib.use("Agg")

from structsept.app import (  # noqa: E402
    runtime,
    tab_explore,
    tab_explore2d,
    tab_train,
    theme,
    widgets,
)

# Debounced jobs are stored under "<tab prefix>_job_<name>"; shutdown cancels
# every key with one of these prefixes.
JOB_PREFIXES = ("ex_job_", "e2_job_", "tr_job_")

# Datasets and training runs are large, regenerable and gitignored, so they
# live at the repo root rather than inside the package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def build_app(root=None):
    """Build the Tk root with every tab wired up, without starting the loop.

    ``root`` lets the desktop launcher hand over the (withdrawn, still empty)
    interpreter its splash screen lives in; without it a fresh ``tk.Tk()`` is
    created. Either way the widgets end up in the returned root.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if root is None:
        theme.enable_dpi_awareness()
        root = tk.Tk()
    # the theme has to be applied before any widget exists: option_add only
    # reaches widgets created afterwards
    palette = theme.apply_theme(root)
    root.title("structsept - DeepSDF lattice explorer")
    theme.clamp_geometry(root, 1500, 980)

    st = runtime.new_state(root)
    st["palette"] = palette
    st["dpi"] = theme.figure_dpi(root)
    root.app_state = st

    header = ttk.Frame(root, padding=(14, 12, 14, 0))
    header.pack(fill="x")
    ttk.Label(header, text="DeepSDF lattice explorer", style="Title.TLabel").pack(
        side="left"
    )
    ttk.Label(
        header,
        text="fθ(λ(x), x)  ·  latent field on a B-spline  ·  "
        "control points are the design variables",
        style="Subtle.TLabel",
    ).pack(side="left", padx=(12, 0), pady=(6, 0))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=14, pady=(8, 14))
    st["notebook"] = notebook

    explore_tab = ttk.Frame(notebook, padding=10)
    explore2d_tab = ttk.Frame(notebook, padding=10)
    train_tab = ttk.Frame(notebook, padding=10)
    notebook.add(explore_tab, text="Explore")
    notebook.add(explore2d_tab, text="Explore 2-D")
    notebook.add(train_tab, text="Train")
    # by name, so nothing depends on the order of the tabs
    st["tab_frames"] = {
        "explore": explore_tab,
        "explore2d": explore2d_tab,
        "train": train_tab,
    }

    tab_explore.build(st, explore_tab, RUNS_DIR)
    tab_explore2d.build(st, explore2d_tab, RUNS_DIR)
    tab_train.build(st, train_tab, DATA_ROOT, RUNS_DIR)

    if palette["_errors"]:
        # the theme swallows its own failures so one bad style name cannot
        # take the app down; say so somewhere rather than nowhere
        widgets.append(
            st["ex_log"],
            f"{len(palette['_errors'])} theme step(s) did not apply: "
            + "; ".join(palette["_errors"][:3]),
        )

    root.protocol("WM_DELETE_WINDOW", lambda: _shutdown(st, root))
    root.after(runtime.POLL_MS, lambda: runtime.poll(st))
    return root


def _shutdown(st, root):
    """Cancel every pending ``after`` before tearing the root down.

    The poll loop and the debounced redraws all re-arm themselves; without
    this, Tk fires them into a destroyed interpreter and prints
    "invalid command name" on the way out.
    """
    st["closing"] = True
    for key, value in list(st.items()):
        # every debounced job is stored as "<tab>_job_<name>"; matching the
        # prefix means a new one does not have to be added here
        if value is not None and (key == "poll_job" or key.startswith(JOB_PREFIXES)):
            try:
                root.after_cancel(value)
            except tk.TclError:
                pass
    root.destroy()


def main():
    build_app().mainloop()


if __name__ == "__main__":
    main()
