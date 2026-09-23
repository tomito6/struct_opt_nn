"""Train tab: turn a dataset of signed-distance samples into a decoder.

Step 2 of the offline pipeline. The dataset itself is produced elsewhere - see
``structsept.app.sdf_maker`` for meshes and the ``datagen`` package for
parametric shapes - so this tab only picks one, states whether it is big
enough for the requested latent dimension, runs the trainer and keeps a record
of the runs that came out.

The dataset also decides the decoder's input: 3-D samples ``(x, y, z, phi)``
train the usual unit-cell decoder, explored on the Explore tab; 2-D samples
``(x, y, phi)`` train a planar one, explored on Explore 2-D. That dimension is
read from the dataset (``datasets.geom_dimension``), never chosen here.

The Decoder card carries the four values changed most often (d, layers, width,
epochs). Every other hyperparameter the trainer reads lives in the window
behind "All hyperparameters..." (``hparam_window``); the card states in one
line which of them differ from the defaults. The full set is held in
``st["tr_hparams"]``, and the four card spinboxes are the authority for their
own four keys - :func:`current_hparams` merges the two.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from structsept.app import (
    datasets,
    hparam_window,
    hyperparams,
    runtime,
    tab_explore,
    tab_explore2d,
    training,
    viz,
    widgets,
)

# A latent space only interpolates sensibly if the trained shapes fill it; the
# paper's d = 2 test case trained 120 unit cells.
MIN_SHAPES_2D = 40
PROGRESS_MS = 1500


def build(st, parent, data_root, runs_dir):
    st["tr_data_root"] = Path(data_root)
    st["tr_runs_dir"] = Path(runs_dir)
    st["tr_hparams"] = hyperparams.defaults()
    hp = st["tr_hparams"]
    st["tr_latent_dim"] = tk.IntVar(value=hp["latent_dim"])
    st["tr_n_layers"] = tk.IntVar(value=hp["n_layers"])
    st["tr_width"] = tk.IntVar(value=hp["width"])
    st["tr_epochs"] = tk.IntVar(value=hp["num_epochs"])
    st["tr_run_name"] = tk.StringVar(value="")

    palette = st["palette"]

    top = ttk.Frame(parent)
    top.pack(fill="x")
    _build_dataset_card(st, top, palette)
    _build_run_card(st, top, palette)

    body = ttk.PanedWindow(parent, orient="horizontal")
    body.pack(fill="both", expand=True, pady=(8, 0))
    left = ttk.Frame(body)
    right = ttk.Frame(body)
    body.add(left, weight=3)
    body.add(right, weight=2)

    _build_progress(st, left, palette)
    _build_runs(st, right, palette)

    refresh_datasets(st)
    refresh_runs(st)
    return parent


def _build_dataset_card(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Dataset", "signed-distance samples")
    outer.pack(side="left", fill="both", expand=True)

    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(fill="x")
    st["tr_combo"] = ttk.Combobox(row, state="readonly", width=34)
    st["tr_combo"].pack(side="left")
    st["tr_combo"].bind("<<ComboboxSelected>>", lambda e: _update_readiness(st))
    ttk.Button(
        row,
        text="Refresh",
        style="CardGhost.TButton",
        command=lambda: refresh_datasets(st),
    ).pack(side="left", padx=(4, 0))

    st["tr_readiness"] = tk.StringVar(value="")
    st["tr_readiness_label"] = ttk.Label(
        body,
        textvariable=st["tr_readiness"],
        style="Card.Subtle.TLabel",
        wraplength=380,
        justify="left",
    )
    st["tr_readiness_label"].pack(fill="x", pady=(8, 0))

    ttk.Label(
        body,
        text="Datasets are built in the SDF maker:\n"
        "uv run python -m structsept.app.sdf_maker",
        style="Card.Subtle.TLabel",
        justify="left",
    ).pack(fill="x", pady=(8, 0))


def _build_run_card(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Decoder", "architecture and budget")
    outer.pack(side="left", fill="both", expand=True, padx=(8, 0))

    grid = ttk.Frame(body, style="Card.TFrame")
    grid.pack(fill="x")
    fields = hyperparams.FIELD_BY_KEY
    for col, (label, key, step) in enumerate(
        (
            ("Latent dim d", "latent_dim", 1),
            ("Layers", "n_layers", 1),
            ("Width", "width", 16),
            ("Epochs", "num_epochs", 10),
        )
    ):
        field = fields[key]
        cell = ttk.Frame(grid, style="Card.TFrame")
        cell.grid(row=0, column=col, sticky="sw", padx=(0, 14))
        ttk.Label(cell, text=label, style="Card.Subtle.TLabel").pack(anchor="w")
        widgets.spinbox(
            cell, st[CARD_VARS[key]], field.low, field.high, step, width=6
        ).pack(anchor="w")
    # CardGhost like the card's other secondary actions: a plain TButton is
    # white-on-white here and reads as a label
    st["tr_btn_hparams"] = ttk.Button(
        grid,
        text="All hyperparameters...",
        style="CardGhost.TButton",
        command=lambda: open_hparams(st),
    )
    st["tr_btn_hparams"].grid(row=0, column=4, sticky="sw")
    st["tr_latent_dim"].trace_add("write", lambda *_: _update_readiness(st))
    for var_name in CARD_VARS.values():
        st[var_name].trace_add("write", lambda *_: _update_hp_summary(st))

    # which of the other ~30 values differ from the defaults - and whether the
    # card values just made the set invalid, e.g. d + 3 >= width with a skip
    st["tr_hp_summary"] = tk.StringVar(value="")
    st["tr_hp_summary_label"] = ttk.Label(
        body,
        textvariable=st["tr_hp_summary"],
        style="Card.Subtle.TLabel",
        wraplength=420,
        justify="left",
    )
    st["tr_hp_summary_label"].pack(fill="x", pady=(8, 0))

    row = ttk.Frame(body, style="Card.TFrame")
    row.pack(fill="x", pady=(10, 0))
    ttk.Label(row, text="Run name", style="Card.TLabel").pack(side="left")
    ttk.Entry(row, textvariable=st["tr_run_name"], width=24).pack(side="left", padx=6)
    ttk.Button(
        row,
        text="Quick preset",
        style="CardGhost.TButton",
        command=lambda: _quick_preset(st),
    ).pack(side="left")
    st["tr_btn_train"] = ttk.Button(
        row, text="Train", style="Accent.TButton", command=lambda: _start_training(st)
    )
    st["tr_btn_train"].pack(side="right")

    # determinate at 0 while idle: an unstarted indeterminate bar still
    # paints its block at the left edge and reads as stuck progress
    st["tr_progress"] = ttk.Progressbar(body, mode="determinate", value=0)
    st["tr_progress"].pack(fill="x", pady=(10, 0))
    st["tr_progress_text"] = tk.StringVar(value="idle")
    ttk.Label(
        body, textvariable=st["tr_progress_text"], style="Card.Subtle.TLabel"
    ).pack(fill="x", pady=(4, 0))
    _update_hp_summary(st)


# card spinbox variable of each hyperparameter that has one
CARD_VARS = {
    "latent_dim": "tr_latent_dim",
    "n_layers": "tr_n_layers",
    "width": "tr_width",
    "num_epochs": "tr_epochs",
}


def _build_progress(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Training loss")
    outer.pack(fill="both", expand=True)
    st["tr_fig"], st["tr_canvas"] = widgets.figure_canvas(
        body, palette, (3.4, 1.9), st["dpi"]
    )
    st["tr_ax"] = viz.single_axes(st["tr_fig"])
    viz.draw_loss_curve(st["tr_ax"], [], palette=palette)
    st["tr_canvas"].draw_idle()

    frame, st["tr_log"] = widgets.log_box(body, palette, height=12)
    frame.pack(fill="both", expand=True, pady=(8, 0))


def _build_runs(st, parent, palette):
    outer, body = widgets.card(
        parent, palette, "Runs", "double-click to open in the Explore tab"
    )
    outer.pack(fill="both", expand=True)

    columns = ("dataset", "d", "epochs", "loss", "date")
    tree = ttk.Treeview(body, columns=columns, show="tree headings", height=10)
    tree.heading("#0", text="run")
    tree.column("#0", width=150, anchor="w")
    for key, title, width in (
        ("dataset", "dataset", 90),
        ("d", "d", 30),
        ("epochs", "epochs", 50),
        ("loss", "final loss", 70),
        ("date", "date", 90),
    ):
        tree.heading(key, text=title)
        tree.column(key, width=width, anchor="w")
    bar = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    tree.bind("<Double-1>", lambda e: _open_in_explore(st))
    st["tr_tree"] = tree


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #


def refresh_datasets(st):
    rows = datasets.list_datasets(st["tr_data_root"])
    st["tr_datasets"] = {
        f"{r['name']} ({r['n_instances']} shapes, {_geom(r)}-D)": r for r in rows
    }
    values = list(st["tr_datasets"])
    combo = st["tr_combo"]
    combo.configure(values=values)
    if values and combo.get() not in st["tr_datasets"]:
        combo.set(values[0])
    _update_readiness(st)


def refresh_runs(st):
    tree = st["tr_tree"]
    tree.delete(*tree.get_children())
    for row in training.list_runs(st["tr_runs_dir"]):
        loss = row.get("final_loss")
        date = (row.get("date") or "")[:16].replace("T", " ")
        tree.insert(
            "",
            "end",
            text=row["name"],
            values=(
                row.get("dataset") or "-",
                row.get("latent_dim") if row.get("latent_dim") is not None else "?",
                row.get("epochs") if row.get("epochs") is not None else "?",
                f"{loss:.4f}" if isinstance(loss, (int, float)) else "-",
                date,
            ),
        )


def _open_in_explore(st):
    """Hand a run to the explorer that can show it: lattice or planar."""
    selection = st["tr_tree"].selection()
    if not selection:
        return
    name = st["tr_tree"].item(selection[0], "text")
    tab_explore.refresh_models(st)
    tab_explore2d.refresh_models(st)
    if tab_explore.select_model(st, name):
        st["notebook"].select(st["tab_frames"]["explore"])
    elif tab_explore2d.select_model(st, name):
        st["notebook"].select(st["tab_frames"]["explore2d"])
    else:
        messagebox.showinfo(
            "Not loadable",
            f"'{name}' has no saved checkpoint yet, so the explorer cannot open it.",
        )


def _update_readiness(st):
    """State the shapes-per-dimension problem before the run, not after it."""
    _update_hp_summary(st)
    row = st.get("tr_datasets", {}).get(st["tr_combo"].get())
    if row is None:
        st["tr_readiness"].set("No dataset found under data/SdfSamples.")
        st["tr_readiness_label"].configure(style="Card.Warning.TLabel")
        return
    # peek, never read_int: this runs from a trace on the spinbox itself
    d = widgets.peek_int(st["tr_latent_dim"])
    if d is None:
        d = st["tr_hparams"]["latent_dim"]
    n = row["n_instances"]
    text = f"{n} shape(s), class(es): {', '.join(row['classes']) or '-'}"
    if _geom(row) == 2:
        text += (
            "\n2-D samples (x, y, φ): trains a planar decoder fθ(λ, x, y); "
            "open the result on Explore 2-D."
        )
    if not row["split"]:
        text += "\nNo split file - this dataset cannot be trained on."
        style = "Card.Danger.TLabel"
    elif d >= 2 and n < MIN_SHAPES_2D:
        text += (
            f"\nd={d} with only {n} shapes. A latent space of that dimension needs "
            f"a grid of shapes to fill it (~{MIN_SHAPES_2D}+; the paper used 120 "
            "for d=2). With fewer, interpolating walks through untrained holes."
        )
        style = "Card.Warning.TLabel"
    else:
        text += "\nReady to train."
        style = "Card.Success.TLabel"
    st["tr_readiness"].set(text)
    st["tr_readiness_label"].configure(style=style)


# --------------------------------------------------------------------------- #
# hyperparameters
# --------------------------------------------------------------------------- #


def current_hparams(st, commit=False) -> dict:
    """The full hyperparameter set as it stands: ``st["tr_hparams"]`` with the
    four card spinboxes laid over it.

    A spinbox accepts typed text, so a card value can be empty or garbage for
    a moment; it then counts as the value last applied. ``commit=True`` also
    writes that value back into the spinbox, so the user sees what is used -
    only for a button press (Train, the window). A live refresh must not: it
    runs from the spinbox's own trace, mid-edit, and a write-back there turns
    "50" typed over "200" into "50200".
    """
    hp = dict(st["tr_hparams"])
    for key, var_name in CARD_VARS.items():
        if commit:
            hp[key] = widgets.read_int(st[var_name], hp[key])
        else:
            value = widgets.peek_int(st[var_name])
            hp[key] = hp[key] if value is None else value
    return hp


def _unreadable_card_fields(st) -> list[str]:
    return [
        hyperparams.FIELD_BY_KEY[key].label
        for key, var_name in CARD_VARS.items()
        if widgets.peek_int(st[var_name]) is None
    ]


def _selected_dataset(st):
    return st.get("tr_datasets", {}).get(st["tr_combo"].get())


def _geom(row):
    """Coordinates per sample of a dataset row; 3 when no dataset is picked."""
    if row is None:
        return hyperparams.GEOM_DIMENSION
    return int(row.get("geom_dimension", hyperparams.GEOM_DIMENSION))


def open_hparams(st):
    """Open the hyperparameter window on the current set."""
    row = _selected_dataset(st)
    split = (row["split"] or "<no split file>") if row is not None else None
    return hparam_window.open_window(
        st,
        current_hparams(st, commit=True),
        lambda hp: apply_hparams(st, hp),
        n_shapes=row["n_instances"] if row is not None else None,
        dataset=row["name"] if row is not None else None,
        sources=training.spec_sources(st["tr_runs_dir"]),
        preview_paths=(
            split or "<split of the selected dataset>",
            str(st["tr_data_root"]),
        ),
        geom_dimension=_geom(row),
    )


def apply_hparams(st, hp):
    """Make ``hp`` the set the next run trains with, card spinboxes included."""
    before = current_hparams(st)
    st["tr_hparams"] = dict(hp)
    for key, var_name in CARD_VARS.items():
        st[var_name].set(hp[key])
    _update_hp_summary(st)
    diff = [
        f"{f.full_label}: {hyperparams.format_value(f, before[f.key]) or f.blank} -> "
        f"{hyperparams.format_value(f, hp[f.key]) or f.blank}"
        for f in hyperparams.FIELDS
        if before[f.key] != hp[f.key]
    ]
    if diff:
        widgets.append(st["tr_log"], "Hyperparameters changed:")
        for line in diff:
            widgets.append(st["tr_log"], f"  {line}")


def _update_hp_summary(st):
    """Refresh the one-line summary under the card spinboxes."""
    if "tr_hp_summary" not in st or "tr_combo" not in st:
        return
    unreadable = _unreadable_card_fields(st)
    if unreadable:
        st["tr_hp_summary"].set(f"{', '.join(unreadable)}: not a whole number.")
        st["tr_hp_summary_label"].configure(style="Card.Danger.TLabel")
        return
    hp = current_hparams(st)
    row = _selected_dataset(st)
    problems = hyperparams.errors(
        hyperparams.validate(
            hp, row["n_instances"] if row is not None else None, _geom(row)
        )
    )
    if problems:
        st["tr_hp_summary"].set(
            problems[0].message + "  Fix it here or under All hyperparameters."
        )
        st["tr_hp_summary_label"].configure(style="Card.Danger.TLabel")
    else:
        st["tr_hp_summary"].set(hyperparams.summary(hp))
        st["tr_hp_summary_label"].configure(style="Card.Subtle.TLabel")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def _quick_preset(st):
    st["tr_epochs"].set(30)
    widgets.append(
        st["tr_log"],
        "Quick preset: 30 epochs, enough to close the loop in about a minute of CPU.",
    )


def _start_training(st):
    row = st.get("tr_datasets", {}).get(st["tr_combo"].get())
    if row is None:
        messagebox.showerror("No dataset", "Pick a dataset from the list.")
        return
    if not row["split"]:
        messagebox.showerror("No split", f"Dataset '{row['name']}' has no split file.")
        return

    hp = current_hparams(st, commit=True)
    geom = _geom(row)
    issues = hyperparams.validate(hp, row["n_instances"], geom)
    problems = hyperparams.errors(issues)
    if problems:
        messagebox.showerror(
            "Hyperparameters",
            "This set cannot be trained:\n\n"
            + "\n".join(f"- {p.message}" for p in problems),
        )
        return
    epochs = hp["num_epochs"]

    name = st["tr_run_name"].get().strip()
    if not name:
        name = f"{row['name']}_d{hp['latent_dim']}_{datetime.now():%Y%m%d_%H%M}"
        st["tr_run_name"].set(name)
    run_dir = st["tr_runs_dir"] / name
    if (run_dir / "specs.json").is_file():
        if not messagebox.askyesno("Existing run", f"Overwrite '{name}'?"):
            return

    def work(log):
        specs = training.write_specs(
            run_dir, row["split"], st["tr_data_root"], hp, geom_dimension=geom
        )
        log(f"specs: {specs}")
        log(
            f"{geom}-D samples: decoder fθ(λ, "
            + ("x, y" if geom == 2 else "x, y, z")
            + f"), input width d + {geom} = {hp['latent_dim'] + geom}"
        )
        changed = hyperparams.describe(hp)
        log(
            "Hyperparameters: all defaults."
            if not changed
            else "Hyperparameters that differ from the defaults:"
        )
        for line in changed:
            log(line)
        for issue in issues:
            log(f"{issue.level}: {issue.message}")
        log(f"Training '{name}' on CPU, {epochs} epochs...")
        with runtime.signals_off():
            info = training.train(run_dir, st["tr_data_root"], log=log)
        training.write_metadata(run_dir, dataset=row["name"], epochs=info["epochs"])
        log(f"Training finished in {info['seconds']:.0f} s -> {run_dir}")

    if st.get("busy"):
        # run_worker would refuse it; do not leave a bar spinning for a run
        # that never started
        messagebox.showinfo("Busy", "Wait for the current task to finish.")
        return

    st["tr_watch_dir"] = run_dir
    st["tr_watch_epochs"] = epochs
    st["tr_progress"].configure(mode="indeterminate")
    st["tr_progress"].start(18)
    st["tr_progress_text"].set("starting...")
    runtime.run_worker(
        st,
        st["tr_btn_train"],
        st["tr_log"],
        work,
        on_done=lambda: _training_done(st),
    )
    _poll_progress(st)


def _poll_progress(st):
    """Follow the run from its own checkpoint file, on the Tk clock.

    The worker thread is inside the trainer for the whole run and cannot report
    anything, and the busy flag blocks a second worker - so the progress read
    runs here instead, slowly enough (1.5 s) not to contend with the 100 ms
    queue drain.
    """
    st["tr_job_progress"] = None
    if not st.get("busy") or st.get("tr_watch_dir") is None:
        return
    info = training.read_progress(st["tr_watch_dir"])
    if info.get("loss"):
        viz.draw_loss_curve(
            st["tr_ax"], info["loss"], epochs=info.get("epoch"), palette=st["palette"]
        )
        st["tr_canvas"].draw_idle()
        total = st.get("tr_watch_epochs") or 0
        epoch = info.get("epoch", 0)
        st["tr_progress_text"].set(
            f"epoch {epoch}/{total}  ·  loss {info['loss'][-1]:.4f}"
            if total
            else f"epoch {epoch}"
        )
    runtime.reschedule(st, "tr_job_progress", PROGRESS_MS, lambda: _poll_progress(st))


def _training_done(st):
    """Reached however the run ended - finished, failed or refused."""
    st["tr_progress"].stop()
    st["tr_progress"].configure(mode="determinate", value=0)
    st["tr_progress_text"].set("done")
    info = training.read_progress(st.get("tr_watch_dir") or "")
    if info.get("loss"):
        viz.draw_loss_curve(
            st["tr_ax"], info["loss"], epochs=info.get("epoch"), palette=st["palette"]
        )
        st["tr_canvas"].draw_idle()
    st["tr_watch_dir"] = None
    refresh_runs(st)
    tab_explore.refresh_models(st)
    tab_explore2d.refresh_models(st)
