"""Train tab: turn a dataset of signed-distance samples into a decoder.

Step 2 of the offline pipeline. The dataset itself is produced elsewhere - see
``structsept.app.sdf_maker`` for meshes and the ``datagen`` package for
parametric shapes - so this tab only picks one, states whether it is big
enough for the requested latent dimension, runs the trainer and keeps a record
of the runs that came out. The datasets listed are those of one data root -
``data/`` unless another was picked with Browse... (remembered between
launches, see ``folders``) - and the run trains from that root.

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
from tkinter import filedialog, messagebox, ttk

from structsept.app import (
    datasets,
    folders,
    hparam_window,
    hyperparams,
    run_editor,
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
    """Populate the Train tab. ``parent`` is an empty padded frame.

    ``data_root`` is the default folder of the dataset picker; the folder
    last picked with Browse... wins over it while it still exists. Runs are
    always written to ``runs_dir``.
    """
    st["tr_data_root"] = folders.remembered(folders.TRAIN_DATA, data_root)
    st["tr_data_text"] = tk.StringVar(value="")
    st["tr_runs_dir"] = Path(runs_dir)
    st["tr_hparams"] = hyperparams.defaults()
    # where the current set was loaded from ("Start from" / "Import sheet"),
    # or None: see hyperparams.origin_summary
    st["tr_hp_origin"] = None
    # where the sheet file dialog opens; the supervisor's templates live here
    st["tr_sheet_dir"] = Path(runs_dir).parent / "docs" / "hyperparameters"
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
    # the runs table gets the wider half: a run name plus its notes is the
    # longest thing on this tab, and the loss curve reads fine at half width
    body.add(left, weight=2)
    body.add(right, weight=3)

    _build_progress(st, left, palette)
    _build_runs(st, right, palette)

    refresh_datasets(st)
    refresh_runs(st)
    return parent


def _build_dataset_card(st, parent, palette):
    outer, body = widgets.card(parent, palette, "Dataset", "signed-distance samples")
    outer.pack(side="left", fill="both", expand=True)

    # the data root the picker lists: data/ or any other SdfSamples + splits
    widgets.folder_row(body, st["tr_data_text"], lambda: browse_data_folder(st)).pack(
        fill="x", pady=(0, 6)
    )
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

    # width is only the requested size, the box still fills the card: at the
    # default 80 columns the log alone claims the wider half of the paned
    # window and squeezes the runs table
    frame, st["tr_log"] = widgets.log_box(body, palette, height=12, width=48)
    frame.pack(fill="both", expand=True, pady=(8, 0))


# Runs table: (row key, tree column, heading, width). "#0" is the tree column.
# Spare width goes to the two text columns; the numbers keep theirs, so a
# wide table does not spread them out.
RUN_COLUMNS = (
    ("name", "#0", "run", 250),
    ("dataset", "dataset", "dataset", 150),
    ("latent_dim", "d", "d", 28),
    ("epochs", "epochs", "epochs", 64),
    ("train_seconds", "time", "time", 78),
    ("final_loss", "loss", "final loss", 66),
    ("date", "date", "date", 132),
    ("description", "notes", "notes", 140),
)
STRETCH_COLUMNS = ("name", "description")
RUNS_ROWS = 18


def _build_runs(st, parent, palette):
    outer, body = widgets.card(
        parent,
        palette,
        "Runs",
        "newest first · click a heading to sort · double-click to open · "
        "right-click for more",
    )
    outer.pack(fill="both", expand=True)

    table = ttk.Frame(body, style="Card.TFrame")
    table.pack(fill="both", expand=True)
    columns = tuple(col for _, col, _, _ in RUN_COLUMNS if col != "#0")
    tree = ttk.Treeview(table, columns=columns, show="tree headings", height=RUNS_ROWS)
    for key, col, title, width in RUN_COLUMNS:
        tree.heading(col, text=title, command=lambda k=key: sort_runs(st, k))
        tree.column(col, width=width, anchor="w", stretch=key in STRETCH_COLUMNS)
    bar = ttk.Scrollbar(table, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    tree.bind("<Double-1>", lambda e: _open_in_explore(st))
    tree.bind("<Button-3>", lambda e: _popup_menu(st, e))
    tree.bind("<F2>", lambda e: edit_run(st))
    tree.bind("<Delete>", lambda e: delete_run(st))
    st["tr_tree"] = tree
    st["tr_runs"] = []
    st["tr_runs_sort"] = ("date", True)  # (row key, descending)

    # the same actions as buttons under the table and as the right-click menu
    actions = ttk.Frame(body, style="Card.TFrame")
    actions.pack(fill="x", pady=(8, 0))
    menu = tk.Menu(tree, tearoff=0)
    st["tr_run_buttons"] = {}
    for key, text, command in RUN_ACTIONS:
        button = ttk.Button(
            actions,
            text=text,
            style="CardGhost.TButton",
            command=lambda c=command: c(st),
        )
        button.pack(side="left", padx=(0, 6))
        st["tr_run_buttons"][key] = button
        menu.add_command(label=text, command=lambda c=command: c(st))
    st["tr_run_menu"] = menu
    ttk.Button(
        actions,
        text="Refresh",
        style="CardGhost.TButton",
        command=lambda: refresh_runs(st),
    ).pack(side="right")
    # a run renamed or deleted on an Explore tab, or finished in another
    # window and refreshed there, is re-read here through run_editor
    st.setdefault("run_listeners", []).append(lambda: refresh_runs(st))


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #


def refresh_datasets(st):
    """Re-read the datasets of the current data root into the picker."""
    root = st["tr_data_root"]
    rows = datasets.list_datasets(root)
    st["tr_datasets"] = {
        f"{r['name']} ({r['n_instances']} shapes, {_geom(r)}-D)": r for r in rows
    }
    values = list(st["tr_datasets"])
    combo = st["tr_combo"]
    combo.configure(values=values)
    if values and combo.get() not in st["tr_datasets"]:
        combo.set(values[0])
    elif not values:
        # not the last root's dataset, which Train could no longer find
        combo.set("")
    n = len(values)
    if n:
        found = f"{n} dataset{'s' if n != 1 else ''}"
    elif (Path(root) / datasets.SDF_SAMPLES_DIR).is_dir():
        found = "no dataset here"
    else:
        found = f"no {datasets.SDF_SAMPLES_DIR} folder here"
    st["tr_data_text"].set(f"{folders.display(root)}  ·  {found}")
    _update_readiness(st)


def browse_data_folder(st):
    """Browse...: pick the data root the dataset picker lists."""
    chosen = filedialog.askdirectory(
        parent=st["root"],
        title="Data folder (holds SdfSamples and splits)",
        initialdir=str(st["tr_data_root"]),
        mustexist=True,
    )
    if not chosen:
        return False
    set_data_folder(st, chosen)
    return True


def set_data_folder(st, folder):
    """List the datasets under ``folder`` from now on, and remember it.

    The data root is what a run's specs name as ``DataSource``, and the
    trainer reads ``<root>/SdfSamples`` and ``<root>/splits`` from it - so a
    pick of ``SdfSamples`` or of one dataset inside it is moved up to its
    root (``datasets.data_root_for``), with that dataset selected.
    """
    root, name = datasets.data_root_for(folder)
    st["tr_data_root"] = root
    folders.remember(folders.TRAIN_DATA, root)
    refresh_datasets(st)
    if name is not None:
        for label, row in st["tr_datasets"].items():
            if row["name"] == name:
                st["tr_combo"].set(label)
                _update_readiness(st)
                break
    n = len(st["tr_datasets"])
    widgets.append(
        st["tr_log"],
        (
            f"Datasets from {root}: {n} found."
            if n
            else f"No dataset under {root}. A data folder holds "
            f"{datasets.SDF_SAMPLES_DIR}/<dataset>/ and {datasets.SPLITS_DIR}/"
            "<dataset>.json, the way datagen --data-root writes them."
        ),
    )


def refresh_runs(st):
    """Re-read the run directories and redraw the table in the current order."""
    st["tr_runs"] = training.list_runs(st["tr_runs_dir"])
    _fill_runs(st)


def sort_runs(st, key):
    """Sort the runs table by a column; the same column again flips the
    direction. Dates start newest first, everything else ascending."""
    current, descending = st["tr_runs_sort"]
    if key == current:
        descending = not descending
    else:
        descending = key == "date"
    st["tr_runs_sort"] = (key, descending)
    _fill_runs(st)


def _run_sort_key(key):
    """Sort key for one column: text case-insensitively, numbers as numbers,
    missing values (a run without metadata) last in either direction."""

    def value(row):
        raw = row.get(key)
        if key in ("latent_dim", "epochs", "train_seconds", "final_loss"):
            missing = not isinstance(raw, (int, float))
            return (missing, 0 if missing else float(raw))
        return (raw is None, str(raw or "").lower())

    return value


def _fill_runs(st):
    tree = st["tr_tree"]
    key, descending = st["tr_runs_sort"]
    selected = selected_run(st)
    keep = selected["name"] if selected else None
    tree.delete(*tree.get_children())
    for row_key, col, title, _ in RUN_COLUMNS:
        mark = "" if row_key != key else ("  ▾" if descending else "  ▴")
        tree.heading(col, text=title + mark)
    rows = sorted(st["tr_runs"], key=_run_sort_key(key), reverse=descending)
    for row in rows:
        loss = row.get("final_loss")
        date = (row.get("date") or "")[:16].replace("T", " ")
        if date and row.get("date_is_estimate"):
            date = "~" + date  # from the specs file, not a finished run
        item = tree.insert(
            "",
            "end",
            text=row["name"],
            values=(
                row.get("dataset") or "-",
                row.get("latent_dim") if row.get("latent_dim") is not None else "?",
                # "80/800" for a run that stopped short of its epochs
                training.epochs_text(row.get("epochs"), row.get("last_epoch")),
                # of the epochs it holds, so "80/800" reads with its own time
                training.duration_text(row.get("train_seconds")),
                f"{loss:.4f}" if isinstance(loss, (int, float)) else "-",
                date,
                # one line in the table; the editor shows the notes in full
                " ".join(str(row.get("description") or "").split()),
            ),
        )
        if row["name"] == keep:
            tree.selection_set(item)


def selected_run(st):
    """The ``training.list_runs`` row selected in the table, or ``None``."""
    selection = st["tr_tree"].selection()
    if not selection:
        return None
    name = st["tr_tree"].item(selection[0], "text")
    return next((r for r in st["tr_runs"] if r["name"] == name), None)


def _require_selection(st):
    row = selected_run(st)
    if row is None:
        messagebox.showinfo("No run selected", "Click a run in the table first.")
    return row


def _select_row(st, name) -> bool:
    """Select and scroll to the row of run ``name``, if the table has it."""
    tree = st["tr_tree"]
    for item in tree.get_children():
        if tree.item(item, "text") == name:
            tree.selection_set(item)
            tree.focus(item)
            tree.see(item)
            return True
    return False


def edit_run(st):
    """Rename the selected run or change its notes, in the shared editor."""
    row = _require_selection(st)
    if row is None:
        return None
    return run_editor.open_editor(
        st, st["tr_runs_dir"], row["name"], on_done=lambda name: _select_row(st, name)
    )


def delete_run(st) -> bool:
    """Delete the selected run directory, after a confirmation."""
    row = _require_selection(st)
    if row is None:
        return False
    name = row["name"]
    if run_editor.training_now(st, name, st["tr_runs_dir"]):
        messagebox.showinfo(
            "Training", f"'{name}' is being trained right now; wait for it to finish."
        )
        return False
    contents = (
        "specs, checkpoints, latent codes and logs"
        if row.get("trained")
        else "its specs and metadata"
    )
    if not messagebox.askyesno(
        "Delete run",
        f"Delete '{name}' - {contents}?\n\n{row['path']}\n\nThis cannot be undone.",
        icon="warning",
        default="no",
    ):
        return False
    try:
        training.delete_run(st["tr_runs_dir"], name)
    except (OSError, ValueError) as exc:
        messagebox.showerror("Delete run", f"Could not delete '{name}':\n{exc}")
        run_editor.runs_changed(st)  # show whatever is left of it
        return False
    widgets.append(st["tr_log"], f"Run '{name}' deleted.")
    run_editor.runs_changed(st)
    return True


def load_run_hparams(st):
    """Open the hyperparameter window with the selected run's settings loaded.

    The same as picking the run under "Start from" and pressing Load - one
    click from the table instead of a search through that list.
    """
    row = _require_selection(st)
    if row is None:
        return None
    window = open_hparams(st)
    label = f"run: {row['name']}"
    if label not in window.sources:
        # the window was already open when this run appeared on disk
        window.status.set(
            f"'{row['name']}' is not in this window's Start from list; "
            "close the window and try again."
        )
        return window
    window.source_combo.set(label)
    window.load_selected()
    return window


def _popup_menu(st, event):
    """Right-click: select the row under the pointer, then show the actions."""
    tree = st["tr_tree"]
    item = tree.identify_row(event.y)
    if item:
        tree.selection_set(item)
        tree.focus(item)
    try:
        st["tr_run_menu"].tk_popup(event.x_root, event.y_root)
    finally:
        st["tr_run_menu"].grab_release()


def _open_in_explore(st):
    """Hand a run to the explorer that can show it: lattice or planar."""
    row = selected_run(st)
    if row is None:
        return
    name = row["name"]
    tab_explore.refresh_models(st)
    tab_explore2d.refresh_models(st)
    # either explorer may be listing another folder; the run is in this one
    if tab_explore.select_model(st, name, st["tr_runs_dir"]):
        st["notebook"].select(st["tab_frames"]["explore"])
    elif tab_explore2d.select_model(st, name, st["tr_runs_dir"]):
        st["notebook"].select(st["tab_frames"]["explore2d"])
    else:
        messagebox.showinfo(
            "Not loadable",
            f"'{name}' has no saved checkpoint yet, so the explorer cannot open it.",
        )


# (state key, button text, action): the buttons under the runs table and the
# entries of its right-click menu, in this order
RUN_ACTIONS = (
    ("open", "Open", _open_in_explore),
    ("edit", "Edit...", edit_run),
    ("hparams", "Load hyperparameters", load_run_hparams),
    ("delete", "Delete...", delete_run),
)


def _update_readiness(st):
    """State the shapes-per-dimension problem before the run, not after it."""
    _update_hp_summary(st)
    row = st.get("tr_datasets", {}).get(st["tr_combo"].get())
    if row is None:
        samples = Path(st["tr_data_root"]) / datasets.SDF_SAMPLES_DIR
        st["tr_readiness"].set(
            f"No dataset found under {folders.display(samples)}. "
            "Browse... to another data folder, or build one in the SDF maker."
        )
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
        lambda hp, origin: apply_hparams(st, hp, origin),
        n_shapes=row["n_instances"] if row is not None else None,
        dataset=row["name"] if row is not None else None,
        sources=training.spec_sources(st["tr_runs_dir"]),
        preview_paths=(
            split or "<split of the selected dataset>",
            str(st["tr_data_root"]),
        ),
        geom_dimension=_geom(row),
        origin=st.get("tr_hp_origin"),
        sheet_dir=st.get("tr_sheet_dir"),
    )


def apply_hparams(st, hp, origin=None):
    """Make ``hp`` the set the next run trains with, card spinboxes included.

    ``origin`` is where the window's draft was loaded from (a run, a shipped
    decoder, a sheet), or None for values typed in or reset; it is kept so
    the card and the next opening of the window can say so.
    """
    before = current_hparams(st)
    previous_origin = st.get("tr_hp_origin")
    st["tr_hparams"] = dict(hp)
    st["tr_hp_origin"] = origin
    if origin and origin.get("path"):
        # the next file dialog opens where the last sheet came from
        path = Path(origin["path"])
        if path.suffix.lower() in hparam_window.SHEET_SUFFIXES and path.is_file():
            st["tr_sheet_dir"] = path.parent
    for key, var_name in CARD_VARS.items():
        st[var_name].set(hp[key])
    _update_hp_summary(st)
    if origin and origin != previous_origin:
        widgets.append(st["tr_log"], f"Hyperparameters loaded from {origin['label']}.")
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
        origin = hyperparams.origin_summary(st.get("tr_hp_origin"), hp)
        st["tr_hp_summary"].set(
            (origin + "\n" if origin else "") + hyperparams.summary(hp)
        )
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
    # read now, on the Tk thread: a Browse... during the run must not change
    # the root the worker hands the trainer
    data_root = Path(st["tr_data_root"])

    def work(log):
        specs = training.write_specs(
            run_dir, row["split"], data_root, hp, geom_dimension=geom
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
            info = training.train(run_dir, data_root, log=log)
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
    run_editor.runs_changed(st)
