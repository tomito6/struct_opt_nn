"""The hyperparameter window of the Train tab.

Opened from the "All hyperparameters..." button on the Decoder card. It shows
every field of :mod:`structsept.app.hyperparams` - architecture, latent codes,
loss, both learning-rate schedules, sampling, budget and checkpoints - each with
a sentence on what it does and the specs.json key it is written to, and checks
the whole set while you type.

It edits a *draft*. Nothing reaches the Train tab until Apply, and Apply is
refused while any value is an error, so the card never holds a set the trainer
would crash on minutes into a run. The four values that also have a spinbox on
the card - d, hidden layers, width, epochs - are the same values: the window
opens with what the card shows and writes them back on Apply. The window holds
the input grab while it is open, so the card cannot change under the draft.

Layout
------
Two pages. *Settings* is the form: one card per group in two columns, one row
per value with its help text under it; a value that differs from its default is
drawn in the accent colour and marked with a dot, an entry that does not parse
gets a red border. *specs.json* is the file the next run will write, rendered
from the draft - the unambiguous answer to "what exactly will the trainer
see?". Under both, the *Checks* panel lists what ``hyperparams.validate``
found; clicking a line jumps to the value it is about.

The learning-rate card is laid out as a table rather than two stacked copies:
the trainer's Adam has two parameter groups, decoder weights and latent codes,
each with its own schedule, and the six rows of a schedule mean the same thing
in both columns. A row the selected schedule type does not use shows a dash in
that column and disappears when neither column uses it.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from structsept.app import hyperparams, runtime, theme, widgets
from structsept.app.hyperparams import ERROR, NOTE, WARNING

VALIDATE_MS = 150
HELP_WRAP = 470
MAX_ISSUES = 6
JOB_KEY = "tr_job_hparams"  # main._shutdown cancels every "tr_job_*" key

_LEVEL_STYLE = {
    ERROR: "Card.Danger.TLabel",
    WARNING: "Card.Warning.TLabel",
    NOTE: "Card.Subtle.TLabel",
}
_LEVEL_WORD = {ERROR: "Error", WARNING: "Warning", NOTE: "Note"}

# Short reminders of the syntax beside entries whose format is not obvious.
_HINTS = {
    "int_list": "e.g. 2, 4 · empty = none",
    "layers": "all · none · 0, 1, 2",
}


def open_window(st, hp, on_apply, **kwargs):
    """Open the window, or raise the one that is already open.

    Parameters
    ----------
    st : dict
        Shared app state; needs ``root`` and ``palette``.
    hp : dict
        Hyperparameter set the draft starts from.
    on_apply : callable
        Called with the new hyperparameter set when Apply succeeds.
    **kwargs
        Passed to :class:`HyperparamWindow`.
    """
    existing = st.get("tr_hp_window")
    if existing is not None and existing.alive():
        existing.top.deiconify()
        existing.top.lift()
        existing.top.focus_set()
        return existing
    window = HyperparamWindow(st, hp, on_apply, **kwargs)
    st["tr_hp_window"] = window
    return window


class HyperparamWindow:
    """A modal Toplevel editing a draft copy of a hyperparameter set.

    Parameters
    ----------
    st : dict
        Shared app state.
    hp : dict
        Starting values. Copied; the caller's dict is never touched.
    on_apply : callable
        Receives the validated hyperparameter set on Apply.
    n_shapes : int, optional
        Shapes in the dataset selected on the Train tab, for the batch check.
    dataset : str, optional
        Name of that dataset, shown in the toolbar.
    sources : sequence of (str, path), optional
        ``(label, specs.json path)`` pairs offered under "Start from".
    preview_paths : (str, str), optional
        TrainSplit and DataSource written into the specs.json preview.
    geom_dimension : int, optional
        Coordinates per sample of that dataset, for the width checks and the
        preview.
    """

    def __init__(
        self,
        st,
        hp,
        on_apply,
        n_shapes=None,
        dataset=None,
        sources=(),
        preview_paths=("<split of the selected dataset>", "<data root>"),
        geom_dimension=hyperparams.GEOM_DIMENSION,
    ):
        self.st = st
        self.palette = st["palette"]
        self.on_apply = on_apply
        self.n_shapes = n_shapes
        self.geom_dimension = int(geom_dimension)
        self.sources = {label: Path(path) for label, path in sources}
        self.preview_paths = preview_paths
        self.initial = {k: _copy(v) for k, v in hp.items()}

        self.vars: dict[str, tk.Variable] = {}
        self.inputs: dict[str, tk.Widget] = {}
        # every row that can be restyled or hidden, keyed by its first field
        self.rows: dict[str, dict] = {}
        self.issues: list = []
        self.draft: dict | None = None
        self._closed = False

        p = self.palette
        root = st["root"]
        top = self.top = tk.Toplevel(root)
        top.title("Training hyperparameters")
        top.configure(background=p["bg"])
        top.transient(root)
        theme.clamp_geometry(top, 1140, 900)
        try:
            top.geometry(f"+{root.winfo_rootx() + 60}+{root.winfo_rooty() + 30}")
        except tk.TclError:
            pass
        top.protocol("WM_DELETE_WINDOW", self.cancel)
        top.bind("<Escape>", lambda e: self.cancel())
        top.bind("<Destroy>", self._on_destroy, add="+")
        top.bind("<Map>", self._on_map, add="+")
        top.bind("<MouseWheel>", self._on_wheel, add="+")

        self._build_header(dataset)
        self._build_footer()
        self._build_checks()
        self._build_pages()

        self._set_draft(self.initial)
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self._schedule())
        self._revalidate()

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #

    def _build_header(self, dataset):
        head = ttk.Frame(self.top, padding=(14, 12, 14, 0))
        head.pack(fill="x")
        ttk.Label(head, text="Training hyperparameters", style="Title.TLabel").pack(
            side="left"
        )
        ttk.Label(
            head,
            text="everything the next run writes to specs.json  ·  "
            "highlighted values differ from the defaults",
            style="Subtle.TLabel",
        ).pack(side="left", padx=(12, 0), pady=(6, 0))

        bar = ttk.Frame(self.top, padding=(14, 10, 14, 0))
        bar.pack(fill="x")
        ttk.Label(bar, text="Start from").pack(side="left")
        self.source_combo = ttk.Combobox(
            bar, state="readonly", width=36, values=list(self.sources)
        )
        self.source_combo.pack(side="left", padx=(6, 4))
        self.source_combo.bind("<MouseWheel>", self._swallow_wheel)
        ttk.Button(
            bar, text="Load", style="Ghost.TButton", command=self.load_selected
        ).pack(side="left")
        ttk.Button(
            bar, text="Reset to defaults", style="Ghost.TButton", command=self.reset
        ).pack(side="left", padx=(10, 0))
        if dataset:
            shapes = "" if self.n_shapes is None else f"  ·  {self.n_shapes} shape(s)"
            ttk.Label(
                bar, text=f"Dataset: {dataset}{shapes}", style="Subtle.TLabel"
            ).pack(side="right")

        self.status = tk.StringVar(value="")
        self.status_label = ttk.Label(
            self.top,
            textvariable=self.status,
            style="Subtle.TLabel",
            wraplength=1000,
            justify="left",
            padding=(14, 4, 14, 0),
        )
        self.status_label.pack(fill="x")

    def _build_footer(self):
        foot = ttk.Frame(self.top, padding=(14, 8, 14, 12))
        foot.pack(side="bottom", fill="x")
        self.btn_apply = ttk.Button(
            foot, text="Apply", style="Accent.TButton", command=self.apply
        )
        self.btn_apply.pack(side="right")
        ttk.Button(foot, text="Cancel", command=self.cancel).pack(
            side="right", padx=(0, 8)
        )
        self.counts = tk.StringVar(value="")
        ttk.Label(foot, textvariable=self.counts, style="Subtle.TLabel").pack(
            side="left"
        )

    def _build_checks(self):
        outer, body = widgets.card(
            self.top, self.palette, "Checks", "errors block Apply · click one to jump"
        )
        outer.pack(side="bottom", fill="x", padx=14)
        self.checks_list = ttk.Frame(body, style="Card.TFrame")
        self.checks_list.pack(fill="x")

    def _build_pages(self):
        p = self.palette
        self.notebook = ttk.Notebook(self.top)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(8, 8))
        self.page_settings = ttk.Frame(self.notebook)
        self.page_preview = ttk.Frame(self.notebook, padding=(0, 8, 0, 0))
        self.notebook.add(self.page_settings, text="Settings")
        self.notebook.add(self.page_preview, text="specs.json")
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._refresh_preview())

        # -- settings: a scrolling canvas holding two columns of cards ------- #
        canvas = tk.Canvas(
            self.page_settings, background=p["bg"], highlightthickness=0, bd=0
        )
        bar = ttk.Scrollbar(self.page_settings, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas, padding=(0, 10, 10, 10))
        item = canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
        self.canvas, self.inner = canvas, inner

        columns = [ttk.Frame(inner), ttk.Frame(inner)]
        for index, column in enumerate(columns):
            inner.columnconfigure(index, weight=1, uniform="column")
            column.grid(
                row=0, column=index, sticky="new", padx=(0, 6) if index == 0 else (6, 0)
            )
        for key, title, subtitle, col in hyperparams.GROUPS:
            outer, body = widgets.card(columns[col], p, title, subtitle)
            outer.pack(fill="x", pady=(0, 10))
            if key == "lr":
                self._build_schedule_table(body)
            else:
                self._build_group(body, key)
        self._build_fixed(columns[1])

        # -- preview: the specs.json the draft would write ------------------- #
        frame, self.preview = widgets.log_box(self.page_preview, p, height=20)
        frame.pack(fill="both", expand=True)

    def _build_group(self, body, group):
        grid = ttk.Frame(body, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(2, weight=1)
        row = 0
        for field in hyperparams.fields_in(group):
            label = ttk.Label(grid, text=field.label, style="Card.TLabel")
            label.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=(4, 0))
            entry = self._make_input(grid, field)
            span = 2 if field.kind == "text" else 1
            entry.grid(row=row, column=1, columnspan=span, sticky="w", pady=(4, 0))
            hint = _HINTS.get(field.kind)
            if field.kind in ("opt_int", "opt_float"):
                hint = f"empty = {field.blank}"
            if hint and span == 1:
                ttk.Label(grid, text=hint, style="Card.Subtle.TLabel").grid(
                    row=row, column=2, sticky="w", padx=(8, 0), pady=(4, 0)
                )
            help_label = ttk.Label(
                grid,
                text=f"{field.help}  [{field.spec}]",
                style="Card.Subtle.TLabel",
                wraplength=HELP_WRAP,
                justify="left",
            )
            help_label.grid(
                row=row + 1, column=0, columnspan=3, sticky="w", pady=(1, 4)
            )
            self.rows[field.key] = {
                "keys": (field.key,),
                "label": label,
                "text": field.label,
                "anchor": label,
            }
            row += 2

    def _build_schedule_table(self, body):
        """Decoder and latent-code schedules side by side, one row per value."""
        grid = ttk.Frame(body, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(3, weight=1)
        columns = hyperparams.subgroups("lr")
        for index, sub in enumerate(columns):
            ttk.Label(grid, text=sub, style="Card.Subtle.TLabel").grid(
                row=0, column=1 + index, sticky="w", padx=(0, 12)
            )
        left, right = (hyperparams.fields_in("lr", sub) for sub in columns)
        row = 1
        for a, b in zip(left, right):
            label = ttk.Label(grid, text=a.label, style="Card.TLabel")
            label.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=(4, 0))
            cells = {}
            for index, field in enumerate((a, b)):
                entry = self._make_input(grid, field, width=11)
                dash = ttk.Label(grid, text="—", style="Card.Subtle.TLabel")
                for widget in (entry, dash):
                    widget.grid(
                        row=row, column=1 + index, sticky="w", padx=(0, 12), pady=(4, 0)
                    )
                dash.grid_remove()
                cells[field.key] = (entry, dash, field)
            help_label = ttk.Label(
                grid,
                text=f"{a.help}  [{a.spec.replace('[0]', '[0|1]')}]",
                style="Card.Subtle.TLabel",
                wraplength=HELP_WRAP,
                justify="left",
            )
            help_label.grid(
                row=row + 1, column=0, columnspan=4, sticky="w", pady=(1, 4)
            )
            self.rows[a.key] = {
                "keys": (a.key, b.key),
                "label": label,
                "text": a.label,
                "anchor": label,
                "cells": cells,
                "widgets": (label, help_label),
                "schedules": a.schedules,
            }
            row += 2

    def _build_fixed(self, parent):
        outer, body = widgets.card(
            parent,
            self.palette,
            "Not editable here",
            "pinned by the app or hardcoded in the trainer",
        )
        outer.pack(fill="x", pady=(0, 10))
        grid = ttk.Frame(body, style="Card.TFrame")
        grid.pack(fill="x")
        row = 0
        for what, value, why in hyperparams.FIXED:
            ttk.Label(grid, text=what, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=(4, 0)
            )
            ttk.Label(grid, text=value, style="Card.Value.TLabel").grid(
                row=row, column=1, sticky="w", pady=(4, 0)
            )
            ttk.Label(
                grid,
                text=why,
                style="Card.Subtle.TLabel",
                wraplength=HELP_WRAP,
                justify="left",
            ).grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(1, 4))
            row += 2

    def _make_input(self, parent, field, width=None):
        if field.kind == "bool":
            var = tk.BooleanVar(value=False)
            widget = ttk.Checkbutton(parent, variable=var, style="Card.TCheckbutton")
        elif field.kind == "choice":
            var = tk.StringVar(value="")
            widget = ttk.Combobox(
                parent,
                textvariable=var,
                values=field.choices,
                state="readonly",
                width=width or 15,
            )
            # a readonly combobox cycles its value on the wheel; in a scrolling
            # form that silently edits whatever the pointer passes over
            widget.bind("<MouseWheel>", self._swallow_wheel)
        else:
            var = tk.StringVar(value="")
            if width is None:
                width = {"text": 44, "int_list": 14, "layers": 14}.get(field.kind, 12)
            widget = ttk.Entry(parent, textvariable=var, width=width)
        self.vars[field.key] = var
        self.inputs[field.key] = widget
        return widget

    # ------------------------------------------------------------------ #
    # draft <-> values
    # ------------------------------------------------------------------ #

    def _set_draft(self, hp):
        for field in hyperparams.FIELDS:
            value = hp.get(field.key, field.default)
            var = self.vars[field.key]
            if field.kind == "bool":
                var.set(bool(value))
            else:
                var.set(hyperparams.format_value(field, value))

    def read_draft(self):
        """``(hp, parse_errors)`` from the entries as they stand.

        ``parse_errors`` maps a field key to the reason its text is not a
        value; those fields keep their default in ``hp`` so the rest of the
        set can still be checked and previewed.

        A schedule value the selected schedule type does not use - a Step
        interval under Warmup - is hidden, so it can never be an error: a
        typo left in it would block Apply with no row to fix it in. It keeps
        the value the window opened with instead.
        """
        hp = hyperparams.defaults()
        bad = {}
        # the schedule types first: they decide which schedule rows count
        order = sorted(hyperparams.FIELDS, key=lambda f: not f.key.endswith("_type"))
        for field in order:
            var = self.vars[field.key]
            try:
                raw = var.get()
            except tk.TclError:
                raw = ""
            try:
                hp[field.key] = hyperparams.parse(field, raw)
            except ValueError as exc:
                if hyperparams.in_use(field, hp):
                    bad[field.key] = str(exc)
                else:
                    kept = self.initial.get(field.key, field.default)
                    if hyperparams.check(field, kept):
                        kept = field.default
                    hp[field.key] = kept
        return hp, bad

    def dirty(self) -> bool:
        hp, bad = self.read_draft()
        return bool(bad) or any(
            hp[f.key] != self.initial.get(f.key, f.default) for f in hyperparams.FIELDS
        )

    # ------------------------------------------------------------------ #
    # validation and restyling
    # ------------------------------------------------------------------ #

    def _schedule(self):
        if self._closed:
            return
        runtime.reschedule(self.st, JOB_KEY, VALIDATE_MS, self._timer_fired)

    def _timer_fired(self):
        self.st[JOB_KEY] = None
        self._revalidate()

    def _revalidate(self):
        """Parse, check and restyle everything. Cheap: 36 fields, no torch."""
        if self._closed:
            return
        hp, bad = self.read_draft()
        issues = [
            hyperparams.Issue(
                ERROR, key, f"{hyperparams.FIELD_BY_KEY[key].full_label}: {msg}."
            )
            for key, msg in bad.items()
        ]
        # An unparseable entry keeps its default in ``hp``, so the rest of the
        # set is still checked: fixing one red field should not reveal a second
        # one only afterwards. Findings about the unparseable fields themselves
        # would describe the default, not the text, and are dropped.
        issues += [
            i
            for i in hyperparams.validate(hp, self.n_shapes, self.geom_dimension)
            if i.key not in bad
        ]
        order = {ERROR: 0, WARNING: 1, NOTE: 2}
        issues.sort(key=lambda i: order[i.level])
        self.issues = issues
        self.draft = None if hyperparams.errors(issues) else hp

        error_keys = {i.key for i in issues if i.level == ERROR}
        for key, widget in self.inputs.items():
            if isinstance(widget, ttk.Entry) and not isinstance(widget, ttk.Combobox):
                widget.configure(
                    style="Invalid.TEntry" if key in error_keys else "TEntry"
                )

        base = hyperparams.defaults()
        for row in self.rows.values():
            differs = any(k in bad or hp[k] != base[k] for k in row["keys"])
            row["label"].configure(
                text=row["text"] + ("  •" if differs else ""),
                style="Card.Accent.TLabel" if differs else "Card.TLabel",
            )
        self._show_schedule_rows()
        self._render_checks()

        n_changed = len(hyperparams.changed(hp, include_card=True))
        n_errors = len(hyperparams.errors(issues))
        self.counts.set(
            f"{n_changed} value(s) differ from the defaults"
            + (f"  ·  {n_errors} error(s)" if n_errors else "")
        )
        self.btn_apply.configure(state="disabled" if n_errors else "normal")
        self._refresh_preview()

    def _show_schedule_rows(self):
        types = {
            prefix: self.vars[f"{prefix}_type"].get()
            for prefix in ("lr_dec", "lr_code")
        }
        for row in self.rows.values():
            schedules = row.get("schedules")
            if "cells" not in row:
                continue
            used_somewhere = False
            for key, (entry, dash, field) in row["cells"].items():
                prefix = key.rsplit("_", 1)[0]
                used = not schedules or types[prefix] in schedules
                used_somewhere |= used
                if used:
                    dash.grid_remove()
                    entry.grid()
                else:
                    entry.grid_remove()
                    dash.grid()
            for widget in row["widgets"]:
                if used_somewhere:
                    widget.grid()
                else:
                    widget.grid_remove()
            if not used_somewhere:
                for entry, dash, _ in row["cells"].values():
                    entry.grid_remove()
                    dash.grid_remove()

    def _render_checks(self):
        for child in self.checks_list.winfo_children():
            child.destroy()
        if not self.issues:
            ttk.Label(
                self.checks_list, text="No problems found.", style="Card.Success.TLabel"
            ).pack(anchor="w")
            return
        for issue in self.issues[:MAX_ISSUES]:
            label = ttk.Label(
                self.checks_list,
                text=f"{_LEVEL_WORD[issue.level]}: {issue.message}",
                style=_LEVEL_STYLE[issue.level],
                wraplength=1040,
                justify="left",
            )
            label.pack(anchor="w", fill="x")
            if issue.key:
                label.configure(cursor="hand2")
                label.bind("<Button-1>", lambda e, k=issue.key: self.focus_field(k))
        extra = len(self.issues) - MAX_ISSUES
        if extra > 0:
            ttk.Label(
                self.checks_list,
                text=f"+{extra} more - fix the ones above first.",
                style="Card.Subtle.TLabel",
            ).pack(anchor="w")

    def _refresh_preview(self):
        if self._closed:
            return
        try:
            if self.notebook.select() != str(self.page_preview):
                return
        except tk.TclError:
            return
        hp, bad = self.read_draft()
        if bad:
            text = "Fix the values marked in red first:\n\n" + "\n".join(
                f"  {hyperparams.FIELD_BY_KEY[k].full_label}: {msg}"
                for k, msg in bad.items()
            )
        else:
            text = compact_json(
                hyperparams.to_specs(hp, *self.preview_paths, self.geom_dimension)
            )
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("end", text)
        self.preview.configure(state="disabled")

    def focus_field(self, key):
        """Bring the row holding ``key`` into view and put the cursor on it."""
        widget = self.inputs.get(key)
        if widget is None:
            return
        self.notebook.select(self.page_settings)
        self.top.update_idletasks()
        total = max(self.inner.winfo_height(), 1)
        y = widget.winfo_rooty() - self.inner.winfo_rooty()
        self.canvas.yview_moveto(max(0.0, (y - 60) / total))
        try:
            widget.focus_set()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #

    def load_selected(self):
        """Replace the draft with the settings of the run picked under
        "Start from"."""
        label = self.source_combo.get()
        path = self.sources.get(label)
        if path is None:
            self.status.set("Pick a run or a shipped decoder under 'Start from' first.")
            return
        try:
            specs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.status.set(f"Could not read {path}: {exc}")
            return
        try:
            hp, notes = hyperparams.from_specs(specs)
        except Exception as exc:  # noqa: BLE001 - a hand-edited file
            # from_specs notes every value it cannot use, but a file whose
            # *structure* is off (NetworkSpecs a list, say) can still raise
            self.status.set(f"Could not use {path}: {exc!r}")
            return
        self._set_draft(hp)
        message = f"Loaded the settings of {label}."
        if notes:
            message += "  " + "  ".join(notes)
        self.status.set(message)
        self._revalidate()

    def reset(self):
        self._set_draft(hyperparams.defaults())
        self.status.set("Every value is back at its default.")
        self._revalidate()

    def apply(self) -> bool:
        """Hand the draft to the Train tab and close. Refused on any error."""
        self._revalidate()
        if self.draft is None:
            self.top.bell()
            first = next((i for i in self.issues if i.level == ERROR), None)
            if first is not None and first.key:
                self.focus_field(first.key)
            return False
        hp = self.draft
        self.close()
        self.on_apply(hp)
        return True

    def cancel(self):
        """Close without applying; asks first when the draft was edited."""
        if self.dirty() and not messagebox.askyesno(
            "Discard changes?",
            "The hyperparameters were edited but not applied. Discard the edits?",
            parent=self.top,
        ):
            return
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.top.grab_release()
        except tk.TclError:
            pass
        try:
            self.top.destroy()
        except tk.TclError:
            pass

    def alive(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def _on_map(self, event):
        if event.widget is not self.top:
            return
        try:
            self.top.grab_set()
            self.top.focus_set()
        except tk.TclError:
            # not viewable yet; the next <Map> tries again
            pass

    def _on_destroy(self, event):
        # <Destroy> on a Toplevel also fires for every child it takes down
        if event.widget is not self.top:
            return
        self._closed = True
        job = self.st.get(JOB_KEY)
        if job is not None:
            try:
                self.st["root"].after_cancel(job)
            except tk.TclError:
                pass
            self.st[JOB_KEY] = None
        if self.st.get("tr_hp_window") is self:
            self.st["tr_hp_window"] = None

    def _on_wheel(self, event):
        try:
            if self.notebook.select() != str(self.page_settings):
                return None
        except tk.TclError:
            return None
        self.canvas.yview_scroll(
            int(-event.delta / 120) or -1 * _sign(event.delta), "units"
        )
        return None

    def _swallow_wheel(self, event):
        self._on_wheel(event)
        return "break"


def compact_json(value, indent=4, _level=0) -> str:
    """``json.dumps(value, indent=4)``, but with lists of plain values on one
    line.

    The preview is read, not parsed: ``dims`` as six lines of ``128`` and the
    dropout indices as eight more push the rest of the file off the page. The
    content is identical to what ``training.write_specs`` writes; only the
    whitespace differs.
    """
    pad = " " * (indent * (_level + 1))
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{pad}{json.dumps(k, ensure_ascii=False)}: "
            f"{compact_json(v, indent, _level + 1)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * (indent * _level) + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [f"{pad}{compact_json(v, indent, _level + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * (indent * _level) + "]"
    return json.dumps(value, ensure_ascii=False)


def _sign(x):
    return (x > 0) - (x < 0)


def _copy(value):
    return list(value) if isinstance(value, list) else value
