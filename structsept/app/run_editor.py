"""Edit a run in place: its name and its notes.

A run is a directory under ``runs/`` - ``specs.json`` plus whatever the
trainer wrote next to it - and the two things about it worth changing
afterwards are the name of that directory and the free text stored as
``Description`` in its specs. A decoder "trained here" on the Explore tabs
*is* such a directory, so the Train tab's runs table and both decoder
pickers open this one window on the same run; the pretrained decoders
shipped with the library are not runs and stay read-only.

The file operations are ``training.rename_run``, ``training.delete_run`` and
``training.write_description``. This module is the window over the first
and the last, and the fan-out afterwards: :func:`runs_changed` makes every
list that shows runs re-read the disk, so a rename made on Explore 2-D is on
the Train tab at once without the tabs importing each other.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import DeepSDFStruct.deep_sdf.workspace as ws

from structsept.app import theme, training, widgets

STATE_KEY = "run_editor"


def runs_changed(st):
    """Re-read the run directories in every list that shows them.

    A tab registers a callable in ``st["run_listeners"]`` when it builds -
    the Train tab's table and the two decoder pickers do - and anything that
    renames, deletes or finishes a run calls this.
    """
    for listener in list(st.get("run_listeners", [])):
        listener()


def training_now(st, name) -> bool:
    """True while the Train tab is training the run called ``name``.

    Its directory is written every few epochs: renamed under the trainer,
    the next checkpoint fails; deleted, the run is gone. Another window
    training the same run cannot be seen from here - Windows then refuses
    the rename with a file-in-use error, which the editor reports as such.
    """
    watched = st.get("tr_watch_dir")
    return bool(st.get("busy")) and watched is not None and Path(watched).name == name


def open_editor(st, runs_dir, name, on_done=None):
    """Open the editor on run ``name`` under ``runs_dir``, or raise the open one.

    ``on_done(new_name)`` runs after a successful save, once the lists were
    refreshed, so the caller can point its picker at the run again under
    whatever it is called now. Returns the :class:`RunEditor`, or ``None``
    when the run is being trained right now.
    """
    existing = st.get(STATE_KEY)
    if existing is not None and existing.alive():
        existing.top.lift()
        existing.top.focus_set()
        return existing
    if training_now(st, name):
        messagebox.showinfo(
            "Training",
            f"'{name}' is being trained right now. Edit it once the run has finished.",
            parent=st["root"],
        )
        return None
    editor = RunEditor(st, runs_dir, name, on_done)
    st[STATE_KEY] = editor
    return editor


def describe(runs_dir, name) -> str:
    """One line saying what the run is: d, architecture, epochs, loss, state."""
    run_dir = Path(runs_dir) / name
    row = next((r for r in training.list_runs(runs_dir) if r["name"] == name), {})
    specs = training.read_specs(run_dir / ws.specifications_filename)
    dims = (specs.get("NetworkSpecs") or {}).get("dims") or []
    parts = []
    if row.get("latent_dim") is not None:
        parts.append(f"d={row['latent_dim']}")
    if dims:
        parts.append(
            f"{len(dims)}x{dims[0]}"
            if len(set(dims)) == 1
            else "x".join(str(x) for x in dims)
        )
    epochs, last = row.get("epochs"), row.get("last_epoch")
    if isinstance(last, int) and isinstance(epochs, int) and last < epochs:
        # stopped, died, or still training: the checkpoint is not the run
        # its specs describe
        parts.append(f"epoch {last} of {epochs}")
    elif epochs is not None:
        parts.append(f"{epochs} epochs")
    if isinstance(row.get("final_loss"), (int, float)):
        parts.append(f"final loss {row['final_loss']:.4f}")
    parts.append("checkpoint saved" if row.get("trained") else "no checkpoint yet")
    date = (row.get("date") or "")[:16].replace("T", " ")
    if date:
        parts.append(("~" if row.get("date_is_estimate") else "") + date)
    return "  ·  ".join(parts)


class RunEditor:
    """A small modal window: the run's name, its notes, and what it is.

    Nothing is written until Save. Notes are written first and the rename
    last, so a rename the file system refuses (a file of the run open
    elsewhere) still leaves the notes saved, and the status line says which
    step failed.
    """

    def __init__(self, st, runs_dir, name, on_done=None):
        self.st = st
        self.runs_dir = Path(runs_dir)
        self.name = name
        self.on_done = on_done
        self._closed = False
        p = st["palette"]
        root = st["root"]

        self.original_notes = training.read_description(self.runs_dir / name)

        top = self.top = tk.Toplevel(root)
        top.title(f"Edit run - {name}")
        top.configure(background=p["bg"])
        top.transient(root)
        top.resizable(True, False)
        try:
            top.geometry(f"+{root.winfo_rootx() + 140}+{root.winfo_rooty() + 140}")
        except tk.TclError:
            pass
        top.protocol("WM_DELETE_WINDOW", self.cancel)
        top.bind("<Escape>", lambda e: self.cancel())
        top.bind("<Destroy>", self._on_destroy, add="+")
        top.bind("<Map>", self._on_map, add="+")

        # the footer claims its height first, like the hyperparameter window
        foot = ttk.Frame(top, padding=(14, 8, 14, 12))
        foot.pack(side="bottom", fill="x")
        self.btn_save = ttk.Button(
            foot, text="Save", style="Accent.TButton", command=self.save
        )
        self.btn_save.pack(side="right")
        ttk.Button(foot, text="Cancel", command=self.cancel).pack(
            side="right", padx=(0, 8)
        )

        outer, body = widgets.card(top, p, "Edit run", name)
        outer.pack(fill="both", expand=True, padx=14, pady=(12, 0))
        wrap = 460

        ttk.Label(body, text="Name", style="Card.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar(value=name)
        self.name_entry = ttk.Entry(body, textvariable=self.name_var, width=60)
        self.name_entry.pack(fill="x", pady=(2, 2))
        self.name_entry.bind("<Return>", lambda e: self.save())
        ttk.Label(
            body,
            text="Renames the directory under runs/. The Explore tabs then list "
            "the decoder under the new name.",
            style="Card.Subtle.TLabel",
            wraplength=wrap,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(body, text="Notes", style="Card.TLabel").pack(anchor="w")
        frame = tk.Frame(
            body,
            background=p["surface"],
            highlightbackground=p["border"],
            highlightcolor=p["border"],
            highlightthickness=1,
            bd=0,
        )
        frame.pack(fill="x", pady=(2, 2))
        self.notes = tk.Text(frame, height=5, width=60, wrap="word")
        theme.style_text(self.notes, p)
        self.notes.pack(fill="both", expand=True)
        self.notes.insert("1.0", self.original_notes)
        ttk.Label(
            body,
            text="Kept as Description in the run's specs.json and shown in the "
            "runs table. Empty: the trainer's own one-liner.",
            style="Card.Subtle.TLabel",
            wraplength=wrap,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            body,
            text=describe(self.runs_dir, name) + f"\n{self.runs_dir / name}",
            style="Card.Subtle.TLabel",
            wraplength=wrap,
            justify="left",
        ).pack(anchor="w")

        self.status = tk.StringVar(value="")
        ttk.Label(
            body,
            textvariable=self.status,
            style="Card.Danger.TLabel",
            wraplength=wrap,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        self.name_entry.focus_set()
        self.name_entry.selection_range(0, "end")

    # ------------------------------------------------------------------ #
    # values
    # ------------------------------------------------------------------ #

    def read(self) -> tuple[str, str]:
        """The name and the notes as typed, without surrounding blanks."""
        return self.name_var.get().strip(), self.notes.get("1.0", "end-1c").strip()

    def dirty(self) -> bool:
        name, notes = self.read()
        return name != self.name or notes != self.original_notes

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #

    def save(self) -> bool:
        """Write the notes, rename the directory, refresh every list.

        Returns True when the window closed with everything saved; on a
        refusal the window stays open with the reason in its status line.
        """
        name, notes = self.read()
        problem = training.check_run_name(self.runs_dir, name, current=self.name)
        if problem:
            self.status.set(problem)
            return False
        if training_now(self.st, self.name):
            self.status.set(f"'{self.name}' is being trained; wait for it to finish.")
            return False
        changes = []
        try:
            if notes != self.original_notes:
                training.write_description(self.runs_dir / self.name, notes)
                self.original_notes = notes
                changes.append("notes updated")
            if name != self.name:
                training.rename_run(self.runs_dir, self.name, name)
                changes.append(f"renamed to '{name}'")
        except (OSError, ValueError) as exc:
            done = f" ({changes[0]} was saved)" if changes else ""
            self.status.set(f"Could not save: {exc}{done}")
            return False
        old, self.name = self.name, name
        self.close()
        if changes:
            runs_changed(self.st)
            log = self.st.get("tr_log")
            if log is not None:
                widgets.append(log, f"Run '{old}': {', '.join(changes)}.")
        if self.on_done is not None:
            self.on_done(name)
        return True

    def cancel(self):
        """Close without saving; asks first when something was edited."""
        if self.dirty() and not messagebox.askyesno(
            "Discard changes?",
            "The run was edited but not saved. Discard the edits?",
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
            pass  # not viewable yet; the next <Map> tries again

    def _on_destroy(self, event):
        # <Destroy> on a Toplevel also fires for every child it takes down
        if event.widget is not self.top:
            return
        self._closed = True
        if self.st.get(STATE_KEY) is self:
            self.st[STATE_KEY] = None
