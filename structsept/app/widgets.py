"""Reusable widgets of the structsept app.

Small, dependency-free building blocks shared by the Explore and Train tabs:
cards, log boxes, metric tiles, embedded matplotlib canvases and the spatial
control-point grid. Nothing here knows about DeepSDF; everything takes plain
numbers and callbacks.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from structsept.app import theme as theme_mod

# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #


def card(parent, palette, title=None, subtitle=None, padding=10):
    """White panel with a 1 px border and an optional heading.

    The border is not decoration: #FFFFFF on the #F2F5F8 page is a 1.09:1
    luminance ratio, so a borderless card is invisible.
    """
    outer = tk.Frame(
        parent,
        background=palette["surface"],
        highlightbackground=palette["border"],
        highlightcolor=palette["border"],
        highlightthickness=1,
        bd=0,
    )
    body = ttk.Frame(outer, style="Card.TFrame", padding=padding)
    body.pack(fill="both", expand=True)
    if title:
        head = ttk.Frame(body, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 6))
        ttk.Label(head, text=title, style="Card.Heading.TLabel").pack(side="left")
        if subtitle:
            ttk.Label(head, text=subtitle, style="Card.Subtle.TLabel").pack(
                side="left", padx=(8, 0)
            )
    return outer, body


def log_box(parent, palette, height=10, width=80):
    """Read-only scrolling text widget in the log's monospace font."""
    frame = tk.Frame(
        parent,
        background=palette["surface"],
        highlightbackground=palette["border"],
        highlightcolor=palette["border"],
        highlightthickness=1,
        bd=0,
    )
    text = tk.Text(frame, height=height, width=width, state="disabled")
    theme_mod.style_text(text, palette)
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    return frame, text


def append(text, line):
    """Append one line to a read-only log box and scroll to it."""
    text.configure(state="normal")
    text.insert("end", str(line) + "\n")
    text.see("end")
    text.configure(state="disabled")


# --------------------------------------------------------------------------- #
# small controls
# --------------------------------------------------------------------------- #


def spinbox(parent, var, low, high, step=1, width=5, command=None):
    return ttk.Spinbox(
        parent,
        textvariable=var,
        from_=low,
        to=high,
        increment=step,
        width=width,
        command=command,
    )


def read_int(var, fallback):
    """Spinboxes accept typed text, so a value can be empty or garbage.

    Snaps an unreadable value back to ``fallback`` - so never call it from a
    trace on ``var``. While the user retypes a field, the entry is empty for
    an instant; putting the old value back then makes "50" typed over "200"
    come out as "50200". Traces use :func:`peek_int`.
    """
    try:
        return int(var.get())
    except (tk.TclError, ValueError):
        var.set(fallback)
        return fallback


def peek_int(var):
    """The integer in ``var``, or ``None`` when it holds none. Never writes."""
    try:
        return int(var.get())
    except (tk.TclError, ValueError):
        return None


def combo_width(labels, low=44, high=96) -> int:
    """Width, in characters, that shows the longest of ``labels`` in full.

    A ``ttk.Combobox`` is sized in average character widths and clips
    anything longer - in the entry and in the drop-down list alike, since
    the list is as wide as the entry. At the old fixed 40 a run called
    ``plate_hole_2d_n134_d2_20260928_1338`` lost its tail. Bounded below so
    an empty list still gives a usable box, and above so one absurd name
    cannot push the buttons off the row.
    """
    longest = max((len(str(label)) for label in labels), default=0)
    return int(min(high, max(low, longest + 2)))


def folder_row(parent, textvariable, command, label="Folder", label_width=None):
    """A row naming the folder a picker lists, with a Browse... button.

    The button comes before the path, not after it: a path outside the repo
    can be long enough to run off the card, and it is the path that may be
    clipped, never the button. ``textvariable`` holds the path and whatever
    the caller says about it (how many entries were found there).
    """
    row = ttk.Frame(parent, style="Card.TFrame")
    ttk.Label(row, text=label, style="Card.TLabel", width=label_width).pack(side="left")
    ttk.Button(row, text="Browse...", style="CardGhost.TButton", command=command).pack(
        side="left", padx=(8, 8)
    )
    ttk.Label(row, textvariable=textvariable, style="Card.Subtle.TLabel").pack(
        side="left"
    )
    return row


def metric(parent, palette, label):
    """Flat tile showing one number. Returns the StringVar and the value label."""
    tile = ttk.Frame(parent, style="Card.TFrame")
    ttk.Label(tile, text=label, style="Card.Subtle.TLabel").pack(anchor="w")
    var = tk.StringVar(value="--")
    value = ttk.Label(tile, textvariable=var, style="Card.Value.TLabel")
    value.pack(anchor="w")
    return tile, var, value


def set_enabled(widget, enabled: bool):
    """Enable or disable a widget subtree, keeping readonly comboboxes readonly."""
    for child in widget.winfo_children():
        set_enabled(child, enabled)
    try:
        if isinstance(widget, ttk.Combobox):
            widget.configure(state="readonly" if enabled else "disabled")
        elif "state" in widget.configure():
            widget.configure(state="normal" if enabled else "disabled")
    except tk.TclError:
        pass


# --------------------------------------------------------------------------- #
# matplotlib canvases
# --------------------------------------------------------------------------- #


def figure_canvas(parent, palette, figsize, dpi=100.0):
    """Embedded matplotlib figure that blends into the surrounding card.

    Built with ``layout="constrained"`` from the start so the engine exists
    before any colorbar is added, and the Tk widget behind it is painted the
    card colour so a resize does not flash white.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
    fig.patch.set_facecolor(palette["surface"])
    canvas = FigureCanvasTkAgg(fig, master=parent)
    tkwidget = canvas.get_tk_widget()
    # FigureCanvasTkAgg hardcodes background="white" on its tk.Canvas, so the
    # option database never reaches it.
    tkwidget.configure(background=palette["surface"], highlightthickness=0, bd=0)
    tkwidget.pack(fill="both", expand=True)

    # A frozen layout keeps fractional axes positions, so it survives a resize
    # but drifts: titles start colliding with the canvas edge. Re-solve once
    # the resize has settled, never on the raw event - a solve costs more than
    # a draw.
    pending = {"job": None}

    def resolve():
        pending["job"] = None
        try:
            fig.set_layout_engine("constrained")
            fig.canvas.draw()
            fig.set_layout_engine("none")
        except Exception:
            pass

    def on_configure(_event):
        if pending["job"] is not None:
            try:
                tkwidget.after_cancel(pending["job"])
            except Exception:
                pass
        try:
            pending["job"] = tkwidget.after(250, resolve)
        except Exception:
            pending["job"] = None

    tkwidget.bind("<Configure>", on_configure, add="+")
    return fig, canvas


def freeze_layout(fig):
    """Solve the constrained layout once, then pin it.

    Re-solving on every draw is the single most expensive thing a panel with a
    colorbar does. The solved axes positions are fractional, so a frozen figure
    survives a resize; it only drifts, which is why callers re-solve on a
    debounced ``<Configure>``.
    """
    try:
        fig.canvas.draw()
        fig.set_layout_engine("none")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# control-point grid
# --------------------------------------------------------------------------- #


def _mix(c0, c1, t):
    """Linear sRGB blend of two #rrggbb strings."""
    t = max(0.0, min(1.0, float(t)))
    a = [int(c0[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


class ControlPointGrid:
    """One nx x ny layer of the latent spline's control net, laid out in space.

    The screen position of a cell is the domain position of the design variable
    it holds: column ``i`` is the x direction, row ``j`` the y direction, and
    ``j = 0`` is drawn on the *bottom* row so +y points up, matching the
    ``origin="lower"`` slice beside it. Clicking a cell selects it; the caller edits the selected value
    with a single slider, so the widget count stays at nx*ny however large the
    design vector gets.
    """

    def __init__(self, parent, palette, on_select):
        self.frame = ttk.Frame(parent, style="Card.TFrame")
        self.palette = palette
        self.on_select = on_select
        self.cells: dict[tuple[int, int], tuple[tk.Frame, tk.Label]] = {}
        self.selected: tuple[int, int] | None = None
        self.nx = self.ny = 0
        # widgets.set_enabled cannot reach these: a tk.Frame has no -state at
        # all and a disabled tk.Label still fires its <Button-1> binding
        self.enabled = True

    def build(self, nx, ny):
        """(Re)build the cell matrix. Cheap: nx*ny <= 36 with the spinbox caps."""
        for child in self.frame.winfo_children():
            child.destroy()
        self.cells.clear()
        previous_nx = self.nx
        self.nx, self.ny = int(nx), int(ny)
        self.selected = None

        for i in range(max(self.nx, previous_nx)):
            self.frame.columnconfigure(
                i, weight=1 if i < self.nx else 0, uniform="cp" if i < self.nx else ""
            )

        for j in range(self.ny):
            for i in range(self.nx):
                cell = tk.Frame(
                    self.frame,
                    background=self.palette["surface_alt"],
                    highlightbackground=self.palette["border"],
                    highlightcolor=self.palette["border"],
                    highlightthickness=1,
                    bd=0,
                )
                label = tk.Label(
                    cell,
                    text="--",
                    background=self.palette["surface_alt"],
                    foreground=self.palette["text"],
                    font=self.palette["font_value"],
                    padx=2,
                    pady=5,
                )
                label.pack(fill="both", expand=True)
                # row 0 of the grid is j = ny-1, so +y points up on screen
                cell.grid(row=self.ny - 1 - j, column=i, sticky="nsew", padx=1, pady=1)
                for widget in (cell, label):
                    widget.bind("<Button-1>", lambda e, i=i, j=j: self._click(i, j))
                self.cells[(i, j)] = (cell, label)

    def _click(self, i, j):
        if not self.enabled:
            return
        self.select(i, j)
        if self.on_select is not None:
            self.on_select(i, j)

    def select(self, i, j):
        self.selected = (i, j)
        self._restyle_borders()

    def _restyle_borders(self):
        for (i, j), (cell, _) in self.cells.items():
            if (i, j) == self.selected:
                colour, width = self.palette["primary"], 2
            elif getattr(cell, "_out_of_range", False):
                colour, width = self.palette["danger"], 2
            else:
                colour, width = self.palette["border"], 1
            try:
                cell.configure(
                    highlightbackground=colour,
                    highlightcolor=colour,
                    highlightthickness=width,
                )
            except tk.TclError:
                pass

    def update_values(self, values, lo, hi, valid_lo=None, valid_hi=None):
        """Repaint the cells from an ``(ny, nx)`` array of latent values.

        ``lo``/``hi`` set the tint ramp; ``valid_lo``/``valid_hi`` mark cells
        outside the trained range with a red outline. When ``hi == lo`` every
        cell gets the same tint rather than a division by zero.
        """
        span = float(hi) - float(lo)
        for (i, j), (cell, label) in self.cells.items():
            try:
                value = float(values[j][i])
            except (IndexError, TypeError, ValueError):
                continue
            t = 0.5 if span <= 1e-12 else (value - float(lo)) / span
            fill = _mix(self.palette["surface_alt"], self.palette["primary"], t)
            fg = self.palette["on_primary"] if t > 0.55 else self.palette["text"]
            bad = (
                valid_lo is not None
                and valid_hi is not None
                and (value < float(valid_lo) - 1e-12 or value > float(valid_hi) + 1e-12)
            )
            cell._out_of_range = bad
            try:
                cell.configure(background=fill)
                label.configure(text=f"{value:+.3f}", background=fill, foreground=fg)
            except tk.TclError:
                pass
        self._restyle_borders()
