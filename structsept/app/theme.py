"""Visual theme of the structsept app: ttk styles, palette and matplotlib.

Stock tkinter/ttk only, no third-party theme package. Every styling step is
guarded, so a failure in one widget family leaves the rest of the theme - and
the app - intact; :func:`apply_theme` always returns a usable palette.

Two facts drive the whole module:

* Windows' native ``vista``/``xpnative`` ttk themes draw buttons, entries,
  comboboxes, scales and progressbars with the OS theme engine and silently
  ignore ``-background`` / ``-fieldbackground`` / ``-troughcolor``. ``clam`` is
  pure-Tk drawn and honours all of them, so it is the only restylable base
  here. ``style.lookup`` is no help in telling the two apart: it echoes back
  whatever was configured even when the option is ignored.
* ttk widgets have no ``-background`` at all (``TclError: unknown option``), so
  anything that has to carry a colour of its own - a tinted control-point cell,
  an out-of-range outline - is a classic ``tk`` widget, not a ttk one.
"""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #

PALETTE: dict = {
    # surfaces
    "bg": "#F2F5F8",  # window / notebook background
    "surface": "#FFFFFF",  # cards, entries, log background
    "surface_alt": "#EAEFF4",  # inset / hover / zebra
    "surface_sunken": "#E1E8EF",  # troughs
    # lines
    "border": "#D4DCE4",
    "border_strong": "#B3BFCB",
    # accent
    "primary": "#2563EB",
    "primary_hover": "#1D4ED8",
    "primary_pressed": "#1E40AF",
    "primary_soft": "#DCE9FE",
    "on_primary": "#FFFFFF",
    # text
    "text": "#16202C",
    "text_muted": "#5B6B7C",
    "text_faint": "#93A1B0",  # decorative only: 2.6:1 on white, fails AA
    # semantic
    "success": "#15803D",
    "warning": "#B45309",
    "danger": "#C0271B",
    "warning_soft": "#FDF2E2",
    # states
    "disabled_bg": "#EDF1F5",
    "disabled_fg": "#A8B4C0",
    "selection": "#CFE0FD",
    # plotting
    "grid": "#E3E9EF",
}

_HEAD_FAMS = ("Segoe UI Semibold", "Segoe UI", "Tahoma")
_BODY_FAMS = ("Segoe UI", "Tahoma", "Arial")
_MONO_FAMS = ("Cascadia Mono", "Consolas", "Courier New")


def enable_dpi_awareness() -> None:
    """Tell Windows we scale ourselves, so Tk is not bitmap-stretched.

    Without this the process is DPI-unaware: on a 125% display Tk sees a
    1536x864 screen, renders everything there and lets Windows blow the bitmap
    up by 1.25 - which is exactly the soft, blurry look of the plot panels.
    Must run before the first ``tk.Tk()``; calling it twice is harmless.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _pick(root, families, default):
    """First installed family out of ``families``."""
    try:
        have = set(tkfont.families(root))
    except tk.TclError:
        return default
    for fam in families:
        if fam in have:
            return fam
    return default


def apply_theme(root: tk.Misc, base_theme: str = "clam") -> dict:
    """Style ``root`` and return the palette, fonts included.

    Call this right after ``tk.Tk()`` and *before* building any widget:
    ``option_add`` only reaches widgets created afterwards, so a later call
    leaves every ``tk.Text``, ``tk.Canvas`` and combobox popup unstyled.

    The returned dict also carries ``_errors``, the list of styling steps that
    did not apply - without it the guards would make a typo invisible.
    """
    p = {k: (list(v) if isinstance(v, list) else v) for k, v in PALETTE.items()}
    errors: list[str] = []
    p["_errors"] = errors

    def safe(label, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a theme must never be fatal
            errors.append(f"{label}: {exc}")

    style = ttk.Style(root)

    # -- base theme --------------------------------------------------------- #
    chosen = base_theme
    try:
        names = style.theme_names()
    except tk.TclError:
        names = ()
    if chosen not in names:
        chosen = "clam" if "clam" in names else (names[0] if names else "")
    if chosen:
        try:
            style.theme_use(chosen)
        except tk.TclError:
            chosen = style.theme_use()
    p["base_theme"] = chosen
    p["restylable"] = chosen in ("clam", "alt", "default", "classic")

    # -- fonts -------------------------------------------------------------- #
    head_fam = _pick(root, _HEAD_FAMS, "Segoe UI")
    body_fam = _pick(root, _BODY_FAMS, "Segoe UI")
    mono_fam = _pick(root, _MONO_FAMS, "Consolas")

    p["font_title"] = (head_fam, 14)
    p["font_heading"] = (head_fam, 10)
    p["font_body"] = (body_fam, 9)
    p["font_small"] = (body_fam, 8)
    p["font_mono"] = (mono_fam, 8)
    p["font_value"] = (mono_fam, 9)
    p["font_family"] = body_fam
    p["font_mono_family"] = mono_fam

    for name, spec in (
        ("TkDefaultFont", p["font_body"]),
        ("TkTextFont", p["font_body"]),
        ("TkMenuFont", p["font_body"]),
        ("TkHeadingFont", p["font_heading"]),
        ("TkTooltipFont", p["font_small"]),
        ("TkFixedFont", p["font_mono"]),
    ):
        safe(
            f"font {name}",
            lambda n=name, s=spec: tkfont.nametofont(n, root=root).configure(
                family=s[0], size=s[1]
            ),
        )

    # -- root and plain tk widget defaults ---------------------------------- #
    safe("root bg", root.configure, background=p["bg"])
    for pat, val in (
        ("*Text.background", p["surface"]),
        ("*Text.foreground", p["text"]),
        ("*Text.insertBackground", p["primary"]),
        ("*Text.selectBackground", p["selection"]),
        ("*Text.selectForeground", p["text"]),
        ("*Text.highlightThickness", "0"),
        ("*Text.borderWidth", "0"),
        ("*Text.relief", "flat"),
        ("*Canvas.background", p["bg"]),
        ("*Canvas.highlightThickness", "0"),
        # The combobox popup is a plain tk Listbox in its own toplevel; ttk
        # styling never reaches it, only the option database does.
        ("*TCombobox*Listbox.background", p["surface"]),
        ("*TCombobox*Listbox.foreground", p["text"]),
        ("*TCombobox*Listbox.selectBackground", p["primary"]),
        ("*TCombobox*Listbox.selectForeground", p["on_primary"]),
        ("*TCombobox*Listbox.borderWidth", "0"),
        ("*TCombobox*Listbox.font", "TkDefaultFont"),
    ):
        safe("option " + pat, root.option_add, pat, val)

    cfg, mp = style.configure, style.map

    # -- frames -------------------------------------------------------------- #
    safe(
        "root style",
        cfg,
        ".",
        background=p["bg"],
        foreground=p["text"],
        bordercolor=p["border"],
        focuscolor=p["primary"],
        font=p["font_body"],
    )
    safe("TFrame", cfg, "TFrame", background=p["bg"])
    safe("Card.TFrame", cfg, "Card.TFrame", background=p["surface"])
    safe("Alt.TFrame", cfg, "Alt.TFrame", background=p["surface_alt"])

    # -- labels -------------------------------------------------------------- #
    # ttk resolves "Card.Heading.TLabel" up to "Heading.TLabel", never to
    # "Card.TLabel", so each surface has to restate its own background.
    for prefix, back in (
        ("", p["bg"]),
        ("Card.", p["surface"]),
        ("Alt.", p["surface_alt"]),
    ):
        for suffix, fg, font in (
            ("TLabel", p["text"], p["font_body"]),
            ("Title.TLabel", p["text"], p["font_title"]),
            ("Heading.TLabel", p["text"], p["font_heading"]),
            ("Subtle.TLabel", p["text_muted"], p["font_small"]),
            ("Value.TLabel", p["text"], p["font_value"]),
            # a value that differs from its default, in the hyperparameter form
            ("Accent.TLabel", p["primary"], p["font_body"]),
            ("Success.TLabel", p["success"], p["font_body"]),
            ("Warning.TLabel", p["warning"], p["font_body"]),
            ("Danger.TLabel", p["danger"], p["font_body"]),
        ):
            safe(
                prefix + suffix,
                cfg,
                prefix + suffix,
                background=back,
                foreground=fg,
                font=font,
            )

    # -- buttons ------------------------------------------------------------- #
    # clam's Button.border takes its fill from -background and its bevel from
    # -lightcolor/-darkcolor; setting those equal to the fill gives a flat 1px
    # outline instead of a 3D bevel.
    safe(
        "TButton",
        cfg,
        "TButton",
        background=p["surface"],
        foreground=p["text"],
        bordercolor=p["border_strong"],
        lightcolor=p["surface"],
        darkcolor=p["surface"],
        focuscolor=p["surface"],
        relief="flat",
        borderwidth=1,
        padding=(10, 5),
        anchor="center",
        font=p["font_body"],
    )
    safe(
        "TButton map",
        mp,
        "TButton",
        background=[
            ("disabled", p["disabled_bg"]),
            ("pressed", p["surface_sunken"]),
            ("active", p["surface_alt"]),
        ],
        foreground=[("disabled", p["disabled_fg"])],
        bordercolor=[("focus", p["primary"]), ("active", p["border_strong"])],
        lightcolor=[("pressed", p["surface_sunken"]), ("active", p["surface_alt"])],
        darkcolor=[("pressed", p["surface_sunken"]), ("active", p["surface_alt"])],
    )

    safe(
        "Accent.TButton",
        cfg,
        "Accent.TButton",
        background=p["primary"],
        foreground=p["on_primary"],
        bordercolor=p["primary"],
        lightcolor=p["primary"],
        darkcolor=p["primary"],
        focuscolor=p["on_primary"],
        relief="flat",
        borderwidth=1,
        padding=(12, 5),
        font=p["font_body"],
    )
    safe(
        "Accent.TButton map",
        mp,
        "Accent.TButton",
        background=[
            ("disabled", p["disabled_bg"]),
            ("pressed", p["primary_pressed"]),
            ("active", p["primary_hover"]),
        ],
        foreground=[("disabled", p["disabled_fg"])],
        bordercolor=[
            ("disabled", p["border"]),
            ("pressed", p["primary_pressed"]),
            ("active", p["primary_hover"]),
        ],
        lightcolor=[("pressed", p["primary_pressed"]), ("active", p["primary_hover"])],
        darkcolor=[("pressed", p["primary_pressed"]), ("active", p["primary_hover"])],
    )

    for prefix, back in (("Ghost.", p["bg"]), ("CardGhost.", p["surface"])):
        safe(
            prefix + "TButton",
            cfg,
            prefix + "TButton",
            background=back,
            foreground=p["primary"],
            bordercolor=back,
            lightcolor=back,
            darkcolor=back,
            focuscolor=back,
            relief="flat",
            borderwidth=0,
            padding=(7, 4),
        )
        safe(
            prefix + "TButton map",
            mp,
            prefix + "TButton",
            background=[("active", p["primary_soft"]), ("pressed", p["primary_soft"])],
            lightcolor=[("active", p["primary_soft"]), ("pressed", p["primary_soft"])],
            darkcolor=[("active", p["primary_soft"]), ("pressed", p["primary_soft"])],
            foreground=[("disabled", p["disabled_fg"])],
        )

    # -- notebook ------------------------------------------------------------ #
    safe(
        "TNotebook",
        cfg,
        "TNotebook",
        background=p["bg"],
        bordercolor=p["border"],
        lightcolor=p["bg"],
        darkcolor=p["bg"],
        borderwidth=0,
        tabmargins=(0, 5, 0, 0),
    )
    safe(
        "TNotebook.Tab",
        cfg,
        "TNotebook.Tab",
        background=p["bg"],
        foreground=p["text_muted"],
        bordercolor=p["border"],
        lightcolor=p["bg"],
        darkcolor=p["bg"],
        focuscolor=p["bg"],
        padding=(16, 7),
        font=p["font_body"],
    )
    safe(
        "TNotebook.Tab map",
        mp,
        "TNotebook.Tab",
        background=[("selected", p["surface"]), ("active", p["surface_alt"])],
        foreground=[("selected", p["primary"]), ("disabled", p["disabled_fg"])],
        lightcolor=[("selected", p["surface"]), ("active", p["surface_alt"])],
        darkcolor=[("selected", p["surface"]), ("active", p["surface_alt"])],
        expand=[("selected", (1, 1, 1, 0))],
    )

    # -- labelframe ---------------------------------------------------------- #
    for prefix, back in (("", p["bg"]), ("Card.", p["surface"])):
        safe(
            prefix + "TLabelframe",
            cfg,
            prefix + "TLabelframe",
            background=back,
            bordercolor=p["border"],
            lightcolor=back,
            darkcolor=back,
            relief="solid",
            borderwidth=1,
            padding=8,
        )
        safe(
            prefix + "TLabelframe.Label",
            cfg,
            prefix + "TLabelframe.Label",
            background=back,
            foreground=p["text_muted"],
            font=p["font_heading"],
        )

    # -- entry / spinbox / combobox ------------------------------------------ #
    field = dict(
        fieldbackground=p["surface"],
        foreground=p["text"],
        bordercolor=p["border_strong"],
        lightcolor=p["border_strong"],
        darkcolor=p["border_strong"],
        insertcolor=p["text"],
        borderwidth=1,
        relief="flat",
        padding=(6, 4),
    )
    focus_ring = dict(
        bordercolor=[("focus", p["primary"])],
        lightcolor=[("focus", p["primary"])],
        darkcolor=[("focus", p["primary"])],
    )

    safe("TEntry", cfg, "TEntry", **field)
    safe(
        "TEntry map",
        mp,
        "TEntry",
        fieldbackground=[
            ("disabled", p["disabled_bg"]),
            ("readonly", p["surface_alt"]),
        ],
        foreground=[("disabled", p["disabled_fg"])],
        **focus_ring,
    )

    # An entry whose text does not parse. The border is the only part of a
    # clam entry that can carry the colour without hurting legibility.
    invalid = dict(
        field,
        bordercolor=p["danger"],
        lightcolor=p["danger"],
        darkcolor=p["danger"],
    )
    safe("Invalid.TEntry", cfg, "Invalid.TEntry", **invalid)
    safe(
        "Invalid.TEntry map",
        mp,
        "Invalid.TEntry",
        fieldbackground=[("disabled", p["disabled_bg"])],
        foreground=[("disabled", p["disabled_fg"])],
        bordercolor=[("focus", p["danger"])],
        lightcolor=[("focus", p["danger"])],
        darkcolor=[("focus", p["danger"])],
    )

    safe("TSpinbox", cfg, "TSpinbox", arrowcolor=p["text_muted"], arrowsize=11, **field)
    safe(
        "TSpinbox map",
        mp,
        "TSpinbox",
        fieldbackground=[("disabled", p["disabled_bg"])],
        foreground=[("disabled", p["disabled_fg"])],
        arrowcolor=[("disabled", p["disabled_fg"]), ("active", p["primary"])],
        **focus_ring,
    )

    safe(
        "TCombobox", cfg, "TCombobox", arrowcolor=p["text_muted"], arrowsize=12, **field
    )
    safe(
        "TCombobox map",
        mp,
        "TCombobox",
        # A readonly combobox otherwise paints its text with the selection
        # colours the moment it takes focus, which reads as a stuck highlight.
        fieldbackground=[("disabled", p["disabled_bg"]), ("readonly", p["surface"])],
        foreground=[("disabled", p["disabled_fg"])],
        selectbackground=[("readonly", p["surface"]), ("!focus", p["surface"])],
        selectforeground=[("readonly", p["text"]), ("!focus", p["text"])],
        arrowcolor=[("disabled", p["disabled_fg"]), ("active", p["primary"])],
        **focus_ring,
    )

    # -- scale ---------------------------------------------------------------- #
    # clam draws trough and grip from one element: -troughcolor is the groove,
    # -background the grip.
    for orient in ("Horizontal", "Vertical"):
        safe(
            orient + ".TScale",
            cfg,
            orient + ".TScale",
            background=p["primary"],
            troughcolor=p["surface_sunken"],
            bordercolor=p["surface_sunken"],
            lightcolor=p["primary"],
            darkcolor=p["primary"],
            gripcount=0,
            sliderlength=16,
            sliderthickness=14,
            borderwidth=0,
        )
        safe(
            orient + ".TScale map",
            mp,
            orient + ".TScale",
            background=[
                ("disabled", p["disabled_fg"]),
                ("pressed", p["primary_pressed"]),
                ("active", p["primary_hover"]),
            ],
            lightcolor=[
                ("pressed", p["primary_pressed"]),
                ("active", p["primary_hover"]),
            ],
            darkcolor=[
                ("pressed", p["primary_pressed"]),
                ("active", p["primary_hover"]),
            ],
            troughcolor=[("disabled", p["disabled_bg"])],
        )

    # -- progressbar ---------------------------------------------------------- #
    # Under clam, -thickness is a silent no-op; the bar height comes from
    # -arrowsize. 8 gives roughly a 12 px bar against clam's default of 18.
    for orient in ("Horizontal", "Vertical"):
        safe(
            orient + ".TProgressbar",
            cfg,
            orient + ".TProgressbar",
            background=p["primary"],
            troughcolor=p["surface_sunken"],
            bordercolor=p["surface_sunken"],
            lightcolor=p["primary"],
            darkcolor=p["primary"],
            borderwidth=0,
            arrowsize=8,
        )

    # -- separator / scrollbar / checkbutton / radiobutton -------------------- #
    safe("TSeparator", cfg, "TSeparator", background=p["border"])

    safe(
        "TScrollbar",
        cfg,
        "TScrollbar",
        background=p["surface_alt"],
        troughcolor=p["bg"],
        bordercolor=p["bg"],
        lightcolor=p["surface_alt"],
        darkcolor=p["surface_alt"],
        arrowcolor=p["text_muted"],
        borderwidth=0,
        arrowsize=12,
    )
    safe(
        "TScrollbar map",
        mp,
        "TScrollbar",
        background=[("pressed", p["text_faint"]), ("active", p["border_strong"])],
        lightcolor=[("pressed", p["text_faint"]), ("active", p["border_strong"])],
        darkcolor=[("pressed", p["text_faint"]), ("active", p["border_strong"])],
        arrowcolor=[("active", p["text"])],
    )

    # The box itself is -indicatorbackground; -background is the strip behind
    # the text.
    for prefix, back in (("", p["bg"]), ("Card.", p["surface"])):
        for kind in ("TCheckbutton", "TRadiobutton"):
            safe(
                prefix + kind,
                cfg,
                prefix + kind,
                background=back,
                foreground=p["text"],
                focuscolor=back,
                indicatorbackground=p["surface"],
                indicatorforeground=p["on_primary"],
                upperbordercolor=p["border_strong"],
                lowerbordercolor=p["border_strong"],
                indicatorsize=12,
                padding=(2, 2),
                font=p["font_body"],
            )
            safe(
                prefix + kind + " map",
                mp,
                prefix + kind,
                background=[("active", back)],
                foreground=[("disabled", p["disabled_fg"])],
                indicatorbackground=[
                    ("disabled", p["disabled_bg"]),
                    ("selected", p["primary"]),
                    ("active", p["surface"]),
                ],
                upperbordercolor=[("selected", p["primary"]), ("active", p["primary"])],
                lowerbordercolor=[("selected", p["primary"]), ("active", p["primary"])],
            )

    # -- treeview -------------------------------------------------------------- #
    safe(
        "Treeview",
        cfg,
        "Treeview",
        background=p["surface"],
        fieldbackground=p["surface"],
        foreground=p["text"],
        bordercolor=p["border"],
        borderwidth=0,
        rowheight=22,
        font=p["font_body"],
    )
    safe(
        "Treeview map",
        mp,
        "Treeview",
        background=[("selected", p["primary_soft"])],
        foreground=[("selected", p["text"])],
    )
    safe(
        "Treeview.Heading",
        cfg,
        "Treeview.Heading",
        background=p["surface_alt"],
        foreground=p["text_muted"],
        relief="flat",
        font=p["font_small"],
    )
    safe(
        "Treeview.Heading map",
        mp,
        "Treeview.Heading",
        background=[("active", p["surface_alt"])],
    )

    apply_matplotlib_style(p)
    return p


def apply_matplotlib_style(palette: dict) -> None:
    """Point matplotlib's rcParams at the palette.

    Done through rcParams rather than per-figure because every draw helper
    rebuilds its axes, and axes created after a ``fig.clear()`` come back at
    rcParams defaults - white face, black ticks - however the figure itself was
    configured.

    ``savefig.facecolor`` deliberately stays white: rcParams is process-global
    and the DeepSDFStruct trainer saves its loss curve through pyplot from the
    worker thread. Grey PNGs on disk would be our fault.
    """
    try:
        import matplotlib as mpl
    except Exception:
        return

    mpl.rcParams.update(
        {
            "figure.facecolor": palette["surface"],
            "figure.edgecolor": palette["surface"],
            "axes.facecolor": palette["surface"],
            "axes.edgecolor": palette["border"],
            "axes.labelcolor": palette["text_muted"],
            "axes.titlecolor": palette["text"],
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.titlesize": 9,
            "axes.titlepad": 5,
            "axes.labelsize": 8,
            "xtick.color": palette["text_muted"],
            "ytick.color": palette["text_muted"],
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "text.color": palette["text"],
            "grid.color": palette["grid"],
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                palette.get("font_family", "Segoe UI"),
                "Tahoma",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.unicode_minus": True,
            # keep exported figures on white, see docstring
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def figure_dpi(root: tk.Misc, fallback: float = 100.0) -> float:
    """Figure DPI that makes matplotlib points equal Tk points.

    With DPI awareness on, ``tk scaling`` reports pixels per point, so
    ``72 * scaling`` is the display's real DPI and a matplotlib 9 pt label ends
    up the same physical size as a Segoe UI 9 label beside it.
    """
    try:
        return 72.0 * float(root.tk.call("tk", "scaling"))
    except Exception:
        return fallback


def style_text(widget, palette: dict, mono: bool = True) -> None:
    """Match a ``tk.Text`` log box to the theme."""
    try:
        widget.configure(
            background=palette["surface"],
            foreground=palette["text"],
            insertbackground=palette["primary"],
            selectbackground=palette["selection"],
            selectforeground=palette["text"],
            font=palette["font_mono"] if mono else palette["font_body"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
            wrap="word",
        )
    except tk.TclError:
        pass


def clamp_geometry(root: tk.Tk, width: int, height: int, margin: int = 80) -> None:
    """Ask for ``width x height`` but never more than the screen holds.

    The old app hardcoded 1360x900 on a display Tk reports as 1536x864, so the
    window opened taller than the desktop and the footer was unreachable.
    """
    try:
        avail_w = root.winfo_screenwidth() - margin
        avail_h = root.winfo_screenheight() - margin
    except tk.TclError:
        avail_w, avail_h = width, height
    w = max(960, min(width, avail_w))
    h = max(640, min(height, avail_h))
    root.geometry(f"{w}x{h}")
    root.minsize(960, 640)
