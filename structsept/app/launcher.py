"""Desktop launcher for the two GUI windows, and the installer of its shortcuts.

    uv run python -m structsept.app.launcher explorer     # what the shortcut runs
    uv run python -m structsept.app.launcher sdf_maker
    uv run python -m structsept.app.launcher --install    # Desktop\\NN\\*.lnk
    uv run python -m structsept.app.launcher --install --dest some\\folder

Nothing here is about the physics; this module exists so the app can be
double-clicked. Three facts shape it:

* **A shortcut without a console has no stdout.** Showing a window without a
  black terminal behind it means running ``pythonw.exe``, and under pythonw
  ``sys.stdout`` and ``sys.stderr`` are ``None``. The trainer's tqdm bar
  (``deep_sdf/training.py``) and the mesh loader (``sampling.py``) write to
  stderr unconditionally, so the Train tab and the SDF maker would die with
  ``AttributeError: 'NoneType' object has no attribute 'write'`` the moment
  they start. The launcher hands the process a log file instead,
  ``outputs/logs/<target>.log``, and switches the bars off: nobody would see
  them, and the GUI reads training progress from ``Logs.pth`` anyway.
* **Nothing on screen for 15-30 s reads as "it did not work".** That is how
  long torch, DeepSDFStruct and matplotlib take to import on this machine. The
  launcher opens a splash within a second and imports the app on a worker
  thread; the Tk loop only draws. Both apps expose ``build_app(root=...)`` so
  the finished window is built into the root the splash already owns.
* **Shortcuts hold absolute paths, so none is committed.** ``--install`` writes
  them for the machine it runs on, pointing at this clone's ``.venv``. Because
  ``structsept`` and ``DeepSDFStruct`` are editable installs, the shortcut
  always runs the current source: edit a file, launch again. Dependencies are
  the one thing an editable install does not cover, so before importing the
  launcher compares a hash of ``pyproject.toml``, ``uv.lock`` and the
  submodule's ``pyproject.toml`` with the one stored after the last
  successful ``uv sync`` and re-syncs when they differ - unless another
  structsept window is open, because Windows cannot replace a DLL that is
  loaded and a sync that stops halfway is worse than a stale environment.
  Whatever it decides is written to the log. ``install_shortcuts.bat`` at the
  repo root is the double-clickable version of ``uv sync`` + ``--install``
  for a fresh clone.

The window icons are drawn once with Pillow into ``structsept/app/icons/``
(``--icons`` redraws them) and are committed, so a clone has them before
anything runs.
"""

from __future__ import annotations

import argparse
import base64
import faulthandler
import gc
import hashlib
import importlib
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "outputs" / "logs"
ICON_DIR = Path(__file__).resolve().parent / "icons"
INSTALL_BAT = REPO_ROOT / "install_shortcuts.bat"
# The environment is wherever we are running, not necessarily <repo>/.venv:
# uv honours UV_PROJECT_ENVIRONMENT, and the shortcut must point at the same
# interpreter that `uv run` used to install it. install() refuses to run
# outside a project environment, so this is never the system Python's.
VENV_PYTHONW = Path(sys.executable).with_name("pythonw.exe")

# Files whose content decides what the environment must contain: the project's
# own, the lock, and the submodule's, whose dependency list uv.lock records
# too. The hash lives inside the environment so it disappears with it.
ENV_FILES = ("pyproject.toml", "uv.lock", "DeepSDFStruct/pyproject.toml")
SYNC_MARKER = Path(sys.prefix) / ".structsept-env-hash"
# Where uv was the last time the environment was synced from a terminal: a
# shortcut runs with Explorer's PATH, which may not be the terminal's.
UV_HINT = SYNC_MARKER.with_name(".structsept-uv-path")
# One byte per running window; see InstanceLock.
LOCK_FILE = SYNC_MARKER.with_name(".structsept-instances")
LOCK_SLOTS = 16

SHORTCUT_FOLDER = "NN"
LOG_MAX_BYTES = 2_000_000
POLL_MS = 100

# Tests set this to False so the splash does not flash on screen.
SPLASH_VISIBLE = True


@dataclass(frozen=True)
class Target:
    """One launchable window."""

    key: str
    module: str  # must expose build_app(root=None) -> tk.Tk
    title: str  # splash headline and window title while loading
    shortcut: str  # .lnk file name, without extension
    description: str  # shortcut tooltip
    dpi_aware: bool  # the explorer scales itself; the SDF maker keeps native Tk
    themed: bool  # apply theme.apply_theme to the splash as the app will


TARGETS = {
    "explorer": Target(
        key="explorer",
        module="structsept.app.main",
        title="DeepSDF lattice explorer",
        shortcut="Lattice explorer",
        description="structsept - explore f(lambda(x), x), train a decoder",
        dpi_aware=True,
        themed=True,
    ),
    "sdf_maker": Target(
        key="sdf_maker",
        module="structsept.app.sdf_maker",
        title="SDF maker",
        shortcut="SDF maker",
        description="structsept - meshes in, SdfSamples dataset out",
        dpi_aware=False,
        themed=False,
    ),
}


# --------------------------------------------------------------------------- #
# console substitute
# --------------------------------------------------------------------------- #


def is_headless() -> bool:
    """True under pythonw, where the standard streams do not exist."""
    return sys.stdout is None or sys.stderr is None


def open_log(target: Target, log_dir: Path = LOG_DIR):
    """Append-mode, line-buffered log for one launch; rotates past 2 MB.

    Line-buffered so that whatever was written before a hard crash is on
    disk; append mode so that yesterday's traceback is still there tomorrow.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{target.key}.log"
    if path.is_file() and path.stat().st_size > LOG_MAX_BYTES:
        try:
            path.replace(path.with_suffix(".log.1"))
        except OSError:
            pass
    log = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}  {target.key}  ===\n")
    log.write(f"python {sys.executable}\ncwd {os.getcwd()}\n")
    return log


def attach_log(target: Target, log_dir: Path = LOG_DIR):
    """Give a console-less process somewhere to print.

    Everything that would have gone to the terminal - tqdm, ``logging``
    handlers created from now on, Tk's own callback tracebacks, a segfault
    report from ``faulthandler`` - lands in the log instead. Must run before
    DeepSDFStruct is imported: ``logging.StreamHandler()`` captures
    ``sys.stderr`` at construction time.
    """
    log = open_log(target, log_dir)
    sys.stdout = log
    sys.stderr = log
    # Progress bars in a file are only \r-noise around the errors that matter;
    # the GUI shows training progress from Logs.pth. tqdm honours this env var.
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        faulthandler.enable(file=log)
    except (OSError, ValueError, AttributeError):
        pass
    return log


# --------------------------------------------------------------------------- #
# environment freshness
# --------------------------------------------------------------------------- #


def env_hash(repo_root: Path = REPO_ROOT) -> str:
    """SHA-256 over the dependency files' contents, in a fixed order."""
    digest = hashlib.sha256()
    for name in ENV_FILES:
        path = repo_root / name
        digest.update(name.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def env_needs_sync(repo_root: Path = REPO_ROOT, marker: Path = SYNC_MARKER) -> bool:
    """True when pyproject/uv.lock changed since the last recorded sync."""
    try:
        return marker.read_text(encoding="utf-8").strip() != env_hash(repo_root)
    except OSError:
        return True


def record_synced(repo_root: Path = REPO_ROOT, marker: Path = SYNC_MARKER) -> None:
    """Remember the dependency files the environment matches, and where uv is."""
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(env_hash(repo_root), encoding="utf-8")
        uv = shutil.which("uv")
        if uv:
            marker.with_name(UV_HINT.name).write_text(uv, encoding="utf-8")
    except OSError:
        pass


def find_uv(hint: Path = UV_HINT) -> str | None:
    """``uv`` on PATH, at the standalone installer's default location, or
    where the last sync from a terminal found it."""
    found = shutil.which("uv")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    ]
    try:
        candidates.append(Path(hint.read_text(encoding="utf-8").strip()))
    except OSError:
        pass
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _stream_for_subprocess():
    """Our stdout when it is a real file (the log), else inherit the console.

    Under pytest or an IDE ``sys.stdout`` is a capture object without a usable
    descriptor; ``None`` lets the child write wherever we would have.
    """
    stream = sys.stdout
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def sync_env(
    report,
    repo_root: Path = REPO_ROOT,
    marker: Path = SYNC_MARKER,
    no_window: bool = False,
) -> bool:
    """Run ``uv sync`` for this clone; output goes to our stdout (or the log).

    Failure is reported, not raised: offline, the old environment is usually
    still good enough to open the window, and the import that follows gives
    the precise error if it is not. ``no_window`` is for the console-less
    launch: a console child of a pythonw parent would otherwise pop up its
    own black window for the duration of the sync. It is a parameter rather
    than ``is_headless()`` because by the time this runs the streams have
    been replaced by the log and no longer say how the process started.
    """
    uv = find_uv()
    if uv is None:
        report("uv not found; skipping the environment update")
        return False
    report("Dependencies changed - updating the environment (may take a while)")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if no_window else 0
    out = _stream_for_subprocess()
    try:
        result = subprocess.run(
            [uv, "sync"],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT if out is not None else None,
            creationflags=flags,
            check=False,
        )
    except OSError as exc:
        report(f"uv sync could not start: {exc}")
        return False
    if result.returncode != 0:
        report(f"uv sync failed (exit {result.returncode}); trying to start anyway")
        return False
    record_synced(repo_root, marker)
    return True


class InstanceLock:
    """One byte of a shared file per running window.

    ``uv sync`` must not replace packages another structsept window has
    loaded: on Windows a loaded ``.pyd`` cannot be deleted, so the sync stops
    halfway and leaves a half-uninstalled package behind. Each window holds
    an exclusive OS lock on one byte of ``LOCK_FILE`` for as long as it lives
    - released by the OS on any kind of exit, crash included - and "is another
    window open?" is answered by probing the other bytes. Windows byte-range
    locks are per handle, so two windows in one process count as two. Without
    ``msvcrt`` (not Windows) nothing is locked and nobody is reported.
    """

    def __init__(self, path: Path = LOCK_FILE):
        self.fd = None
        self.slot = None
        try:
            import msvcrt  # noqa: F401 - Windows only
        except ImportError:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(path, os.O_RDWR | os.O_CREAT)
            if os.fstat(self.fd).st_size < LOCK_SLOTS:
                os.write(self.fd, b"\0" * LOCK_SLOTS)
        except OSError:
            self.close()
            return
        for slot in range(LOCK_SLOTS):
            if self._try_lock(slot):
                self.slot = slot
                break

    def _try_lock(self, slot: int) -> bool:
        import msvcrt

        os.lseek(self.fd, slot, os.SEEK_SET)
        try:
            msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(self, slot: int) -> None:
        import msvcrt

        os.lseek(self.fd, slot, os.SEEK_SET)
        try:
            msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

    def others_running(self) -> bool:
        """True when any byte but ours is held by another window."""
        if self.fd is None:
            return False
        for slot in range(LOCK_SLOTS):
            if slot == self.slot:
                continue
            if self._try_lock(slot):
                self._unlock(slot)
            else:
                return True
        return False

    def close(self) -> None:
        if self.fd is None:
            return
        if self.slot is not None:
            self._unlock(self.slot)
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = None
        self.slot = None


# --------------------------------------------------------------------------- #
# splash
# --------------------------------------------------------------------------- #


class Splash:
    """A small undecorated window that lives while the app imports.

    A ``Toplevel`` of the withdrawn real root rather than a second ``tk.Tk()``:
    the SDF maker creates master-less ``tk.StringVar()`` objects, which bind
    to whichever root is the default, and two interpreters would split the
    app across them.
    """

    def __init__(self, root, target: Target, palette: dict | None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.started = time.perf_counter()
        bg = palette["bg"] if palette else None

        top = tk.Toplevel(root)
        self.top = top
        # Position it before it is first shown: on Windows an override-redirect
        # window that is already mapped ignores a later "+x+y" and stays in
        # the top-left corner.
        top.withdraw()
        top.overrideredirect(True)
        if bg:
            top.configure(background=bg)
        frame = ttk.Frame(top, padding=(28, 22, 28, 20))
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text=target.title,
            style="Title.TLabel" if palette else None,
            font=None if palette else ("Segoe UI", 13, "bold"),
        ).pack(anchor="w")
        # Plain labels, no StringVar: a Variable's finaliser talks to Tcl, and
        # if the collector runs it from another thread Tcl is left corrupted.
        self.status = ttk.Label(
            frame,
            text="Starting",
            style="Subtle.TLabel" if palette else None,
            wraplength=360,
        )
        self.status.pack(anchor="w", pady=(6, 10))
        self.bar = ttk.Progressbar(frame, mode="indeterminate", length=360)
        self.bar.pack(fill="x")
        self.bar.start(12)
        self.elapsed = ttk.Label(frame, style="Subtle.TLabel" if palette else None)
        self.elapsed.pack(anchor="e", pady=(6, 0))

        top.update_idletasks()
        w, h = top.winfo_reqwidth(), top.winfo_reqheight()
        x = (top.winfo_screenwidth() - w) // 2
        y = (top.winfo_screenheight() - h) // 2
        top.geometry(f"{w}x{h}+{x}+{y}")
        if SPLASH_VISIBLE:
            top.deiconify()
            top.lift()
            top.attributes("-topmost", True)

    def set_status(self, text: str) -> None:
        try:
            self.status.configure(text=text)
        except Exception:  # noqa: BLE001 - splash already closed
            pass

    def tick(self) -> None:
        """Elapsed seconds - the sign that the process is alive, not hung."""
        try:
            self.elapsed.configure(text=f"{time.perf_counter() - self.started:.0f} s")
        except Exception:  # noqa: BLE001 - splash already closed
            pass

    def close(self) -> None:
        try:
            self.bar.stop()
            self.top.destroy()
        except Exception:  # noqa: BLE001 - already gone is fine
            pass


# --------------------------------------------------------------------------- #
# launching
# --------------------------------------------------------------------------- #


def _alert(title: str, message: str) -> None:
    """Module-level so tests can replace the blocking dialog."""
    from tkinter import messagebox

    messagebox.showerror(title, message)


def _show_centered(root) -> None:
    """Map the finished window in the middle of the screen.

    Left to ``deiconify`` alone the window lands wherever Windows cascades
    it - on a 1536-px display that was x=233 for a 1456-px window, off the
    right edge. The size cannot be read beforehand: until a window is mapped,
    ``geometry()`` and ``winfo_width()`` report the 200x200 default, not what
    ``build_app`` requested. So it is mapped fully transparent, measured,
    moved, and only then made opaque; nothing visibly jumps. The 100 px keep
    the title bar and the taskbar out of the vertical sum.
    """
    import tkinter as tk

    faded = False
    try:
        root.attributes("-alpha", 0.0)
        faded = True
    except tk.TclError:
        pass
    root.deiconify()
    try:
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        if w > 1 and h > 1:
            x = max(0, (root.winfo_screenwidth() - w) // 2)
            y = max(0, (root.winfo_screenheight() - h - 100) // 2)
            root.geometry(f"+{x}+{y}")
            root.update_idletasks()
    except tk.TclError:
        pass
    if faded:
        try:
            root.attributes("-alpha", 1.0)
        except tk.TclError:
            pass
    try:
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass


def _failure_message(exc: BaseException, log) -> str:
    where = f"\n\nDetails: {log.name}" if log is not None else ""
    if isinstance(exc, ImportError):  # ModuleNotFoundError included
        name = getattr(exc, "name", None)
        what = (
            f"A Python package is missing: {name}."
            if name
            else f"A Python package failed to import: {exc}."
        )
        return (
            f"{what}\n\n"
            "The environment is out of date, or was changed while a window "
            f"was open. Close every structsept window, double-click "
            f"{INSTALL_BAT.name} in the repo folder (or run `uv sync`) and "
            f"launch again.{where}"
        )
    return f"{type(exc).__name__}: {exc}{where}"


def run(target_key: str) -> int:
    """Splash now, import on a thread, then build the app into the same root.

    Returns the process exit code: 0 when the window closed normally, 1 when
    the app could not start (the reason is in the log and in a dialog).
    """
    target = TARGETS[target_key]
    # decided once: attach_log replaces the None streams, after which nothing
    # else tells a shortcut launch from a terminal one
    headless = is_headless()
    log = attach_log(target, LOG_DIR) if headless else None
    if log is None:
        try:
            faulthandler.enable()
        except (OSError, ValueError, AttributeError):
            pass

    import tkinter as tk

    from structsept.app import theme

    if target.dpi_aware:
        # process-wide and only honoured before the first Tk window exists
        theme.enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    root.title(target.title)
    icon = ICON_DIR / f"{target.key}.ico"
    if icon.is_file():
        try:
            root.iconbitmap(default=str(icon))
        except tk.TclError:
            pass
    # Tk half of the theme only: the matplotlib half would load numpy and
    # Pillow too, and a DLL that is loaded cannot be replaced by the uv sync
    # the worker may run next. build_app applies the full theme afterwards.
    palette = theme.apply_theme(root, matplotlib=False) if target.themed else None
    splash = Splash(root, target, palette)

    events: queue.Queue = queue.Queue()
    # held until run() returns, i.e. for the life of the window
    instances = InstanceLock(LOCK_FILE)

    def report(msg):
        # the splash line is overwritten within a tick; the log keeps it
        print(f"[launcher] {msg}", file=sys.stderr, flush=True)
        events.put(("status", msg))

    def work():
        try:
            loading = "Loading torch, DeepSDFStruct and matplotlib"
            if env_needs_sync():
                if instances.others_running():
                    report(
                        "Dependencies changed, but another structsept window is "
                        "open - not updating the environment now. Close every "
                        "window and launch again."
                    )
                    synced = False
                else:
                    synced = sync_env(report, no_window=headless)
                if not synced:
                    loading += " (environment not updated - see the log)"
            events.put(("status", loading))
            module = importlib.import_module(target.module)
            events.put(("done", module))
        except BaseException as exc:  # noqa: BLE001 - reported on the Tk side
            events.put(("error", (exc, traceback.format_exc())))

    outcome = {"code": 0}

    def fail(exc, text):
        print(text, file=sys.stderr, flush=True)
        splash.close()
        outcome["code"] = 1
        _alert(f"{target.title} could not start", _failure_message(exc, log))
        root.destroy()

    def hand_over(module):
        splash.set_status("Building the window")
        splash.tick()
        root.update_idletasks()
        try:
            module.build_app(root=root)
        except BaseException as exc:  # noqa: BLE001
            fail(exc, traceback.format_exc())
            return
        splash.close()
        _show_centered(root)

    def poll():
        splash.tick()
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                splash.set_status(payload)
            elif kind == "done":
                hand_over(payload)
                return
            else:
                fail(*payload)
                return
        root.after(POLL_MS, poll)

    threading.Thread(target=work, name="launcher-import", daemon=True).start()
    root.after(POLL_MS, poll)
    root.mainloop()
    # The nested functions above hold each other and the splash through their
    # closure cells - a cycle only the collector frees. Free it here, on the
    # Tk thread. Left to chance it is freed wherever an allocation next trips
    # the collector, e.g. on the import thread of a later run() in one test
    # process, and a Tk object finalised from the wrong thread leaves Tcl
    # corrupted ("Tcl_AsyncDelete: async handler deleted by the wrong
    # thread", then "can't find a usable init.tcl" for the next interpreter).
    splash = hand_over = fail = poll = work = report = None
    gc.collect()
    return outcome["code"]


# --------------------------------------------------------------------------- #
# icons
# --------------------------------------------------------------------------- #

ICON_SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]


def draw_icons(icon_dir: Path = ICON_DIR, force: bool = False) -> list[Path]:
    """Draw ``explorer.ico`` and ``sdf_maker.ico`` with Pillow.

    Explorer: a round-cross unit cell - the paper's first test case - in the
    theme's primary blue. SDF maker: nested level sets around a filled zero
    set, the picture of a signed distance field.
    """
    from PIL import Image, ImageDraw

    from structsept.app.theme import PALETTE

    icon_dir.mkdir(parents=True, exist_ok=True)
    blue, white, soft = PALETTE["primary"], PALETTE["surface"], PALETTE["primary_soft"]
    S = 1024  # supersampled, then downscaled: smooth edges at 16 px
    written = []

    def finish(img, path):
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(path, format="ICO", sizes=ICON_SIZES)
        written.append(path)

    path = icon_dir / "explorer.ico"
    if force or not path.is_file():
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, S - 1, S - 1), radius=S // 5, fill=blue)
        arm, r = S // 6, S // 12  # strut half-width, end radius
        c = S // 2
        d.rounded_rectangle(
            (c - arm, S // 8, c + arm, S - S // 8), radius=r, fill=white
        )
        d.rounded_rectangle(
            (S // 8, c - arm, S - S // 8, c + arm), radius=r, fill=white
        )
        d.ellipse((c - arm, c - arm, c + arm, c + arm), fill=soft)
        finish(img, path)

    path = icon_dir / "sdf_maker.ico"
    if force or not path.is_file():
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, S - 1, S - 1), radius=S // 5, fill=white)
        d.rounded_rectangle(
            (S // 32, S // 32, S - S // 32, S - S // 32),
            radius=S // 5,
            outline=blue,
            width=S // 32,
        )
        c = S // 2
        for k, rad in enumerate((S * 0.40, S * 0.30, S * 0.20)):
            d.ellipse(
                (c - rad, c - rad, c + rad, c + rad),
                outline=blue if k else soft,
                width=S // 40,
            )
        d.ellipse((c - S * 0.11, c - S * 0.11, c + S * 0.11, c + S * 0.11), fill=blue)
        finish(img, path)
    return written


# --------------------------------------------------------------------------- #
# shortcuts
# --------------------------------------------------------------------------- #


# PowerShell ends a single-quoted string at these typographic quotes as well
# as at the ASCII one, and Explorer happily puts them in folder names.
_PS_QUOTES = "‘’‚‛"


def _ps_quote(text: str) -> str:
    """PowerShell literal for ``text``: single-quoted, ``'`` doubled.

    A typographic quote inside the text is spliced in as ``[char]``, so the
    result is then a ``$( ... )`` expression, which is valid in the same
    places as a plain literal (assignments and arguments alike).
    """
    text = str(text)
    if not any(ch in text for ch in _PS_QUOTES):
        return "'" + text.replace("'", "''") + "'"
    parts = []
    for ch in text:
        if ch in _PS_QUOTES:
            parts.append(f"[char]0x{ord(ch):04X}")
        elif parts and parts[-1].startswith("'"):
            parts[-1] = parts[-1][:-1] + ch.replace("'", "''") + "'"
        else:
            parts.append("'" + ch.replace("'", "''") + "'")
    return "$(" + " + ".join(parts) + ")"


def run_powershell(script: str) -> subprocess.CompletedProcess:
    """Run a PowerShell script from Python, text in and out as UTF-8.

    ``-EncodedCommand`` avoids a temp file and every quoting rule but
    PowerShell's own. Windows PowerShell writes piped output in the console's
    OEM code page (IBM437 here), which Python would decode as cp1252 - an
    umlaut in a path then either comes back wrong or raises - so the script
    is told to emit UTF-8 and the reply is decoded as such.
    """
    script = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n" + script
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def shortcut_script(dest: Path | None, pythonw: Path = VENV_PYTHONW) -> str:
    """PowerShell that writes one .lnk per target and prints the folder.

    WScript.Shell is the only shortcut writer that ships with Windows; going
    through PowerShell keeps pywin32 out of our dependencies.
    """
    if dest is None:
        folder = "Join-Path ([Environment]::GetFolderPath('Desktop')) " + _ps_quote(
            SHORTCUT_FOLDER
        )
    else:
        folder = _ps_quote(Path(dest).resolve())
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"$dest = {folder}",
        "New-Item -ItemType Directory -Force -Path $dest | Out-Null",
        "$shell = New-Object -ComObject WScript.Shell",
    ]
    for target in TARGETS.values():
        lines += [
            f"$lnk = $shell.CreateShortcut((Join-Path $dest {_ps_quote(target.shortcut + '.lnk')}))",
            f"$lnk.TargetPath = {_ps_quote(pythonw)}",
            f"$lnk.Arguments = {_ps_quote('-m structsept.app.launcher ' + target.key)}",
            f"$lnk.WorkingDirectory = {_ps_quote(REPO_ROOT)}",
            f"$lnk.IconLocation = {_ps_quote(str(ICON_DIR / (target.key + '.ico')) + ',0')}",
            f"$lnk.Description = {_ps_quote(target.description)}",
            "$lnk.Save()",
        ]
    lines.append("Write-Output $dest")
    return "\n".join(lines)


def install(dest: Path | None = None) -> Path:
    """Create the shortcut folder for this clone on this machine.

    Returns the folder that now holds the .lnk files. Raises ``SystemExit``
    with a readable message when the environment or the OS cannot do it.
    """
    if os.name != "nt":
        raise SystemExit(
            "--install writes Windows .lnk files. On this system run the "
            "windows directly:  uv run python -m structsept.app.launcher explorer"
        )
    if sys.prefix == sys.base_prefix:
        raise SystemExit(
            "This is not the project's environment, so the shortcuts would point "
            "at the wrong Python. Run it as:  uv run python -m "
            "structsept.app.launcher --install"
        )
    if not VENV_PYTHONW.is_file():
        raise SystemExit(
            f"{VENV_PYTHONW} does not exist - run `uv sync` in {REPO_ROOT} first "
            f"(or double-click {INSTALL_BAT.name})."
        )
    draw_icons()
    result = run_powershell(shortcut_script(dest))
    if result.returncode != 0:
        raise SystemExit(
            "PowerShell could not write the shortcuts:\n"
            + (result.stderr or result.stdout).strip()
        )
    folder = Path(result.stdout.strip().splitlines()[-1])
    record_synced()
    return folder


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m structsept.app.launcher",
        description="Open a structsept window from a shortcut, or create the shortcuts.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=sorted(TARGETS),
        help="window to open (what the desktop shortcuts pass)",
    )
    install_group = parser.add_argument_group("installing")
    install_group.add_argument(
        "--install",
        action="store_true",
        help=f"write the .lnk shortcuts for this clone into Desktop\\{SHORTCUT_FOLDER}",
    )
    install_group.add_argument(
        "--dest",
        type=Path,
        default=None,
        metavar="DIR",
        help="put the shortcuts in DIR instead of the desktop folder",
    )
    install_group.add_argument(
        "--icons",
        action="store_true",
        help="redraw structsept/app/icons/*.ico and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.icons:
        for path in draw_icons(force=True):
            print(f"wrote {path}")
        return 0
    if args.install:
        folder = install(args.dest)
        names = ", ".join(f"{t.shortcut}.lnk" for t in TARGETS.values())
        print(f"Shortcuts written to {folder}: {names}")
        return 0
    if args.target is None:
        _parser().print_help()
        return 2
    return run(args.target)


if __name__ == "__main__":
    sys.exit(main())
