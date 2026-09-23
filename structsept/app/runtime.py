"""Worker-thread plumbing shared by the app's tabs.

Sampling, training and meshing all take seconds to minutes; the Tk loop only
draws. Everything a worker wants to say to the interface goes through one
queue that the Tk loop drains on a timer, so no widget is ever touched off the
main thread.
"""

from __future__ import annotations

import contextlib
import queue
import signal
import threading
import tkinter as tk
import traceback
from tkinter import messagebox

from structsept.app import widgets

POLL_MS = 100


def new_state(root) -> dict:
    """Fresh shared-state dict for one Tk root."""
    return {"root": root, "queue": queue.Queue(), "busy": False}


def ui(st, fn):
    """Run ``fn`` on the Tk loop from anywhere."""
    st["queue"].put(("call", fn))


def log_to(st, widget, line):
    """Append a line to a log box from anywhere."""
    st["queue"].put(("log", (widget, line)))


def poll(st):
    """Drain the worker queue on the Tk loop."""
    while True:
        try:
            kind, payload = st["queue"].get_nowait()
        except queue.Empty:
            break
        if kind == "log":
            widgets.append(*payload)
        else:
            try:
                payload()
            except Exception as exc:  # noqa: BLE001 - one bad callback must not
                # stop the pump, or the app silently stops updating. It still
                # has to be visible: a swallowed exception here used to leave
                # panels frozen with no trace anywhere.
                st.setdefault("callback_errors", []).append(repr(exc))
                traceback.print_exc()
                widget = st.get("error_log")
                if widget is not None:
                    widgets.append(widget, f"INTERNAL ERROR: {exc!r}")
    if st.get("closing"):
        return
    st["poll_job"] = st["root"].after(POLL_MS, lambda: poll(st))


def run_worker(st, button, log_widget, work, lock=(), on_done=None):
    """Run ``work(log)`` off the Tk loop, funnelling output through the queue.

    ``lock`` lists widget subtrees to disable for the duration. That is not
    cosmetic: a worker meshing the lattice and a slider redrawing the same
    lattice both mutate ``microtile.latvec``, and the loser gets a shape
    mismatch. Whoever hands the SDF to a worker locks the controls that touch
    it.

    ``on_done`` runs on the Tk loop when the worker finishes, whether it
    succeeded or raised - the place to release whatever the worker held.
    """
    if st["busy"]:
        messagebox.showinfo("Busy", "Wait for the current task to finish.")
        return
    st["busy"] = True
    if button is not None:
        button.configure(state="disabled")
    for widget in lock:
        widgets.set_enabled(widget, False)

    def log(line):
        log_to(st, log_widget, line)

    def target():
        try:
            work(log)
        except Exception as exc:  # noqa: BLE001 - report, never crash the app
            log(f"ERROR: {exc}")
            ui(st, lambda exc=exc: messagebox.showerror("Error", str(exc)))
        finally:
            ui(st, lambda: _worker_done(st, button, lock, on_done))

    threading.Thread(target=target, daemon=True).start()


def _worker_done(st, button, lock=(), on_done=None):
    st["busy"] = False
    if button is not None:
        button.configure(state="normal")
    for widget in lock:
        widgets.set_enabled(widget, True)
    if on_done is not None:
        on_done()


@contextlib.contextmanager
def signals_off():
    """Neutralize signal.signal while a worker runs.

    The DeepSDFStruct trainer installs a SIGINT handler so a terminal run can
    be stopped with Ctrl-C; off the main thread that raises instead.
    """
    original = signal.signal
    signal.signal = lambda *args, **kwargs: None
    try:
        yield
    finally:
        signal.signal = original


def reschedule(st, key, delay, fn):
    """Trailing debounce: cancel any pending call under ``key`` and re-arm it."""
    if st.get(key) is not None:
        try:
            st["root"].after_cancel(st[key])
        except tk.TclError:
            pass
    st[key] = st["root"].after(delay, fn)
