"""Tests for the desktop launcher.

Each test is one way a double-clicked shortcut can fail that a terminal run
never shows: no console to print to, an environment that drifted from the
lock file, a splash that never hands over, a shortcut pointing at the wrong
interpreter. No window ever appears on screen.

The launch scenarios run in a fresh interpreter each (``_run_scenario``), for
two reasons. One is fidelity: the console-less case can then be the real
thing - a child started with no standard handles, so ``sys.stdout`` really is
``None`` - instead of a simulation. The other is Tcl: creating and destroying
several Tk interpreters inside one pytest process, with pytest's fd-level
capture swapping the process's std handles around every test, made every
third run fail to read ``init.tcl`` ("couldn't read file ... No error").
One interpreter per process, which is what a real launch is, never does.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from structsept.app import launcher

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# --------------------------------------------------------------------------- #
# console substitute
# --------------------------------------------------------------------------- #


def test_headless_streams_get_a_log(tmp_path, monkeypatch):
    """Under pythonw both streams are None and tqdm dies on its first bar -
    the crash that made the Train tab unusable from a shortcut. After
    attach_log, printing and a fresh logging handler reach the log, tqdm is
    silenced, and faulthandler is pointed at the log."""
    tqdm = pytest.importorskip("tqdm")
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    # setenv-then-delenv makes monkeypatch restore the *absence* on teardown;
    # a bare delenv of a missing variable records nothing, and the value
    # attach_log sets would leak into every later test
    monkeypatch.setenv("TQDM_DISABLE", "x")
    monkeypatch.delenv("TQDM_DISABLE")
    # faulthandler is process-wide state shared with pytest: record the call
    # rather than flipping it
    enabled = []
    monkeypatch.setattr(
        launcher.faulthandler, "enable", lambda **kw: enabled.append(kw)
    )
    assert launcher.is_headless()

    log = launcher.attach_log(launcher.TARGETS["explorer"], log_dir=tmp_path)
    try:
        print("hello from a console-less process")
        assert sum(1 for _ in tqdm.trange(3, desc="Training")) == 3
        handler = logging.StreamHandler()  # binds sys.stderr at construction
        handler.emit(
            logging.LogRecord("t", logging.INFO, __file__, 1, "via handler", None, None)
        )
    finally:
        log.close()

    assert os.environ["TQDM_DISABLE"] == "1", "tqdm must be silenced, not redirected"
    assert enabled and enabled[0].get("file") is log
    text = (tmp_path / "explorer.log").read_text(encoding="utf-8")
    assert "=== " in text and "explorer" in text
    assert "hello from a console-less process" in text
    assert "via handler" in text


def test_console_runs_are_left_alone():
    """From a terminal the streams exist and must stay the terminal's."""
    assert not launcher.is_headless()


def test_log_rotates_past_the_cap(tmp_path):
    big = tmp_path / "explorer.log"
    big.write_bytes(b"x" * (launcher.LOG_MAX_BYTES + 1))
    launcher.open_log(launcher.TARGETS["explorer"], log_dir=tmp_path).close()
    assert (tmp_path / "explorer.log.1").stat().st_size == launcher.LOG_MAX_BYTES + 1
    assert big.stat().st_size < 1000, "the new log starts with a header only"


def test_icons_are_drawn_at_every_size(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    written = launcher.draw_icons(tmp_path, force=True)
    assert {p.name for p in written} == {"explorer.ico", "sdf_maker.ico"}
    for path in written:
        with Image.open(path) as ico:
            assert ico.size == (256, 256)
            assert (16, 16) in ico.info["sizes"]


# --------------------------------------------------------------------------- #
# environment freshness
# --------------------------------------------------------------------------- #


def test_env_hash_follows_the_dependency_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (repo / "uv.lock").write_text("version = 1\n")
    marker = repo / ".venv" / "hash"

    assert launcher.env_needs_sync(repo, marker), "never synced"
    launcher.record_synced(repo, marker)
    assert not launcher.env_needs_sync(repo, marker)
    (repo / "uv.lock").write_text("version = 2\n")
    assert launcher.env_needs_sync(repo, marker), "a git pull changed the lock"
    launcher.record_synced(repo, marker)
    (repo / "pyproject.toml").write_text("[project]\nname = 'y'\n")
    assert launcher.env_needs_sync(repo, marker), "pyproject counts too"
    launcher.record_synced(repo, marker)
    (repo / "DeepSDFStruct").mkdir()
    (repo / "DeepSDFStruct" / "pyproject.toml").write_text("[project]\nname = 'lib'\n")
    assert launcher.env_needs_sync(repo, marker), "the submodule's pyproject too"


def test_sync_without_uv_reports_instead_of_raising(monkeypatch):
    monkeypatch.setattr(launcher, "find_uv", lambda: None)
    said = []
    assert launcher.sync_env(said.append) is False
    assert said and "uv" in said[0]


def test_sync_hides_the_console_only_when_asked(tmp_path, monkeypatch):
    """Under pythonw a console child pops its own window unless told not to;
    the flag has to come from run(), since by then the streams are the log."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    marker = repo / ".venv" / "hash"
    monkeypatch.setattr(launcher, "find_uv", lambda *a, **k: str(tmp_path / "uv"))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    assert launcher.sync_env(lambda m: None, repo, marker, no_window=True)
    assert calls[-1][1]["creationflags"] == no_window
    assert calls[-1][1]["cwd"] == repo and calls[-1][0][1:] == ["sync"]
    assert not launcher.env_needs_sync(repo, marker), "success records the hash"
    assert launcher.sync_env(lambda m: None, repo, marker, no_window=False)
    assert calls[-1][1]["creationflags"] == 0


def test_find_uv_falls_back_to_the_path_recorded_at_sync(tmp_path, monkeypatch):
    """A shortcut runs with Explorer's PATH, which may lack the terminal's uv;
    record_synced leaves a note of where it was."""
    fake_uv = tmp_path / "somewhere" / "uv.exe"
    fake_uv.parent.mkdir()
    fake_uv.write_text("")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    marker = repo / ".venv" / "hash"
    hint = marker.with_name(launcher.UV_HINT.name)

    monkeypatch.setattr(launcher.shutil, "which", lambda name: str(fake_uv))
    launcher.record_synced(repo, marker)
    assert hint.read_text(encoding="utf-8") == str(fake_uv)

    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path / "nohome")
    assert launcher.find_uv(hint=hint) == str(fake_uv)
    assert launcher.find_uv(hint=tmp_path / "missing") is None


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-range locks")
def test_instance_lock_sees_other_windows(tmp_path):
    path = tmp_path / "instances"
    first = launcher.InstanceLock(path)
    try:
        assert first.slot is not None
        assert not first.others_running()
        second = launcher.InstanceLock(path)
        try:
            assert second.slot not in (None, first.slot)
            assert first.others_running() and second.others_running()
        finally:
            second.close()
        assert not first.others_running(), "a closed window must not count"
    finally:
        first.close()


# --------------------------------------------------------------------------- #
# splash -> app handover, one fresh interpreter per scenario
# --------------------------------------------------------------------------- #

# Runs before each scenario in the child. Everything that would show or block
# is replaced: the splash stays withdrawn, the window is never made opaque,
# the error dialog is recorded, the instance lock and the log go to tmp_path.
_SCENARIO_PRELUDE = r"""
import json, os, pathlib, sys, threading, types
from structsept.app import launcher

out = {"alerts": [], "headless": launcher.is_headless(), "sync_calls": 0}
launcher.SPLASH_VISIBLE = False
launcher._show_centered = lambda root: root.deiconify()
launcher._alert = lambda title, msg: out["alerts"].append([title, msg])
launcher.LOCK_FILE = pathlib.Path(os.environ["LAUNCHER_TEST_LOCK"])
launcher.LOG_DIR = pathlib.Path(os.environ["LAUNCHER_TEST_LOG"])
launcher.env_needs_sync = lambda *a, **k: False


def fake_module(name, build_app):
    module = types.ModuleType(name)
    module.build_app = build_app
    sys.modules[name] = module
    return name


def closing_app(root=None):
    out["root_seen"] = root is not None
    out["build_thread"] = threading.current_thread().name
    try:
        root.attributes("-alpha", 0.0)  # deiconified by the launcher, unseen
    except Exception:
        pass
    root.after(150, root.destroy)
    return root


def target(key, module, themed=True):
    launcher.TARGETS[key] = launcher.Target(key, module, key, key, "", False, themed)
    return key


def finish(code):
    out["code"] = code
    out["stderr_is_none"] = sys.stderr is None
    with open(os.environ["LAUNCHER_TEST_RESULT"], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
"""


def _run_scenario(tmp_path, code, *, headless=False):
    """Run ``code`` after the prelude in a fresh interpreter.

    ``headless=True`` runs the child under ``pythonw.exe`` with nothing
    attached - what a shortcut does - so its ``sys.stdout``/``sys.stderr``
    are ``None`` and ``attach_log`` has to do its job for real (a console
    ``python.exe`` always gets streams, even detached). Returns the child's
    result dict, its stderr text and the concatenated launcher logs.
    """
    result_file = tmp_path / "result.json"
    log_dir = tmp_path / "logs"
    env = dict(
        os.environ,
        LAUNCHER_TEST_RESULT=str(result_file),
        LAUNCHER_TEST_LOCK=str(tmp_path / "instances"),
        LAUNCHER_TEST_LOG=str(log_dir),
    )
    exe = Path(sys.executable)
    if headless:
        exe = exe.with_name("pythonw.exe")
        if not exe.is_file():
            pytest.skip("no pythonw.exe next to the interpreter")
        kwargs = {}  # a pipe would be a stream; the point is to have none
    else:
        kwargs = {"capture_output": True, "text": True}
    proc = subprocess.run(
        [str(exe), "-c", _SCENARIO_PRELUDE + code],
        cwd=launcher.REPO_ROOT,
        env=env,
        timeout=180,
        check=False,
        **kwargs,
    )
    stderr = proc.stderr or ""
    assert result_file.is_file(), f"scenario died with {proc.returncode}:\n{stderr}"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    logs = (
        "".join(p.read_text(encoding="utf-8") for p in sorted(log_dir.glob("*.log")))
        if log_dir.is_dir()
        else ""
    )
    return result, stderr, logs


HANDOVER = 'finish(launcher.run(target("fake", fake_module("_fake", closing_app))))\n'


def test_splash_hands_its_root_to_build_app(tmp_path):
    result, _, _ = _run_scenario(tmp_path, HANDOVER)
    assert result["code"] == 0
    assert result["root_seen"], "build_app never ran"
    assert result["build_thread"] == "MainThread", "widgets built off the Tk thread"
    assert result["alerts"] == []
    assert result["headless"] is False


def test_console_less_launch_logs_instead_of_crashing(tmp_path):
    """The shortcut case for real: no std handles, so attach_log must give
    the process a log, and the launch must still reach build_app."""
    result, _, logs = _run_scenario(tmp_path, HANDOVER, headless=True)
    assert result["headless"] is True
    assert result["stderr_is_none"] is False, "attach_log did not replace stderr"
    assert result["code"] == 0 and result["root_seen"]
    assert "=== " in logs and "fake" in logs


def test_skipped_environment_update_reaches_the_log(tmp_path):
    """A sync that could not run is flashed on the splash for one tick; the
    log line is the only durable trace, and the docs promise it."""
    code = """
launcher.env_needs_sync = lambda *a, **k: True
def failing_sync(report, *a, **k):
    report("uv not found; skipping the environment update")
    return False
launcher.sync_env = failing_sync
""" + HANDOVER
    result, stderr, _ = _run_scenario(tmp_path, code)
    assert result["code"] == 0
    assert "[launcher] uv not found" in stderr
    result, _, logs = _run_scenario(tmp_path, code, headless=True)
    assert result["code"] == 0
    assert "[launcher] uv not found" in logs, "under a shortcut it goes to the log"


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-range locks")
def test_no_sync_while_another_window_is_open(tmp_path):
    code = """
launcher.env_needs_sync = lambda *a, **k: True
def counting_sync(*a, **k):
    out["sync_calls"] += 1
    return True
launcher.sync_env = counting_sync
""" + HANDOVER
    other = launcher.InstanceLock(
        tmp_path / "instances"
    )  # this process is "the other window"
    try:
        result, stderr, _ = _run_scenario(tmp_path, code)
    finally:
        other.close()
    assert result["code"] == 0
    assert result["sync_calls"] == 0, "uv sync ran under an open window"
    assert "another structsept window is open" in stderr


def test_missing_package_is_explained(tmp_path):
    """A stale environment after a git pull must produce a dialog that says
    what to do, not a process that silently exits."""
    code = 'finish(launcher.run(target("broken", "structsept_no_such_module_xyz", themed=False)))\n'
    result, _, _ = _run_scenario(tmp_path, code)
    assert result["code"] == 1
    assert len(result["alerts"]) == 1
    title, message = result["alerts"][0]
    assert "could not start" in title
    assert "structsept_no_such_module_xyz" in message
    assert "uv sync" in message


def test_build_app_failure_is_a_dialog_not_a_hang(tmp_path):
    code = """
def exploding(root=None):
    raise TypeError("wrong root")
finish(launcher.run(target("brokenbuild", fake_module("_broken", exploding), themed=False)))
"""
    result, _, _ = _run_scenario(tmp_path, code)
    assert result["code"] == 1
    assert result["alerts"] and "TypeError" in result["alerts"][0][1]


def test_cli_without_a_target_prints_help(capsys):
    assert launcher.main([]) == 2
    assert "explorer" in capsys.readouterr().out


def test_sdf_maker_builds_into_a_given_root():
    tk = pytest.importorskip("tkinter")
    from structsept.app import sdf_maker

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    app = sdf_maker.build_app(root=root)
    try:
        assert app is root
        root.update()
    finally:
        sdf_maker._shutdown(app.app_state, app)


# --------------------------------------------------------------------------- #
# shortcuts
# --------------------------------------------------------------------------- #


def test_ps_quote_doubles_single_quotes():
    assert launcher._ps_quote("O'Neil") == "'O''Neil'"
    assert launcher._ps_quote("Tom’s") == "$('Tom' + [char]0x2019 + 's')"
    assert launcher._ps_quote("‘x’") == "$([char]0x2018 + 'x' + [char]0x2019)"


@pytest.mark.skipif(os.name != "nt", reason="needs powershell.exe")
def test_powershell_round_trips_umlauts_and_typographic_quotes():
    """The two ways a real Desktop path broke the installer: an umlaut in the
    user name came back in the wrong code page, and a typographic apostrophe
    ended the PowerShell string early."""
    text = "Jürgen’s NN (O'Neil)"
    result = launcher.run_powershell("Write-Output " + launcher._ps_quote(text))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == text


def test_install_refuses_outside_the_project_environment(monkeypatch):
    monkeypatch.setattr(launcher.sys, "prefix", launcher.sys.base_prefix)
    monkeypatch.setattr(launcher.os, "name", "nt")
    with pytest.raises(SystemExit, match="uv run"):
        launcher.install(dest=Path("unused"))


def test_shortcut_script_points_at_this_clone(tmp_path):
    script = launcher.shortcut_script(tmp_path)
    assert str(launcher.VENV_PYTHONW) in script
    assert str(launcher.REPO_ROOT) in script
    assert str(tmp_path.resolve()) in script
    for target in launcher.TARGETS.values():
        assert f"-m structsept.app.launcher {target.key}" in script
        assert f"{target.key}.ico" in script
    assert "GetFolderPath" not in script
    assert "GetFolderPath('Desktop')" in launcher.shortcut_script(None)


@pytest.mark.skipif(os.name != "nt", reason="writes Windows .lnk files")
def test_install_writes_shortcuts_that_resolve(tmp_path, monkeypatch):
    if not launcher.VENV_PYTHONW.is_file():
        pytest.skip("no .venv pythonw.exe to point at")
    # the real install stamps the real .venv as in sync; a test must not
    monkeypatch.setattr(launcher, "record_synced", lambda *a, **k: None)
    folder = launcher.install(dest=tmp_path / "NN")
    assert folder == (tmp_path / "NN").resolve()
    for target in launcher.TARGETS.values():
        lnk = folder / f"{target.shortcut}.lnk"
        assert lnk.is_file()
        read_back = (
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
            f"{launcher._ps_quote(lnk)}); "
            "$s.TargetPath; $s.Arguments; $s.WorkingDirectory; $s.IconLocation"
        )
        out = launcher.run_powershell(read_back).stdout.splitlines()
        assert Path(out[0]) == launcher.VENV_PYTHONW
        assert out[1] == f"-m structsept.app.launcher {target.key}"
        assert Path(out[2]) == launcher.REPO_ROOT
        assert Path(out[3].rsplit(",", 1)[0]).is_file(), "icon file missing"
