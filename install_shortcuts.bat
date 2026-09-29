@echo off
rem Creates the desktop shortcuts for THIS clone, on THIS machine:
rem     Desktop\NN\Lattice explorer.lnk    (Explore, Explore 2-D, Train)
rem     Desktop\NN\SDF maker.lnk           (meshes -> SdfSamples dataset)
rem
rem Shortcuts hold absolute paths, so they are generated here rather than
rem committed. Safe to run again at any time, e.g. after moving the folder.
rem The shortcuts run the source in this folder directly (editable install):
rem code changes need nothing; the launcher re-runs `uv sync` by itself when
rem pyproject.toml, uv.lock or DeepSDFStruct/pyproject.toml change.
rem Details: structsept/app/launcher.py.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install it from https://docs.astral.sh/uv/ , open a new window and run this again.
    goto :fail
)

if not exist "DeepSDFStruct\pyproject.toml" (
    echo The DeepSDFStruct submodule is empty - fetching it ^(clone with --recursive next time^)...
    git submodule update --init --recursive || goto :fail
)

echo Syncing the Python environment. The first time this downloads torch and
echo friends and takes several minutes; afterwards it is instant.
uv sync || goto :fail

uv run python -m structsept.app.launcher --install || goto :fail
echo.
echo Done. Open the NN folder on your desktop and double-click a shortcut.
pause
exit /b 0

:fail
echo.
echo Something failed - the messages above say what.
pause
exit /b 1
