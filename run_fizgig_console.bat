@echo off
REM Foreground launcher with a visible console.
REM
REM "Launch Fizgig.bat" is the normal way in: it starts fizgig.py under pythonw, so no console
REM window hangs around. That hides stdout, where the server and the caching, training and
REM Studio subprocesses log, so a run that dies early leaves nothing to read.
REM
REM Use this one when something is failing and you need the traceback. It runs the same
REM fizgig.py (the app window) under python.exe, and `pause` keeps the window open after a
REM crash instead of letting it vanish with the error.
REM
REM Kept as its own file, not an edit of the shipped launchers: the updater restores those
REM with `git checkout -- <launcher>` before every pull, so any change to them is reverted.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Fizgig's Python environment is missing.
    echo.
    echo Run install_fizgig.bat first, then try again.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" fizgig.py %*
pause
