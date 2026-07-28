@echo off
REM Foreground launcher with a visible console.
REM
REM run_fizgig.bat is the normal way in: it hands off to run_silent.vbs so no console
REM window hangs around. That hides stdout, and the caching and training subprocesses
REM log there -- so a run that dies early leaves nothing to read.
REM
REM Use this one when something is failing and you need the traceback. `pause` keeps
REM the window open after a crash instead of it vanishing with the error.
REM
REM Kept as a separate file on purpose: update_fizgig.bat runs
REM `git checkout -- run_fizgig.bat` to undo old installers that clobbered the
REM consoleless launcher, so editing that file directly gets reverted on every update.
cd /d "%~dp0"
call venv\Scripts\activate
python lora_trainer_gui.py
pause
