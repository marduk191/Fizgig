@echo off
rem Fizgig's web UI in your web browser: this window runs it (close it to stop Fizgig) and the
rem browser opens on it once it is up. Already running? The browser just opens on it.
title Fizgig
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Fizgig's Python environment is missing.
    echo.
    echo Run install_fizgig.bat first, then try again.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" fizgig.py --web --open %*
pause
