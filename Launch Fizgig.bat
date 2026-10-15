@echo off
rem Fizgig in its own app window. In your web browser instead: "Launch Fizgig in browser.bat".
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Fizgig's Python environment is missing.
    echo.
    echo Run install_fizgig.bat first, then try again.
    pause
    exit /b 1
)
start "" "venv\Scripts\pythonw.exe" fizgig.py
