@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
    echo venv not found - run install_fizgig.bat first.
    pause
    exit /b 1
)
start "" "venv\Scripts\pythonw.exe" diff_to_lora_gui.py
