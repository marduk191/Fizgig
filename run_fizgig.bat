@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python lora_trainer_gui.py
pause
