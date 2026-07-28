@echo off
cd /d "%~dp0"
start "" /b wscript //nologo //b "%~dp0run_silent.vbs"
