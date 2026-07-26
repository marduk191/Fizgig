@echo off
cd /d "%~dp0"
echo Updating Fizgig...
git pull
echo.
echo Installing/updating dependencies...
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found. Skipping dependency installation.
    echo.
    goto skip_deps
)

if not exist "venv\Scripts\python.exe" (
    echo WARNING: venv not found - run install_fizgig.bat to set it up.
    echo.
    goto skip_deps
)

"venv\Scripts\python.exe" -m uv --version >nul 2>&1
if errorlevel 1 (
    "venv\Scripts\python.exe" -m pip install --upgrade uv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install uv package.
        echo Aborting update.
        exit /b 1
    )
)

"venv\Scripts\python.exe" -m uv pip install --link-mode=copy --index-strategy unsafe-best-match -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install/update dependencies using uv. See output above.
    echo Aborting update.
    exit /b 1
)

:skip_deps

REM ---- MSVC C++ Build Tools check (needed by torch.compile's inductor backend) ----
REM triton installs via requirements.txt, but inductor also needs a C++ compiler on
REM Windows. Detection via vswhere (ships with any VS/Build Tools install).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set MSVC_FOUND=0
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set MSVC_FOUND=1
)
if "%MSVC_FOUND%"=="0" (
    echo.
    echo ------------------------------------------------------------------------
    echo  OPTIONAL: MSVC C++ Build Tools not detected.
    echo.
    echo  The torch.compile training speedup needs them. Training works fine
    echo  without - you just skip the compile speedup.
    echo.
    echo  Direct download ^(exact installer, no hunting on the MS site^):
    echo    https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo  During install, tick the "Desktop development with C++" workload.
    echo.
    echo  Or install unattended from a terminal:
    echo    winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
    echo.
    echo  Install it any time - no need to re-run this update. The next training
    echo  run detects it automatically.
    echo ------------------------------------------------------------------------
)

echo.
echo Update complete! Run Fizgig with run_fizgig.bat
pause
