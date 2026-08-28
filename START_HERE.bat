@echo off
setlocal
cd /d "%~dp0"
title Pitcher Research Lab

echo ==============================================
echo        PITCHER RESEARCH LAB - STARTUP
echo ==============================================
echo.

set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py -3"

if not defined PY_CMD (
    where python >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo ERROR: Python was not found on this computer.
    echo.
    echo Install Python 3 from python.org and make sure
    echo "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

echo Found Python. Preparing the app...

if not exist ".venv\Scripts\python.exe" (
    echo Creating the app environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto :failed
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo Checking required packages...
"%VENV_PY%" -c "import flask, pandas, numpy, requests" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    "%VENV_PY%" -m pip install --upgrade pip
    if errorlevel 1 goto :failed
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

echo.
echo Starting Pitcher Research Lab...
echo Browser address: http://127.0.0.1:5050
echo.
echo IMPORTANT: Keep this black window open while using the app.
echo Close this window when you are finished.
echo.

if not defined PRL_PORT set "PRL_PORT=5050"
if not defined PRL_OPEN_BROWSER set "PRL_OPEN_BROWSER=1"
"%VENV_PY%" app.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo ==============================================
echo STARTUP FAILED
echo ==============================================
echo.
echo The error is shown above. Keep this window open while
echo troubleshooting the startup failure.
echo.
pause
exit /b 1
