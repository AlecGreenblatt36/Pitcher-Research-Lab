@echo off
setlocal
cd /d "%~dp0"
title Pitcher Research Lab - Validation

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo The project environment was not found.
    echo Run START_HERE.bat once, then run this file again.
    echo.
    pause
    exit /b 1
)

echo Running project validation...
"%PY%" validate_project.py
if errorlevel 1 goto :failed

echo.
echo Running regression tests when pytest is available...
"%PY%" -c "import pytest" >nul 2>&1
if errorlevel 1 (
    echo pytest is not installed in this environment.
    echo To run the full suite: "%PY%" -m pip install -r requirements-dev.txt
    echo Then run run_checks.bat again.
    echo.
    pause
    exit /b 0
)

"%PY%" -m pytest -q
if errorlevel 1 goto :failed

echo.
echo All checks passed.
pause
exit /b 0

:failed
echo.
echo Validation failed. Review the messages above.
pause
exit /b 1
