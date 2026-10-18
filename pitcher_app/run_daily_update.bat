@echo off
cd /d %~dp0

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe update_all_pitchers.py
) else (
    python update_all_pitchers.py
)
