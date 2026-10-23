@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo FRIDAY is not set up yet. Run: py -3.12 setup_friday.py
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.main
if errorlevel 1 pause
