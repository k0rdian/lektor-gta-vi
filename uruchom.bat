@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom instaluj.bat
  pause
  exit /b 1
)
".venv\Scripts\python.exe" aplikacja.py
pause
