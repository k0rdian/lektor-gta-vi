@echo off
setlocal
cd /d "%~dp0"
title Lektor GTA VI
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom setup.bat
  pause
  exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m lektor %*
if errorlevel 1 pause
