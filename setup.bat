@echo off
setlocal
cd /d "%~dp0"
title Lektor GTA VI - instalacja
echo.
echo  === Lektor GTA VI - instalacja ===
echo.

set "PY="
for %%V in (3.13 3.12 3.14) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Nie znalazlem Pythona 3.12-3.14.
  echo Zainstaluj Pythona 3.13 z https://www.python.org/downloads/windows/
  echo i przy instalacji zaznacz "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)
echo Uzywam: %PY%

if not exist ".venv\Scripts\python.exe" (
  echo Tworze srodowisko .venv ...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
)

echo Instaluje biblioteki, to chwile potrwa ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Sprawdzam polski OCR w Windows ...
".venv\Scripts\python.exe" -c "import sys; from winrt.windows.media.ocr import OcrEngine; from winrt.windows.globalization import Language; sys.exit(0 if OcrEngine.is_language_supported(Language('pl')) else 1)"
if errorlevel 1 goto :no_ocr
echo Polski OCR jest dostepny.
goto :done

:no_ocr
echo.
echo  BRAK polskiego OCR. Otworz PowerShell jako administrator i wpisz:
echo    Get-WindowsCapability -Online ^| Where-Object Name -like 'Language.OCR*pl-PL*' ^| Add-WindowsCapability -Online
echo  albo dodaj jezyk polski: Ustawienia - Czas i jezyk - Jezyk i region.

:done
echo.
echo Gotowe. Dalej:
echo   test-glosu.bat       - test glosu Marek
echo   lista-urzadzen.bat   - karty, monitory i wyjscia audio
echo   start.bat            - uruchom lektora
echo.
pause
exit /b 0

:error
echo.
echo Instalacja nie powiodla sie - przewin wyzej, zeby zobaczyc blad.
pause
exit /b 1
