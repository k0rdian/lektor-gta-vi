@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
echo ==================================================
echo   Lektor GTA VI v2 - instalacja
echo ==================================================
set "BEZPAUZY="
if /i "%~1"=="bezpauzy" set "BEZPAUZY=1"

call :znajdz_pythona
if defined PY goto :mam_pythona
echo Nie znalazlem Pythona 3.11-3.13 - instaluje Pythona 3.12 przez winget...
where winget >nul 2>&1
if errorlevel 1 goto :brak_pythona
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :znajdz_pythona
if defined PY goto :mam_pythona
:brak_pythona
echo.
echo Nie udalo sie zainstalowac Pythona automatycznie.
echo Zainstaluj Python 3.12 ze strony https://www.python.org/downloads/windows/
echo i uruchom ten plik ponownie.
goto :blad

:mam_pythona
echo Uzywam: %PY%
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if not exist ".venv\Scripts\python.exe" goto :blad
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :blad
echo.
echo Pobieram domyslny glos lektora - Piper, ok. 65 MB...
".venv\Scripts\python.exe" -c "from lektor.glos import pobierz_glos; pobierz_glos('meski')"
if errorlevel 1 echo Glosu nie udalo sie pobrac teraz - aplikacja sprobuje przy pierwszym uruchomieniu.
echo.
echo Glowna aplikacja jest zainstalowana.
if defined BEZPAUZY goto :koniec
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :koniec
echo.
echo Masz karte NVIDIA, wiec mozesz doinstalowac naturalny, klonowany glos Chatterbox.
echo Zajmie ok. 10 GB na dysku i 15-40 minut pobierania. Mozna to zrobic tez pozniej
echo plikiem instaluj_chatterbox.bat.
choice /c TN /m "Zainstalowac Chatterbox teraz"
if errorlevel 2 goto :koniec
call instaluj_chatterbox.bat bezpauzy

:koniec
echo.
echo Gotowe. Uruchom aplikacje plikiem uruchom.bat
if not defined BEZPAUZY pause
exit /b 0

:blad
echo.
echo Instalacja nie powiodla sie - sprawdz komunikaty powyzej.
if not defined BEZPAUZY pause
exit /b 1

:znajdz_pythona
set "PY="
for %%V in (3.12 3.13 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if defined PY exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY exit /b 0
python -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13)" >nul 2>&1 && set "PY=python"
exit /b 0
