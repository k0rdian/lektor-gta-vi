@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo   Lektor GTA VI v2 - instalacja glosu Chatterbox
echo ==================================================
echo  Osobne srodowisko .venv-chatterbox - PyTorch z CUDA, ok. 6 GB
echo  + modele Chatterbox Multilingual V3 i szybki dekoder - ok. 4 GB w modele\chatterbox.
echo  Wymaga karty NVIDIA. Glowna aplikacja (.venv) nie jest zmieniana.
echo.
call :znajdz_pythona
if defined PY goto :mam_pythona
echo Nie znalazlem Pythona 3.12 ani 3.11 - instaluje Pythona 3.12 przez winget...
where winget >nul 2>&1
if errorlevel 1 goto :brak_pythona
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :znajdz_pythona
if defined PY goto :mam_pythona
:brak_pythona
echo PyTorch wymaga Pythona 3.12 albo 3.11. Zainstaluj Python 3.12 ze strony
echo https://www.python.org/downloads/windows/ i uruchom ten plik ponownie.
goto :blad

:mam_pythona
echo Uzywam: %PY%
if not exist ".venv-chatterbox\Scripts\python.exe" %PY% -m venv .venv-chatterbox
set "VPY=.venv-chatterbox\Scripts\python.exe"
if not exist "%VPY%" goto :blad
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :blad
echo.
echo [1/4] PyTorch z CUDA 12.8 - dziala z RTX 50xx, ok. 3 GB...
"%VPY%" -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :blad
echo.
echo [2/4] Biblioteki Chatterboxa...
"%VPY%" -m pip install -r requirements-chatterbox.txt
if errorlevel 1 goto :blad
echo.
echo [3/4] Chatterbox - GitHub, wersja z Multilingual V3...
"%VPY%" -m pip install --no-deps https://github.com/resemble-ai/chatterbox/archive/ab55923c27757c90cfcc407baa57dc25a7d25217.zip
if errorlevel 1 goto :blad
echo.
echo [4/4] Pobieranie modeli i test karty graficznej...
"%VPY%" lektor\chatterbox_worker.py --pobierz --test
if errorlevel 1 goto :blad
echo.
echo Gotowe. Uruchom aplikacje (uruchom.bat) i wybierz glos "Chatterbox".
if /i not "%~1"=="bezpauzy" pause
exit /b 0

:blad
echo.
echo Instalacja nie powiodla sie - sprawdz komunikaty powyzej.
if /i not "%~1"=="bezpauzy" pause
exit /b 1

:znajdz_pythona
set "PY="
for %%V in (3.12 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if defined PY exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
exit /b 0
