@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
  echo Najpierw uruchom instaluj.bat
  pause
  exit /b 1
)
echo ==================================================
echo   Lektor GTA VI v2 - testy bez gry
echo ==================================================
echo.
echo [1/4] Logika: pamiec przeczytanych linii, filtry tablic i ruchu (kilka sekund)
".venv\Scripts\python.exe" testy\test_logiki.py
echo.
echo [2/4] Scenariusze z prawdziwym OCR (ok. 1,5 min): druga linia, przewijanie,
echo       tablica rejestracyjna, mrugniecie napisu...
".venv\Scripts\python.exe" testy\scenariusze.py
echo.
if not exist ".venv-chatterbox\Scripts\python.exe" goto bez_chatterboxa
echo [3/4] Chatterbox: zgodnosc i szybkosc dekodera na karcie graficznej - ok. 1 min
".venv-chatterbox\Scripts\python.exe" lektor\chatterbox_worker.py --test-szybki
echo.
echo [4/4] Chatterbox: strumieniowanie - czas do pierwszego dzwieku w trzech trybach,
echo       bez odtwarzania (nagrania do odsluchu: probki\strumien_*.wav) - ok. 2 min
".venv-chatterbox\Scripts\python.exe" testy\strumien_test.py
goto koniec
:bez_chatterboxa
echo [3/4] Chatterbox nie jest zainstalowany - pomijam. Instalacja: instaluj_chatterbox.bat
:koniec
echo.
echo Gotowe. Filmy ze scenariuszami do obejrzenia w aplikacji: testy\scenariusze\*.mp4
pause
