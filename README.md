# Lektor GTA VI

Czyta polskie napisy z gry na głos — jak lektor w telewizji. Gra chodzi na PS5,
obraz trafia do PC przez kartę Elgato HD60 X (albo z ekranu), a PC rozpoznaje
napisy (OCR Windowsa) i czyta je głosem **Marek** z Edge TTS.

```
obraz z karty → wycinek z napisami → obróbka → OCR Windowsa → odsiew powtórek → Edge TTS (Marek) → głośniki
```

- każda kwestia jest czytana raz, nawet jeśli wisi na ekranie kilka sekund albo OCR na chwilę ją zgubi,
- lektor czyta płynnie: bez pauz na przecinkach i z krótkimi pauzami między zdaniami,
- gdy napisy lecą szybciej, niż lektor czyta, przyspiesza (do +60%), a wszystkie rozpoznane kwestie czekają na odczyt w kolejności,
- nieprzeczytane teksty są zapisywane w `kolejka.json` i wracają po ponownym uruchomieniu,
- nazwy własne możesz „spolszczyć” w `config.toml` (domyślnie *Vice City → Wajs Siti*, *Jason → Dżejson*),
- nic nie dotyka konsoli ani gry — lektor tylko patrzy na obraz.

## Instalacja (raz)

1. Zainstaluj **Pythona 3.13** z [python.org](https://www.python.org/downloads/windows/) i zaznacz *Add python.exe to PATH*.
2. Skopiuj ten folder na PC i kliknij dwukrotnie **`setup.bat`**.
3. Jeśli setup napisze, że brakuje polskiego OCR, otwórz PowerShell **jako administrator** i wklej:
   ```powershell
   Get-WindowsCapability -Online | Where-Object Name -like 'Language.OCR*pl-PL*' | Add-WindowsCapability -Online
   ```
4. Kliknij **`test-glosu.bat`** — powinieneś usłyszeć Marka. Jeśli tak, Edge TTS i dźwięk działają.

## Podłączenie obrazu

- **HD60 X**: kabel USB z karty do **PC**. Monitor dalej dostaje obraz z wyjścia HDMI OUT karty, bez opóźnień.
- Kartę może używać naraz tylko jeden program. Zamknij OBS i Elgato Studio **albo** włącz w OBS
  *Uruchom wirtualną kamerę* i wpisz w `config.toml`: `device = "OBS Virtual Camera"`.
- W PS5 musi być wyłączone HDCP: *Ustawienia → System → HDMI → Włącz HDCP* (wyłącz).
- W GTA włącz napisy po polsku.
- Bez karty: `start.bat --source screen` czyta z monitora (PS Remote Play, gra na PC w trybie okna bez ramki).

`lista-urzadzen.bat` pokazuje karty i kamery, monitory oraz wyjścia audio.

## Uruchomienie

### Test na YouTube

Kliknij **`youtube.bat`**. Otworzy się widoczny panel **Lektor — sterowanie**.
Na początku lektor jest zatrzymany. Przycisk **Pokaż numery** wyświetla przez
chwilę duży numer na każdym monitorze. Lista podaje także położenie monitora
(główny, po lewej, po prawej), więc jednakowa rozdzielczość nie przeszkadza w wyborze.

W przeglądarce włącz polskie napisy (CC) i zatrzymaj film z widocznym napisem.
Wybierz monitor i kliknij **Wybierz napisy**. W ciągu 7 sekund pokaż przeglądarkę.
Na zrzucie zaznacz pas napisów z zapasem na dwie linie i zatwierdź Enterem.
Lektor nadal milczy: dopiero **Czytaj** uruchamia odczyt. Wznów też film.

Panel pozostaje na wierzchu i można go przeciągnąć poza obszar napisów.
**Pauza** ucisza głos i zachowuje kolejkę. Czerwony **ZATRZYMAJ** kończy proces
odczytu, również podczas zaznaczania obszaru lub odliczania. **Zamknij** oraz
krzyżyk okna kończą odczyt i zamykają panel. Ponowne uruchomienie programu
nie tworzy drugiego panelu ani drugiego głosu.

### Kolejka bez pomijania kwestii

Wszystkie przyjęte przez lektora kwestie są czytane po kolei, nawet gdy
opóźnienie rośnie. Wiek kwestii nie powoduje jej usunięcia. Po zakończeniu
pliku wideo lektor czeka na odczyt całej kolejki, bez limitu 30 sekund.

Panel pokazuje liczbę oczekujących kwestii (wlicza też bieżącą). Przycisk
**Pokaż kolejkę** otwiera pełną listę w kolejności odczytu. Teksty są zapisywane
w `kolejka.json` w folderze aplikacji i usuwane z zapisu dopiero po pełnym
odtworzeniu. Chwilowy błąd syntezy lub dźwięku powoduje ponowienie próby tej
samej kwestii; następne czekają.

Pauza, zatrzymanie i zamknięcie aplikacji zachowują zaległe teksty. Po ponownym
wybraniu źródła i kliknięciu **Czytaj** lektor odczyta najpierw zapisane
zaległości. Przerwana kwestia jest czytana od początku, aby nie zgubić jej
końcówki. Dopiero ręczne **Wyczyść kolejkę…** w oknie podglądu usuwa wszystkie
zaległości; skrót **Pomiń** nadal pozwala świadomie pominąć bieżącą kwestię.
Podczas pauzy rozpoznawanie nowych napisów także jest wstrzymane.

Lektor czyta napisy widoczne na ekranie; film musi pozostać odsłonięty,
na tym samym monitorze, w tym samym miejscu i rozmiarze. Po zmianie układu
wybierz obszar ponownie przez **Ctrl+Alt+W**. **Ctrl+Alt+P** wstrzymuje czytanie,
a **Ctrl+Alt+Q** kończy lektora. Obszar ekranu jest zapisywany osobno od karty Elgato.
Ten tryb nie pobiera filmu ani nie tłumaczy dźwięku. Możesz też użyć
`start.bat --youtube` lub `start.bat --source screen --monitor 2 --select`.

### Tempo i pauzy

Lektor czyta z tempem bazowym `rate` (+20%). Przecinki, myślniki i wielokropki w środku zdania
nie robią pauz (`skip_commas`), a dłuższe pauzy syntezatora, np. między zdaniami, są skracane
do `max_pause_ms` (180 ms). Gdy napisy lecą szybciej, niż lektor czyta, przyspiesza o `rate_step`
(15%) za każdą sekundę spóźnienia ponad `catch_up_s` (1,5 s), najwyżej do `rate_max` (+60%),
i po nadrobieniu stopniowo zwalnia. Kolejne kwestie grają przez jeden stały strumień audio,
bez dodatkowej ciszy między nimi (`gapless`). Wszystko to ustawisz w sekcji `[tts]` w `config.toml`.

Przy każdej czytanej kwestii lektor wypisuje opóźnienie (od pojawienia się napisu do początku
czytania), tempo i czas syntezy — widać to w ostatniej linii panelu. Opóźnienie stale rośnie →
podnieś `rate` albo `rate_max`; lektor brzmi zbyt pośpiesznie → obniż `rate`.

### Gra / karta Elgato

Kliknij **`start.bat`**. W widocznym panelu wybierz **Karta Elgato / gra**,
następnie **Wybierz napisy**. Na klatce z gry zaznacz myszką pas napisów
(z zapasem na dwie linie) i wciśnij **ENTER**. Kliknij **Czytaj**, aby uruchomić głos.
Obszar zapisuje się w `roi.json`. Każde uruchomienie zaczyna się bez odczytu.

| Skrót (globalny) | W oknie konsoli | Działanie |
|---|---|---|
| Ctrl+Alt+P | p | pauza / wznowienie |
| Ctrl+Alt+R | r | powtórz ostatnią kwestię |
| Ctrl+Alt+K | k | pomiń bieżącą kwestię |
| Ctrl+Alt+W | w | zaznacz obszar od nowa |
| Ctrl+Alt+D | d | podgląd: co widzi OCR |
| Ctrl+Alt+Q | q | wyjście |

Przydatne opcje: `start.bat --preview` (od razu z podglądem), `--verbose` (każdy odczyt OCR w konsoli),
`--select` (nowy obszar), `--voice pl-PL-ZofiaNeural`, `--voices` (lista polskich głosów).

## Gdy coś nie czyta — strojenie

Tryb `outline` zachowuje cienkie litery oraz polskie akcenty przy antyaliasingu.
Szerokość szukanej obwódki dopasowuje do rozmiaru liter; małe napisy na ekranie
4K są automatycznie powiększane do OCR. Pozostaw `upscale = 0`, aby korzystać
z tego doboru. Po zmianie wielkości odtwarzacza wybierz obszar napisów ponownie
i zostaw zapas nad oraz pod literami, aby nie obciąć kresek, kropek i ogonków.

1. Włącz podgląd (**Ctrl+Alt+D**). Górny pasek to wycinek z gry, dolny to obraz dla OCR:
   powinien pokazywać **czarne litery na białym tle** i niewiele więcej.
2. Litery giną albo są dziurawe → w `config.toml` obniż `brightness` (np. 150) lub podnieś `outline` (np. 140).
3. Łapie śmieci z tła → podnieś `brightness` (np. 190), obniż `outline` (np. 90) albo zawęź obszar.
4. Napisy leżą na ciemnym pasku tła → spróbuj `preprocess = "gray"`.
5. Zrób zrzut ekranu z napisem i uruchom `start.bat --test-ocr zrzut.png` — porówna wszystkie tryby
   obróbki i zapisze obrazy do `debug_samples`.
6. `save_samples = true` w sekcji `[debug]` zapisuje wycinek i tekst każdej przeczytanej kwestii — przyślij
   kilka takich plików, to dostroimy progi.

Lektor startuje mniej więcej sekundę po pojawieniu się napisu (OCR + synteza w chmurze). W konsoli przy każdej
kwestii widać czas syntezy i opóźnienie.

## Problemy

| Objaw | Co zrobić |
|---|---|
| Błąd 403 albo brak głosu | Microsoft zmienił zabezpieczenia Edge TTS: `.venv\Scripts\pip install -U edge-tts` |
| Obraz z karty czarny | Wyłącz HDCP w PS5, sprawdź, czy konsola jest włączona |
| „Nie mogę otworzyć urządzenia” | Kartę trzyma OBS albo Elgato Studio — zamknij je albo użyj wirtualnej kamery OBS |
| Obraz z karty zepsuty | Ustaw `fourcc = "NV12"` (albo `"YUY2"`, `"MJPG"`) w sekcji `[source]` |
| Brak polskiego OCR | Komenda PowerShell z punktu 3 instalacji |
| Lektor gra nie tam, gdzie trzeba | `lista-urzadzen.bat`, potem `output_device = "Focusrite"` w `[tts]` |
| Dźwięk lektora trzeszczy, ucina się albo znika | W `[tts]` ustaw `gapless = false` |
| Lektor czyta dziwne znaki albo słowa | Zajrzyj do `lektor.log`: linie `OCR …` to surowe odczyty, `Napis:` to tekst przyjęty do kolejki, `do syntezatora:` to dokładnie to, co dostał Edge |
| Skróty globalne nie działają | Używaj liter w oknie konsoli; nie uruchamiaj gry ani lektora jako administrator |

## Pliki

- `start.bat`, `setup.bat`, `test-glosu.bat`, `lista-urzadzen.bat` — uruchamianie i instalacja
- `config.toml` — wszystkie ustawienia (z opisami)
- `roi.json` — zapamiętany obszar napisów (tworzy się sam)
- `kolejka.json` — zapis nieprzeczytanych kwestii, automatycznie przywracany po restarcie
- `lektor.log` — przebieg pracy: każdy odczyt OCR, przyjęte napisy, tekst wysłany do syntezatora, tempo i opóźnienie
- `lektor/` — kod: `sources.py` (obraz), `preprocess.py` (obróbka), `ocr.py`, `textproc.py` (odsiew powtórek),
  `tts.py` (Edge TTS), `speaker.py` (kolejka i tempo), `audio.py`, `app.py` (główna pętla)
- `tests/` — testy (`pip install pytest`, potem `python -m pytest tests`)
