# Lektor GTA VI

Czyta na głos polskie napisy z GTA VI – jak lektor w telewizji. Aplikacja patrzy na obraz z gry (karta przechwytująca, np. Elgato HD60 X przy PS5, albo ekran PC), rozpoznaje napisy i czyta je polskim głosem. Nic nie dotyka konsoli ani gry.

```
obraz z gry → wycinek z napisami → OCR (PaddleOCR PP-OCRv6) → odsiew powtórek i śmieci → głos (Piper albo Chatterbox) → głośniki
```

- **Dwa rodzaje głosu:**
  - **Piper** – szybki, działa na każdym komputerze.
  - **Chatterbox** – naturalny głos, który można sklonować z dowolnego nagrania. Wymaga karty NVIDIA. Pierwszy dźwięk po ok. 0,1 s.
- **Każda linia raz.** Lektor pamięta, co już przeczytał, także gdy w grze zmienia się tylko druga linia napisu.
- **Kolejka dialogów.** Przy szybkiej wymianie zdań czyta wszystko po kolei, a gdy zostaje w tyle, mówi szybciej i skraca pauzy.
- **Odsiew śmieci.** Odrzuca tablice rejestracyjne, szyldy, tekst w ruchu i tekst poza środkiem ekranu.
- **Słownik wymowy** (`slownik_wymowy.txt`), np. `Jason = Dżejson`, `Vice City = Wajs Siti`.
- **Lokalnie i za darmo.** Internet jest potrzebny tylko do instalacji i pobrania modeli.

## Wymagania

- Windows 10 albo 11 (64-bit).
- Źródło obrazu:
  - karta przechwytująca, np. Elgato HD60 X przy PS5,
  - gra na ekranie PC (także PS Remote Play),
  - plik wideo albo zrzut ekranu (do testów).
- Głos **Chatterbox**:
  - karta NVIDIA (sprawdzone na RTX 5080 16 GB; Chatterbox zajmuje ok. 8 GB jej pamięci),
  - ok. 10 GB miejsca na dysku.
- Głos **Piper**: wystarczy procesor.

## Instalacja

1. Pobierz aplikację: zielony przycisk **Code → Download ZIP**, a potem rozpakuj. Możesz też użyć `git clone https://github.com/k0rdian/lektor-gta-vi.git`. Najlepiej do folderu bez polskich znaków w ścieżce, np. `C:\Lektor`.
2. Kliknij dwukrotnie **`instaluj.bat`**. Instalator:
   - zainstaluje Pythona 3.12, jeśli go nie masz (przez `winget`),
   - utworzy środowisko `.venv` i pobierze pakiety (ok. 150 MB) oraz domyślny głos Pipera,
   - na komputerze z kartą NVIDIA zapyta, czy od razu doinstalować głos **Chatterbox**. To ok. 10 GB i 15–40 minut pobierania. Możesz to zrobić później plikiem **`instaluj_chatterbox.bat`**.
3. Uruchom **`uruchom.bat`**. Otworzy się przeglądarka z panelem aplikacji (`http://127.0.0.1:8765`). Okno konsoli musi zostać otwarte, bo jego zamknięcie wyłącza lektora.

## Pierwszy test bez gry (2 minuty)

1. W polu **Źródło obrazu** wybierz „Obraz (zrzut ekranu)” i plik `klatka1.png`.
2. Kliknij **▶ Start**. Lektor przeczyta napis z obrazka, a w **Historii** pojawią się czasy.
3. Przeciągnij na podgląd własny zrzut ekranu z GTA. Aplikacja od razu go odczyta.
4. Zaznacz myszką na podglądzie obszar z napisami albo kliknij **Obszar GTA VI**.

## Gra na PS5 przez kartę przechwytującą

1. Zamknij OBS i Elgato Studio, bo kartę może naraz używać tylko jeden program.
   Jeśli chcesz zostawić OBS, włącz w nim „Wirtualną kamerę” i wybierz ją jako kartę przechwytującą.
2. Wybierz źródło „Karta przechwytująca”, kliknij **Wykryj** i wybierz kartę.
3. Na PS5 wyłącz HDCP: *Ustawienia → System → HDMI → Włącz HDCP*. Inaczej obraz będzie czarny.
4. Kliknij **Start** i zaznacz obszar napisów.

**Ustawienia napisów w GTA VI**, które najbardziej pomagają OCR:
- Napisy: włączone.
- Rozmiar: duży.
- Tło: pełne.
- Imiona mówiących: wyłączone.

## Głosy

### Piper (na procesorze)

Pięć polskich głosów. Domyślny pobiera instalator, pozostałe pobierają się same przy pierwszym wyborze. Własny model Pipera (`.onnx` + `.onnx.json`) wrzuć do `modele\piper\`.

### Chatterbox (na karcie NVIDIA)

Chatterbox Multilingual V3 firmy Resemble AI mówi po polsku dowolnym głosem na podstawie ok. 10 s nagrania i nie wymaga trenowania. W folderze `glosy\` są gotowe głosy:

- **Janusz**
- **Lektor PL**
- **Przykład – klon głosu Piper „darkman”**

Wybierasz je w sekcji „Głos lektora”.

**Własny głos.** W sekcji „Własny głos (Chatterbox)” wybierz plik, wpisz nazwę i kliknij **Utwórz głos**. Przyjmuje mp3, wav, m4a, ogg, flac, a nawet mp4/mkv (dźwięk z filmu).
- Daj 15–60 s czystej mowy jednej osoby, bez muzyki i drugiego głosu w tle.
- Aplikacja sama wybierze najlepsze 10 s.
- Klonuj tylko głos, do którego masz prawo: własny albo za zgodą tej osoby.
- Chatterbox dodaje do dźwięku niesłyszalny znak wodny (Perth).

**Szybkość czy jakość** (sekcja „Głos lektora”):

| Tryb | Pierwszy dźwięk (RTX 5080) | Opis |
|---|---|---|
| Najszybszy | ok. 60 ms | najmniejsze kawałki mowy |
| Zrównoważony (domyślny) | ok. 75 ms | zalecany |
| Najlepsza jakość | ok. 230 ms | oryginalny dekoder Chatterboxa |

Głos rusza po pierwszych ~0,5 s gotowej mowy, a reszta zdania liczy się w tym czasie. *Tempo* przyspiesza mowę bez zmiany wysokości głosu.

## Strojenie

| Problem | Co zmienić |
|---|---|
| Lektor czyta urywki napisów | Stabilizacja: 3 (wymaga 3 zgodnych odczytów; domyślnie 2) |
| Lektor zaczyna za późno | Stabilizacja: 1, gdy napisy pojawiają się od razu w całości. Do tego rozdzielczość wykrywania 640 i model OCR „PP-OCRv6 tiny”. |
| Lektor powtarza linię, która została na ekranie | Włącz „Nie czytaj drugi raz linii…” (sekcja „Co czytać, a co pomijać”) |
| Czyta tablice rejestracyjne, szyldy | „Odrzucaj tablice…” i „Czytaj tylko tekst na środku ekranu” (domyślnie włączone). Odrzucony tekst ma w podglądzie czerwoną ramkę z powodem. |
| Czyta elementy HUD albo tło | Zawęź obszar napisów (przycisk **Obszar GTA VI**) albo podnieś minimalną pewność |
| Nie zauważa zmiany napisu | Zwiększ czułość. Najpóźniej po 0,6 s napis i tak złapie odczyt kontrolny. |
| Przy szybkich dialogach lektor nie nadąża | Polityka „Kolejka”, „Przyspieszaj, gdy lektor zostaje w tyle” i „Maksymalne tempo przy nadrabianiu” |
| Źle wymawia nazwę | Dopisz ją do `slownik_wymowy.txt`. Działa od razu. |

**Kolumny w Historii:**
- **OCR** – czas odczytów potrzebnych do stabilnego wyniku.
- **Głos** – czas do pierwszego kawałka dźwięku.
- **Razem** – od klatki, na której zmienił się napis, do pierwszego dźwięku lektora.

**Diagnostyka.** Zaznacz „Zapisuj diagnostykę odczytów”, a zobaczysz przy każdej kwestii wycinek obrazu: co lektor przeczytał, co pominął i co odrzucił. Przycisk **⏺ Nagraj** zapisuje scenę do pliku MP4 w folderze `nagrania`, żeby sprawdzić ją potem z innymi ustawieniami.

## Testy

**`testuj.bat`** sprawdza aplikację bez gry, w czterech częściach:
1. logika: pamięć przeczytanych linii, kolejka, filtry,
2. scenariusze z prawdziwym OCR na wygenerowanych filmach,
3. zgodność szybkiego dekodera Chatterboxa z oryginałem,
4. czas do pierwszego dźwięku w trzech trybach. Nagrania do odsłuchu trafiają do `probki\`.

## Problemy

- **„Failed to fetch”, zawieszony podgląd.** Niektóre programy (np. AdGuard, antywirusy) przechwytują połączenia z `127.0.0.1`. Dodaj w nich wyjątek dla `python.exe` albo dla adresu `127.0.0.1`.
- **Czarny obraz z karty.** Na PS5 włączone jest HDCP albo kartę zajmuje inny program (OBS, Elgato Studio).
- **Chatterbox się nie pojawia na liście głosów.** Uruchom `instaluj_chatterbox.bat`.
  - Pierwsze ładowanie po starcie trwa ok. 20 s. W tym czasie lektor mówi Piperem.
  - Kilka sekund trwa też zmiana trybu „Szybkość czy jakość”.

## Z czego korzysta

- OCR: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (modele PP-OCRv6), [RapidOCR](https://github.com/RapidAI/RapidOCR), [MNN](https://github.com/alibaba/MNN).
- Głos:
  - [Piper](https://github.com/OHF-Voice/piper1-gpl) (polskie głosy z [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)),
  - [Chatterbox](https://github.com/resemble-ai/chatterbox) firmy Resemble AI, z dekoderem z Chatterbox-Turbo.
- Szybka ścieżka Chatterboxa (grafy CUDA, strumieniowanie) jest napisana dla tej aplikacji (`lektor/chatterbox_*.py`).

Każdy z tych projektów ma własną licencję (w jego repozytorium). Lektor tylko czyta obraz z gry i nie jest związany z Rockstar Games.

Poprzednia wersja (Edge TTS + OCR Windows) nie jest już rozwijana. Zostaje w historii repozytorium.
