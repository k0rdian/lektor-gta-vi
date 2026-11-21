"""Zapis diagnostyki odczytów: wycinek obrazu z ramkami OCR + opis, co lektor przeczytał,
co pominął jako powtórkę i co odrzucił (tablica, ruch, poza środkiem…).

Włączany w interfejsie („Zapisuj diagnostykę odczytów”). Pliki trafiają do
diagnostyka/RRRR-MM-DD/: obrazy .jpg i odczyty.jsonl. Podgląd: http://127.0.0.1:8765/diagnostyka
Zapis odbywa się w osobnym wątku, żeby nie spowalniać lektora."""
from __future__ import annotations

import html
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

POWODY_WARTE_ZAPISU = {"tablica rejestracyjna", "ruch", "poza środkiem", "obce znaki",
                       "ciemny tekst na jasnym tle"}
ODSTEP_ODRZUCONYCH = 1.0      # s – najwyżej tyle zapisów „odrzucono” na sekundę


class Diagnostyka:
    def __init__(self, katalog: Path, ustawienia, log=print):
        self.katalog = katalog
        self.ust = ustawienia
        self.log = log
        self._kolejka: queue.Queue = queue.Queue(maxsize=50)
        self._t_odrzuconych = 0.0
        self._ost_odrzucone = ""
        self.zapisane = 0
        threading.Thread(target=self._petla, name="diagnostyka", daemon=True).start()

    def wlaczona(self) -> bool:
        return bool(self.ust.get("zapis_diagnostyki", False))

    # --- zdarzenia z potoku ---
    def odczyt(self, klatka, roi, pola, typ: str, tekst: str, do_przeczytania: str, pominiete: list[str]):
        if not self.wlaczona():
            return
        self._dodaj(klatka, roi, pola, {"typ": typ, "tekst": tekst, "lektor": do_przeczytania,
                                        "pominiete": pominiete})

    def odrzucone(self, klatka, roi, pola):
        if not self.wlaczona():
            return
        odrz = [p for p in pola if p.get("odrzucone") in POWODY_WARTE_ZAPISU]
        if not odrz:
            return
        teraz = time.perf_counter()
        klucz = "|".join(sorted(p["tekst"] for p in odrz))
        if teraz - self._t_odrzuconych < ODSTEP_ODRZUCONYCH or klucz == self._ost_odrzucone:
            return
        self._t_odrzuconych, self._ost_odrzucone = teraz, klucz
        self._dodaj(klatka, roi, pola, {"typ": "odrzucono",
                                        "tekst": " · ".join(f"{p['tekst']} ({p['odrzucone']})" for p in odrz)})

    def _dodaj(self, klatka, roi, pola, opis: dict):
        try:
            self._kolejka.put_nowait((klatka, list(roi), [dict(p) for p in pola], opis, time.time()))
        except queue.Full:
            pass

    # --- zapis w tle ---
    def _petla(self):
        while True:
            klatka, roi, pola, opis, t = self._kolejka.get()
            try:
                self._zapisz(klatka, roi, pola, opis, t)
            except Exception as e:
                self.log(f"Diagnostyka: nie udało się zapisać ({e}).")

    def _zapisz(self, klatka: np.ndarray, roi, pola, opis: dict, t: float):
        dzien = time.strftime("%Y-%m-%d", time.localtime(t))
        folder = self.katalog / dzien
        folder.mkdir(parents=True, exist_ok=True)
        H, W = klatka.shape[:2]
        x0, y0, x1, y1 = roi
        # wycinek z marginesem, żeby było widać kontekst (np. auto pod napisem)
        a, b = max(0, int((x0 - 0.03) * W)), min(W, int((x1 + 0.03) * W))
        c, d = max(0, int((y0 - 0.05) * H)), min(H, int((y1 + 0.02) * H))
        obraz = klatka[c:d, a:b].copy()
        cv2.rectangle(obraz, (int(x0 * W) - a, int(y0 * H) - c), (int(x1 * W) - a, int(y1 * H) - c),
                      (47, 122, 255), 1)
        for p in pola:
            kolor = (80, 220, 80) if not p.get("odrzucone") else (60, 60, 255)
            if p.get("odrzucone") in ("niska pewność", "bez liter"):
                kolor = (0, 200, 255)
            pa = (int(p["x0"] * W) - a, int(p["y0"] * H) - c)
            pb = (int(p["x1"] * W) - a, int(p["y1"] * H) - c)
            cv2.rectangle(obraz, pa, pb, kolor, 2)
            if p.get("odrzucone") in POWODY_WARTE_ZAPISU:
                cv2.putText(obraz, _ascii(p["odrzucone"]), (pa[0], max(12, pa[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, kolor, 1, cv2.LINE_AA)
        nazwa = time.strftime("%H%M%S", time.localtime(t)) + f"_{int((t % 1) * 1000):03d}_{opis['typ']}.jpg"
        ok, buf = cv2.imencode(".jpg", obraz, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            (folder / nazwa).write_bytes(buf.tobytes())
        wpis = dict(opis)
        wpis.update({"czas": time.strftime("%H:%M:%S", time.localtime(t)), "obraz": nazwa,
                     "pola": [{"tekst": p["tekst"], "pewnosc": round(p["pewnosc"], 2),
                               "odrzucone": p.get("odrzucone")} for p in pola]})
        with open(folder / "odczyty.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(wpis, ensure_ascii=False) + "\n")
        self.zapisane += 1

    # --- strona z podglądem ---
    def strona(self, dzien: str | None = None, limit: int = 300) -> str:
        dni = sorted((p.name for p in self.katalog.glob("????-??-??") if p.is_dir()), reverse=True) \
            if self.katalog.exists() else []
        dzien = dzien if dzien in dni else (dni[0] if dni else None)
        wpisy = []
        if dzien:
            plik = self.katalog / dzien / "odczyty.jsonl"
            if plik.exists():
                for linia in plik.read_text(encoding="utf-8").splitlines()[-limit:]:
                    try:
                        wpisy.append(json.loads(linia))
                    except ValueError:
                        pass
        wpisy.reverse()
        e = html.escape
        kolory = {"przeczytano": "#3ecf8e", "powtorka": "#f5b84a", "odrzucono": "#ff6161"}
        nazwy = {"przeczytano": "przeczytano", "powtorka": "pominięto powtórkę", "odrzucono": "odrzucono"}
        wiersze = []
        for w in wpisy:
            szczegoly = ""
            if w.get("lektor"):
                szczegoly += f"<div><span class=s>Lektor:</span> {e(w['lektor'])}</div>"
            if w.get("pominiete"):
                szczegoly += f"<div><span class=s>Pominięte (już było):</span> {e(' · '.join(w['pominiete']))}</div>"
            pola = " · ".join(f"{e(p['tekst'])} [{p['pewnosc']}]" + (f" <i>{e(p['odrzucone'])}</i>" if p.get('odrzucone') else "")
                              for p in w.get("pola", []))
            wiersze.append(
                f"<tr><td class=s>{e(w.get('czas', ''))}</td>"
                f"<td><b style='color:{kolory.get(w.get('typ'), '#ccc')}'>{e(nazwy.get(w.get('typ'), w.get('typ', '')))}</b></td>"
                f"<td><div>{e(w.get('tekst', ''))}</div>{szczegoly}<div class='s m'>OCR: {pola}</div>"
                f"<img loading=lazy src='/diagnostyka/{e(dzien)}/{e(w.get('obraz', ''))}'></td></tr>")
        wybor = " ".join(f"<a href='/diagnostyka?dzien={d}'>{d}</a>" for d in dni[:14])
        return f"""<!DOCTYPE html><html lang=pl><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Diagnostyka odczytów</title>
<style>body{{margin:0;padding:16px;background:#0e1014;color:#e8eaef;font:14px/1.45 system-ui,Segoe UI,sans-serif}}
table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #262b36;padding:8px;vertical-align:top}}
img{{max-width:100%;margin-top:6px;border-radius:6px}}.s{{color:#9aa3b2}}.m{{font-size:12px;margin-top:4px}}
a{{color:#ff7a2f}} i{{color:#ff6161;font-style:normal}}</style></head><body>
<h2 style="margin:0 0 6px">Diagnostyka odczytów {e(dzien or '')}</h2>
<p class=s>Najnowsze na górze (ostatnie {limit}). Zielone ramki – tekst przyjęty, czerwone – odrzucony (powód nad ramką),
żółte – za niska pewność. Folder: diagnostyka\\{e(dzien or '')}. Dni: {wybor or '—'}</p>
<table>{''.join(wiersze) or '<tr><td class=s>Brak zapisów. Włącz „Zapisuj diagnostykę odczytów” w sekcji OCR.</td></tr>'}</table>
</body></html>"""

    def plik(self, dzien: str, nazwa: str) -> Path | None:
        """Bezpieczna ścieżka do obrazu diagnostyki (bez wychodzenia poza folder)."""
        if not (len(dzien) == 10 and dzien.replace("-", "").isdigit()):
            return None
        if "/" in nazwa or "\\" in nazwa or ".." in nazwa or not nazwa.endswith(".jpg"):
            return None
        p = self.katalog / dzien / nazwa
        return p if p.exists() else None


def _ascii(t: str) -> str:
    return t.translate(str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ"))
