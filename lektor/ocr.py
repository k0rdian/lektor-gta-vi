"""OCR napisów: modele PaddleOCR (PP-OCR) uruchamiane przez RapidOCR na silniku MNN."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

KATALOG_OCR = Path(__file__).resolve().parents[1] / "modele" / "ocr"

MODELE_OCR = {
    "v6_small_mnn": "PP-OCRv6 small · MNN (dołączony, domyślny)",
    "v6_tiny_mnn": "PP-OCRv6 tiny · MNN (szybszy, pobierany)",
    "v6_medium_mnn": "PP-OCRv6 medium · MNN (dokładniejszy, pobierany)",
    "v5_latin_mnn": "PP-OCRv5 latin · MNN (pobierany)",
    "v6_small_onnx": "PP-OCRv6 small · ONNX Runtime (dla porównania)",
}


def _parametry(klucz: str, rozdzielczosc_det: int) -> dict:
    from rapidocr import EngineType, ModelType, OCRVersion

    p = {
        "Global.log_level": "error",
        "Global.model_root_dir": str(KATALOG_OCR),
        "Global.text_score": 0.3,           # odsiewamy sami, żeby widzieć słabe odczyty w podglądzie
        "Det.limit_type": "max",            # wycinek napisów zmniejszamy, a nie powiększamy
        "Det.limit_side_len": int(rozdzielczosc_det),
        "Det.unclip_ratio": 2.2,            # skleja słowa w całe linie
        "Det.box_thresh": 0.5,
    }
    mnn = EngineType.MNN
    if klucz == "v6_small_mnn":
        p.update({
            "Det.engine_type": mnn, "Rec.engine_type": mnn, "Cls.engine_type": mnn,
            "Det.model_path": str(KATALOG_OCR / "PP-OCRv6_det_small.mnn"),
            "Rec.model_path": str(KATALOG_OCR / "PP-OCRv6_rec_small.mnn"),
            "Rec.rec_keys_path": str(KATALOG_OCR / "ppocrv6_dict.txt"),
        })
    elif klucz in ("v6_tiny_mnn", "v6_medium_mnn"):
        typ = ModelType.TINY if klucz == "v6_tiny_mnn" else ModelType.MEDIUM
        p.update({
            "Det.engine_type": mnn, "Rec.engine_type": mnn, "Cls.engine_type": mnn,
            "Det.ocr_version": OCRVersion.PPOCRV6, "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": typ, "Rec.model_type": typ,
            "Det.lang_type": "pl", "Rec.lang_type": "pl",
        })
    elif klucz == "v5_latin_mnn":
        from rapidocr import LangDet, LangRec
        p.update({
            "Det.engine_type": mnn, "Rec.engine_type": mnn, "Cls.engine_type": mnn,
            "Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE,
            "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.LATIN,
        })
    elif klucz == "v6_small_onnx":
        pass  # modele ONNX są w paczce rapidocr
    else:
        raise ValueError(f"Nieznany model OCR: {klucz}")
    return p


class Ocr:
    def __init__(self, log=print):
        self.log = log
        self.silnik = None
        self.klucz = None
        self.rozdzielczosc = None
        self._lock = threading.Lock()

    def zaladuj(self, klucz: str, rozdzielczosc_det: int = 960) -> None:
        from rapidocr import RapidOCR

        KATALOG_OCR.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        silnik = RapidOCR(params=_parametry(klucz, rozdzielczosc_det))
        # rozgrzewka na pustym obrazie
        silnik(np.zeros((120, 600, 3), dtype=np.uint8), use_cls=False)
        with self._lock:
            self.silnik, self.klucz, self.rozdzielczosc = silnik, klucz, rozdzielczosc_det
        self.log(f"OCR gotowy: {MODELE_OCR.get(klucz, klucz)} ({(time.perf_counter() - t0) * 1000:.0f} ms).")

    def przeladuj(self) -> None:
        """Tworzy silnik od nowa z tymi samymi ustawieniami (po serii błędów)."""
        if self.klucz is not None:
            self.zaladuj(self.klucz, self.rozdzielczosc or 960)

    def rozpoznaj(self, obraz_bgr: np.ndarray, min_pewnosc: float = 0.6) -> tuple[list[str], float, list[dict]]:
        """Zwraca (linie od góry do dołu, czas w ms, surowe pola z wynikami).

        Silnik nie jest bezpieczny wątkowo: przycisk „Odczytaj teraz” (wątek serwera) i potok
        nie mogą liczyć jednocześnie, dlatego całe rozpoznawanie jest pod blokadą."""
        with self._lock:
            silnik = self.silnik
            if silnik is None:
                raise RuntimeError("OCR nie jest załadowany")
            t0 = time.perf_counter()
            wynik = silnik(np.ascontiguousarray(obraz_bgr), use_cls=False)
            ms = (time.perf_counter() - t0) * 1000
        pola = []
        if wynik.boxes is not None and wynik.txts:
            for box, txt, score in zip(wynik.boxes, wynik.txts, wynik.scores):
                b = np.asarray(box, dtype=np.float32)
                pola.append({
                    "tekst": txt.strip(), "pewnosc": float(score),
                    "x": float(b[:, 0].min()), "y": float(b[:, 1].min()),
                    "w": float(np.ptp(b[:, 0])), "h": float(np.ptp(b[:, 1])),
                })
        dobre = [p for p in pola if p["pewnosc"] >= min_pewnosc and any(c.isalpha() for c in p["tekst"])]
        return _w_linie(dobre), ms, pola


def _w_linie(pola: list[dict]) -> list[str]:
    """Grupuje wykryte pola w linie (po wysokości), a w linii sortuje od lewej."""
    if not pola:
        return []
    pola = sorted(pola, key=lambda p: p["y"] + p["h"] / 2)
    linie: list[list[dict]] = []
    for p in pola:
        srodek = p["y"] + p["h"] / 2
        if linie:
            ost = linie[-1]
            sr = np.mean([q["y"] + q["h"] / 2 for q in ost])
            wys = np.mean([q["h"] for q in ost])
            if abs(srodek - sr) < 0.5 * max(wys, p["h"]):
                ost.append(p)
                continue
        linie.append([p])
    return [" ".join(q["tekst"] for q in sorted(l, key=lambda q: q["x"])) for l in linie]
