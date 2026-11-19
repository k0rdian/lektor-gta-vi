from lektor import textproc
from lektor.textproc import RepeatFilter, Stabilizer


def test_clean_lines_joins_and_drops_small_noise():
    lines = [("Jason, nie panikuj.", 40.0), ("Wszystko jest pod kontrolą.", 41.0), ("ii", 9.0)]
    assert textproc.clean_lines(lines) == "Jason, nie panikuj. Wszystko jest pod kontrolą."


def test_clean_lines_hyphenation_and_junk():
    lines = [("| Musimy to prze-", 40.0), ("czytać jeszcze raz _", 40.0)]
    assert textproc.clean_lines(lines) == "Musimy to przeczytać jeszcze raz"


def test_clean_lines_empty():
    assert textproc.clean_lines([]) == ""
    assert textproc.clean_lines([("...", 30.0), ("| |", 30.0)]) == ""


def test_looks_like_text():
    assert textproc.looks_like_text("Musimy się stąd wynosić, zanim przyjedzie policja.")
    assert textproc.looks_like_text("Co?")
    assert textproc.looks_like_text("W porządku, a ty?")
    assert not textproc.looks_like_text("ll ff kk")
    assert not textproc.looks_like_text("#@! 12 %%")
    assert not textproc.looks_like_text("a")
    assert not textproc.looks_like_text("Co?", min_letters=3)
    assert not textproc.looks_like_text("")


def test_prepare_for_speech_replacements_with_declension():
    repl = {"Vice City": "Wajs Siti", "Jason": "Dżejson"}
    assert (
        textproc.prepare_for_speech("Witaj w Vice City, Jasonie. Daj Jasonowi spokój.", repl)
        == "Witaj w Wajs Siti, Dżejsonie. Daj Dżejsonowi spokój."
    )
    # nie zamienia w środku innego słowa
    assert textproc.prepare_for_speech("MiniJason", repl) == "MiniJason"


def test_prepare_for_speech_strips_speaker():
    assert textproc.prepare_for_speech("Lucia: Gdzie jesteś?", strip_speaker=True) == "Gdzie jesteś?"
    assert textproc.prepare_for_speech("Lucia: Gdzie jesteś?", strip_speaker=False) == "Lucia: Gdzie jesteś?"


def test_stabilizer_needs_two_reads_and_emits_once():
    s = Stabilizer(required_reads=2, similarity=90)
    assert s.feed("Musimy się stąd wynosić.") is None
    assert s.feed("Musimy się stąd wynosić.") == "Musimy się stąd wynosić."
    assert s.feed("Musimy się stad wynosić.") is None  # drobna różnica OCR — ta sama kwestia
    assert s.feed("Musimy się stąd wynosić.") is None
    assert s.feed("Teraz w lewo!") is None
    assert s.feed("Teraz w lewo!") == "Teraz w lewo!"


def test_stabilizer_resets_on_blank():
    s = Stabilizer(2, 90)
    s.feed("Tak.")
    assert s.feed("") is None
    assert s.feed("Tak.") is None
    assert s.feed("Tak.") == "Tak."


def test_repeat_filter_skips_duplicates_in_window():
    r = RepeatFilter(window_s=12, similarity=88)
    assert r.process("Witamy w Vice City.", 0.0) == "Witamy w Vice City."
    assert r.process("Witamy w Vice City.", 3.0) is None
    assert r.process("Witamy w Vlce City.", 4.0) is None  # błąd OCR
    assert r.process("Witamy w Vice City.", 20.0) == "Witamy w Vice City."  # po oknie wolno


def test_repeat_filter_fragment_and_extension():
    r = RepeatFilter(window_s=12, similarity=88)
    assert r.process("Jason, nie panikuj.", 0.0) == "Jason, nie panikuj."
    # OCR złapał teraz obie linie — czytamy tylko dalszy ciąg
    assert r.process("Jason, nie panikuj. Wszystko jest pod kontrolą.", 0.5) == "Wszystko jest pod kontrolą."
    # a potem tylko fragment — pomijamy
    assert r.process("Wszystko jest pod kontrolą.", 1.0) is None
    assert r.process("Wszystko jest pod", 1.2) is None


def test_repeat_filter_extension_with_garbage_tail_is_silent():
    r = RepeatFilter(window_s=12, similarity=88)
    r.process("Musimy się stąd wynosić.", 0.0)
    assert r.process("Musimy się stąd wynosić. ll kk", 0.5) is None


def test_repeat_filter_different_lines_pass():
    r = RepeatFilter(window_s=12, similarity=88)
    assert r.process("Gdzie jest samochód?", 0.0)
    assert r.process("Za rogiem, jak zawsze.", 1.0)
    assert r.process("To jedziemy.", 2.0)


def test_repeat_filter_long_visible_subtitle_is_not_reread():
    r = RepeatFilter(window_s=5, similarity=88)
    assert r.process("Ta kwestia wisi długo na ekranie.", 0.0)
    # migotanie OCR co 3 s odświeża czas — po 12 s nadal pomijamy
    for t in (3.0, 6.0, 9.0, 12.0):
        assert r.process("Ta kwestia wisi długo na ekranie.", t) is None


def test_repeat_filter_short_line_inside_longer_one_is_spoken():
    r = RepeatFilter(window_s=12, similarity=88)
    assert r.process("Nie wiem, gdzie on teraz jest.", 0.0)
    assert r.process("Nie wiem.", 2.0) == "Nie wiem."


def test_stabilizer_remembers_first_seen():
    s = Stabilizer(2, 90)
    s.feed("Gdzie jesteś?", 10.0)
    assert s.feed("Gdzie jesteś?", 10.2) == "Gdzie jesteś?"
    assert s.first_seen == 10.0
