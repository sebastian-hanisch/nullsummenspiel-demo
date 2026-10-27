"""PDF-Export: baut für jedes Spiel ein gültiges PDF, ohne Zeichen, die die Kernschriften nicht kennen."""

import pytest

import nsm_constants as C
import nsm_evaluation as ev
import nsm_pdf_export as X
from nsm_scenario import Settings


@pytest.mark.parametrize("game", list(C.GAMES))
def test_pdf_is_built_for_every_game(game):
    pdf = X.build_pdf(ev.analyse(Settings(game=game)))
    assert pdf[:5] == b"%PDF-" and len(pdf) > 1500


def test_text_replaces_characters_the_core_fonts_do_not_know():
    assert X._text("Wert – η² ≥ 2·x → „y“ … Σ") == 'Wert - eta^2 >= 2*x -> "y" ... Summe'
    assert X._text("Ünicode ß ä") == "Ünicode ß ä"                         # Umlaute gehen durch (Latin-1)
    X._text("𝔸").encode("latin-1")                                          # nichts Unbekanntes bleibt übrig
