"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nsm_constants as C
import nsm_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"


def _candidates(spec, current):
    """Andere gültige Werte als der aktuelle: bei Auswahlreglern alle übrigen Auswahlen, bei Schiebereglern die beiden Enden des Bereichs."""
    if spec.choices is not None:
        return [o for o in spec.choices if o != current]
    if spec.cleaner is not None:
        return ["1; 0,5"]
    return [v for v in (spec.hi, spec.lo) if v != current]


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    """Die Änderung eines Reglers hebt das aktive Preset auf. Ergibt die Änderung zufällig genau die Einstellungen eines anderen Presets (Spiel „Stein-Schere-Papier“ statt Verstecken und Suchen),
    wird dieses markiert und nicht das alte: auch das ist richtig. Es muss mindestens einen Wert geben, der keines trifft, und nie darf das alte Preset markiert bleiben."""
    first = next(iter(C.PRESETS))
    seen_unmarked = False
    for value in _candidates(P.SETTING_SPECS[state_key], P.SETTING_SPECS[state_key].default):
        at = _app()
        _regler(at, state_key).set_value(value).run()
        assert not at.exception, [str(e) for e in at.exception]
        marked = [n for n, typ in _typen(at).items() if typ == "primary"]
        assert first not in marked, (state_key, value)
        if not marked:
            seen_unmarked = True
            # erneutes Anklicken markiert wieder
            at.sidebar.button(key=f"preset_{first}").click().run()
            assert _typen(at)[first] == "primary"
            break
    assert seen_unmarked or state_key in ("game_select",), f"jeder Wert von {state_key} trifft ein anderes Preset"
