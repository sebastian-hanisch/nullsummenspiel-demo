"""App: jedes Preset, der Permalink, der ▶️-Knopf des Shapley-Laufs, die Reglermaxima und die Verdikt-Zeile je Preset (DEMO-PLAYBOOK §1d und §11a Punkt 7)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nsm_constants as C
import nsm_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
CHARTS = 10                          # Abbildungen der Grundansicht: 2 Strategien, 2 Regeln, E1, Beobachter (Verlauf, E2), Lernen (Verlauf, E3), E4


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning] + [c.value for c in at.caption])


def _expander(at, label):
    return next(e for e in at.expander if e.label == label)               # nie per Index (DEMO-PLAYBOOK §2)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_sets_its_values(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    key_of = {state_key: key for key, state_key in P.PRESET_KEYS.items()}
    for state_key, spec in P.SETTING_SPECS.items():
        assert at.session_state[state_key] == C.PRESETS[name].get(key_of[state_key], spec.default), (name, state_key)
    assert len(at.get("plotly_chart")) == CHARTS and len(at.tabs) == 3


def test_the_expanders_and_the_footer_are_found_by_title():
    at = _app()
    for label in ("📐 Mathematische Formulierung", "⚠️ Annahmen, Grenzen und bekannte Probleme", "So funktioniert ein Nullsummenspiel mit gemischten Strategien"):
        assert _expander(at, label)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


def test_the_assumptions_expander_shows_the_constant_text():
    at = _app()
    assert _expander(at, "⚠️ Annahmen, Grenzen und bekannte Probleme").markdown[0].value.strip() == C.ASSUMPTIONS_MARKDOWN.strip()


def test_permalink_is_loaded_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["game"] = "random"
    at.query_params["m"] = "99"
    at.query_params["routes"] = "1;0.5;0.25"
    at.query_params["rounds"] = "1234"
    at.query_params["learn"] = "adam"                              # ungültige Auswahl: wird ignoriert
    at.query_params["tie"] = "random"
    at.run()
    assert not at.exception
    assert at.session_state["game_select"] == "random" and at.session_state["m_slider"] == C.M_MAX and at.session_state["routes_input"] == "1; 0,5; 0,25"
    assert at.session_state["rounds_slider"] == 1200 and at.session_state["learn_select"] == C.DEFAULT_LEARN and at.session_state["tie_radio"] == "random"


def test_default_metrics_and_verdict():
    at = _app()
    assert [m.label for m in at.metric] == ["Wert des Spiels", "Sicherheitsniveau", "Zertifikat: Lücke", "Geschlossene Form gegen LP"]
    assert at.metric[0].value == "0,750" and at.metric[1].value == "0,000" and (at.metric[2].value == "0" or at.metric[2].value.endswith(("⁻¹⁶", "⁻¹⁵", "⁻¹⁴"))) and "garantiert 0,7500, zugelassen 0,7500" in at.metric[2].delta
    assert "Mischen bringt +0,750" in _texts(at)


def test_the_game_selection_drives_the_main_metrics():
    at = _app()
    at.sidebar.selectbox(key="game_select").set_value("saddle").run()
    assert not at.exception and at.metric[0].value == "3,000" and [m.label for m in at.metric][3] == "Sattelpunkte" and at.metric[3].value == "1"
    assert "Zufall bringt hier nichts" in _texts(at)
    at.sidebar.selectbox(key="game_select").set_value("rps").run()
    assert at.metric[0].value == "0,000" and at.metric[1].value == "-1,000"
    at.sidebar.selectbox(key="game_select").set_value("random").run()
    assert not at.exception and at.metric[0].value != "0,000"


def test_invalid_routes_warn_and_fall_back_to_the_default():
    at = _app()
    at.sidebar.text_input(key="routes_input").set_value("1; 7").run()
    assert not at.exception and any("keine gültige Liste" in w.value for w in at.warning) and at.metric[0].value == "0,750"
    at.sidebar.text_input(key="routes_input").set_value("1; 0,1").run()
    assert not any("keine gültige Liste" in w.value for w in at.warning) and at.metric[0].value == "0,091"


def test_main_verdict_per_preset():
    expected = {"Vier gleiche Routen (Voreinstellung)": "Mischen bringt +0,750", "Münzwurf (zwei gleiche Routen)": "Mischen bringt +0,500", "Ungleiche Werte (10 : 1)": "Mischen bringt +0,091",
                "Vier Routen mit Streichen": "Mischen bringt +0,462", "Stein-Schere-Papier": "Mischen bringt +1,000", "Sattelpunkt": "Zufall bringt hier nichts", "Zufallsspiel 5 × 5": "Mischen bringt +"}
    for name, snippet in expected.items():
        at = _app()
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert snippet in _texts(at), name


def test_observer_verdict_per_preset():
    at = _app()
    assert "Das ist im Band" in _texts(at)                                                     # Voreinstellung: Rotation gegen Häufigkeitszähler, kleinster Index: Lieferung gleich dem Wert
    at.sidebar.button(key="preset_Rotation gegen Folgezähler").click().run()
    assert "kostet" in _texts(at) and "Rotation gegen Folgezähler" in _texts(at)
    at.sidebar.button(key="preset_Rotation gegen Häufigkeitszähler").click().run()
    assert "nutzt das Muster nicht aus" in _texts(at)


def test_every_combination_of_pattern_counter_and_tie_runs():
    at = _app()
    for pattern in C.PATTERNS:
        for counter in C.COUNTERS:
            at.sidebar.selectbox(key="pattern_select").set_value(pattern)
            at.sidebar.selectbox(key="counter_select").set_value(counter)
            at.sidebar.radio(key="tie_radio").set_value("random").run()
            assert not at.exception, (pattern, counter, [str(e) for e in at.exception])


def test_every_learning_method_and_start_runs_on_every_game():
    at = _app()
    for game in C.GAMES:
        at.sidebar.selectbox(key="game_select").set_value(game)
        for method in C.LEARN_METHODS:
            for start in C.STARTS:
                at.sidebar.selectbox(key="learn_select").set_value(method)
                at.sidebar.radio(key="start_radio").set_value(start)
                at.sidebar.slider(key="rounds_slider").set_value(300).run()
                assert not at.exception, (game, method, start, [str(e) for e in at.exception])


def test_the_start_in_the_equilibrium_is_reported_as_standing_still():
    at = _app()
    at.sidebar.button(key="preset_Stein-Schere-Papier").click().run()
    at.sidebar.radio(key="start_radio").set_value("uniform").run()
    assert "Der Start liegt schon im Gleichgewicht" in _texts(at)
    at.sidebar.radio(key="start_radio").set_value("random").run()
    assert "Der Start liegt schon im Gleichgewicht" not in _texts(at)


def test_slider_maxima_of_the_whole_app():
    at = _app()
    at.sidebar.selectbox(key="game_select").set_value("random")
    at.sidebar.slider(key="m_slider").set_value(C.M_MAX)
    at.sidebar.slider(key="n_slider").set_value(C.N_MAX)
    at.sidebar.slider(key="rounds_slider").set_value(C.ROUNDS_MAX)
    at.sidebar.slider(key="obs_rounds_slider").set_value(C.OBS_ROUNDS_MAX)
    at.sidebar.slider(key="bm_n_slider").set_value(C.BM_N_MAX)
    at.sidebar.text_input(key="routes_input").set_value("; ".join(["0,5"] * 8)).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("plotly_chart")) == CHARTS


def test_the_bimatrix_tab_runs_every_game_kind():
    at = _app()
    for kind in C.BM_KINDS:
        at.sidebar.selectbox(key="bm_select").set_value(kind).run()
        assert not at.exception, (kind, [str(e) for e in at.exception])
    assert "Shapley" in _texts(at)


def test_the_shapley_button_runs_and_shows_the_cycle():
    at = _app()
    assert "Noch kein Lauf" in _texts(at)
    at.button(key="shapley_start").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    assert "Shapley-Beispiel, Fictitious Play, 20000 Runden" in text and "Rechenzeit dieses Laufs" in text and len(at.get("plotly_chart")) == CHARTS + 1


def test_the_pdf_button_makes_the_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1
