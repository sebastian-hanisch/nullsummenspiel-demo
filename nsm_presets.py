"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in der Seitenleiste und in SETTING_SPECS: ihre Änderung hebt die Markierung des aktiven Presets auf (tests/test_layout.py prüft das für jeden Regler)."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

import nsm_constants as C
from nsm_scenario import Settings, format_routes, parse_routes


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple | None = None
    cleaner: Callable | None = None       # eigene Bereinigung eines Wertes aus der Adresszeile (Text); None bei Unsinn


def _clean_routes(raw):
    values = parse_routes(raw)
    return format_routes(values) if values is not None else None


SETTING_SPECS = {
    "game_select": SettingSpec("game", str, C.DEFAULT_GAME, choices=tuple(C.GAMES)),
    "routes_input": SettingSpec("routes", str, C.DEFAULT_ROUTES, cleaner=_clean_routes),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX, 1),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, 1),
    "game_seed_input": SettingSpec("gseed", int, C.DEFAULT_GAME_SEED, 0, C.SEED_MAX, 1),
    "learn_select": SettingSpec("learn", str, C.DEFAULT_LEARN, choices=tuple(C.LEARN_METHODS)),
    "rounds_slider": SettingSpec("rounds", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX, 100),
    "start_radio": SettingSpec("start", str, C.DEFAULT_START, choices=tuple(C.STARTS)),
    "pattern_select": SettingSpec("pattern", str, C.DEFAULT_PATTERN, choices=tuple(C.PATTERNS)),
    "counter_select": SettingSpec("counter", str, C.DEFAULT_COUNTER, choices=tuple(C.COUNTERS)),
    "tie_radio": SettingSpec("tie", str, C.DEFAULT_TIE, choices=tuple(C.TIES)),
    "obs_rounds_slider": SettingSpec("orounds", int, C.DEFAULT_OBS_ROUNDS, C.OBS_ROUNDS_MIN, C.OBS_ROUNDS_MAX, 100),
    "bm_select": SettingSpec("bm", str, C.DEFAULT_BM, choices=tuple(C.BM_KINDS)),
    "bm_n_slider": SettingSpec("bmn", int, C.DEFAULT_BM_N, C.BM_N_MIN, C.BM_N_MAX, 1),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
}
# Schlüssel der Presets (Feldnamen von Settings, `routes` als Text) -> Schlüssel im Session State
PRESET_KEYS = {
    "game": "game_select", "routes": "routes_input", "m": "m_slider", "n": "n_slider", "game_seed": "game_seed_input", "learn": "learn_select", "rounds": "rounds_slider",
    "start": "start_radio", "pattern": "pattern_select", "counter": "counter_select", "tie": "tie_radio", "obs_rounds": "obs_rounds_slider", "bm": "bm_select", "bm_n": "bm_n_slider",
    "seed": "seed_input",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean_value(spec: SettingSpec, raw):
    """Wert aus der Adresszeile: umwandeln, auf Reglergrenzen begrenzen, auf die Stufen des Reglers einrasten; unbekannte Auswahl oder Unsinn gibt None."""
    if spec.cleaner is not None:
        return spec.cleaner(raw)
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if isinstance(value, float) and value != value:                         # nan
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step and spec.lo is not None and isinstance(value, int) and spec.step > 1:
        value = int(round((value - spec.lo) / spec.step) * spec.step + spec.lo)
        value = min(spec.hi, max(spec.lo, value))
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = clean_value(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt ALLE Regler: die des Presets auf seine Werte, die übrigen auf ihren Standard (so bleibt die Hervorhebung des Presets nach dem Anklicken gültig)."""
    preset = C.PRESETS[name]
    key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(key_of.get(state_key), spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_game_seed():
    st.session_state["game_seed_input"] = random.randint(0, C.SEED_MAX)


def settings_from_state() -> tuple[Settings, bool]:
    """Die Einstellungen aus dem Session State (nach den Reglern der Seitenleiste) und ob der Routentext gültig war (sonst gelten die Standardwerte)."""
    values = {field: st.session_state.get(state_key, SETTING_SPECS[state_key].default) for field, state_key in PRESET_KEYS.items()}
    routes = parse_routes(values["routes"])
    valid = routes is not None
    values["routes"] = routes if valid else parse_routes(C.DEFAULT_ROUTES)
    return Settings(**values), valid
