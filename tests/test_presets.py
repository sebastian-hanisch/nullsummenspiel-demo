"""Presets: gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ist unterscheidbar; Permalink-Bereinigung; Routentext lesen und schreiben."""

import dataclasses

import pytest

import nsm_constants as C
import nsm_presets as P
from nsm_scenario import Settings, format_routes, parse_routes

DEFAULT_PRESET = "Vier gleiche Routen (Voreinstellung)"


def _settings_of(preset: dict) -> Settings:
    values = {"routes": parse_routes(C.DEFAULT_ROUTES)} | {k: v for k, v in preset.items() if k != "routes"}
    if "routes" in preset:
        values["routes"] = parse_routes(preset["routes"])
    return Settings(**values)


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_keys_are_exactly_the_fields_of_settings():
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_values_are_inside_the_regler_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if key == "routes":
                assert parse_routes(value) is not None and format_routes(parse_routes(value)) == value, (name, value)      # kanonischer Text, sonst hebt aktives_preset nie hervor
            elif spec.choices is not None:
                assert value in spec.choices, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_equals_the_default_settings():
    assert next(iter(C.PRESETS)) == DEFAULT_PRESET
    assert _settings_of(C.PRESETS[DEFAULT_PRESET]) == Settings()
    assert C.DEFAULT_ROUTES == format_routes(Settings().routes)


def test_every_other_preset_differs_from_the_default_and_from_each_other():
    seen = {Settings()}
    for name, preset in list(C.PRESETS.items())[1:]:
        s = _settings_of(preset)
        assert s not in seen, name                    # sonst hebt aktives_preset zwei Presets zugleich hervor und das spätere wäre nie erkennbar
        seen.add(s)


def test_every_preset_changes_the_settings_it_describes():
    default = Settings()
    changed = {name: {f.name for f in dataclasses.fields(Settings) if getattr(_settings_of(p), f.name) != getattr(default, f.name)} for name, p in C.PRESETS.items()}
    assert changed["Münzwurf (zwei gleiche Routen)"] == {"routes"} and changed["Ungleiche Werte (10 : 1)"] == {"routes"} and changed["Vier Routen mit Streichen"] == {"routes"}
    assert changed["Stein-Schere-Papier"] == {"game"} and changed["Sattelpunkt"] == {"game"} and changed["Zufallsspiel 5 × 5"] == {"game"}
    assert changed["Rotation gegen Folgezähler"] == {"counter", "obs_rounds"} and changed["Rotation gegen Häufigkeitszähler"] == {"tie", "obs_rounds"}


def test_sliders_with_steps_have_their_default_on_the_grid():
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.step and spec.lo is not None:
            assert abs((spec.default - spec.lo) / spec.step - round((spec.default - spec.lo) / spec.step)) < 1e-9, state_key


def test_parse_and_format_routes():
    assert parse_routes("1; 0,75; 0.5") == (1.0, 0.75, 0.5)
    assert parse_routes(" 1 ;0,1;; ") == (1.0, 0.1)
    assert format_routes((1.0, 0.75, 0.1)) == "1; 0,75; 0,1"
    for bad in ("", "1", "abc; 1", "1; 2", "1; 0,01", "1; nan", "1;1;1;1;1;1;1;1;1", "1; -0,5"):
        assert parse_routes(bad) is None, bad
    assert parse_routes("0,05; 1") == (0.05, 1.0)                    # Grenzen eingeschlossen
    assert len(parse_routes("; ".join(["0,5"] * 8))) == 8


@pytest.mark.parametrize("state_key,raw,expected", [
    ("game_select", "rps", "rps"), ("game_select", "poker", None), ("routes_input", "1;0.5", "1; 0,5"), ("routes_input", "7; 1", None),
    ("m_slider", "99", C.M_MAX), ("m_slider", "0", C.M_MIN), ("m_slider", "x", None), ("n_slider", "7", 7),
    ("rounds_slider", "123456", C.ROUNDS_MAX), ("rounds_slider", "1234", 1200), ("rounds_slider", "5", C.ROUNDS_MIN), ("obs_rounds_slider", "1999", 2000),
    ("learn_select", "rmplus", "rmplus"), ("learn_select", "adam", None), ("start_radio", "uniform", "uniform"), ("tie_radio", "random", "random"), ("tie_radio", "x", None),
    ("pattern_select", "habit", "habit"), ("counter_select", "fixed", "fixed"), ("bm_select", "shapley", "shapley"), ("bm_n_slider", "20", C.BM_N_MAX), ("bm_n_slider", "1", C.BM_N_MIN),
    ("seed_input", "-5", 0), ("game_seed_input", "9999999", C.SEED_MAX),
])
def test_permalink_values_are_cleaned(state_key, raw, expected):
    assert P.clean_value(P.SETTING_SPECS[state_key], raw) == expected


def test_permalink_value_of_every_regler_round_trips():
    for state_key, spec in P.SETTING_SPECS.items():
        assert P.clean_value(spec, str(spec.default)) == spec.default, state_key
