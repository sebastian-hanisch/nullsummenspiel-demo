"""Verstecken und Suchen: Handbeispiele (conftest.py), geschlossene Form gegen das lineare Programm und gegen die exakte Aufzählung, Streichen, Faustregeln."""


import pytest

import nsm_game as G
import nsm_hideseek as H
import nsm_lp as L
from exact_games import value_and_extremes
from nsm_rng import SplitMix64
from nsm_scenario import uniform01


def test_matrix_has_zeros_on_the_diagonal_and_the_route_value_in_each_row():
    assert H.hide_seek_matrix((10.0, 1.0)).tolist() == [[0.0, 10.0], [1.0, 0.0]]
    assert H.hide_seek_matrix((3.0, 2.0, 1.0)).tolist() == [[0, 3, 3], [2, 0, 2], [1, 1, 0]]


def test_closed_form_hand_example_ten_to_one(hide_10_1):
    k, x, y, active = H.closed_form(hide_10_1["v"])
    assert k == pytest.approx(float(hide_10_1["value"])) and active == [0, 1]
    assert list(x) == pytest.approx([float(t) for t in hide_10_1["x"]]) and list(y) == pytest.approx([float(t) for t in hide_10_1["y"]])


def test_closed_form_drops_the_fourth_route_in_the_hand_example(hide_4321):
    k, x, y, active = H.closed_form(hide_4321["v"])
    assert active == hide_4321["active"]
    assert k == pytest.approx(float(hide_4321["value"]))
    assert list(x) == pytest.approx([float(t) for t in hide_4321["x"]]) and list(y) == pytest.approx([float(t) for t in hide_4321["y"]])
    assert x[3] == 0.0 and y[3] == 0.0


def test_equal_routes_give_n_minus_1_over_n_times_the_value():
    for n in (2, 3, 4, 6, 10):
        k, x, y, active = H.closed_form([0.8] * n)
        assert k == pytest.approx(0.8 * (n - 1) / n) and active == list(range(n)) and list(x) == pytest.approx([1 / n] * n) and list(y) == pytest.approx([1 / n] * n)


def test_the_best_pure_route_guarantees_zero():
    for v in ((10.0, 1.0), (4.0, 3.0, 2.0, 1.0), (0.9, 0.8, 0.7, 0.6)):
        assert G.pure_security(H.hide_seek_matrix(v)) == 0.0              # der Kontrolleur prüft genau diese Route


def _random_values(n, low, seed):
    rng = SplitMix64(seed)
    return [round(low + (1.0 - low) * uniform01(rng), 4) for _ in range(n)]


def test_closed_form_matches_the_linear_program_on_random_values():
    """Wert, x und y der Formel gegen das lineare Programm (Primal und Dual getrennt): 2 Wertebereiche, n = 2, 3, 4, 6, 8, je 15 Spiele. Eindeutig, wenn alle Werte verschieden sind."""
    dropped = 0
    for low in (0.3, 0.05):
        for n in (2, 3, 4, 6, 8):
            for t in range(15):
                v = _random_values(n, low, 7000 + 31 * n + t + int(low * 100))
                A = H.hide_seek_matrix(v)
                sol = L.solve_zero_sum(A)
                k, x, y, active = H.closed_form(v)
                assert sol.value == pytest.approx(k, abs=1e-9) and sol.dual_value == pytest.approx(k, abs=1e-9)
                assert list(sol.x) == pytest.approx(list(x), abs=1e-6) and list(sol.y) == pytest.approx(list(y), abs=1e-6), (v,)
                assert G.guaranteed(A, x) == pytest.approx(k, abs=1e-9) and G.ceiling(A, y) == pytest.approx(k, abs=1e-9)
                dropped += len(active) < n
    assert dropped >= 10, "kein Streichen gesehen: der Zweig wäre ungeprüft"


def test_closed_form_matches_the_exact_enumeration_on_integer_values():
    """Ganzzahlige Werte (Brüche rechnen exakt): geschlossene Form gleich der exakten Basislösung. Mit und ohne Streichen."""
    cases = [(10, 1), (4, 3, 2, 1), (5, 4, 3), (9, 2, 1), (6, 6, 6), (8, 5, 5, 2), (7, 3, 2), (3, 2, 1, 1)]
    for v in cases:
        value, xs, ys = value_and_extremes(H.hide_seek_matrix(v).astype(int).tolist())
        k, x, y, _active = H.closed_form(v)
        assert len(xs) == 1 and len(ys) == 1, v
        assert k == pytest.approx(float(value), abs=1e-12)
        assert list(x) == pytest.approx([float(t) for t in xs[0]], abs=1e-12) and list(y) == pytest.approx([float(t) for t in ys[0]], abs=1e-12), v


def test_loader_rules_hand_values():
    """Werte (10, 1): „proportional zum Wert“ ist x = (10/11, 1/11) und garantiert x·v − max_j x_j v_j = 101/11 − 100/11 = 1/11 = 0,0909; gleichverteilt: 0,5 · 11 − 0,5 · 10 = 0,5."""
    A = H.hide_seek_matrix((10.0, 1.0))
    rules = H.loader_rules((10.0, 1.0))
    assert G.guaranteed(A, rules["proportional zum Wert"]) == pytest.approx(1 / 11)
    assert G.guaranteed(A, rules["gleichverteilt"]) == pytest.approx(0.5)
    assert G.guaranteed(A, rules["nur die wertvollste Route"]) == 0.0
    assert list(rules["umgekehrt proportional zum Wert (alle Routen)"]) == pytest.approx([1 / 11, 10 / 11])       # ist hier das Gleichgewicht
    assert G.guaranteed(A, rules["umgekehrt proportional zum Wert (alle Routen)"]) == pytest.approx(10 / 11)


def test_controller_rules_hand_values_for_four_routes():
    """Werte (4, 3, 2, 1), Wert 24/13: „nur die wertvollste Route prüfen“ (y = e₁) lässt der Verlader die beste andere Route: max(3, 2, 1) · 1 = 3 zu (Zeile 2: 3 · (1 − 0) = 3).
    Gleichverteilt prüfen: Zeile i bringt v_i · 3/4: max = 4 · 3/4 = 3. Proportional (y = v/10): Zeile i bringt v_i (1 − v_i/10): 4 · 0,6 = 2,4; 3 · 0,7 = 2,1; 2 · 0,8 = 1,6; 0,9: max 2,4."""
    v = (4.0, 3.0, 2.0, 1.0)
    A = H.hide_seek_matrix(v)
    rules = H.controller_rules(v)
    assert G.ceiling(A, rules["nur die wertvollste Route"]) == pytest.approx(3.0)
    assert G.ceiling(A, rules["gleichverteilt"]) == pytest.approx(3.0)
    assert G.ceiling(A, rules["proportional zum Wert"]) == pytest.approx(2.4)
    assert 24 / 13 < 2.4


def test_with_two_routes_the_proportional_rule_of_the_controller_is_the_equilibrium():
    for v in ((10.0, 1.0), (0.9, 0.3), (0.5, 0.5)):
        k, _x, y, _a = H.closed_form(v)
        assert list(H.controller_rules(v)["proportional zum Wert"]) == pytest.approx(list(y))


def test_rules_are_probability_vectors():
    for v in ((0.9, 0.8, 0.7, 0.6), (1.0, 0.05), (0.3,) * 8):
        for rules in (H.loader_rules(v), H.controller_rules(v)):
            for p in rules.values():
                assert p.min() >= 0 and p.sum() == pytest.approx(1.0)
