"""Orakel für das lineare Programm: Zertifikat ohne Löser, exakte Stützenaufzählung mit Brüchen, Primal gegen Dual getrennt, Bereich der optimalen Strategien, Sattelpunkt-Formel.

Jeder Test benutzt einen anderen Rechenweg als `nsm_lp` (ORACLE-PLAYBOOK §2: LP gegen exakt rational und gegen die geschlossene Formel); die Orakel selbst sind an den Handbeispielen aus conftest.py geprüft."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import nsm_game as G
import nsm_lp as L
from exact_games import basic_solutions, value_and_extremes
from nsm_rng import SplitMix64
from nsm_scenario import draw_matrix, uniform01


def _int_matrix(m, n, seed, top=9):
    rng = SplitMix64(seed)
    return [[rng.below(top + 1) for _ in range(n)] for _ in range(m)]


def test_exact_oracle_reproduces_the_hand_examples(mixed2, saddle):
    assert value_and_extremes([[3, 1], [2, 4]]) == (F(5, 2), [(F(1, 2), F(1, 2))], [(F(3, 4), F(1, 4))])
    value, xs, ys = value_and_extremes([[4, 3], [2, 1]])
    assert value == 3 and xs == [(F(1), F(0))] and ys == [(F(0), F(1))]


def test_lp_value_equals_the_exact_value_on_integer_games():
    """40 ganzzahlige Spiele von 2 × 2 bis 4 × 5 (Einträge 0 bis 9): LP-Wert gleich dem exakten Bruch, und die LP-Strategien erfüllen die exakten Garantien."""
    sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (2, 4), (5, 2)]
    for t, (m, n) in enumerate(sizes * 4):
        A = _int_matrix(m, n, 1000 + t)
        value, _xs, _ys = value_and_extremes(A)
        sol = L.solve_zero_sum(np.array(A, float))
        assert sol.value == pytest.approx(float(value), abs=1e-9), (A, value)
        assert sol.dual_value == pytest.approx(float(value), abs=1e-9)
        assert sol.lower == pytest.approx(float(value), abs=1e-9) and sol.upper == pytest.approx(float(value), abs=1e-9)


def test_optimal_ranges_equal_min_and_max_over_the_exact_extreme_strategies():
    """Der Bereich jeder Koordinate über die optimale Menge ist das Minimum und Maximum über die exakten Ecken (Shapley–Snow). Ganzzahlige Spiele haben oft mehrere optimale Ecken."""
    multiple = 0
    for t in range(60):
        m, n = 2 + t % 3, 2 + (t // 3) % 3
        A = _int_matrix(m, n, 2000 + t, top=4)
        value, xs, ys = value_and_extremes(A)
        x_lo, x_hi, y_lo, y_hi = L.optimal_ranges(np.array(A, float), float(value))
        for k in range(m):
            assert x_lo[k] == pytest.approx(float(min(x[k] for x in xs)), abs=1e-7) and x_hi[k] == pytest.approx(float(max(x[k] for x in xs)), abs=1e-7), (A, k)
        for k in range(n):
            assert y_lo[k] == pytest.approx(float(min(y[k] for y in ys)), abs=1e-7) and y_hi[k] == pytest.approx(float(max(y[k] for y in ys)), abs=1e-7), (A, k)
        multiple += len(xs) > 1 or len(ys) > 1
        assert L.is_unique(x_lo, x_hi) == (len(xs) == 1)
        assert L.is_unique(y_lo, y_hi) == (len(ys) == 1)
    assert multiple >= 5, "der Test sieht keine Spiele mit mehreren optimalen Ecken: das Orakel wäre blind für die Mehrdeutigkeit"


def test_certificate_does_not_use_the_solver_value():
    """Zertifikat: x garantiert `lower`, y hält `upper`, und lower ≤ upper; beides aus der Matrix gerechnet. Auf Zufallsspielen 2 × 2 bis 10 × 10 ist die Lücke höchstens 1e-9."""
    for t in range(60):
        m, n = 2 + t % 9, 2 + (t * 7) % 9
        A = draw_matrix(m, n, 3000 + t)
        sol = L.solve_zero_sum(A)
        assert np.isclose(sol.x.sum(), 1.0) and np.isclose(sol.y.sum(), 1.0) and sol.x.min() >= 0 and sol.y.min() >= 0
        assert float(np.min(sol.x @ A)) == sol.lower and float(np.max(A @ sol.y)) == sol.upper
        assert sol.lower <= sol.upper + 1e-12 and sol.upper - sol.lower <= 1e-9
        assert sol.lower - 1e-9 <= sol.value <= sol.upper + 1e-9 and abs(sol.value - sol.dual_value) <= 1e-9


def test_primal_and_dual_are_solved_separately_and_agree_on_the_minimax_theorem():
    for t in range(20):
        A = draw_matrix(4, 6, 4000 + t)
        v, x = L.lp_row(A)
        w, y = L.lp_col(A)
        assert v == pytest.approx(w, abs=1e-9)
        assert G.pure_security(A) - 1e-12 <= v <= G.pure_ceiling(A) + 1e-12      # Mischen kann das Sicherheitsniveau nur verbessern


def test_value_is_unchanged_by_positive_affine_maps_and_flips_sign_by_transposing():
    A = draw_matrix(3, 4, 5000)
    v = L.solve_zero_sum(A).value
    assert L.solve_zero_sum(3.0 * A + 2.0).value == pytest.approx(3.0 * v + 2.0, abs=1e-9)
    assert L.solve_zero_sum(-A.T).value == pytest.approx(-v, abs=1e-9)               # Rollentausch: der Spaltenspieler wird zum Zeilenspieler


def test_flat_game_has_the_whole_simplex_as_solution_set(flat2):
    sol = L.solve_zero_sum(flat2["A"])
    x_lo, x_hi, y_lo, y_hi = L.optimal_ranges(flat2["A"], sol.value)
    assert list(x_lo) == pytest.approx([0, 0], abs=1e-12) and list(x_hi) == pytest.approx([1, 1], abs=1e-12)
    assert list(y_lo) == pytest.approx([0, 0], abs=1e-12) and list(y_hi) == pytest.approx([1, 1], abs=1e-12)
    assert not L.is_unique(x_lo, x_hi) and not L.is_unique(y_lo, y_hi)


def test_saddle_game_and_mixed_game_have_unique_solutions(saddle, mixed2):
    for case in (saddle, mixed2):
        sol = L.solve_zero_sum(case["A"])
        assert sol.value == pytest.approx(float(case["value"]))
        assert list(sol.x) == pytest.approx([float(t) for t in case["x"]], abs=1e-9) and list(sol.y) == pytest.approx([float(t) for t in case["y"]], abs=1e-9)
        lo_hi = L.optimal_ranges(case["A"], sol.value)
        assert L.is_unique(lo_hi[0], lo_hi[1]) and L.is_unique(lo_hi[2], lo_hi[3])


def test_saddle_point_share_of_random_games_follows_n_factorial_squared_over_2n_minus_1_factorial():
    """Bei stetigen Einträgen hat ein n × n-Spiel mit Wahrscheinlichkeit n!² / (2n − 1)! einen Sattelpunkt (zählt die Anordnungen in Zeile und Spalte). 2 000 Spiele je Größe, Marge 4 Standardfehler."""
    for n in (2, 3, 4):
        games = 2000
        count = sum(bool(G.saddle_points(draw_matrix(n, n, 6000 + 97 * n + t))) for t in range(games))
        p = math.factorial(n) ** 2 / math.factorial(2 * n - 1)
        assert abs(count / games - p) <= 4 * math.sqrt(p * (1 - p) / games), (n, count / games, p)


def test_uniform01_stays_in_the_half_open_interval():
    rng = SplitMix64(1)
    draws = [uniform01(rng) for _ in range(5000)]
    assert 0.0 <= min(draws) and max(draws) < 1.0 and abs(sum(draws) / 5000 - 0.5) < 0.03


def test_basic_solutions_of_degenerate_square_games_are_found_again_and_again():
    sols = basic_solutions([[1, 1], [1, 1]])
    assert len(sols) >= 4 and all(s[0] == 1 for s in sols)
