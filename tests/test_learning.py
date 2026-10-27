"""Die Einheiten von nsm_learning an der Handrechnung (conftest.py: A = [[3, 1], [2, 4]]), der Zufallsstart und die Eigenschaften der Verfahren."""

import math

import numpy as np
import pytest

import nsm_game as G
import nsm_learning as Ln
import nsm_lp as L
from nsm_rng import SplitMix64
from nsm_scenario import draw_matrix, random_simplex

UNIFORM2 = Ln.Start(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0, 0)


def test_checkpoints_are_sorted_unique_and_end_with_the_last_round():
    for rounds in (1, 2, 7, 100, 10000):
        pts = Ln.checkpoints(rounds)
        assert pts[0] == 1 and pts[-1] == rounds and list(pts) == sorted(set(pts)) and len(pts) <= 65
    assert list(Ln.checkpoints(1)) == [1] and list(Ln.checkpoints(3)) == [1, 2, 3]
    assert {10, 100, 1000, 10000} <= set(Ln.checkpoints(10000).tolist())


def test_positive_strategy_hand_values():
    assert list(Ln.positive_strategy([-0.5, 0.5])) == [0.0, 1.0]
    assert list(Ln.positive_strategy([1.0, 3.0])) == [0.25, 0.75]
    assert list(Ln.positive_strategy([-1.0, -2.0, 0.0])) == pytest.approx([1 / 3] * 3)         # kein positiver Regret: Gleichverteilung


def test_first_regret_step_on_the_mixed_game(mixed2):
    """Aus der Gleichverteilung: A·y = (2, 3), Mittel 2,5, Regrets der Zeile (−0,5; +0,5); xᵀA = (2,5; 2,5), also Regrets der Spalte (0, 0)."""
    gx, gy = Ln.regret_update(mixed2["A"], np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert list(gx) == pytest.approx([-0.5, 0.5]) and list(gy) == pytest.approx([0.0, 0.0])


def test_regret_matching_first_four_strategies_by_hand(mixed2):
    """Handrechnung: Runde 1 beide gleichverteilt; Regrets der Zeile (−0,5; 0,5) → x₂ = (0, 1), Spalte bleibt y₂ = (½, ½). Runde 2: A y = (2, 3), x = (0, 1): gx = (−1, 0), Rx = (−1,5; 0,5);
    xᵀA = (2, 4), gy = (1, −1), Ry = (1, −1) → x₃ = (0, 1), y₃ = (1, 0). Runde 3: A y = (3, 2), gx = (1, 0), Rx = (−0,5; 0,5); xᵀA = (2, 4), y·u = −2, gy = (0, −2), Ry = (1, −3) → x₄ = (0, 1), y₄ = (1, 0)."""
    run = Ln.regret_matching(mixed2["A"], 10, UNIFORM2)
    assert np.allclose(run.first_x, [[0.5, 0.5], [0, 1], [0, 1], [0, 1]]) and np.allclose(run.first_y, [[0.5, 0.5], [0.5, 0.5], [1, 0], [1, 0]])


def test_rm_plus_first_strategies_and_linear_average_by_hand(mixed2):
    """Regrets auf null abgeschnitten: Rx nach Runde 1 (0; 0,5), nach Runde 2 (0; 0,5) → x₃ = (0, 1); Ry nach Runde 2 (1; 0) → y₃ = (1, 0). Durchschnitt nach 3 Runden mit Gewicht t:
    (1 · (½, ½) + 2 · (0, 1) + 3 · (0, 1)) / 6 = (0,5; 5,5) / 6."""
    run = Ln.regret_matching(mixed2["A"], 3, UNIFORM2, plus=True)
    assert np.allclose(run.first_x[:3], [[0.5, 0.5], [0, 1], [0, 1]]) and np.allclose(run.first_y[:3], [[0.5, 0.5], [0.5, 0.5], [1, 0]])
    assert list(run.avg_x) == pytest.approx([0.5 / 6, 5.5 / 6])


def test_rm_plus_differs_from_rm_when_regrets_go_negative(mixed2):
    """Nach Runde 2 hat Zeile 1 bei RM den Regret −1,5, bei RM+ 0: nach weiteren Runden mit positivem Zuwachs unterscheiden sich die Strategien."""
    rm = Ln.regret_matching(mixed2["A"], 200, UNIFORM2)
    plus = Ln.regret_matching(mixed2["A"], 200, UNIFORM2, plus=True)
    assert not np.allclose(rm.avg_x, plus.avg_x, atol=1e-4)


def test_hedge_rate_formula_and_first_step(mixed2):
    A = mixed2["A"]
    eta = math.sqrt(8 * math.log(2) / 100) / 3.0                  # Spannweite 4 − 1 = 3
    assert Ln.hedge_rate(A, 2, 100) == pytest.approx(eta)
    run = Ln.hedge(A, 100, UNIFORM2)
    # Runde 2: Gewichte exp(η · A y₁) mit A y₁ = (2, 3): Verhältnis e^η
    assert run.first_x[1][1] / run.first_x[1][0] == pytest.approx(math.exp(eta * 1.0))
    # Spalte: xᵀA = (2,5; 2,5), Auszahlung der Spalte = −2,5 für beide: bleibt gleichverteilt
    assert list(run.first_y[1]) == pytest.approx([0.5, 0.5])


def test_hedge_starts_in_the_given_start_strategy(mixed2):
    start = Ln.Start(np.array([0.2, 0.8]), np.array([0.9, 0.1]), 0, 0)
    run = Ln.hedge(mixed2["A"], 50, start)
    assert list(run.first_x[0]) == pytest.approx([0.2, 0.8]) and list(run.first_y[0]) == pytest.approx([0.9, 0.1])


def test_fictitious_play_first_actions_by_hand(mixed2):
    """Start (0, 0). Nach Runde 1: Häufigkeiten x = (1, 0), y = (1, 0): Zeile: argmax A y = argmax(3, 2) = 0; Spalte: argmin xᵀA = argmin(3, 1) = 1. Nach Runde 2: x = (2, 0), y = (1, 1): argmax A y = argmax(4, 6) = 1,
    argmin xᵀA = argmin(6, 2) = 1. Verlader (0, 0, 1), Kontrolleur (0, 1, 1)."""
    run, acts_a, acts_b = Ln.fictitious_play(mixed2["A"], 6, UNIFORM2)
    assert list(acts_a[:3]) == [0, 0, 1] and list(acts_b[:3]) == [0, 1, 1]
    assert run.avg_x.sum() == pytest.approx(1.0) and run.avg_y.sum() == pytest.approx(1.0)


def test_first_best_breaks_ties_to_the_smallest_index():
    assert Ln.first_best(np.array([1.0, 3.0, 3.0])) == 1 and Ln.first_best(np.array([2.0, 2.0 - 1e-13, 1.0])) == 0


def test_sampled_regret_is_unbiased_for_the_expected_regret(mixed2):
    """Über alle Paare (a, b), gewichtet mit x_a y_b, ist der gezogene Regret gleich dem erwarteten (Handrechnung der Fixture, x = (0,3; 0,7), y = (0,6; 0,4))."""
    A = mixed2["A"]
    x, y = np.array([0.3, 0.7]), np.array([0.6, 0.4])
    gx = sum(x[a] * y[b] * Ln.sampled_regret_update(A, a, b)[0] for a in range(2) for b in range(2))
    gy = sum(x[a] * y[b] * Ln.sampled_regret_update(A, a, b)[1] for a in range(2) for b in range(2))
    ex, ey = Ln.regret_update(A, x, y)
    assert list(gx) == pytest.approx(list(ex)) and list(gy) == pytest.approx(list(ey))
    # Einzelfall von Hand: a = 0, b = 0: Gewinn 3; Zeile: A[:, 0] − 3 = (0, −1); Spalte: 3 − A[0, :] = (0, 2)
    assert np.allclose(Ln.sampled_regret_update(A, 0, 0), [[0.0, -1.0], [0.0, 2.0]])


def test_draw_action_follows_the_distribution():
    rng = SplitMix64(5)
    counts = np.bincount([Ln.draw_action([0.2, 0.5, 0.3], rng) for _ in range(6000)], minlength=3) / 6000
    assert list(counts) == pytest.approx([0.2, 0.5, 0.3], abs=0.03)
    assert Ln.draw_action([0.0, 1.0], SplitMix64(1)) == 1 and Ln.draw_action([1.0, 0.0], SplitMix64(1)) == 0


def test_random_start_is_a_new_point_each_time_and_uniform_start_is_uniform():
    starts = [Ln.make_start(4, 3, "random", SplitMix64(s)) for s in range(20)]
    assert len({tuple(np.round(s.x1, 9)) for s in starts}) == 20
    for s in starts:
        assert s.x1.sum() == pytest.approx(1.0) and s.y1.sum() == pytest.approx(1.0) and s.x1.min() > 0 and 0 <= s.a0 < 4 and 0 <= s.b0 < 3
    u = Ln.make_start(4, 3, "uniform", SplitMix64(1))
    assert list(u.x1) == [0.25] * 4 and list(u.y1) == pytest.approx([1 / 3] * 3) and (u.a0, u.b0) == (0, 0)
    with pytest.raises(ValueError):
        Ln.make_start(2, 2, "unbekannt", SplitMix64(1))


def test_random_simplex_has_the_right_mean():
    rng = SplitMix64(9)
    pts = np.array([random_simplex(4, rng) for _ in range(3000)])
    assert list(pts.mean(axis=0)) == pytest.approx([0.25] * 4, abs=0.01)


@pytest.mark.parametrize("method", ["rm", "rmplus", "mw"])
def test_starting_in_the_equilibrium_stands_still(method, rps):
    """Stein-Schere-Papier aus der Gleichverteilung: alle Auszahlungen gleich, alle Regrets null; Ausnutzbarkeit 0 an jeder Stützstelle (der „tote Zweig“)."""
    run = Ln.run_method(method, rps["A"], 500, "uniform", 3)
    assert np.all(run.expl_avg <= 1e-12) and np.all(run.expl_last <= 1e-12)
    assert list(run.avg_x) == pytest.approx([1 / 3] * 3, abs=1e-12)


@pytest.mark.parametrize("method", ["rm", "rmplus", "mw", "fp", "rm_sampled"])
def test_a_random_start_leaves_the_equilibrium_in_every_method(method, rps):
    run = Ln.run_method(method, rps["A"], 500, "random", 3)
    assert run.expl_avg[0] > 1e-3 or run.expl_last.max() > 1e-3


def test_average_exploitability_is_never_below_zero_and_brackets_the_value():
    for t, method in enumerate(["rm", "rmplus", "mw", "fp", "rm_sampled"] * 2):
        A = draw_matrix(3 + t % 3, 4, 100 + t)
        v = L.solve_zero_sum(A).value
        run = Ln.run_method(method, A, 400, "random", t)
        assert np.all(run.expl_avg >= -1e-12) and np.all(run.expl_last >= -1e-12)
        assert G.guaranteed(A, run.avg_x) <= v + 1e-9 and G.ceiling(A, run.avg_y) >= v - 1e-9           # Minimax-Satz: kein Paar liegt außerhalb [garantiert, zugelassen]


def test_every_method_gets_close_to_the_equilibrium_on_the_mixed_game(mixed2):
    for method in Ln.C.LEARN_METHODS:
        run = Ln.run_method(method, mixed2["A"], 5000, "random", 4)
        assert run.expl_avg[-1] < 0.05, method


def test_methods_end_in_different_places_on_a_random_game():
    A = draw_matrix(5, 5, 77)
    finals = {m: Ln.run_method(m, A, 300, "random", 5).avg_x for m in Ln.C.LEARN_METHODS}
    names = list(finals)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(finals[names[i]], finals[names[j]], atol=1e-6), (names[i], names[j])         # jede Variante läuft wirklich (kein toter Zweig)


def test_runs_are_reproducible_and_depend_on_the_seed():
    A = draw_matrix(4, 4, 8)
    for method in Ln.C.LEARN_METHODS:
        a = Ln.run_method(method, A, 200, "random", 1)
        b = Ln.run_method(method, A, 200, "random", 1)
        c = Ln.run_method(method, A, 200, "random", 2)
        assert np.array_equal(a.avg_x, b.avg_x) and np.array_equal(a.expl_last, b.expl_last)
        assert not np.array_equal(a.expl_avg, c.expl_avg), method


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        Ln.run_method("unbekannt", np.eye(2), 10, "uniform", 1)


def test_constant_matrix_does_not_break_hedge():
    run = Ln.hedge(np.ones((3, 3)), 50, Ln.make_start(3, 3, "random", SplitMix64(1)))
    assert np.all(np.isfinite(run.expl_avg)) and np.all(run.expl_avg <= 1e-12)


def test_largest_game_and_longest_run_finish_for_every_method():
    A = draw_matrix(10, 10, 5)
    for method in Ln.C.LEARN_METHODS:
        run = Ln.run_method(method, A, 10000, "random", 1)
        assert run.steps[-1] == 10000 and np.all(np.isfinite(run.expl_avg))


def test_every_method_keeps_the_first_four_strategies_and_never_more_than_it_played():
    A = draw_matrix(3, 3, 12)
    for method in Ln.C.LEARN_METHODS:
        for rounds in (3, 10):
            run = Ln.run_method(method, A, rounds, "random", 1)
            assert run.first_x.shape == (min(rounds, 4), 3) and run.first_y.shape == (min(rounds, 4), 3), (method, rounds)


def test_checkpoints_include_the_powers_of_ten_even_when_the_logarithmic_grid_misses_them():
    pts = set(Ln.checkpoints(200, count=5).tolist())
    assert {10, 100, 200, 1} <= pts and 11 not in pts and 121 not in pts


def test_draw_action_never_returns_an_index_outside_the_distribution():
    class Top:
        """Zufallsstrom, dessen Zahl immer die größte ist (u nahe 1): auch wenn die Summe der Wahrscheinlichkeiten unter u liegt, gilt der letzte Index."""
        def next(self):
            return (1 << 64) - 1
    assert Ln.draw_action([0.3, 0.3], Top()) == 1 and Ln.draw_action([0.1] * 10, Top()) == 9 and Ln.draw_action([1.0], Top()) == 0


def test_checkpoints_powers_of_ten_just_above_the_threshold_and_above_ten_thousand():
    """Runde 110 mit drei Stützstellen: logarithmisch 1, 10, 110; die Zehnerpotenz 100 liegt knapp unter der Rundenzahl und muss dazu. Bei 20 000 Runden gehört 10 000 dazu."""
    assert list(Ln.checkpoints(110, count=3)) == [1, 10, 100, 110]
    assert 10000 in set(Ln.checkpoints(20000, count=5).tolist()) and 1000 in set(Ln.checkpoints(1500, count=3).tolist())
