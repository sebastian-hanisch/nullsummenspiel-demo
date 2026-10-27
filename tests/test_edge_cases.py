"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): zwei Routen, Gleichstände, Streichen genau an der Grenze, 1 × n, Reglermaxima.

Die häufigsten Lücken, die der Mutationstest in anderen Demos fand: ein Zweig `n < 2` greift bei genau zwei Elementen, eine Grenze gilt nicht inklusive, ein Gleichstand wird anders aufgelöst als dokumentiert."""

import numpy as np
import pytest

import nsm_bimatrix as Bm
import nsm_constants as C
import nsm_evaluation as ev
import nsm_game as G
import nsm_hideseek as H
import nsm_learning as Ln
import nsm_lp as L
import nsm_observers as O
from nsm_rng import SplitMix64
from nsm_scenario import Settings, draw_matrix


def test_exactly_two_routes_never_drop_a_route():
    """Mit zwei Routen ist y₂ = 1 − k/v₂ = v₁/(v₁ + v₂) ≥ 0: nie wird gestrichen (auch nicht bei Wertverhältnis 20 : 1)."""
    for v in ((1.0, 0.05), (0.05, 1.0), (0.3, 0.3)):
        k, x, y, active = H.closed_form(v)
        assert active == [0, 1] and x.sum() == pytest.approx(1.0) and y.sum() == pytest.approx(1.0)
    k, x, y, _a = H.closed_form((1.0, 0.05))
    assert k == pytest.approx(0.05 / 1.05) and list(x) == pytest.approx([0.05 / 1.05, 1 / 1.05]) and list(y) == pytest.approx([1 / 1.05, 0.05 / 1.05])


def test_dropping_at_the_boundary_keeps_the_route_with_probability_zero():
    """Werte (1, 1, 0,5): c = 1/(1 + 1 + 2) = 1/4, k = 2 c = 1/2, y₃ = 1 − (1/2)/(1/2) = 0 genau: die Route 3 bleibt (Grenze eingeschlossen), x₃ = c/v₃ = 1/2 mit y₃ = 0.
    Dasselbe Gleichgewicht ergibt das lineare Programm: Wert 1/2. (Ein x₃ > 0 ist hier optimal, weil Route 3 gegen y gerade den Wert k zahlt.)"""
    k, x, y, active = H.closed_form((1.0, 1.0, 0.5))
    assert active == [0, 1, 2] and k == pytest.approx(0.5) and y[2] == pytest.approx(0.0, abs=1e-12) and list(x) == pytest.approx([0.25, 0.25, 0.5])
    A = H.hide_seek_matrix((1.0, 1.0, 0.5))
    assert L.solve_zero_sum(A).value == pytest.approx(0.5) and G.guaranteed(A, x) == pytest.approx(0.5) and G.ceiling(A, y) == pytest.approx(0.5)


def test_two_equal_cheap_routes_are_both_dropped_one_after_the_other():
    """Werte (1, 1, 0,05, 0,05): alle vier Routen: c = 1/42, k = 3/42, y₃ = 1 − (3/42)/0,05 < 0 → eine der beiden billigen fällt; Rest (1, 1, 0,05): c = 1/22, k = 2/22 = 0,0909, y = 1 − 0,0909/0,05 < 0 →
    auch die zweite fällt. Es bleiben die beiden Routen mit Wert 1: Wert 1/2, beide gleichverteilt."""
    k, x, y, active = H.closed_form((1.0, 1.0, 0.05, 0.05))
    assert active == [0, 1] and k == pytest.approx(0.5) and list(x) == pytest.approx([0.5, 0.5, 0, 0]) and list(y) == pytest.approx([0.5, 0.5, 0, 0])
    assert L.solve_zero_sum(H.hide_seek_matrix((1.0, 1.0, 0.05, 0.05))).value == pytest.approx(0.5)


def test_one_by_n_and_n_by_one_games():
    row = np.array([[3.0, 1.0, 2.0]])                      # Verlader hat nur eine Zeile: der Kontrolleur nimmt die kleinste Spalte
    sol = L.solve_zero_sum(row)
    assert sol.value == pytest.approx(1.0) and list(sol.x) == [1.0] and list(sol.y) == pytest.approx([0.0, 1.0, 0.0])
    col = np.array([[3.0], [1.0], [2.0]])                  # Kontrolleur hat nur eine Spalte: der Verlader nimmt die größte Zeile
    sol = L.solve_zero_sum(col)
    assert sol.value == pytest.approx(3.0) and list(sol.x) == pytest.approx([1.0, 0.0, 0.0]) and list(sol.y) == [1.0]
    assert G.saddle_points(row) == [(0, 1)] and G.saddle_points(col) == [(0, 0)]


def test_learning_on_a_one_by_one_game():
    for method in C.LEARN_METHODS:
        run = Ln.run_method(method, np.array([[2.0]]), 50, "random", 1)
        assert run.expl_avg[-1] == 0.0 and list(run.avg_x) == [1.0]


def test_minimal_rounds_for_every_method():
    A = draw_matrix(3, 3, 4)
    for method in C.LEARN_METHODS:
        run = Ln.run_method(method, A, C.ROUNDS_MIN, "random", 2)
        assert run.steps[-1] == C.ROUNDS_MIN and run.steps[0] == 1 and np.all(np.isfinite(run.expl_avg))
    run = Ln.run_method("rm", A, 1, "random", 2)
    assert list(run.steps) == [1] and len(run.first_x) == 1


def test_observers_with_two_routes_and_the_minimal_number_of_rounds():
    v = (1.0, 1.0)
    k, x, y, _a = H.closed_form(v)
    for pattern in C.PATTERNS:
        for counter in C.COUNTERS:
            obs = O.simulate(pattern, counter, "random", v, x, y, C.OBS_ROUNDS_MIN, 3)
            assert len(obs.routes) == C.OBS_ROUNDS_MIN and set(obs.routes.tolist()) <= {0, 1} and 0.0 <= obs.mean_delivery <= 1.0
    assert set(O.loader_sequence("norepeat", 2, None, 20, SplitMix64(1)).tolist()) == {0, 1}
    assert np.all(np.abs(np.diff(O.loader_sequence("norepeat", 2, None, 20, SplitMix64(1)))) == 1)         # bei zwei Routen heißt „kein Wiederholen“ abwechseln


def test_observers_with_eight_routes_at_the_maximum_rounds():
    v = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
    k, x, y, _a = H.closed_form(v)
    for counter in ("freq", "window", "follow", "fixed"):
        obs = O.simulate("habit", counter, "low", v, x, y, C.OBS_ROUNDS_MAX, 1)
        assert obs.mean_delivery >= 0 and obs.running_mean[-1] == pytest.approx(obs.mean_delivery)


def test_largest_games_for_the_lp_and_the_ranges():
    A = draw_matrix(C.M_MAX, C.N_MAX, 3)
    a = ev.analyse(Settings(game="random", m=C.M_MAX, n=C.N_MAX, game_seed=3))
    assert a.A.shape == (10, 10) and a.sol.upper - a.sol.lower < 1e-9
    assert a.value == pytest.approx(L.solve_zero_sum(A).value)
    big = ev.analyse(Settings(routes=(0.5,) * 8))
    assert big.value == pytest.approx(0.5 * 7 / 8) and big.form["active"] == list(range(8))


def test_enumeration_at_the_slider_maximum():
    A, B = ev.bimatrix_game("random", C.BM_N_MAX, 5)
    eq = Bm.enumerate_nash(A, B)
    assert len(eq) % 2 == 1 and all(Bm.is_nash(A, B, x, y, tol=1e-8) for x, y in eq)
    assert Bm.lemke_howson(A, B, 2 * C.BM_N_MAX - 1)[2] >= 1


def test_lemke_howson_on_a_game_with_a_dominant_pair_takes_two_pivots():
    """Eine strikt dominante Aktion je Spieler: das einzige Gleichgewicht ist rein; Lemke–Howson findet es von jedem Label."""
    A = np.array([[3.0, 2.0], [1.0, 0.0]])
    B = np.array([[3.0, 1.0], [2.0, 0.0]])
    assert len(Bm.enumerate_nash(A, B)) == 1
    for label in range(4):
        x, y, pivots, _path = Bm.lemke_howson(A, B, label)
        assert list(x) == pytest.approx([1, 0]) and list(y) == pytest.approx([1, 0]) and pivots >= 1


def test_zero_payoff_game_is_all_ties():
    A = np.zeros((3, 3))
    sol = L.solve_zero_sum(A)
    assert sol.value == 0.0 and len(G.saddle_points(A)) == 9
    run = Ln.run_method("fp", A, 100, "uniform", 1)
    assert np.all(run.expl_avg == 0.0)
