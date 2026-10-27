"""Beobachter: Muster, Zähler, Gleichstandsregel und das Schätzen aus Beobachtungen, an Handsequenzen und Eigenschaften."""

import numpy as np
import pytest

import nsm_game as G
import nsm_hideseek as H
import nsm_lp as L
import nsm_observers as O
from nsm_rng import SplitMix64

EQUAL3 = (1.0, 1.0, 1.0)


def test_rotation_visits_the_routes_in_order():
    assert O.loader_sequence("rotation", 3, None, 7, SplitMix64(1)).tolist() == [0, 1, 2, 0, 1, 2, 0]


def test_norepeat_never_repeats_and_uses_all_other_routes():
    seq = O.loader_sequence("norepeat", 4, None, 2000, SplitMix64(2))
    assert np.all(seq[1:] != seq[:-1])
    transitions = {(int(a), int(b)) for a, b in zip(seq[:-1], seq[1:])}
    assert transitions == {(a, b) for a in range(4) for b in range(4) if a != b}


def test_habit_repeats_with_probability_one_half_plus_one_half_over_n():
    seq = O.loader_sequence("habit", 4, None, 20000, SplitMix64(3))
    assert np.mean(seq[1:] == seq[:-1]) == pytest.approx(0.5 + 0.5 / 4, abs=0.02)


def test_iid_equilibrium_follows_x_star():
    seq = O.loader_sequence("iid_eq", 3, np.array([0.2, 0.3, 0.5]), 20000, SplitMix64(4))
    assert list(np.bincount(seq, minlength=3) / 20000) == pytest.approx([0.2, 0.3, 0.5], abs=0.02)


def test_unknown_pattern_and_counter_raise():
    with pytest.raises(ValueError):
        O.loader_sequence("unbekannt", 3, None, 5, SplitMix64(1))
    with pytest.raises(ValueError):
        O.controller_sequence("unbekannt", "low", EQUAL3, None, np.array([0, 1]), SplitMix64(1))


def test_choose_check_ties():
    w = np.array([2.0, 5.0, 5.0, 1.0])
    assert O.choose_check(w, "low", SplitMix64(1)) == 1
    picks = {O.choose_check(w, "random", SplitMix64(s)) for s in range(30)}
    assert picks == {1, 2}                                           # nur die Gleichstände, beide kommen vor
    assert O.choose_check(np.array([1.0, 3.0]), "random", SplitMix64(1)) == 1


def test_follow_counter_catches_the_rotation_from_round_n_plus_1_by_hand():
    """Rotation über drei gleiche Routen, Gleichstand „kleinster Index“. Runden 1 bis 3: noch keine Folge bekannt (Gleichstand → Route 0): Runde 1 zufällig gefangen, 2 und 3 nicht. Ab Runde 4 hat der Zähler
    die Folge 0→1, 1→2, 2→0 gesehen und prüft immer die richtige nächste Route."""
    routes = O.loader_sequence("rotation", 3, None, 9, SplitMix64(1))
    checks = O.controller_sequence("follow", "low", EQUAL3, None, routes, SplitMix64(1))
    assert checks.tolist() == [0, 0, 0, 0, 1, 2, 0, 1, 2]
    caught = routes == checks
    assert caught.tolist() == [True, False, False, True, True, True, True, True, True]
    assert O.first_certain_round(routes, checks) == 4


def test_first_certain_round_for_two_to_eight_routes_is_n_plus_1():
    for n in range(2, 9):
        routes = O.loader_sequence("rotation", n, None, 60, SplitMix64(1))
        checks = O.controller_sequence("follow", "low", [1.0] * n, None, routes, SplitMix64(1))
        assert O.first_certain_round(routes, checks) == n + 1, n


def test_first_certain_round_edge_cases():
    assert O.first_certain_round([0, 1, 2], [0, 1, 2]) == 1
    assert O.first_certain_round([0, 1, 2], [0, 1, 0]) is None
    assert O.first_certain_round([0, 1, 2, 3], [1, 1, 2, 3]) == 2


def test_frequency_counter_with_lowest_index_always_checks_route_zero_against_the_rotation():
    """Gleichstand „kleinster Index“: nach jedem vollen Umlauf stehen alle Zähler gleich, davor führen die niedrigen Routen; der Zähler prüft immer Route 0 und fängt genau jede n-te Sendung."""
    for n in (3, 4):
        routes = O.loader_sequence("rotation", n, None, 12 * n, SplitMix64(1))
        checks = O.controller_sequence("freq", "low", [1.0] * n, None, routes, SplitMix64(1))
        assert set(checks.tolist()) == {0} and int(np.sum(routes == checks)) == 12


def test_random_ties_make_the_frequency_counter_worse_against_the_rotation():
    """Bei zufälligem Gleichstand prüft der Zähler Routen, die im Umlauf schon dran waren: die Lieferung steigt gegenüber dem kleinsten Index (vier gleiche Routen, 1 000 Runden, 4 Zufalls-Seeds)."""
    v, x, y = (1.0,) * 4, np.full(4, 0.25), np.full(4, 0.25)
    low = np.mean([O.simulate("rotation", "freq", "low", v, x, y, 1000, s).mean_delivery for s in range(4)])
    rnd = np.mean([O.simulate("rotation", "freq", "random", v, x, y, 1000, s).mean_delivery for s in range(4)])
    assert low == pytest.approx(0.75, abs=1e-12) and rnd > low + 0.1


def test_window_counter_sees_only_the_last_twenty_routes():
    routes = np.array([0] * 30 + [1] * 5)
    checks = O.controller_sequence("window", "low", (1.0, 1.0), None, routes, SplitMix64(1))
    assert checks[29] == 0 and checks[30] == 0                       # Fenster t−20 … t−1 enthält bei t = 30 nur Route 0
    assert checks[34] == 0                                           # bei t = 34: Fenster hat 16 · Route 0 und 4 · Route 1
    full = O.controller_sequence("freq", "low", (1.0, 1.0), None, np.array([0] * 30 + [1] * 25), SplitMix64(1))
    win = O.controller_sequence("window", "low", (1.0, 1.0), None, np.array([0] * 30 + [1] * 25), SplitMix64(1))
    assert full[-1] == 0 and win[-1] == 1                            # nach 25 Einsen führt im Fenster die Route 1, in der Gesamtzählung (30 : 24) noch die Route 0


def test_values_weight_the_counts():
    routes = np.array([0, 0, 1, 1, 1, 0])
    checks = O.controller_sequence("freq", "low", (1.0, 5.0), None, routes, SplitMix64(1))
    assert checks[5] == 1                                            # Zählerstand 2 · 1 gegen 3 · 5


def test_delivered_is_value_unless_checked():
    assert O.delivered((1.0, 0.5), np.array([0, 1, 1]), np.array([0, 0, 1])).tolist() == [0.0, 0.5, 0.0]


def test_equilibrium_play_is_not_exploitable_by_any_counter():
    """Gegen iid x* liefert jeder Kontrolleur im Mittel höchstens um Rauschen mehr als der Wert (Wert 0,75 bei vier Routen mit dem Wert 1, 4 · 1 000 Runden)."""
    v = (1.0,) * 4
    for counter in O.C.COUNTERS:
        mean = np.mean([O.simulate("iid_eq", counter, "low", v, np.full(4, 0.25), np.full(4, 0.25), 1000, s).mean_delivery for s in range(4)])
        assert mean == pytest.approx(0.75, abs=0.05), counter


def test_simulation_is_deterministic_and_seed_dependent():
    v = (0.9, 0.8, 0.7, 0.6)
    k, x, y, _ = H.closed_form(v)
    a = O.simulate("habit", "follow", "random", v, x, y, 300, 5)
    b = O.simulate("habit", "follow", "random", v, x, y, 300, 5)
    c = O.simulate("habit", "follow", "random", v, x, y, 300, 6)
    assert np.array_equal(a.routes, b.routes) and np.array_equal(a.checks, b.checks) and not np.array_equal(a.routes, c.routes)
    assert a.running_mean[-1] == pytest.approx(a.mean_delivery) and len(a.running_mean) == 300


def test_estimation_against_the_equilibrium_gains_nothing_when_all_routes_pay_the_same():
    """Vier Routen mit dem Wert 1: jede Route zahlt gegen y* = Gleichverteilung dasselbe (1 · ¾ = Wert), jede Antwort ist beste Antwort: Gewinn 0 bei jedem T."""
    v = (1.0,) * 4
    A = H.hide_seek_matrix(v)
    k, x, y, _ = H.closed_form(v)
    for side, opp in (("loader", y), ("controller", x)):
        for T in (1, 5, 50):
            r = O.estimate_and_respond(A, k, side, opp, T, 40, 3)
            assert abs(r["gain"]) < 1e-12 and abs(r["full"]) < 1e-12 and r["share"] is None


def test_estimation_hand_value_for_a_uniform_controller():
    """Werte (0,9; 0,8; 0,7; 0,6): ein gleichverteilt prüfender Kontrolleur lässt Route i den Wert v_i · ¾ (0,675; 0,6; 0,525; 0,45); die beste Antwort (Route 1) liegt um 0,675 − Wert über dem Gleichgewicht.
    Wert = 3c mit c = 1 / (1/0,9 + 1/0,8 + 1/0,7 + 1/0,6) = 0,18327…, also 0,5498…; voller Gewinn 0,1252…"""
    v = (0.9, 0.8, 0.7, 0.6)
    A = H.hide_seek_matrix(v)
    value = 3 / sum(1 / t for t in v)
    full = O.estimate_and_respond(A, value, "loader", np.full(4, 0.25), 1, 1, 1)["full"]
    assert full == pytest.approx(0.675 - value) and full == pytest.approx(0.1252, abs=5e-5)
    big = O.estimate_and_respond(A, value, "loader", np.full(4, 0.25), 1000, 20, 1)
    assert big["gain"] == pytest.approx(full, abs=1e-12) and big["share"] == pytest.approx(1.0)


def test_estimation_gain_is_never_above_the_full_gain():
    v = (0.9, 0.8, 0.7, 0.6)
    A = H.hide_seek_matrix(v)
    sol = L.solve_zero_sum(A)
    for opp in (np.full(4, 0.25), np.array([0.7, 0.1, 0.1, 0.1])):
        for T in (1, 5, 20, 200):
            r = O.estimate_and_respond(A, sol.value, "loader", opp, T, 60, 7)
            assert r["gain"] <= r["full"] + 1e-12
    assert G.saddle_points(A) == []
