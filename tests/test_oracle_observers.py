"""Orakel für die Beobachter: Zähler und Lieferung gegen eine zweite Umsetzung mit reinen Python-Listen und Wörterbüchern (kein NumPy in der Rechnung), und das Schätzen gegen eine Schleifenfassung, die dieselben
Zufallszahlen der Reihe nach verbraucht."""

import numpy as np
import pytest

import nsm_constants as C
import nsm_hideseek as H
import nsm_observers as O
from nsm_learning import first_best
from nsm_rng import SplitMix64
from nsm_scenario import uniform01


def _reference_checks(counter, tie, v, routes):
    """Prüfroute je Runde nach der Definition: größter (Zählerstand · Wert), Gleichstand: kleinster Index. Zähler: alle bisherigen Routen, die letzten 20 oder die Nachfolger der letzten Route."""
    n = len(v)
    out = []
    candidates = []
    for t in range(len(routes)):
        if counter == "freq":
            counts = [sum(1 for r in routes[:t] if r == j) for j in range(n)]
        elif counter == "window":
            counts = [sum(1 for r in routes[max(0, t - 20):t] if r == j) for j in range(n)]
        elif counter == "follow":
            last = routes[t - 1] if t > 0 else None
            counts = [sum(1 for s in range(1, t) if routes[s - 1] == last and routes[s] == j) if last is not None else 0 for j in range(n)]
        weights = [c * w for c, w in zip(counts, v)]
        top = max(weights)
        best = [j for j in range(n) if weights[j] >= top - 1e-12]
        candidates.append(best)
        out.append(best[0])
    return out, candidates


def _random_routes(n, length, seed):
    rng = SplitMix64(seed)
    return [rng.below(n) for _ in range(length)]


@pytest.mark.parametrize("counter", ["freq", "window", "follow"])
def test_counters_with_the_lowest_index_equal_the_reference(counter):
    for t in range(12):
        n = 2 + t % 5
        rng = SplitMix64(500 + t)
        v = [round(0.3 + 0.7 * uniform01(rng), 2) for _ in range(n)]
        routes = _random_routes(n, 60 + 7 * t, 700 + t)
        expected, _cands = _reference_checks(counter, "low", v, routes)
        assert O.controller_sequence(counter, "low", v, None, np.array(routes), SplitMix64(1)).tolist() == expected, (counter, t)


@pytest.mark.parametrize("counter", ["freq", "window", "follow"])
def test_counters_with_random_ties_choose_only_among_the_best_routes_and_use_all_of_them(counter):
    chosen_other_than_first = 0
    for t in range(10):
        n = 3 + t % 4
        routes = [i % n for i in range(80)] if t % 2 == 0 else _random_routes(n, 80, 900 + t)
        v = [1.0] * n
        _exp, cands = _reference_checks(counter, "random", v, routes)
        got = O.controller_sequence(counter, "random", v, None, np.array(routes), SplitMix64(40 + t)).tolist()
        assert all(g in c for g, c in zip(got, cands)), (counter, t)
        chosen_other_than_first += sum(g != c[0] for g, c in zip(got, cands))
    assert chosen_other_than_first > 20                                                                       # die Zufallswahl weicht wirklich vom kleinsten Index ab


def test_delivery_follows_the_definition():
    for t in range(10):
        n = 2 + t % 5
        v = [round(0.2 + 0.8 * uniform01(SplitMix64(t * 7 + k)), 2) for k in range(n)]
        routes = _random_routes(n, 50, 1100 + t)
        checks = _random_routes(n, 50, 1200 + t)
        got = O.delivered(v, np.array(routes), np.array(checks))
        assert list(got) == [0.0 if r == c else v[r] for r, c in zip(routes, checks)]


def test_simulate_equals_the_manual_pipeline_and_its_summaries():
    v = (0.9, 0.8, 0.7, 0.6)
    k, x, y, _a = H.closed_form(v)
    for pattern in C.PATTERNS:
        for counter in C.COUNTERS:
            obs = O.simulate(pattern, counter, "low", v, x, y, 200, 9)
            assert obs.delivered.tolist() == [0.0 if r == c else v[r] for r, c in zip(obs.routes, obs.checks)]
            assert obs.mean_delivery == pytest.approx(sum(obs.delivered) / 200) and obs.caught_share == pytest.approx(sum(r == c for r, c in zip(obs.routes, obs.checks)) / 200)
            assert list(obs.running_mean) == pytest.approx([sum(obs.delivered[:i]) / i for i in range(1, 201)])


def test_rotation_catches_by_the_follow_counter_for_every_route_count_and_value_weights_do_not_matter():
    for n in range(2, 9):
        v = [0.2 + 0.1 * j for j in range(n)]
        routes = np.arange(60) % n
        checks = O.controller_sequence("follow", "low", v, None, routes, SplitMix64(1))
        assert O.first_certain_round(routes, checks) == n + 1
        assert np.all(routes[n:] == checks[n:])


def _reference_estimate(A, value, side, opponent, observed, samples, seed):
    A = np.asarray(A, float)
    rng = SplitMix64(seed)
    obj = A if side == "loader" else -A.T
    payoff = [sum(obj[i][j] * opponent[j] for j in range(len(opponent))) for i in range(len(obj))]
    total = 0.0
    cdf = np.cumsum(opponent)
    for _ in range(samples):
        counts = [0] * len(opponent)
        for _ in range(observed):
            u = uniform01(rng)
            idx = next((j for j in range(len(opponent)) if cdf[j] > u), len(opponent) - 1)
            counts[idx] += 1
        scores = [sum(obj[i][j] * counts[j] for j in range(len(opponent))) for i in range(len(obj))]
        total += payoff[first_best(np.array(scores))]
    ref = value if side == "loader" else -value
    return total / samples - ref, max(payoff) - ref


def test_estimation_equals_the_loop_reference_including_the_random_stream():
    v = (0.9, 0.8, 0.7, 0.6)
    A = H.hide_seek_matrix(v)
    k, x, y, _a = H.closed_form(v)
    for side, opp in (("loader", [0.25] * 4), ("controller", [0.7, 0.1, 0.1, 0.1]), ("loader", y), ("controller", x)):
        for observed in (1, 5, 30):
            got = O.estimate_and_respond(A, k, side, opp, observed, 25, 77)
            gain, full = _reference_estimate(A, k, side, np.array(opp), observed, 25, 77)
            assert got["gain"] == pytest.approx(gain, abs=1e-12) and got["full"] == pytest.approx(full, abs=1e-12), (side, observed)
            assert (got["share"] is None) == (full <= 1e-9) and (got["share"] is None or got["share"] == pytest.approx(gain / full))


def test_loader_patterns_are_reproducible_and_depend_on_the_seed():
    for pattern in ("habit", "norepeat", "iid_eq"):
        a = O.loader_sequence(pattern, 4, np.full(4, 0.25), 100, SplitMix64(3))
        b = O.loader_sequence(pattern, 4, np.full(4, 0.25), 100, SplitMix64(3))
        c = O.loader_sequence(pattern, 4, np.full(4, 0.25), 100, SplitMix64(4))
        assert np.array_equal(a, b) and not np.array_equal(a, c) and a.min() >= 0 and a.max() < 4


def test_estimation_clamps_draws_that_fall_beyond_a_distribution_summing_to_less_than_one():
    """Wie `draw_action`: liegt die Zahl über der letzten kumulierten Wahrscheinlichkeit (hier 0,6), gilt die letzte Aktion; das Ergebnis gleicht der Schleifenfassung."""
    A = np.array([[0.0, 2.0], [1.0, 0.0]])
    got = O.estimate_and_respond(A, 0.5, "loader", [0.3, 0.3], 40, 6, 3)
    gain, full = _reference_estimate(A, 0.5, "loader", np.array([0.3, 0.3]), 40, 6, 3)
    assert got["gain"] == pytest.approx(gain, abs=1e-12) and got["full"] == pytest.approx(full, abs=1e-12)
