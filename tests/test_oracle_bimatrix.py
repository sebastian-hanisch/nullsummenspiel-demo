"""Orakel für die allgemeinen Spiele: Stützenaufzählung gegen exakte Brüche, Parität, Best-Response-Prüfung, Nullsumme gegen das lineare Programm, Lemke–Howson gegen Aufzählung und `is_nash`.

Handbeispiel (Kampf der Geschlechter): A = [[2, 0], [0, 1]], B = [[1, 0], [0, 2]]. Gleichgewichte: (Zeile 1, Spalte 1), (Zeile 2, Spalte 2) und gemischt x = (2/3, 1/3), y = (1/3, 2/3)
(Zeile indifferent: 2 y₁ = y₂ → y₁ = 1/3; Spalte indifferent: x₁ = 2 x₂ → x₁ = 2/3)."""

from fractions import Fraction as F

import numpy as np
import pytest

import nsm_bimatrix as Bm
import nsm_lp as L
from exact_games import nash_equilibria
from nsm_rng import SplitMix64
from nsm_scenario import draw_bimatrix

BOS_A = np.array([[2.0, 0.0], [0.0, 1.0]])
BOS_B = np.array([[1.0, 0.0], [0.0, 2.0]])


def _sorted(profiles):
    return sorted((tuple(np.round(np.asarray(x, float), 6)), tuple(np.round(np.asarray(y, float), 6))) for x, y in profiles)


def _int_game(m, n, seed, top=99):
    rng = SplitMix64(seed)
    return [[rng.below(top + 1) for _ in range(n)] for _ in range(m)], [[rng.below(top + 1) for _ in range(n)] for _ in range(m)]


def test_battle_of_the_sexes_has_the_three_hand_equilibria():
    eq = Bm.enumerate_nash(BOS_A, BOS_B)
    assert _sorted(eq) == _sorted([((1, 0), (1, 0)), ((0, 1), (0, 1)), ((2 / 3, 1 / 3), (1 / 3, 2 / 3))])
    assert all(Bm.is_nash(BOS_A, BOS_B, x, y) for x, y in eq) and Bm.count_pure(eq) == 2


def test_lemke_howson_on_the_battle_of_the_sexes_by_hand():
    """Handrechnung (Tableau): Start Label 0 → x₁ tritt ein, Label 2 verlässt, y₁ tritt ein, Label 0 verlässt: Pfad [0, 2], 2 Pivots, Gleichgewicht (Zeile 1, Spalte 1). Label 1 → [1, 3] → (Zeile 2, Spalte 2),
    Label 2 → [2, 0] → (Zeile 1, Spalte 1), Label 3 → [3, 1] → (Zeile 2, Spalte 2). Das gemischte Gleichgewicht findet kein Label."""
    expected = {0: ([0, 2], (1, 0)), 1: ([1, 3], (0, 1)), 2: ([2, 0], (1, 0)), 3: ([3, 1], (0, 1))}
    for label, (path, pure) in expected.items():
        x, y, pivots, found_path = Bm.lemke_howson(BOS_A, BOS_B, label)
        assert found_path == path and pivots == 2, label
        assert list(x) == pytest.approx(pure) and list(y) == pytest.approx(pure)


def test_enumeration_matches_the_exact_fraction_oracle_on_integer_games():
    for t, (m, n) in enumerate([(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 4)] * 4):
        A, B = _int_game(m, n, 300 + t)
        exact = [(tuple(float(p) for p in x), tuple(float(p) for p in y)) for x, y in nash_equilibria(A, B)]
        mine = Bm.enumerate_nash(np.array(A, float), np.array(B, float))
        assert _sorted(mine) == _sorted(exact), (A, B)


def test_exact_oracle_reproduces_the_hand_example():
    eq = nash_equilibria([[2, 0], [0, 1]], [[1, 0], [0, 2]])
    assert len(eq) == 3 and ((F(2, 3), F(1, 3)), (F(1, 3), F(2, 3))) in eq


def test_number_of_equilibria_is_odd_for_random_games():
    """Nicht entartete Bimatrixspiele haben eine ungerade Zahl von Gleichgewichten (Wilson). 120 Zufallsspiele n = 2 bis 6, A und B unabhängig."""
    counts = []
    for t in range(120):
        n = 2 + t % 5
        A, B = draw_bimatrix(n, n, 5000 + t)
        counts.append(len(Bm.enumerate_nash(A, B)))
    assert all(c % 2 == 1 for c in counts), counts
    assert max(counts) >= 3 and min(counts) >= 1


def test_every_enumerated_profile_passes_the_best_response_check():
    for t in range(40):
        m, n = 2 + t % 4, 2 + (t // 4) % 4
        A, B = draw_bimatrix(m, n, 6000 + t)
        for x, y in Bm.enumerate_nash(A, B):
            assert Bm.is_nash(A, B, x, y, tol=1e-8) and Bm.nash_gap(A, B, x, y) <= 1e-8
            assert x.sum() == pytest.approx(1.0) and y.sum() == pytest.approx(1.0)


def test_zero_sum_games_have_exactly_one_equilibrium_and_it_pays_the_lp_value():
    for t in range(60):
        n = 2 + t % 4
        A, _B = draw_bimatrix(n, n, 7000 + t)
        A, B = Bm.zero_sum_partner(A)
        eq = Bm.enumerate_nash(A, B)
        assert len(eq) == 1, (t, len(eq))
        assert eq[0][0] @ A @ eq[0][1] == pytest.approx(L.solve_zero_sum(A).value, abs=1e-9)


def test_lemke_howson_returns_an_equilibrium_from_every_label_and_it_is_in_the_enumeration():
    covered = set()
    for t in range(40):
        m, n = 2 + t % 4, 2 + (t // 3) % 4
        A, B = draw_bimatrix(m, n, 8000 + t)
        eq = Bm.enumerate_nash(A, B)
        for label in range(m + n):
            x, y, pivots, path = Bm.lemke_howson(A, B, label)
            assert Bm.is_nash(A, B, x, y, tol=1e-8) and pivots >= 1 and path[0] == label and len(path) == pivots
            hits = [i for i, (xe, ye) in enumerate(eq) if Bm.same_profile(x, y, xe, ye)]
            assert len(hits) == 1, (t, label)
            covered.add((t, hits[0]))
    assert len(covered) >= 40


def test_lemke_howson_does_not_find_every_equilibrium_from_every_label():
    """Es gibt Spiele, in denen mehrere Gleichgewichte existieren, die kein Label erreicht (Kampf der Geschlechter: das gemischte)."""
    missed = 0
    for label in range(4):
        x, y, _p, _path = Bm.lemke_howson(BOS_A, BOS_B, label)
        missed += not Bm.same_profile(x, y, [2 / 3, 1 / 3], [1 / 3, 2 / 3])
    assert missed == 4


def test_shapley_game_has_the_uniform_profile_as_the_only_equilibrium_and_lemke_howson_finds_it_from_all_six_labels():
    A, B = Bm.shapley_game()
    assert A.tolist() == np.eye(3).tolist() and B.tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    eq = Bm.enumerate_nash(A, B)
    assert len(eq) == 1 and list(eq[0][0]) == pytest.approx([1 / 3] * 3) and list(eq[0][1]) == pytest.approx([1 / 3] * 3)
    for label in range(6):
        x, y, _p, _path = Bm.lemke_howson(A, B, label)
        assert list(x) == pytest.approx([1 / 3] * 3) and list(y) == pytest.approx([1 / 3] * 3)
    assert len(nash_equilibria(A.astype(int).tolist(), B.astype(int).tolist())) == 1          # exakt, mit Brüchen


def test_nash_gap_and_distance_hand_values():
    assert Bm.nash_gap(BOS_A, BOS_B, np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0               # Zeile: A y = (0, 1), Zeile 1 bringt 0, Zeile 2 brächte 1; Spalte: x B = (1, 0), Spalte 2 bringt 0, Spalte 1 brächte 1
    assert Bm.nash_gap(BOS_A, BOS_B, np.array([2 / 3, 1 / 3]), np.array([1 / 3, 2 / 3])) == pytest.approx(0.0, abs=1e-12)
    assert Bm.distance_to_uniform([1, 0, 0]) == pytest.approx(2 / 3) and Bm.distance_to_uniform([5, 5]) == 0.0


def test_support_pairs_formula():
    from math import comb

    assert [Bm.support_pairs(n) for n in (2, 3, 10)] == [5, 19, 184755]
    for n in range(1, 9):
        assert Bm.support_pairs(n) == sum(comb(n, k) ** 2 for k in range(1, n + 1))             # Summe der Paare gleich großer, nicht leerer Stützen


# --- Handwerte der Hilfsfunktionen (Mutationsprüfung: jede Teilbedingung einzeln) ----------------------------------------------------------------

def test_is_nash_needs_both_sides():
    """Kampf der Geschlechter. (Zeile 1, y = (½, ½)): Zeile 1 ist beste Antwort (A y = (1; 0,5)), aber die Spalte mischt, obwohl x B = (1, 0) Spalte 1 allein belohnt: kein Gleichgewicht.
    (x = (½, ½), Spalte 1): die Spalte ist beste Antwort, aber die Zeile mischt bei A y = (2, 0): kein Gleichgewicht."""
    assert not Bm.is_nash(BOS_A, BOS_B, np.array([1.0, 0.0]), np.array([0.5, 0.5]))
    assert not Bm.is_nash(BOS_A, BOS_B, np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert Bm.is_nash(BOS_A, BOS_B, np.array([1.0, 0.0]), np.array([1.0, 0.0]))


def test_nash_gap_takes_the_larger_of_the_two_gaps():
    """x = (1, 0), y = (½, ½): Zeilenlücke max(A y) − xᵀAy = 1 − 1 = 0; Spaltenlücke max(xᵀB) − xᵀBy = 1 − 0,5 = 0,5."""
    assert Bm.nash_gap(BOS_A, BOS_B, np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert Bm.nash_gap(BOS_A, BOS_B, np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(1.0)       # Zeilenlücke 2 − 1 = 1, Spaltenlücke 0 − ... = 0


def test_same_profile_needs_both_strategies_to_agree():
    assert Bm.same_profile([1, 0], [0, 1], [1, 0], [0, 1]) and not Bm.same_profile([1, 0], [0, 1], [1, 0], [1, 0]) and not Bm.same_profile([1, 0], [0, 1], [0, 1], [0, 1])
    assert Bm.same_profile([0.5, 0.5], [0.5, 0.5], [0.5 + 1e-8, 0.5 - 1e-8], [0.5, 0.5]) and not Bm.same_profile([0.5, 0.5], [0.5, 0.5], [0.5 + 1e-3, 0.5 - 1e-3], [0.5, 0.5])


def test_count_pure_needs_both_supports_of_size_one():
    eq = [(np.array([1.0, 0.0]), np.array([0.5, 0.5])), (np.array([0.5, 0.5]), np.array([1.0, 0.0])), (np.array([1.0, 0.0]), np.array([0.0, 1.0])), (np.array([0.5, 0.5]), np.array([0.5, 0.5]))]
    assert Bm.count_pure(eq) == 1


def test_lemke_howson_handles_negative_payoffs_by_shifting():
    """Stein-Schere-Papier als Bimatrixspiel (B = −A): die Auszahlungen sind negativ und werden für das Tableau auf positive verschoben; jedes Label führt zur Gleichverteilung."""
    A = np.array([[0.0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    for label in range(6):
        x, y, pivots, _path = Bm.lemke_howson(A, -A, label)
        assert list(x) == pytest.approx([1 / 3] * 3) and list(y) == pytest.approx([1 / 3] * 3) and pivots >= 1


def test_regret_matching_in_a_bimatrix_game_by_hand():
    """Kampf der Geschlechter, Start gleichverteilt. Runde 1: A y = (1; 0,5), im Mittel 0,75, Regrets der Zeile (0,25; −0,25); xᵀB = (0,5; 1), im Mittel 0,75, Regrets der Spalte (−0,25; 0,25) → x₂ = (1, 0), y₂ = (0, 1).
    Runde 2: A y = (0, 1) gegen x = (1, 0): Mittel 0, Regrets +(0; 1) → (0,25; 0,75); xᵀB = (1, 0) gegen y = (0, 1): Mittel 0, Regrets +(1; 0) → (0,75; 0,25). Nach zwei Runden: Durchschnitt x = (0,75; 0,25), y = (0,25; 0,75),
    Nash-Lücke 0,1875 (Zeile: A y = (0,5; 0,75), Mittel 0,5625; Spalte: xᵀB = (0,75; 0,5), Mittel 0,5625 → beide 0,1875); letzte Strategien (1, 0) und (0, 1): Lücke 1."""
    out = Bm.regret_matching_bimatrix(BOS_A, BOS_B, 2)
    assert list(out["avg_x"]) == pytest.approx([0.75, 0.25]) and list(out["avg_y"]) == pytest.approx([0.25, 0.75])
    assert out["gap_avg"] == pytest.approx(0.1875) and out["gap_last"] == pytest.approx(1.0)
    one = Bm.regret_matching_bimatrix(BOS_A, BOS_B, 1)
    assert list(one["avg_x"]) == pytest.approx([0.5, 0.5]) and one["gap_last"] == pytest.approx(Bm.nash_gap(BOS_A, BOS_B, np.array([0.5, 0.5]), np.array([0.5, 0.5])))


def _reference_fp(A, B, rounds):
    """Fictitious Play im Bimatrixspiel mit reinen Schleifen: Abstand je Runde und Zahl der Wechsel des Zeilenspielers."""
    m, n = len(A), len(A[0])
    cx, cy, a, b = [0] * m, [0] * n, 0, 0
    dist, switches = [], 0
    for t in range(1, rounds + 1):
        cx[a] += 1
        cy[b] += 1
        tv = lambda c: 0.5 * sum(abs(v / sum(c) - 1 / len(c)) for v in c)           # noqa: E731
        dist.append(max(tv(cx), tv(cy)))
        rows = [sum(A[i][j] * cy[j] for j in range(n)) for i in range(m)]
        cols = [sum(cx[i] * B[i][j] for i in range(m)) for j in range(n)]
        a2 = next(i for i in range(m) if rows[i] >= max(rows) - 1e-12)
        b2 = next(j for j in range(n) if cols[j] >= max(cols) - 1e-12)
        switches += a2 != a
        a, b = a2, b2
    return dist, switches


def test_fictitious_play_in_a_bimatrix_game_matches_the_loop_reference():
    """Hand: Shapley-Beispiel, Runde 1: beide Häufigkeiten (1, 0, 0): Abstand 2/3 für beide; Runde 2: Zeile (1, 0, 0) → 2/3, Spalte (½, ½, 0) → 1/3: der größere zählt (2/3)."""
    A, B = Bm.shapley_game()
    out = Bm.fictitious_play_bimatrix(A, B, 3)
    assert out["dist"][0] == pytest.approx(2 / 3) and out["dist"][1] == pytest.approx(2 / 3)
    for game_A, game_B, rounds in ((A, B, 400), (BOS_A, BOS_B, 50), *[(*draw_bimatrix(3, 4, 40 + t), 300) for t in range(4)]):
        out = Bm.fictitious_play_bimatrix(game_A, game_B, rounds)
        dist, switches = _reference_fp(np.asarray(game_A).tolist(), np.asarray(game_B).tolist(), rounds)
        assert list(out["dist"]) == pytest.approx(dist, abs=1e-12) and out["switches"] == switches
        assert out["freq_x"].sum() == pytest.approx(1.0) and out["freq_y"].sum() == pytest.approx(1.0)
