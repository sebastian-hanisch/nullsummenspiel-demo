"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, Spieltheorie): Aussagen über das Verfahren, die für jede Eingabe gelten müssen; je Eigenschaft 20 bis 50 feste Zufallsspiele.

Jede Eigenschaft ist eine belegte Aussage über Nullsummenspiele, nicht das heutige Verhalten des Codes: Umbenennen der Aktionen, positive affine Umrechnung der Auszahlungen, zusätzliche
dominierte Aktion, zusätzliche Aktion kann den Wert nur zugunsten ihres Spielers ändern, Rollentausch durch Transponieren und Negieren, Monotonie der Routenwerte im Verstecken-Spiel."""

import numpy as np
import pytest

import nsm_game as G
import nsm_hideseek as H
import nsm_lp as L
from nsm_rng import SplitMix64
from nsm_scenario import draw_matrix, uniform01


def _games(count=30):
    for t in range(count):
        yield t, draw_matrix(2 + t % 4, 2 + (t * 3) % 4, 15000 + t)


def test_permuting_rows_and_columns_changes_only_the_names():
    for t, A in _games():
        rng = SplitMix64(t)
        rows = sorted(range(A.shape[0]), key=lambda _i: rng.next())
        cols = sorted(range(A.shape[1]), key=lambda _j: rng.next())
        B = A[np.ix_(rows, cols)]
        sol, solB = L.solve_zero_sum(A), L.solve_zero_sum(B)
        assert solB.value == pytest.approx(sol.value, abs=1e-9)
        assert G.guaranteed(A, np.eye(A.shape[0])[rows].T @ solB.x) == pytest.approx(sol.value, abs=1e-8)            # die zurückübersetzte Strategie ist auch in A optimal
        assert len(G.saddle_points(A)) == len(G.saddle_points(B))


def test_positive_affine_maps_keep_the_optimal_strategies_and_transform_the_value():
    for t, A in _games():
        sol = L.solve_zero_sum(A)
        B = 2.5 * A + 3.0
        solB = L.solve_zero_sum(B)
        assert solB.value == pytest.approx(2.5 * sol.value + 3.0, abs=1e-8)
        assert G.guaranteed(B, sol.x) == pytest.approx(solB.value, abs=1e-8) and G.ceiling(B, sol.y) == pytest.approx(solB.value, abs=1e-8)     # dieselben Strategien bleiben optimal


def test_role_swap_by_transposing_and_negating():
    for t, A in _games():
        sol = L.solve_zero_sum(A)
        swapped = L.solve_zero_sum(-A.T)
        assert swapped.value == pytest.approx(-sol.value, abs=1e-8)
        assert G.guaranteed(-A.T, sol.y) == pytest.approx(-sol.value, abs=1e-8)                    # y ist als Zeilenstrategie des getauschten Spiels optimal


def test_an_added_row_cannot_hurt_the_row_player_and_an_added_column_cannot_hurt_the_column_player():
    for t, A in _games():
        rng = SplitMix64(900 + t)
        v = L.solve_zero_sum(A).value
        row = np.array([[uniform01(rng) for _ in range(A.shape[1])]])
        col = np.array([[uniform01(rng)] for _ in range(A.shape[0])])
        assert L.solve_zero_sum(np.vstack([A, row])).value >= v - 1e-9
        assert L.solve_zero_sum(np.hstack([A, col])).value <= v + 1e-9


def test_a_dominated_row_does_not_change_the_value_and_gets_probability_zero():
    for t, A in _games():
        dominated = A.min(axis=0, keepdims=True) - 0.5 * np.ones((1, A.shape[1]))              # in jeder Spalte schlechter als jede vorhandene Zeile
        B = np.vstack([A, dominated])
        sol = L.solve_zero_sum(B)
        assert sol.value == pytest.approx(L.solve_zero_sum(A).value, abs=1e-9) and sol.x[-1] == pytest.approx(0.0, abs=1e-9)


def test_duplicating_a_row_keeps_the_value():
    for t, A in _games():
        B = np.vstack([A, A[:1]])
        assert L.solve_zero_sum(B).value == pytest.approx(L.solve_zero_sum(A).value, abs=1e-9)


def test_raising_a_route_value_never_lowers_the_value_of_hide_and_seek_and_never_raises_it_above_the_better_route():
    for t in range(40):
        rng = SplitMix64(33 + t)
        n = 2 + t % 5
        v = [0.2 + 0.8 * uniform01(rng) for _ in range(n)]
        k = H.closed_form(v)[0]
        i = rng.below(n)
        w = list(v)
        w[i] = min(1.0, v[i] + 0.1)
        assert H.closed_form(w)[0] >= k - 1e-12
        assert k <= (n - 1) / n * max(v) + 1e-12 and k >= 0


def test_value_of_hide_and_seek_is_symmetric_in_the_routes():
    for t in range(30):
        rng = SplitMix64(77 + t)
        n = 2 + t % 6
        v = [0.1 + 0.9 * uniform01(rng) for _ in range(n)]
        perm = sorted(range(n), key=lambda _i: rng.next())
        assert H.closed_form([v[i] for i in perm])[0] == pytest.approx(H.closed_form(v)[0], abs=1e-12)


def test_value_scales_linearly_with_the_route_values():
    for t in range(30):
        rng = SplitMix64(99 + t)
        n = 2 + t % 6
        v = np.array([0.1 + 0.4 * uniform01(rng) for _ in range(n)])
        k, x, y, active = H.closed_form(v)
        k2, x2, y2, active2 = H.closed_form(2 * v)
        assert k2 == pytest.approx(2 * k, rel=1e-12) and active2 == active and list(x2) == pytest.approx(list(x)) and list(y2) == pytest.approx(list(y))
