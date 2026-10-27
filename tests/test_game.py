"""Die Einheiten von nsm_game an den Mini-Instanzen (Handrechnung in conftest.py) und an Randfällen."""

import numpy as np
import pytest

import nsm_game as G


def test_pure_security_and_ceiling_on_the_mixed_game(mixed2):
    A = mixed2["A"]
    assert G.pure_security(A) == mixed2["security"] == 2.0          # Zeilenminima 1 und 2
    assert G.pure_ceiling(A) == mixed2["ceiling"] == 3.0            # Spaltenmaxima 3 und 4
    assert G.saddle_points(A) == []


def test_saddle_point_of_the_saddle_game(saddle):
    assert G.saddle_points(saddle["A"]) == [saddle["point"]]
    assert G.pure_security(saddle["A"]) == G.pure_ceiling(saddle["A"]) == 3.0


def test_flat_game_has_four_saddle_points(flat2):
    assert G.saddle_points(flat2["A"]) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_rock_paper_scissors_has_no_saddle_point_and_negative_security(rps):
    assert G.saddle_points(rps["A"]) == []
    assert G.pure_security(rps["A"]) == -1.0 and G.pure_ceiling(rps["A"]) == 1.0


def test_guaranteed_and_ceiling_at_the_equilibrium_of_the_mixed_game(mixed2):
    A = mixed2["A"]
    x = np.array([0.5, 0.5])
    y = np.array([0.75, 0.25])
    assert G.guaranteed(A, x) == pytest.approx(2.5)                 # Spalten: 0,5 · 3 + 0,5 · 2 = 2,5 und 0,5 · 1 + 0,5 · 4 = 2,5
    assert G.ceiling(A, y) == pytest.approx(2.5)                    # Zeilen: 2,5 und 2,5
    assert G.exploitability(A, x, y) == pytest.approx(0.0, abs=1e-12)


def test_exploitability_of_a_pure_pair_is_the_sum_of_both_gaps(mixed2):
    A = mixed2["A"]
    x, y = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    # x = Zeile 1 sichert min(3, 1) = 1; y = Spalte 1 lässt max(3, 2) = 3 zu: Lücken 2,5 − 1 = 1,5 und 3 − 2,5 = 0,5, Summe 2
    assert G.guaranteed(A, x) == 1.0 and G.ceiling(A, y) == 3.0
    assert G.exploitability(A, x, y) == 2.0


def test_best_responses_report_all_ties(mixed2):
    A = mixed2["A"]
    assert G.best_rows(A, np.array([0.75, 0.25])) == [0, 1]         # beide Zeilen 2,5
    assert G.best_rows(A, np.array([1.0, 0.0])) == [0]              # Spalte 1: 3 gegen 2
    assert G.best_cols(A, np.array([0.5, 0.5])) == [0, 1]
    assert G.best_cols(A, np.array([1.0, 0.0])) == [1]              # Zeile 1: Spalte 1 = 3, Spalte 2 = 1


def test_support_counts_above_the_noise_threshold():
    assert G.support([0.5, 0.5, 0.0]) == 2
    assert G.support([1.0, 1e-12, 0.0]) == 1                        # Rauschen zählt nicht
    assert G.support([1e-9, 0.2], tol=1e-10) == 2
