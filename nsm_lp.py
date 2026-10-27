"""Lineares Programm des Nullsummenspiels (HiGHS): Primal und Dual getrennt gelöst, Zertifikat ohne Rückgriff auf den Löser, Bereich der optimalen Strategien.

Der Minimax-Satz sagt: max_x min_j (xᵀA)_j = min_y max_i (A y)_i. `solve_zero_sum` löst beide Seiten als eigenes Programm; das Zertifikat prüft, dass x den Wert `lower`
garantiert, y den Wert `upper` hält und der Wert dazwischen liegt. Gibt es mehrere optimale Strategien, sind sie eine Menge: `optimal_ranges` zeigt den Bereich jeder Koordinate."""

from dataclasses import dataclass

import numpy as np

from nsm_game import ceiling, guaranteed

LP_TOL = 1e-9
UNIQUE_TOL = 1e-6         # eine Koordinate gilt als eindeutig, wenn ihr Bereich über die optimale Menge nicht breiter ist


@dataclass(frozen=True)
class Solution:
    value: float           # Wert des primalen Programms (max_x min_j)
    dual_value: float      # Wert des dualen Programms (min_y max_i)
    x: np.ndarray          # eine optimale Strategie des Verladers
    y: np.ndarray          # eine optimale Strategie des Kontrolleurs
    lower: float           # guaranteed(A, x): was x sichert
    upper: float           # ceiling(A, y): was y hält


def _linprog(c, A_ub, b_ub, A_eq, b_eq, bounds):
    from scipy.optimize import linprog

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"Lineares Programm nicht gelöst: {res.message}")
    return res


def _simplex(p):
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def lp_row(A) -> tuple[float, np.ndarray]:
    """max v unter (xᵀA)_j ≥ v für alle j, x in der Wahrscheinlichkeitssimplex: Wert und eine optimale Strategie des Verladers."""
    A = np.asarray(A, float)
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    res = _linprog(c, A_ub, np.zeros(n), A_eq, [1.0], [(0, None)] * m + [(None, None)])
    return float(res.x[-1]), _simplex(res.x[:m])


def lp_col(A) -> tuple[float, np.ndarray]:
    """min w unter (A y)_i ≤ w für alle i, y in der Wahrscheinlichkeitssimplex: Wert und eine optimale Strategie des Kontrolleurs."""
    A = np.asarray(A, float)
    m, n = A.shape
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub = np.hstack([A, -np.ones((m, 1))])
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    res = _linprog(c, A_ub, np.zeros(m), A_eq, [1.0], [(0, None)] * n + [(None, None)])
    return float(res.x[-1]), _simplex(res.x[:n])


def solve_zero_sum(A) -> Solution:
    """Beide Programme lösen und das Zertifikat bilden."""
    A = np.asarray(A, float)
    value, x = lp_row(A)
    dual_value, y = lp_col(A)
    return Solution(value, dual_value, x, y, guaranteed(A, x), ceiling(A, y))


def optimal_ranges(A, value: float, tol: float = LP_TOL):
    """Bereich jeder Koordinate über alle optimalen Strategien: (x_lo, x_hi, y_lo, y_hi).

    Verlader: Menge {x in Simplex : (xᵀA)_j ≥ value − tol für alle j}; Kontrolleur: {y in Simplex : (A y)_i ≤ value + tol für alle i}. Je Koordinate ein Minimum- und ein Maximum-Programm."""
    A = np.asarray(A, float)
    m, n = A.shape
    out = []
    for size, A_ub, b_ub in ((m, -A.T, np.full(n, -(value - tol))), (n, A, np.full(m, value + tol))):
        A_eq = np.ones((1, size))
        lo, hi = np.zeros(size), np.zeros(size)
        for k in range(size):
            c = np.zeros(size)
            c[k] = 1.0
            lo[k] = _linprog(c, A_ub, b_ub, A_eq, [1.0], [(0, None)] * size).fun
            hi[k] = -_linprog(-c, A_ub, b_ub, A_eq, [1.0], [(0, None)] * size).fun
        out += [lo, hi]
    return tuple(out)


def is_unique(lo, hi, tol: float = UNIQUE_TOL) -> bool:
    """Ist die optimale Strategie eindeutig? Genau dann, wenn kein Koordinatenbereich breiter als tol ist."""
    return bool(np.max(np.asarray(hi) - np.asarray(lo)) <= tol)
