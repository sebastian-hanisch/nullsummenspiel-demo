"""Matrixspiele mit Nullsumme: Sicherheitsniveau, Sattelpunkte, garantierter Wert, Ausnutzbarkeit und beste Antworten.

Konvention: A hat m Zeilen (Verlader, maximiert) und n Spalten (Kontrolleur, minimiert); der Eintrag A[i, j] ist die Auszahlung an den Verlader. Eine Strategie ist ein
Vektor von Wahrscheinlichkeiten (Summe 1). Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import numpy as np

TOL = 1e-9


def pure_security(A) -> float:
    """Sicherheitsniveau der besten reinen Zeile: max_i min_j A[i, j] (was der Verlader ohne Zufall garantiert)."""
    return float(np.max(np.min(np.asarray(A, float), axis=1)))


def pure_ceiling(A) -> float:
    """Was die beste reine Spalte dem Verlader höchstens lässt: min_j max_i A[i, j]."""
    return float(np.min(np.max(np.asarray(A, float), axis=0)))


def saddle_points(A, tol: float = 1e-12) -> list[tuple[int, int]]:
    """Alle Sattelpunkte (i, j): A[i, j] ist das Minimum seiner Zeile und das Maximum seiner Spalte."""
    A = np.asarray(A, float)
    row_min = A.min(axis=1, keepdims=True)
    col_max = A.max(axis=0, keepdims=True)
    hits = (np.abs(A - row_min) <= tol) & (np.abs(A - col_max) <= tol)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(hits))]


def guaranteed(A, x) -> float:
    """Was die Strategie x des Verladers gegen die beste Antwort des Kontrolleurs sichert: min_j (xᵀA)_j."""
    return float(np.min(np.asarray(x, float) @ np.asarray(A, float)))


def ceiling(A, y) -> float:
    """Was die Strategie y des Kontrolleurs dem Verlader höchstens lässt: max_i (A y)_i."""
    return float(np.max(np.asarray(A, float) @ np.asarray(y, float)))


def exploitability(A, x, y) -> float:
    """Ausnutzbarkeit des Paares (x, y): ceiling(A, y) − guaranteed(A, x); null genau im Gleichgewicht, sonst größer (Summe beider Lücken)."""
    return ceiling(A, y) - guaranteed(A, x)


def best_rows(A, y, tol: float = TOL) -> list[int]:
    """Alle besten Antworten des Verladers auf y (Zeilen mit dem größten (A y)_i, bis auf tol)."""
    u = np.asarray(A, float) @ np.asarray(y, float)
    return [int(i) for i in np.nonzero(u >= u.max() - tol)[0]]


def best_cols(A, x, tol: float = TOL) -> list[int]:
    """Alle besten Antworten des Kontrolleurs auf x (Spalten mit dem kleinsten (xᵀA)_j, bis auf tol)."""
    u = np.asarray(x, float) @ np.asarray(A, float)
    return [int(j) for j in np.nonzero(u <= u.min() + tol)[0]]


def support(p, tol: float = TOL) -> int:
    """Zahl der Aktionen mit Wahrscheinlichkeit über tol (Rauschgrenze, DEMO-PLAYBOOK §4)."""
    return int(np.sum(np.asarray(p, float) > tol))
