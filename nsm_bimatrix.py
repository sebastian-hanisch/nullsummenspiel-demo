"""Allgemeine Zwei-Personen-Spiele (Bimatrixspiele): Nash-Gleichgewichte durch Stützenaufzählung, Lemke–Howson-Pfade, das Shapley-Beispiel und Lernverfahren, die dort kreisen.

A ist die Auszahlungsmatrix des Zeilenspielers (m × n), B die des Spaltenspielers. Nullsumme ist der Sonderfall B = −A (oder B = c − A). Die Aufzählung prüft alle Paare gleich großer Stützen;
das ist für nicht entartete Spiele vollständig (dort haben alle Gleichgewichte gleich große Stützen) und für Zufallsmatrizen mit Wahrscheinlichkeit 1 anwendbar. Lemke–Howson folgt dem Pfad
vom Anfangslabel zum nächsten Gleichgewicht (Tableauumsetzung, positive Matrizen durch Verschieben)."""

from itertools import combinations
from math import comb

import numpy as np

from nsm_game import TOL
from nsm_learning import first_best, positive_strategy

SUPPORT_EPS = 1e-10
ENUM_N_MAX = 8                      # Aufzählung bis n = 8: 12 869 Stützenpaare (Zehntelsekunden)


def is_nash(A, B, x, y, tol: float = 1e-7) -> bool:
    """Ist (x, y) ein Nash-Gleichgewicht? Keine reine Zeile lohnt sich gegen y, keine reine Spalte gegen x (bis tol)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return bool(np.all(A @ y <= x @ A @ y + tol) and np.all(x @ B <= x @ B @ y + tol))


def nash_gap(A, B, x, y) -> float:
    """Größter Gewinn, den eine Seite durch einseitiges Abweichen erzielen könnte (null genau im Gleichgewicht)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return float(max(np.max(A @ y) - x @ A @ y, np.max(x @ B) - x @ B @ y))


def support_pairs(n: int) -> int:
    """Zahl der Paare gleich großer Stützen eines n × n-Spiels: Summe über k von C(n, k)² = C(2n, n) − 1."""
    return comb(2 * n, n) - 1


def enumerate_nash(A, B) -> list[tuple[np.ndarray, np.ndarray]]:
    """Alle Nash-Gleichgewichte (x, y) mit gleich großen Stützen; je Stützengröße k werden alle Paare gebündelt gelöst (Gleichungssysteme (k + 1) × (k + 1))."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    m, n = A.shape
    found = []
    for k in range(1, min(m, n) + 1):
        rows_ = np.array(list(combinations(range(m), k)))
        cols_ = np.array(list(combinations(range(n), k)))
        Ii, Jj = np.repeat(rows_, len(cols_), axis=0), np.tile(cols_, (len(rows_), 1))           # (P, k) je Stützenpaar
        P = len(Ii)
        My, Mx = np.zeros((P, k + 1, k + 1)), np.zeros((P, k + 1, k + 1))
        for r in range(k):
            for c in range(k):
                My[:, r, c] = A[Ii[:, r], Jj[:, c]]            # Zeile i ∈ rows_: Auszahlung gegen y auf cols_
                Mx[:, r, c] = B[Ii[:, c], Jj[:, r]]            # Spalte j ∈ cols_: Auszahlung gegen x auf rows_
        My[:, :k, k] = Mx[:, :k, k] = -1.0
        My[:, k, :k] = Mx[:, k, :k] = 1.0
        rhs = np.zeros((P, k + 1, 1))
        rhs[:, k, 0] = 1.0
        regular = np.flatnonzero((np.abs(np.linalg.det(My)) > 1e-12) & (np.abs(np.linalg.det(Mx)) > 1e-12))
        if len(regular) == 0:
            continue
        sy = np.linalg.solve(My[regular], rhs[regular])[:, :, 0]
        sx = np.linalg.solve(Mx[regular], rhs[regular])[:, :, 0]
        for t in np.flatnonzero(np.all(sy[:, :k] > SUPPORT_EPS, axis=1) & np.all(sx[:, :k] > SUPPORT_EPS, axis=1)):
            p = regular[t]
            x, y = np.zeros(m), np.zeros(n)
            x[Ii[p]], y[Jj[p]] = sx[t, :k], sy[t, :k]
            if np.all(A @ y <= sy[t, k] + 1e-9) and np.all(x @ B <= sx[t, k] + 1e-9):
                found.append((x, y))
    return found


def lemke_howson(A, B, label: int) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Lemke–Howson mit dem Anfangslabel `label` (0 … m + n − 1; Labels 0 … m − 1 sind die Zeilen, m … m + n − 1 die Spalten).

    Rückgabe: (x, y, Zahl der Pivots, Pfad der eintretenden Labels vom Anfangslabel an). Der Pfad endet, sobald das Anfangslabel wieder die Basis verlässt: dann ist (x, y) ein Gleichgewicht."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    m, n = A.shape
    A1, B1 = A - A.min() + 1.0, B - B.min() + 1.0                  # positive Auszahlungen ändern die Gleichgewichte nicht
    table = {"P": np.hstack([np.ones((n, 1)), B1.T]), "Q": np.hstack([np.ones((m, 1)), A1])}
    basic = {"P": [m + j for j in range(n)], "Q": list(range(m))}
    nonbasic = {"P": list(range(m)), "Q": [m + j for j in range(n)]}
    side = "P" if label < m else "Q"
    entering, pivots, path = label, 0, [label]
    while True:
        T = table[side]
        q = nonbasic[side].index(entering)
        column = T[:, 1 + q]
        rows = np.flatnonzero(column > 1e-12)
        r = int(rows[int(np.argmin(T[rows, 0] / column[rows]))])         # kleinster Quotient (Minimum-Ratio-Test)
        piv = T[r, 1 + q]
        f = column.copy()
        pivot_row = T[r] / piv
        T -= np.outer(f, pivot_row)
        T[:, 1 + q] = -f / piv
        T[r] = pivot_row
        T[r, 1 + q] = 1.0 / piv
        leaving = basic[side][r]
        basic[side][r], nonbasic[side][q] = entering, leaving
        pivots += 1
        if leaving == label:
            break
        entering = leaving
        path.append(entering)
        side = "Q" if side == "P" else "P"
    x, y = np.zeros(m), np.zeros(n)
    for r, lab in enumerate(basic["P"]):
        if lab < m:
            x[lab] = table["P"][r, 0]
    for r, lab in enumerate(basic["Q"]):
        if lab >= m:
            y[lab - m] = table["Q"][r, 0]
    return x / x.sum(), y / y.sum(), pivots, path


def same_profile(x1, y1, x2, y2, tol: float = 1e-6) -> bool:
    """Sind zwei Strategiepaare gleich (jede Koordinate bis tol)?"""
    return bool(np.max(np.abs(np.asarray(x1) - x2)) < tol and np.max(np.abs(np.asarray(y1) - y2)) < tol)


def shapley_game() -> tuple[np.ndarray, np.ndarray]:
    """Shapleys 3 × 3-Beispiel: A = Einheitsmatrix, B[i, (i + 1) mod 3] = 1. Einziges Gleichgewicht: beide gleichverteilt."""
    A = np.eye(3)
    B = np.zeros((3, 3))
    for i in range(3):
        B[i, (i + 1) % 3] = 1.0
    return A, B


def zero_sum_partner(A) -> tuple[np.ndarray, np.ndarray]:
    """Das Nullsummen-Gegenstück: B = 1 − A (konstante Summe, dieselben Gleichgewichte wie bei B = −A)."""
    A = np.asarray(A, float)
    return A, 1.0 - A


def distance_to_uniform(counts) -> float:
    """Halber L1-Abstand der Häufigkeiten zur Gleichverteilung (Totalvariation), 0 bis 1 − 1/k."""
    c = np.asarray(counts, float)
    return float(0.5 * np.sum(np.abs(c / c.sum() - 1.0 / len(c))))


def fictitious_play_bimatrix(A, B, rounds: int, a0: int = 0, b0: int = 0) -> dict:
    """Fictitious Play in einem Bimatrixspiel: beide spielen die beste Antwort auf die Häufigkeiten des Gegners (kleinster Index bei Gleichstand), gleichzeitig.

    Rückgabe: Abstand der Häufigkeiten zur Gleichverteilung je Runde (größerer der beiden Spieler), Zahl der Aktionswechsel des Zeilenspielers und die Endhäufigkeiten."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    m, n = A.shape
    cx, cy = np.zeros(m), np.zeros(n)
    a, b = a0, b0
    dist = np.zeros(rounds)
    switches = 0
    for t in range(1, rounds + 1):
        cx[a] += 1
        cy[b] += 1
        dist[t - 1] = max(distance_to_uniform(cx), distance_to_uniform(cy))
        a_next, b_next = first_best(A @ cy), first_best(cx @ B)
        switches += a_next != a
        a, b = a_next, b_next
    return {"dist": dist, "switches": int(switches), "freq_x": cx / rounds, "freq_y": cy / rounds}


def regret_matching_bimatrix(A, B, rounds: int, x1=None, y1=None) -> dict:
    """Regret Matching mit erwarteten Auszahlungen in einem Bimatrixspiel (beide maximieren ihre eigene Matrix). Rückgabe: Nash-Lücke des Durchschnitts und der letzten Iteration."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    m, n = A.shape
    x = np.full(m, 1.0 / m) if x1 is None else np.asarray(x1, float)
    y = np.full(n, 1.0 / n) if y1 is None else np.asarray(y1, float)
    Rx, Ry = np.zeros(m), np.zeros(n)
    sx, sy = np.zeros(m), np.zeros(n)
    for _ in range(rounds):
        sx += x
        sy += y
        u_x, u_y = A @ y, x @ B
        Rx = Rx + u_x - x @ u_x
        Ry = Ry + u_y - y @ u_y
        x_last, y_last = x, y
        x, y = positive_strategy(Rx), positive_strategy(Ry)
    return {"gap_avg": nash_gap(A, B, sx / rounds, sy / rounds), "gap_last": nash_gap(A, B, x_last, y_last), "avg_x": sx / rounds, "avg_y": sy / rounds}


def count_pure(equilibria) -> int:
    """Zahl der reinen Gleichgewichte (beide Stützen haben die Größe 1) in einer Liste von Gleichgewichten."""
    return sum(int(np.sum(x > TOL) == 1 and np.sum(y > TOL) == 1) for x, y in equilibria)
