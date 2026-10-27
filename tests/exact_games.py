"""Exakte Orakel mit Brüchen (fractions) für Matrix- und Bimatrixspiele mit ganzzahligen Einträgen; benutzt nichts aus den Modulen der Demo (ORACLE-PLAYBOOK §2: Support-Enumeration, exakt rational).

`solve_exact` löst ein quadratisches lineares Gleichungssystem mit Gauß–Jordan in Brüchen. `basic_solutions` zählt alle Paare gleich großer Stützen eines Nullsummenspiels
(Shapley–Snow: die Ecken der optimalen Mengen sind genau solche Basislösungen) und liefert (Wert, x, y). `nash_equilibria` zählt alle Gleichgewichte eines nicht entarteten
Bimatrixspiels (dort haben alle Gleichgewichte gleich große Stützen)."""

from fractions import Fraction as F
from itertools import combinations


def solve_exact(M, b):
    """Löst M z = b in Brüchen; None, wenn M singulär ist."""
    n = len(M)
    a = [[F(v) for v in row] + [F(b[i])] for i, row in enumerate(M)]
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None:
            return None
        a[c], a[p] = a[p], a[c]
        piv = a[c][c]
        a[c] = [v / piv for v in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [vr - f * vc for vr, vc in zip(a[r], a[c])]
    return [a[i][n] for i in range(n)]


def basic_solutions(A):
    """Alle Basislösungen (Wert, x, y) eines Nullsummenspiels mit ganzzahliger Matrix A (Zeilen maximieren): gleich große Stützen rows_, cols_, Gleichungen auf der Stütze, Nichtnegativität,
    keine lohnende Abweichung. Bei einem Wert, der zu keinem Paar passt, bleibt die Liste leer (kommt bei Matrizen ohne Entartung nicht vor)."""
    A = [[F(int(v)) for v in row] for row in A]
    m, n = len(A), len(A[0])
    out = []
    for k in range(1, min(m, n) + 1):
        for rows_ in combinations(range(m), k):
            for cols_ in combinations(range(n), k):
                sx = solve_exact([[A[i][j] for i in rows_] + [F(-1)] for j in cols_] + [[F(1)] * k + [F(0)]], [F(0)] * k + [F(1)])
                sy = solve_exact([[A[i][j] for j in cols_] + [F(-1)] for i in rows_] + [[F(1)] * k + [F(0)]], [F(0)] * k + [F(1)])
                if sx is None or sy is None:
                    continue
                xs, v = sx[:k], sx[k]
                ys, w = sy[:k], sy[k]
                if v != w or any(t < 0 for t in xs) or any(t < 0 for t in ys):
                    continue
                x, y = [F(0)] * m, [F(0)] * n
                for t, i in enumerate(rows_):
                    x[i] = xs[t]
                for t, j in enumerate(cols_):
                    y[j] = ys[t]
                if any(sum(x[i] * A[i][j] for i in range(m)) < v for j in range(n)):
                    continue
                if any(sum(A[i][j] * y[j] for j in range(n)) > v for i in range(m)):
                    continue
                out.append((v, tuple(x), tuple(y)))
    return out


def value_and_extremes(A):
    """Exakter Wert und die Mengen der extremen optimalen Strategien beider Seiten (ohne Doppelte)."""
    sols = basic_solutions(A)
    value = sols[0][0]
    assert all(s[0] == value for s in sols)
    return value, sorted({s[1] for s in sols}), sorted({s[2] for s in sols})


def nash_equilibria(A, B):
    """Alle Nash-Gleichgewichte (x, y) eines nicht entarteten Bimatrixspiels mit ganzzahligen Einträgen, in Brüchen. A: Auszahlung der Zeilen, B: der Spalten."""
    A = [[F(int(v)) for v in row] for row in A]
    B = [[F(int(v)) for v in row] for row in B]
    m, n = len(A), len(A[0])
    out = []
    for k in range(1, min(m, n) + 1):
        for rows_ in combinations(range(m), k):
            for cols_ in combinations(range(n), k):
                sy = solve_exact([[A[i][j] for j in cols_] + [F(-1)] for i in rows_] + [[F(1)] * k + [F(0)]], [F(0)] * k + [F(1)])
                sx = solve_exact([[B[i][j] for i in rows_] + [F(-1)] for j in cols_] + [[F(1)] * k + [F(0)]], [F(0)] * k + [F(1)])
                if sx is None or sy is None:
                    continue
                xs, w = sx[:k], sx[k]
                ys, u = sy[:k], sy[k]
                if any(t <= 0 for t in xs) or any(t <= 0 for t in ys):
                    continue
                x, y = [F(0)] * m, [F(0)] * n
                for t, i in enumerate(rows_):
                    x[i] = xs[t]
                for t, j in enumerate(cols_):
                    y[j] = ys[t]
                if any(sum(A[i][j] * y[j] for j in range(n)) > u for i in range(m)):
                    continue
                if any(sum(x[i] * B[i][j] for i in range(m)) > w for j in range(n)):
                    continue
                out.append((tuple(x), tuple(y)))
    return out
