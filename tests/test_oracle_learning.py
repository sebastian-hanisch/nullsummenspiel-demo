"""Orakel für die Lernverfahren: eine zweite Umsetzung mit reinen Python-Schleifen über die Aktionen (kein numpy in der Rechnung), verglichen mit der vektorisierten Fassung, dazu die
Regret-Schranke von Hedge als Band (nur mit Gleichverteilung als Start gültig) und die Ausnutzbarkeit nach der Definition."""

import math

import pytest

import nsm_game as G
import nsm_learning as Ln
from nsm_rng import SplitMix64
from nsm_scenario import draw_matrix


def _expl(A, x, y):
    """Ausnutzbarkeit nach Definition mit Schleifen: größter Zeilenwert gegen y minus kleinster Spaltenwert gegen x."""
    m, n = len(A), len(A[0])
    top = max(sum(A[i][j] * y[j] for j in range(n)) for i in range(m))
    low = min(sum(x[i] * A[i][j] for i in range(m)) for j in range(n))
    return top - low


def _positive(R):
    p = [max(r, 0.0) for r in R]
    s = sum(p)
    return [q / s for q in p] if s > 0 else [1.0 / len(R)] * len(R)


def _reference_rm(A, rounds, x1, y1, plus):
    m, n = len(A), len(A[0])
    Rx, Ry = [0.0] * m, [0.0] * n
    x, y = list(x1), list(y1)
    sx, sy, w = [0.0] * m, [0.0] * n, 0.0
    history = {}
    for t in range(1, rounds + 1):
        wt = float(t) if plus else 1.0
        sx = [s + wt * xi for s, xi in zip(sx, x)]
        sy = [s + wt * yj for s, yj in zip(sy, y)]
        w += wt
        history[t] = (_expl(A, [s / w for s in sx], [s / w for s in sy]), _expl(A, x, y))
        ux = [sum(A[i][j] * y[j] for j in range(n)) for i in range(m)]
        uy = [-sum(x[i] * A[i][j] for i in range(m)) for j in range(n)]
        mx, my = sum(x[i] * ux[i] for i in range(m)), sum(y[j] * uy[j] for j in range(n))
        Rx = [r + u - mx for r, u in zip(Rx, ux)]
        Ry = [r + u - my for r, u in zip(Ry, uy)]
        if plus:
            Rx, Ry = [max(r, 0.0) for r in Rx], [max(r, 0.0) for r in Ry]
        x, y = _positive(Rx), _positive(Ry)
    return [s / w for s in sx], [s / w for s in sy], history


def _reference_hedge(A, rounds, x1, y1):
    m, n = len(A), len(A[0])
    spread = max(max(r) for r in A) - min(min(r) for r in A)
    eta_x, eta_y = math.sqrt(8 * math.log(m) / rounds) / spread, math.sqrt(8 * math.log(n) / rounds) / spread
    cx, cy = [math.log(p) / eta_x for p in x1], [math.log(p) / eta_y for p in y1]
    sx, sy = [0.0] * m, [0.0] * n
    history = {}
    for t in range(1, rounds + 1):
        mx, my = max(cx), max(cy)
        x = [math.exp(eta_x * (c - mx)) for c in cx]
        y = [math.exp(eta_y * (c - my)) for c in cy]
        x, y = [p / sum(x) for p in x], [p / sum(y) for p in y]
        sx, sy = [s + p for s, p in zip(sx, x)], [s + p for s, p in zip(sy, y)]
        history[t] = (_expl(A, [s / t for s in sx], [s / t for s in sy]), _expl(A, x, y))
        cx = [c + sum(A[i][j] * y[j] for j in range(n)) for i, c in enumerate(cx)]
        cy = [c - sum(x[i] * A[i][j] for i in range(m)) for j, c in enumerate(cy)]
    return [s / rounds for s in sx], [s / rounds for s in sy], history


def _reference_fp(A, rounds, a, b):
    m, n = len(A), len(A[0])
    cx, cy = [0] * m, [0] * n
    history, acts = {}, []
    for t in range(1, rounds + 1):
        cx[a] += 1
        cy[b] += 1
        acts.append((a, b))
        ex, ey = [1.0 if i == a else 0.0 for i in range(m)], [1.0 if j == b else 0.0 for j in range(n)]
        history[t] = (_expl(A, [c / t for c in cx], [c / t for c in cy]), _expl(A, ex, ey))
        rows = [sum(A[i][j] * cy[j] for j in range(n)) for i in range(m)]
        cols = [-sum(cx[i] * A[i][j] for i in range(m)) for j in range(n)]
        a = next(i for i in range(m) if rows[i] >= max(rows) - 1e-12)
        b = next(j for j in range(n) if cols[j] >= max(cols) - 1e-12)
    return [c / rounds for c in cx], [c / rounds for c in cy], history, acts


def _games():
    for t in range(12):
        m, n = 2 + t % 4, 2 + (t * 3) % 4
        yield t, draw_matrix(m, n, 900 + t)


@pytest.mark.parametrize("plus", [False, True])
def test_regret_matching_matches_the_loop_reference(plus):
    for t, A in _games():
        start = Ln.make_start(A.shape[0], A.shape[1], "random", SplitMix64(t + 1))
        run = Ln.regret_matching(A, 120, start, plus=plus)
        ax, ay, hist = _reference_rm(A.tolist(), 120, start.x1.tolist(), start.y1.tolist(), plus)
        assert list(run.avg_x) == pytest.approx(ax, abs=1e-9) and list(run.avg_y) == pytest.approx(ay, abs=1e-9)
        for k, step in enumerate(run.steps):
            assert run.expl_avg[k] == pytest.approx(hist[int(step)][0], abs=1e-9) and run.expl_last[k] == pytest.approx(hist[int(step)][1], abs=1e-9)


def test_hedge_matches_the_loop_reference():
    for t, A in _games():
        start = Ln.make_start(A.shape[0], A.shape[1], "random", SplitMix64(t + 1))
        run = Ln.hedge(A, 120, start)
        ax, ay, hist = _reference_hedge(A.tolist(), 120, start.x1.tolist(), start.y1.tolist())
        assert list(run.avg_x) == pytest.approx(ax, abs=1e-9) and list(run.avg_y) == pytest.approx(ay, abs=1e-9)
        for k, step in enumerate(run.steps):
            assert run.expl_avg[k] == pytest.approx(hist[int(step)][0], abs=1e-9) and run.expl_last[k] == pytest.approx(hist[int(step)][1], abs=1e-9)


def test_fictitious_play_matches_the_loop_reference_including_the_actions():
    for t, A in _games():
        start = Ln.make_start(A.shape[0], A.shape[1], "random", SplitMix64(t + 1))
        run, acts_a, acts_b = Ln.fictitious_play(A, 150, start)
        ax, ay, hist, acts = _reference_fp(A.tolist(), 150, start.a0, start.b0)
        assert [(int(a), int(b)) for a, b in zip(acts_a, acts_b)] == acts
        assert list(run.avg_x) == pytest.approx(ax, abs=1e-12) and list(run.avg_y) == pytest.approx(ay, abs=1e-12)
        for k, step in enumerate(run.steps):
            assert run.expl_avg[k] == pytest.approx(hist[int(step)][0], abs=1e-9) and run.expl_last[k] == pytest.approx(hist[int(step)][1], abs=1e-9)


def test_exploitability_function_equals_the_loop_definition():
    for t, A in _games():
        rng = SplitMix64(t)
        x, y = Ln.random_simplex(A.shape[0], rng), Ln.random_simplex(A.shape[1], rng)
        assert G.exploitability(A, x, y) == pytest.approx(_expl(A.tolist(), x.tolist(), y.tolist()), abs=1e-12)


def test_hedge_average_exploitability_respects_the_regret_bound_from_the_uniform_start():
    """Mit Gleichverteilung als Start und η = √(8 ln N / T)/Spannweite gilt für den Durchschnitt: Ausnutzbarkeit ≤ Spannweite · (√(ln m / (2T)) + √(ln n / (2T)))
    (Regret-Schranke von Hedge je Seite, durch T geteilt). Das Band gilt für jede Gegnerfolge, also auch hier."""
    for t, A in _games():
        m, n = A.shape
        T = 400
        start = Ln.make_start(m, n, "uniform", SplitMix64(1))
        run = Ln.hedge(A, T, start)
        spread = A.max() - A.min()
        bound = spread * (math.sqrt(math.log(m) / (2 * T)) + math.sqrt(math.log(n) / (2 * T)))
        assert run.expl_avg[-1] <= bound + 1e-12, (t, run.expl_avg[-1], bound)
