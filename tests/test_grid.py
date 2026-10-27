"""Die vorgerechneten Messreihen (data/nullsummen_grid.json): Aufbau, Zählungen und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import nsm_constants as C
import nsm_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/nullsummen_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-9), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Rechnung oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_grid_has_every_cell_exactly_once(grid):
    seeds = list(C.GRID_SEEDS)
    assert sorted((c["m"], c["n"], c["seed"]) for c in grid["e1"]) == sorted((m, n, s) for m, n in C.E1_SIZES for s in seeds)
    assert all(c["games"] == C.E1_GAMES for c in grid["e1"])
    assert sorted((c["n"], c["seed"]) for c in grid["e1_int"]) == sorted((n, s) for n in (2, 3, 4, 5) for s in seeds)
    assert sorted((c["n"], c["low"], c["seed"]) for c in grid["e1b"]) == sorted((n, low, s) for low, _ in C.E1B_RANGES for n in C.E1B_NS for s in seeds)
    assert sorted((c["profile"], c["pattern"], c["counter"], c["tie"]) for c in grid["e2"]) == sorted(
        (name, p, c, t) for name in C.E2_PROFILES for p in C.PATTERNS for c in C.COUNTERS for t in (("low", "random") if c != "fixed" else ("low",)))
    assert sorted(grid["capture"]) == [str(n) for n in range(2, 9)]
    assert len(grid["e2est"]) == 5 * len(C.E2_EST_TS) and {r["T"] for r in grid["e2est"]} == set(C.E2_EST_TS)
    assert sorted((c["method"], c["seed"]) for c in grid["e3"]) == sorted((m, s) for m in C.LEARN_METHODS for s in seeds) and all(c["horizons"] == list(C.E3_HORIZONS) for c in grid["e3"])
    assert len(grid["e3start"]) == 3 * len(C.LEARN_METHODS) * len(C.STARTS) * len(seeds)
    assert sorted((c["n"], c["seed"]) for c in grid["e4"]) == sorted((n, s) for n in C.E4_NS for s in seeds) and all(c["games"] == C.E4_GAMES[c["n"]] for c in grid["e4"])
    assert sorted((c["n"], c["seed"]) for c in grid["e4null"]) == sorted((n, s) for n in C.E4_NULLSUM_NS for s in seeds)
    assert sorted((c["n"], c["seed"]) for c in grid["e4big"]) == sorted((n, s) for n in C.E4_LH_BIG for s in seeds)
    assert len(grid["rules"]) == 4 and grid["shapley"]["rounds"] == C.SHAPLEY_ROUNDS_GRID


def test_grid_meta(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)


def test_no_all_zero_column_in_the_sweeps(grid):
    """Nullspalten-Regel (DEMO-PLAYBOOK §0): eine Spalte mit lauter exakten Nullen hieße, dass die Variante nie läuft. Wo Nullen erwartet sind (Start im Gleichgewicht, Gleichgewichtsgegner), stehen sie gezielt in den Tests."""
    start = ev.e3_start_summary(grid)
    for game in ("rps", "coin", "four"):
        for method in C.LEARN_METHODS:
            assert start[(game, method, "random")]["last"] > 0.1, (game, method)               # mit Zufallsstart läuft jedes Verfahren und bleibt in Bewegung
    summary = ev.e3_summary(grid)
    for method, s in summary.items():
        assert all(v > 0 for v in s["avg"] + s["last"]), method
    assert all(c["mean"] > 0 for c in grid["e2"] if c["counter"] != "follow" or c["pattern"] != "rotation")
    assert all(r["coverage"] > 0 and r["mean_pivots"] > 0 for r in ev.e4_summary(grid))
    assert len({round(c["saddle"], 6) for c in grid["e1"]}) > 5
