"""Rechnet die Messreihen der Experimente vor und schreibt data/nullsummen_grid.json.

    python tools/make_grid.py            # volle Messreihen, mehrere Prozesse
    python tools/make_grid.py --quick    # kleine Probe zum Ausprobieren (data/nullsummen_grid_quick.json, nicht für Tests)

Alles ist deterministisch (SplitMix64 mit festen Zufalls-Seeds aus nsm_constants.GRID_SEEDS). tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import nsm_constants as C  # noqa: E402
import nsm_evaluation as ev  # noqa: E402
import nsm_hideseek as H  # noqa: E402

EST_SCENARIOS = (
    ("loader", "Kontrolleur prüft gleichverteilt", [0.25] * 4),
    ("controller", "Verlader wählt gleichverteilt", [0.25] * 4),
    ("controller", "Verlader wählt meist Route 1 (0,7)", [0.7, 0.1, 0.1, 0.1]),
    ("loader", "Kontrolleur im Gleichgewicht y*", "y_star"),
    ("controller", "Verlader im Gleichgewicht x*", "x_star"),
)
EST_PROFILE = (0.9, 0.8, 0.7, 0.6)
E1_INT_SIZES = ((2, 2), (3, 3), (4, 4), (5, 5))
E1_INT_GAMES = 100
E2_PATTERNS = tuple(C.PATTERNS)
E3_METHODS = tuple(C.LEARN_METHODS)
E3_ROUNDS = 10000
E3_START_GAMES = ("rps", "coin", "four")
E4_BIG_GAMES = {20: 10, 40: 6}
E4_NULLSUM_GAMES = 60


def _tasks(quick: bool):
    seeds = C.GRID_SEEDS
    tasks = []
    for m, n in (C.E1_SIZES if not quick else ((3, 3),)):
        for seed in seeds:
            tasks.append(("e1", "e1_cell", (m, n, seed, C.E1_GAMES if not quick else 20)))
    if quick:
        return tasks + [("rules", "rules_rows", (EST_PROFILE,)), ("capture", "capture_rounds", ())]
    for m, n in E1_INT_SIZES:
        for seed in seeds:
            tasks.append(("e1_int", "e1_integer_cell", (m, n, seed, E1_INT_GAMES)))
    for low, _hi in C.E1B_RANGES:
        for n in C.E1B_NS:
            for seed in seeds:
                tasks.append(("e1b", "e1b_cell", (n, low, seed, C.E1B_GAMES)))
    for name, v in C.E2_PROFILES.items():
        for pattern in E2_PATTERNS:
            for counter in C.COUNTERS:
                for tie in (("low", "random") if counter != "fixed" else ("low",)):
                    tasks.append(("e2", "e2_cell", (name, v, pattern, counter, tie)))
    for side, name, opp in EST_SCENARIOS:
        for T in C.E2_EST_TS:
            tasks.append(("e2est", "e2_estimate_cell", (side, name, opp, T)))
    for method in E3_METHODS:
        for seed in seeds:
            tasks.append(("e3", "e3_game_runs", (method, C.E3_SIZE, seed, C.E3_GAMES)))
    for game in E3_START_GAMES:
        for method in E3_METHODS:
            for start in C.STARTS:
                for seed in seeds:
                    tasks.append(("e3start", "e3_start_cell", (game, method, start, seed, E3_ROUNDS)))
    for n in C.E4_NS:
        for seed in seeds:
            tasks.append(("e4", "e4_cell", (n, seed, C.E4_GAMES[n])))
    for n in C.E4_NULLSUM_NS:
        for seed in seeds:
            tasks.append(("e4null", "e4_nullsum_cell", (n, seed, E4_NULLSUM_GAMES)))
    for n, games in E4_BIG_GAMES.items():
        for seed in seeds:
            tasks.append(("e4big", "e4_lh_big", (n, seed, games)))
    tasks.append(("shapley", "e4_shapley", (C.SHAPLEY_ROUNDS_GRID,)))
    for v in (C.E2_PROFILES["vier gleiche Routen"], (1.0, 0.1), (1.0, 0.75, 0.5, 0.25), EST_PROFILE):
        tasks.append(("rules", "rules_rows", (v,)))
    tasks.append(("capture", "capture_rounds", ()))
    return tasks


def _run(task):
    key, func, args = task
    if func == "e2_cell":
        name, v, pattern, counter, tie = args
        return key, ev.e2_cell(v, pattern, counter, tie, C.GRID_SEEDS, C.E2_T) | {"profile": name}
    if func == "e2_estimate_cell":
        side, name, opp, T = args
        if isinstance(opp, str):
            k, x, y, _a = H.closed_form(EST_PROFILE)
            opp = x if opp == "x_star" else y
        return key, ev.e2_estimate_cell(EST_PROFILE, side, name, opp, T, C.GRID_SEEDS, C.E2_EST_SAMPLES)
    if func == "rules_rows":
        return key, ev.rules_rows(*args)
    return key, getattr(ev, func)(*args)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main() -> None:
    quick = "--quick" in sys.argv
    t0 = time.time()
    out: dict = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for key, value in pool.map(_run, _tasks(quick), chunksize=1):
            if key in ("capture", "shapley"):
                out[key] = value
            else:
                out.setdefault(key, []).append(value)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("nullsummen_grid_quick.json" if quick else "nullsummen_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    print(f"{dest} geschrieben: {', '.join(out)}; {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
