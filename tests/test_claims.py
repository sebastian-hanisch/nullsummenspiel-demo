"""Jede Zahl der README steht hier und wird aus den vorgerechneten Messreihen (data/nullsummen_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft; die Zeilen-Bauer (`e1_table` usw.) und `numbers()` dienen auch dazu, die README-Tabellen und
-Zahlen zu erzeugen. Alles ist deterministisch (feste Zufalls-Seeds, SplitMix64). Jede Zahl nennt ihre Bedingung (Spiel, Größe, Verfahren, Start, Gleichstandsregel, Zahl der Spiele und Läufe) in der README
direkt daneben; die Tests unten prüfen die Zahl unter genau dieser Bedingung."""

import math
from pathlib import Path

import pytest

import nsm_bimatrix as Bm
import nsm_constants as C
import nsm_evaluation as ev
from nsm_scenario import Settings

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
DEFAULT_PROFILE = "vier gleiche Routen"
UNEQUAL_PROFILE = "ungleich (0,9; 0,8; 0,7; 0,6)"


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _p(v, d=1):
    return f"{round(100 * v, d) + 0.0:.{d}f}".replace(".", ",") + " %"


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def e1_table(grid) -> list[str]:
    rows = []
    for r in ev.aggregate_e1(grid["e1"]):
        gn = "–" if r["gain_nonsaddle"] is None else _c(r["gain_nonsaddle"], 3)
        formula = "–" if r["formula"] is None else _p(r["formula"], 2)
        rows.append(f"| {r['m']} × {r['n']} | {r['games']} | {_p(r['saddle'])} ({_p(r['saddle_std'])}) | {formula} | {_c(r['value'], 3)} | {_c(r['security'], 3)} | {_c(r['gain'], 3)} | {gn} | {_c(r['support'], 2)} |")
    return rows


def e1b_table(grid) -> list[str]:
    summary = ev.e1b_summary(grid)
    return [f"| {_c(low, 2)} | {n} | {ev.sci(v['deviation'])} | {_p(v['kept_all'])} |" for (low, n), v in sorted(summary.items(), key=lambda kv: (-kv[0][0], kv[0][1]))]


def ints_table(grid) -> list[str]:
    return [f"| {n} × {n} | {v['games']} | {_p(v['multi_x'], 1)} | {_p(v['multi_y'], 1)} |" for n, v in sorted(ev.e1_int_summary(grid).items())]


def rules_table(grid) -> list[str]:
    rows = []
    for r in grid["rules"]:
        rows.append(f"| {'; '.join(_c(t, 2) for t in r['v'])} | {_c(r['value'], 4)} | " + " | ".join(_c(r["loader"][k], 4) for k in ("gleichverteilt", "proportional zum Wert", "umgekehrt proportional zum Wert (alle Routen)", "nur die wertvollste Route"))
                    + " | " + " | ".join(_c(r["controller"][k], 4) for k in ("gleichverteilt", "proportional zum Wert", "nur die wertvollste Route")) + " |")
    return rows


def e2_table(grid, profile) -> list[str]:
    lookup = ev.e2_lookup(grid, profile)
    cols = [("freq", "low"), ("freq", "random"), ("window", "low"), ("window", "random"), ("follow", "low"), ("follow", "random"), ("fixed", "low")]
    rows = []
    for pattern in C.PATTERNS:
        cells = [f"{_c(lookup[(pattern, c, t)]['mean'], 3)} ({_c(lookup[(pattern, c, t)]['std'], 3)})" for c, t in cols]
        rows.append(f"| {C.PATTERNS[pattern].split(' (')[0]} | " + " | ".join(cells) + " |")
    return rows


def capture_table(grid) -> list[str]:
    return [f"| {n} | {r} |" for n, r in sorted(((int(k), v) for k, v in grid["capture"].items()))]


def estimate_table(grid) -> list[str]:
    rows = []
    for name in dict.fromkeys(r["opponent"] for r in grid["e2est"]):
        pts = sorted((r for r in grid["e2est"] if r["opponent"] == name), key=lambda r: r["T"])
        cells = " | ".join(("–" if r["share"] is None else _p(r["share"], 0)) for r in pts)
        rows.append(f"| {name} | {_c(pts[0]['full'], 4)} | {cells} |")
    return rows


def e3_table(grid) -> list[str]:
    rows = []
    for m, s in ev.e3_summary(grid).items():
        rows.append(f"| {C.LEARN_METHODS[m]} | " + " | ".join(_c(v, 4) for v in s["avg"]) + f" | {_c(s['slope'], 2)} | " + " | ".join(_c(v, 3) for v in s["last"]) + " |")
    return rows


def start_table(grid) -> list[str]:
    start = ev.e3_start_summary(grid)
    rows = []
    for game, label in (("rps", "Stein-Schere-Papier"), ("four", "vier gleiche Routen")):
        for m in C.LEARN_METHODS:
            u, r = start[(game, m, "uniform")], start[(game, m, "random")]
            rows.append(f"| {label} | {C.LEARN_METHODS[m]} | {_c(u['avg'], 4)} | {_c(u['last'], 3)} | {_c(r['avg'], 4)} | {_c(r['last'], 3)} |")
    return rows


def e4_table(grid) -> list[str]:
    return [f"| {r['n']} | {r['games']} | {_c(r['mean_count'], 2)} | {r['max_count']} | {_p(r['only_one'], 0)} | {r['even']} | {_p(r['with_pure'], 0)} | {_c(r['mean_pivots'], 1)} | {r['max_pivots']} | {r['lh_errors']} | {_p(r['coverage'], 0)} | {_pairs(r['pairs'])} |"
            for r in ev.e4_summary(grid)]


def _pairs(v):
    return f"{v:,}".replace(",", " ")


def e4_extra_table(grid) -> list[str]:
    rows = [f"| Nullsumme B = 1 − A, n = {n} | {v['games']} | {_p(v['only_one'], 0)} | {ev.sci(v['deviation'])} |" for n, v in ev.e4_nullsum_summary(grid).items()]
    rows += [f"| Lemke–Howson allein, n = {n} | {v['paths']} Pfade | Mittel {_c(v['mean'], 1)}, Median {_c(v['median'], 0)}, Maximum {v['max']} | {v['errors']} Fehler |" for n, v in ev.e4_big_summary(grid).items()]
    return rows


def numbers(grid) -> dict:
    """Die Zahlen im Fließtext der README (Name -> Zeichenkette in der Rundung der App)."""
    e1 = {(r["m"], r["n"]): r for r in ev.aggregate_e1(grid["e1"])}
    e1b = ev.e1b_summary(grid)
    ints = ev.e1_int_summary(grid)
    cert = max(r["certificate"] for r in ev.aggregate_e1(grid["e1"]))
    dev = max(v["deviation"] for v in e1b.values())
    d4 = ev.e2_lookup(grid, DEFAULT_PROFILE)
    d4u = ev.e2_lookup(grid, UNEQUAL_PROFILE)
    e3 = ev.e3_summary(grid)
    st = ev.e3_start_summary(grid)
    e4 = {r["n"]: r for r in ev.e4_summary(grid)}
    nulls = ev.e4_nullsum_summary(grid)
    big = ev.e4_big_summary(grid)
    est = {(r["opponent"], r["T"]): r for r in grid["e2est"]}
    shap = grid["shapley"]
    rules = {tuple(r["v"]): r for r in grid["rules"]}
    iid = [c["mean"] for (p, _c_, _t), c in d4.items() if p == "iid_eq"]
    iidu = [c["mean"] for (p, _c_, _t), c in d4u.items() if p == "iid_eq"]
    lo = min(w[0] for w in shap["windows"].values())
    hi = max(w[1] for w in shap["windows"].values())
    return {
        "sad22": _p(e1[(2, 2)]["saddle"]), "sad22_f": _p(e1[(2, 2)]["formula"]), "sad1010": _p(e1[(10, 10)]["saddle"]), "sad1010_f": _p(e1[(10, 10)]["formula"], 2), "sad55": _p(e1[(5, 5)]["saddle"]), "sad55_f": _p(e1[(5, 5)]["formula"]),
        "gain22": _c(e1[(2, 2)]["gain"], 3), "gain1010": _c(e1[(10, 10)]["gain"], 3), "cert": ev.sci(cert), "dev_form": ev.sci(dev), "dev_cells": str(len(e1b)),
        "multi_lo": _p(min(v["multi_x"] for v in ints.values()), 0), "multi_hi": _p(max(v["multi_x"] for v in ints.values()), 0),
        "kept_n4_03": _p(e1b[(0.3, 4)]["kept_all"], 1), "kept_n8_03": _p(e1b[(0.3, 8)]["kept_all"], 1), "kept_n3_03": _p(e1b[(0.3, 3)]["kept_all"], 1), "kept_n4_005": _p(e1b[(0.05, 4)]["kept_all"], 1), "kept_n3_005": _p(e1b[(0.05, 3)]["kept_all"], 1),
        "rule_prop_101": _c(rules[(1.0, 0.1)]["loader"]["proportional zum Wert"], 4), "val_101": _c(rules[(1.0, 0.1)]["value"], 4), "rule_prop_4321": _c(rules[(1.0, 0.75, 0.5, 0.25)]["loader"]["proportional zum Wert"], 4),
        "val_4321": _c(rules[(1.0, 0.75, 0.5, 0.25)]["value"], 4), "ctl_prop_4321": _c(rules[(1.0, 0.75, 0.5, 0.25)]["controller"]["proportional zum Wert"], 4), "ctl_prop_101": _c(rules[(1.0, 0.1)]["controller"]["proportional zum Wert"], 4),
        "rot_follow": _c(d4[("rotation", "follow", "low")]["mean"], 3), "rot_freq_low": _c(d4[("rotation", "freq", "low")]["mean"], 3), "rot_freq_rand": _c(d4[("rotation", "freq", "random")]["mean"], 3),
        "rot_follow_u": _c(d4u[("rotation", "follow", "low")]["mean"], 3), "iid_lo": _c(min(iid), 3), "iid_hi": _c(max(iid), 3), "iid_lo_u": _c(min(iidu), 3), "iid_hi_u": _c(max(iidu), 3),
        "capture2": str(grid["capture"]["2"]), "capture8": str(grid["capture"]["8"]),
        "habit_follow": _c(d4[("habit", "follow", "low")]["mean"], 3), "norep_follow": _c(d4[("norepeat", "follow", "low")]["mean"], 3),
        "est_uni_1": _p(est[("Kontrolleur prüft gleichverteilt", 1)]["share"], 0), "est_uni_5": _p(est[("Kontrolleur prüft gleichverteilt", 5)]["share"], 0), "est_uni_1000": _p(est[("Kontrolleur prüft gleichverteilt", 1000)]["share"], 0),
        "est_uni_full": _c(est[("Kontrolleur prüft gleichverteilt", 1)]["full"], 4), "est_ctl_1000": _p(est[("Verlader wählt gleichverteilt", 1000)]["share"], 0), "est_ctl_full": _c(est[("Verlader wählt gleichverteilt", 1)]["full"], 4),
        "est_main_5": _p(est[("Verlader wählt meist Route 1 (0,7)", 5)]["share"], 0), "est_main_full": _c(est[("Verlader wählt meist Route 1 (0,7)", 1)]["full"], 4),
        "slope_rm": _c(e3["rm"]["slope"], 2), "slope_rmplus": _c(e3["rmplus"]["slope"], 2), "slope_mw": _c(e3["mw"]["slope"], 2), "slope_fp": _c(e3["fp"]["slope"], 2), "slope_rms": _c(e3["rm_sampled"]["slope"], 2),
        "avg_lo": _c(min(s["avg"][3] for s in e3.values()), 4), "avg_hi": _c(max(s["avg"][3] for s in e3.values()), 4), "last_lo": _c(min(s["last"][3] for s in e3.values()), 3), "last_hi": _c(max(s["last"][3] for s in e3.values()), 3),
        "rps_last_rm": _c(st[("rps", "rm", "random")]["last"], 3), "rps_last_mw": _c(st[("rps", "mw", "random")]["last"], 3), "rps_last_rmplus": _c(st[("rps", "rmplus", "random")]["last"], 3),
        "e3_games": str(e3["rm"]["games"]),
        "n2_count": _c(e4[2]["mean_count"], 2), "n8_count": _c(e4[8]["mean_count"], 2), "bm_games": str(sum(r["games"] for r in e4.values())), "bm_max": str(max(r["max_count"] for r in e4.values())),
        "cov2": _p(e4[2]["coverage"], 0), "cov8": _p(e4[8]["coverage"], 0), "piv8": _c(e4[8]["mean_pivots"], 1), "pivmax8": str(e4[8]["max_pivots"]), "only1_2": _p(e4[2]["only_one"], 0), "only1_8": _p(e4[8]["only_one"], 0),
        "ns_games": str(sum(v["games"] for v in nulls.values())), "big40_mean": _c(big[40]["mean"], 1), "big40_med": _c(big[40]["median"], 0), "big40_max": str(big[40]["max"]), "big20_mean": _c(big[20]["mean"], 1),
        "pairs10": _pairs(Bm.support_pairs(10)), "pairs8": _pairs(Bm.support_pairs(8)),
        "shap_lo": _c(lo, 3), "shap_hi": _c(hi, 3), "shap_sw": str(shap["switches"]), "shap_zero": _c(shap["zero_last"], 4), "shap_rm_avg": _c(shap["rm_gap_avg"], 3), "shap_rm_last": _c(shap["rm_gap_last"], 3),
        "shap_rounds": _pairs(shap["rounds"]),
    }


# --- README gegen die Zeilen-Bauer ------------------------------------------------------------------------------------------------------------------

def _in_readme(lines):
    missing = [line for line in lines if line not in README]
    assert not missing, "fehlt in der README: " + " || ".join(missing[:3])


def test_readme_tables_equal_the_rows_of_the_app(grid):
    for rows in (e1_table(grid), e1b_table(grid), ints_table(grid), rules_table(grid), e2_table(grid, DEFAULT_PROFILE), e2_table(grid, UNEQUAL_PROFILE), capture_table(grid), estimate_table(grid),
                 e3_table(grid), start_table(grid), e4_table(grid), e4_extra_table(grid)):
        assert rows
        _in_readme(rows)


def test_readme_numbers_appear_in_the_readme(grid):
    missing = [f"{k} = {v}" for k, v in numbers(grid).items() if v not in README]
    assert not missing, "fehlt in der README: " + "; ".join(missing)


def test_the_assumptions_text_is_the_same_in_app_and_readme():
    assert C.ASSUMPTIONS_MARKDOWN.strip() in README


def test_the_readme_names_the_hypotheses():
    for h in range(1, 9):
        assert f"**H{h}**" in README, h


# --- Die Zahlen selbst, unter ihrer Bedingung ----------------------------------------------------------------------------------------------

def test_e1_the_saddle_share_follows_the_formula_within_four_standard_errors(grid):
    for r in ev.aggregate_e1(grid["e1"]):
        if r["formula"] is not None:
            p = r["formula"]
            assert abs(r["saddle"] - p) <= 4 * math.sqrt(p * (1 - p) / r["games"]) + 1e-12, (r["n"], r["saddle"], p)         # 4 000 Spiele je Größe
    n = numbers(grid)
    assert (n["sad22"], n["sad55"], n["sad1010"], n["gain22"], n["gain1010"]) == ("65,5 %", "4,0 %", "0,0 %", "0,035", "0,253")
    rows = {(r["m"], r["n"]): r for r in ev.aggregate_e1(grid["e1"])}
    gains = [rows[(k, k)]["gain"] for k in (2, 3, 4, 5, 7, 10)]
    assert gains == sorted(gains)                                                                                       # der Gewinn wächst mit n (gemessene Größen)
    assert all(rows[(k, k)]["value"] > rows[(k, k)]["security"] for k in (2, 3, 4, 5, 7, 10))


def test_e1_certificate_and_closed_form(grid):
    n = numbers(grid)
    assert (n["cert"], n["dev_form"], n["dev_cells"]) == ("3,2·10⁻¹⁴", "7,5·10⁻¹⁶", "10")
    assert all(v["deviation"] < 1e-12 for v in ev.e1b_summary(grid).values())
    assert (n["kept_n3_03"], n["kept_n4_03"], n["kept_n8_03"], n["kept_n4_005"], n["kept_n3_005"]) == ("85,5 %", "42,5 %", "0,0 %", "13,2 %", "42,2 %")
    summary = ev.e1b_summary(grid)
    assert all(summary[(0.3, k)]["kept_all"] > 0.5 for k in (2, 3)) and all(summary[(0.3, k)]["kept_all"] < 0.5 for k in (4, 6, 8)) and summary[(0.05, 3)]["kept_all"] < 0.5
    assert (n["multi_lo"], n["multi_hi"]) == ("10 %", "15 %")


def test_h4_the_rules_by_value(grid):
    n = numbers(grid)
    assert (n["val_101"], n["rule_prop_101"], n["val_4321"], n["rule_prop_4321"], n["ctl_prop_4321"], n["ctl_prop_101"]) == ("0,0909", "0,0091", "0,4615", "0,3500", "0,6000", "0,0909")
    for r in grid["rules"]:
        assert r["loader"]["Gleichgewicht"] == pytest.approx(r["value"]) and r["controller"]["Gleichgewicht"] == pytest.approx(r["value"])
        assert r["loader"]["nur die wertvollste Route"] == 0.0 and all(v <= r["value"] + 1e-9 for v in r["loader"].values()) and all(v >= r["value"] - 1e-9 for v in r["controller"].values())
    two = next(r for r in grid["rules"] if len(r["v"]) == 2)
    assert two["controller"]["proportional zum Wert"] == pytest.approx(two["value"])                                  # Ausnahme bei zwei Routen
    for r in grid["rules"]:
        if len(r["v"]) >= 3 and len(set(r["v"])) > 1:
            assert r["controller"]["proportional zum Wert"] > r["value"] + 1e-6                                       # ab drei verschiedenen Routenwerten ausnutzbar


def test_h5_patterns_against_counters(grid):
    n = numbers(grid)
    assert (n["rot_follow"], n["rot_freq_low"], n["rot_freq_rand"], n["rot_follow_u"]) == ("0,003", "0,750", "0,947", "0,002")
    assert (n["capture2"], n["capture8"]) == ("3", "9") and [grid["capture"][str(k)] for k in range(2, 9)] == [k + 1 for k in range(2, 9)]
    d4 = ev.e2_lookup(grid, DEFAULT_PROFILE)
    assert d4[("rotation", "freq", "low")]["mean"] == pytest.approx(0.75, abs=1e-12) and d4[("rotation", "freq", "low")]["caught"] == pytest.approx(0.25, abs=1e-12)
    assert d4[("rotation", "freq", "random")]["mean"] > 0.9 and d4[("rotation", "freq", "random")]["caught"] < 0.1                  # Zähler mit zufälligem Gleichstand fängt weniger als der Zufall (1/4)
    iid = [c["mean"] for (p, _c_, _t), c in d4.items() if p == "iid_eq"]
    assert len(iid) == 7 and (n["iid_lo"], n["iid_hi"]) == ("0,744", "0,766") and all(abs(m - 0.75) < 0.03 for m in iid)       # Rauschen: Standardfehler je Zelle etwa 0,007
    assert d4[("habit", "follow", "low")]["mean"] < 0.7 and d4[("norepeat", "follow", "low")]["mean"] < 0.7
    assert (n["habit_follow"], n["norep_follow"]) == ("0,377", "0,680")


def test_h6_averages_converge_and_last_iterates_do_not(grid):
    s = ev.e3_summary(grid)
    n = numbers(grid)
    assert (n["avg_lo"], n["avg_hi"], n["last_lo"], n["last_hi"], n["e3_games"]) == ("0,0011", "0,0092", "0,084", "0,456", "20")
    for m, row in s.items():
        assert row["avg"] == sorted(row["avg"], reverse=True) and row["last"][3] > 0.05, m                       # Durchschnitt fällt bei jedem Horizont, letzte Iteration bleibt über 0,05
    assert (n["slope_rm"], n["slope_rmplus"], n["slope_mw"], n["slope_fp"], n["slope_rms"]) == ("-0,52", "-0,64", "-0,51", "-0,53", "-0,52")
    assert all(-0.56 <= s[m]["slope"] <= -0.49 for m in ("rm", "mw", "fp", "rm_sampled")) and s["rmplus"]["slope"] < -0.6


def test_h6_headline_with_other_random_seeds():
    """Gegenprobe mit anderen Zufalls-Seeds (21 bis 24, kleiner Umfang): Durchschnitt unter 0,02 und letzte Iteration über 0,03 bei T = 10 000 für jedes Verfahren; Rotation gegen Folgezähler unter 0,01, gegen Häufigkeitszähler mit
    zufälligem Gleichstand über 0,9 und mit kleinstem Index genau 0,75; Sattelpunktanteil 2 × 2 in 2 000 Spielen im Vierfehlerband um 2/3."""
    for method in C.LEARN_METHODS:
        cell = ev.e3_game_runs(method, 5, 21, 3)
        assert cell["avg_sum"][3] / 3 < 0.02 and cell["last_sum"][3] / 3 > 0.03, method
    v4 = (1.0,) * 4
    assert ev.e2_cell(v4, "rotation", "follow", "low", (21, 22, 23, 24), 1000)["mean"] < 0.01
    assert ev.e2_cell(v4, "rotation", "freq", "random", (21, 22, 23, 24), 1000)["mean"] > 0.9
    assert ev.e2_cell(v4, "rotation", "freq", "low", (21, 22, 23, 24), 1000)["mean"] == pytest.approx(0.75, abs=1e-12)
    games = sum(ev.e1_cell(2, 2, seed, 500)["saddle"] * 500 for seed in (21, 22, 23, 24))
    assert abs(games / 2000 - 2 / 3) <= 4 * math.sqrt(2 / 3 / 3 / 2000)


def test_b1_the_start_in_the_equilibrium_stands_still(grid):
    st = ev.e3_start_summary(grid)
    for game in ("rps", "coin", "four"):
        for m in ("rm", "rmplus", "mw"):
            assert st[(game, m, "uniform")] == {"avg": 0.0, "last": 0.0, "last_max": 0.0, "seeds": 4}, (game, m)                 # exakt null: alle Regrets bleiben null
            assert st[(game, m, "random")]["last"] > 0.2 and st[(game, m, "random")]["avg"] > 0
        for m in ("fp", "rm_sampled"):
            assert st[(game, m, "uniform")]["avg"] > 0 and st[(game, m, "uniform")]["last_max"] > 0                                 # diese Verfahren stehen nie still
    n = numbers(grid)
    assert (n["rps_last_rm"], n["rps_last_mw"], n["rps_last_rmplus"]) == ("1,673", "0,799", "0,969")


def test_h7_estimation(grid):
    n = numbers(grid)
    assert (n["est_uni_1"], n["est_uni_5"], n["est_uni_1000"], n["est_uni_full"], n["est_ctl_1000"], n["est_ctl_full"], n["est_main_5"], n["est_main_full"]) == ("85 %", "62 %", "100 %", "0,1252", "89 %", "0,0248", "94 %", "0,3398")
    eq = [r for r in grid["e2est"] if "Gleichgewicht" in r["opponent"]]
    assert len(eq) == 14 and all(abs(r["gain"]) < 1e-12 and r["share"] is None for r in eq)
    uni = {r["T"]: r["share"] for r in grid["e2est"] if r["opponent"] == "Kontrolleur prüft gleichverteilt"}
    assert uni[5] < uni[1] < uni[1000]                                                                                  # nicht monoton: nach 5 Zügen weniger als nach 1
    ctl = {r["T"]: r["share"] for r in grid["e2est"] if r["opponent"] == "Verlader wählt gleichverteilt"}
    assert ctl[1] < 0 and 0.85 < ctl[1000] < 0.9 and ctl[500] < 0.9                                                       # 90 % des Gewinns werden bis T = 1 000 nicht erreicht
    main = {r["T"]: r["share"] for r in grid["e2est"] if r["opponent"] == "Verlader wählt meist Route 1 (0,7)"}
    assert main[1] < 0.9 < main[5]


def test_h8_the_tab(grid):
    n = numbers(grid)
    assert (n["ns_games"], n["bm_games"], n["bm_max"], n["n2_count"], n["n8_count"], n["cov2"], n["cov8"]) == ("960", "4440", "17", "1,26", "6,05", "96 %", "56 %")
    assert (n["piv8"], n["pivmax8"], n["only1_2"], n["only1_8"]) == ("8,3", "79", "87 %", "9 %")
    assert (n["big40_mean"], n["big40_med"], n["big40_max"], n["big20_mean"], n["pairs10"], n["pairs8"]) == ("97,2", "40", "777", "23,6", "184 755", "12 869")
    rows = ev.e4_summary(grid)
    assert all(r["even"] == 0 and r["lh_errors"] == 0 for r in rows) and [r["mean_count"] for r in rows] == sorted(r["mean_count"] for r in rows)         # nie gerade, kein Fehler, die Zahl wächst mit n
    assert all(v["only_one"] == 1.0 and v["deviation"] < 1e-12 for v in ev.e4_nullsum_summary(grid).values())
    assert all(v["errors"] == 0 for v in ev.e4_big_summary(grid).values())
    s = grid["shapley"]
    assert (n["shap_lo"], n["shap_hi"], n["shap_sw"], n["shap_zero"], n["shap_rm_avg"], n["shap_rm_last"], n["shap_rounds"]) == ("0,173", "0,267", "13", "0,0015", "0,206", "1,000", "200 000")
    assert min(w[0] for w in s["windows"].values()) > 0.1 and s["zero_last"] < 0.01 and s["rm_gap_avg"] > 0.1 and s["rm_gap_last"] == pytest.approx(1.0)


def test_the_shapley_equilibrium_is_unique_and_uniform_and_fp_does_not_reach_it():
    A, B = Bm.shapley_game()
    ((x, y),) = Bm.enumerate_nash(A, B)
    assert list(x) == pytest.approx([1 / 3] * 3) and list(y) == pytest.approx([1 / 3] * 3)
    cyc = ev.shapley_cycle(5000)
    lo, hi = ev.window_range(cyc["dist"], 2501, 5000)
    assert lo > 0.1 and hi < 0.4 and cyc["zero_dist"][-1] < 0.02


def test_the_preset_help_texts_carry_the_measured_numbers():
    h = C.PRESET_HELP
    assert "Wert des Spiels ist 3/4" in h["Vier gleiche Routen (Voreinstellung)"] and ev.analyse(Settings()).value == pytest.approx(0.75)
    assert "Wert des Spiels ist 1/2" in h["Münzwurf (zwei gleiche Routen)"] and ev.analyse(Settings(routes=(1.0, 1.0))).value == pytest.approx(0.5)
    a = ev.analyse(Settings(routes=(1.0, 0.75, 0.5, 0.25)))
    assert a.form["active"] == [0, 1, 2] and a.sol.x[3] == pytest.approx(0.0, abs=1e-9) and a.sol.y[3] == pytest.approx(0.0, abs=1e-9) and "vierte Route" in h["Vier Routen mit Streichen"]
    b = ev.analyse(Settings(routes=(1.0, 0.1)))
    assert b.sol.x[1] == pytest.approx(10 * b.sol.x[0]) and "10-mal so oft" in h["Ungleiche Werte (10 : 1)"]
    assert ev.analyse(Settings(game="rps")).value == pytest.approx(0.0, abs=1e-12) and "Wert 0" in h["Stein-Schere-Papier"]
    assert ev.analyse(Settings(game="saddle")).value == pytest.approx(3.0) and "Wert 3" in h["Sattelpunkt"]
    assert "1 000 Runden" in h["Rotation gegen Folgezähler"] and "1 000 Runden" in h["Rotation gegen Häufigkeitszähler"]
    assert C.PRESETS["Rotation gegen Folgezähler"]["obs_rounds"] == 1000 and C.PRESETS["Rotation gegen Häufigkeitszähler"]["obs_rounds"] == 1000
    follow = ev.observer_run(Settings(counter="follow", obs_rounds=1000))["obs"].mean_delivery
    freq_random = ev.observer_run(Settings(counter="freq", tie="random", obs_rounds=1000))["obs"].mean_delivery
    assert follow < 0.01 and freq_random > 0.9


def test_the_verdict_numbers_in_the_experiments_are_the_readme_numbers(grid):
    n = numbers(grid)
    text1 = ev.verdict_e1(ev.aggregate_e1(grid["e1"]), ev.e1b_summary(grid), ev.e1_int_summary(grid))[1]
    assert n["sad22"] in text1 and n["sad1010"] in text1 and n["dev_form"] in text1
    text2 = ev.verdict_e2(ev.e2_lookup(grid, DEFAULT_PROFILE), grid["capture"], "Vier gleiche Routen", C.E2_T)[1]
    assert n["rot_follow"] in text2 and n["rot_freq_low"] in text2 and n["iid_lo"] in text2 and n["iid_hi"] in text2
    text4 = ev.verdict_e4(ev.e4_summary(grid), ev.e4_nullsum_summary(grid), ev.e4_big_summary(grid), grid["shapley"])[1]
    assert n["n2_count"] in text4 and n["n8_count"] in text4 and n["cov2"] in text4 and n["big40_max"] in text4 and n["shap_lo"] in text4
    text3 = ev.verdict_e3(ev.e3_summary(grid), ev.e3_start_summary(grid))[1]
    assert n["avg_lo"] in text3 and n["avg_hi"] in text3 and n["last_hi"] in text3
