"""Verdikte der Experimente an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Bedingung steht hier."""


import nsm_evaluation as ev


def _e1_rows(sad2=0.65, sad10=0.0):
    return [{"m": 2, "n": 2, "games": 4000, "saddle": sad2, "formula": 2 / 3, "gain": 0.035}, {"m": 10, "n": 10, "games": 4000, "saddle": sad10, "formula": 0.0001, "gain": 0.253}]


def test_verdict_e1_quotes_measurement_formula_and_deviation():
    e1b = {(0.3, 2): {"deviation": 2e-16}, (0.3, 4): {"deviation": 3e-16}}
    ints = {2: {"multi_x": 0.15}, 3: {"multi_x": 0.10}}
    kind, text = ev.verdict_e1(_e1_rows(), e1b, ints)
    assert kind == "success"
    assert "4000 je Größe" in text and "**65,0 %** (2 × 2; Formel n!²/(2n−1)! gibt 66,7 %)" in text and "**0,0 %** (10 × 10; Formel 0,01 %)" in text
    assert "steigt von 0,035 auf 0,253" in text and "in 2 Zellen höchstens um 3,0·10⁻¹⁶" in text and "in 10 % bis 15 % der Spiele" in text
    e1b[(0.3, 4)]["deviation"] = 1e-6
    assert ev.verdict_e1(_e1_rows(), e1b, ints)[0] == "warning"                       # Schwelle 1e-9


def _lookup(**over):
    base = {("rotation", "follow", "low"): 0.003, ("rotation", "freq", "low"): 0.75, ("rotation", "freq", "random"): 0.947}
    base.update(over)
    cells = {k: {"mean": v, "value": 0.75} for k, v in base.items()}
    for counter, tie, v in (("freq", "low", 0.7545), ("window", "low", 0.7662), ("fixed", "low", 0.7442), ("follow", "random", 0.7548)):
        cells[("iid_eq", counter, tie)] = {"mean": v, "value": 0.75}
    return cells


def test_verdict_e2_quotes_rotation_values_and_the_equilibrium_band():
    capture = {"2": 3, "8": 9}
    kind, text = ev.verdict_e2(_lookup(), capture, "Vier gleiche Routen", 1000)
    assert kind == "info" and "Wert des Spiels 0,750" in text and "nur **0,003** je Sendung" in text and "ab Runde 3 bei zwei, ab Runde 9 bei acht gleichen Routen" in text
    assert "Gleichstand „kleinster Index“ 0,750, mit zufälligem Gleichstand **0,947**" in text and "zwischen 0,744 und 0,766" in text      # kleinster und größter Wert der vier iid-Zellen (0,7442 → 0,744)
    assert "Mittel über 4 Zufalls-Seeds" in text and "1000 Runden" in text


def _est_rows():
    rows = []
    for T, full_share in ((1, 0.85), (1000, 1.0)):
        rows.append({"opponent": "Kontrolleur prüft gleichverteilt", "T": T, "gain": 0.1252 * full_share, "full": 0.1252, "share": full_share})
        rows.append({"opponent": "Verlader wählt gleichverteilt", "T": T, "gain": 0.0248 * (0.895 if T == 1000 else -0.5), "full": 0.0248, "share": 0.895 if T == 1000 else -0.5})
        rows.append({"opponent": "Kontrolleur im Gleichgewicht y*", "T": T, "gain": 0.0, "full": 0.0, "share": None})
    return rows


def test_verdict_e2_estimate_names_zero_gain_and_shares():
    kind, text = ev.verdict_e2_estimate(_est_rows())
    assert kind == "info" and "(1 bis 1000) 0:" in text and "nach 1000 Zügen 100 % des vollen Gewinns 0,1252 (nach 1 Zug 85 %)" in text and "nach 1000 Zügen 90 % des vollen Gewinns 0,0248" in text
    rows = _est_rows()
    rows[2]["gain"] = 0.01
    assert "(1 bis 1000) nicht 0:" in ev.verdict_e2_estimate(rows)[1]                # die Aussage „0“ hängt an der Bedingung


def _e3(avg_last=0.0092, last=0.456):
    s = {m: {"horizons": [10, 100, 1000, 10000], "avg": [0.2, 0.05, 0.01, 0.003], "last": [0.4, 0.4, 0.4, 0.3], "games": 20} for m in ("rm", "rmplus", "mw", "fp", "rm_sampled")}
    s["mw"]["avg"][3] = avg_last
    s["fp"]["last"][3] = last
    return s


def _start(stand=True):
    out = {}
    for m in ("rm", "rmplus", "mw"):
        out[("rps", m, "uniform")] = {"avg": 0.0 if stand else 0.01, "last": 0.0, "last_max": 0.0}
        out[("rps", m, "random")] = {"avg": 0.02, "last": {"rm": 1.673, "rmplus": 0.969, "mw": 0.799}[m], "last_max": 2.0}
    return out


def test_verdict_e3_counts_methods_and_quotes_ranges():
    kind, text = ev.verdict_e3(_e3(), _start())
    assert kind == "info" and "20 Spiele" in text and "bei allen fünf Verfahren von T = 10 auf T = 10 000 (dort 0,0030 bis 0,0092)" in text
    assert "bei T = 10 000 zwischen 0,300 und 0,456" in text and "bleiben Regret Matching, Regret Matching+ (RM+), Multiplicative Weights (Hedge) bei Start in der Gleichverteilung" in text
    assert "im Mittel um 0,799 bis 1,673 ausnutzbar" in text
    bad = _e3()
    bad["rm"]["avg"][3] = 0.5                                                           # steigt statt zu fallen
    assert "bei nicht allen Verfahren" in ev.verdict_e3(bad, _start())[1]
    assert "bleiben  bei Start" in ev.verdict_e3(_e3(), _start(stand=False))[1]         # keine Verfahren stehen still, die Liste ist leer


def test_verdict_e4_quotes_counts_parity_and_pivots():
    rows = [{"n": 2, "games": 1200, "mean_count": 1.26, "even": 0, "lh_errors": 0, "coverage": 0.956, "mean_pivots": 2.9, "max_pivots": 5},
            {"n": 8, "games": 80, "mean_count": 6.05, "even": 0, "lh_errors": 0, "coverage": 0.557, "mean_pivots": 8.3, "max_pivots": 79}]
    nulls = {2: {"games": 240, "only_one": 1.0}, 3: {"games": 240, "only_one": 1.0}}
    big = {20: {"mean": 23.6, "median": 13.0, "max": 301}, 40: {"mean": 97.2, "median": 40.0, "max": 777}}
    shap = {"rounds": 200000, "windows": {"1000-9999": (0.1735, 0.2671), "10000-99999": (0.1727, 0.2619)}, "switches": 13, "zero_last": 0.0015}
    kind, text = ev.verdict_e4(rows, nulls, big, shap)
    assert kind == "info" and "1280 Spiele für n = 2 bis 8" in text and "von **1,26** auf **6,05**" in text and "darunter 0 Spiele mit gerader Zahl" in text
    assert "in 480 von 480 Spielen **genau ein** Gleichgewicht" in text and "(0 Fehler)" in text and "finden 96 % (n = 2) bis 56 % (n = 8)" in text
    assert "Mittel 8,3 Pivots bei n = 8 (Maximum 79), bei n = 40 Mittel 97,2, Median 40, Maximum 777" in text
    assert "ab Runde 1 000 bis Runde 200 000 zwischen 0,173 und 0,267 (13 Wechsel" in text and "am Ende 0,0015" in text
    rows[0]["even"] = 3
    assert "darunter 3 Spiele mit gerader Zahl" in ev.verdict_e4(rows, nulls, big, shap)[1]
