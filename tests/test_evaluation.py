"""Auswertung: Formatierung, Analyse der Hauptansicht an Handwerten, Projektionen für den Zwischenspeicher, Reiter-Rechnungen und die Zusammenfassungen der Messreihen an synthetischen Zeilen."""

import numpy as np
import pytest

import nsm_evaluation as ev
import nsm_game as G
from nsm_scenario import Settings, draw_matrix


def test_formatting():
    assert ev.de(0.5) == "0,500" and ev.de(-0.0004) == "0,000" and ev.de(1234.5678, 2) == "1234,57"          # kein „-0,000“
    assert ev.pct(0.1234) == "12,3 %" and ev.pct(-0.00001, 1) == "0,0 %" and ev.pct(1.0, 0) == "100 %"
    assert ev.sci(1.2e-14) == "1,2·10⁻¹⁴" and ev.sci(0) == "0" and ev.sci(7.5e-16) == "7,5·10⁻¹⁶"


def test_analysis_of_the_ten_to_one_route_game_by_hand():
    """Werte (1; 0,1) sind (10; 1) / 10: Wert 1/11, x = (1/11, 10/11), Sicherheitsniveau 0, Gewinn durch Mischen 1/11. Faustregel „proportional zum Wert“: x = (10/11, 1/11) garantiert 0,1/11 = 1/110."""
    a = ev.analyse(Settings(routes=(1.0, 0.1)))
    assert a.value == pytest.approx(1 / 11) and a.security == 0.0 and a.unique and a.saddles == ()
    assert list(a.sol.x) == pytest.approx([1 / 11, 10 / 11]) and list(a.sol.y) == pytest.approx([10 / 11, 1 / 11])
    assert a.form["active"] == [0, 1] and a.form["deviation"] < 1e-9
    rows = dict(a.loader_rows)
    assert rows["proportional zum Wert"] == pytest.approx(1 / 110) and rows["Gleichgewicht"] == pytest.approx(1 / 11) and rows["gleichverteilt"] == pytest.approx(0.05)
    assert rows["nur die wertvollste Route"] == 0.0 and rows["beste reine Zeile (Sicherheitsniveau)"] == 0.0
    ctl = dict(a.controller_rows)
    assert ctl["proportional zum Wert"] == pytest.approx(1 / 11) and ctl["nur die wertvollste Route"] == pytest.approx(0.1) and ctl["gleichverteilt"] == pytest.approx(0.5)


def test_analysis_of_rock_paper_scissors_and_the_saddle_game():
    rps = ev.analyse(Settings(game="rps"))
    assert rps.value == pytest.approx(0.0, abs=1e-12) and rps.security == -1.0 and rps.pure_ceiling == 1.0 and rps.form is None and rps.unique
    assert list(rps.sol.x) == pytest.approx([1 / 3] * 3)
    saddle = ev.analyse(Settings(game="saddle"))
    assert saddle.value == pytest.approx(3.0) and saddle.security == 3.0 and saddle.saddles == ((0, 1),) and saddle.unique
    assert [n for n, _ in saddle.loader_rows] == ["gleichverteilt", "beste reine Zeile (Sicherheitsniveau)", "Gleichgewicht"]


def test_random_game_matches_the_drawn_matrix():
    s = Settings(game="random", m=3, n=4, game_seed=11)
    assert np.array_equal(ev.analyse(s).A, draw_matrix(3, 4, 11))


def test_verdict_main_branches():
    kind, text = ev.verdict_main(0.5, 0.0, 0, True)
    assert kind == "success" and "Wert des Spiels ist **0,500**" in text and "Mischen bringt +0,500" in text and "nicht eindeutig" not in text
    kind, text = ev.verdict_main(3.0, 3.0, 1, True)
    assert kind == "info" and "Zufall bringt hier nichts" in text and "(1 Sattelpunkt)" in text
    assert "(4 Sattelpunkte)" in ev.verdict_main(1.0, 1.0, 4, False)[1] and "nicht eindeutig" in ev.verdict_main(1.0, 1.0, 4, False)[1]
    assert ev.verdict_main(0.5, 0.5 - 1e-10, 1, True)[0] == "info"          # innerhalb der Rauschgrenze kein Gewinn


def test_rule_shortfalls_and_verdict_rules():
    loader = (("a", 0.5), ("b", 0.4), ("Gleichgewicht", 0.5))
    controller = (("c", 0.5), ("d", 0.6))
    low, high = ev.rule_shortfalls(0.5, loader, controller)
    assert low == [("b", 0.4)] and high == [("d", 0.6)]
    kind, text = ev.verdict_rules(0.5, loader, controller)
    assert kind == "warning" and "b 0,400" in text and "d 0,600" in text and "Wert des Spiels 0,500" in text
    assert ev.verdict_rules(0.5, (("a", 0.5),), (("c", 0.5),)) == ("info", "Alle gezeigten Regeln erreichen den Wert 0,500.")


def test_verdict_learning_branches():
    kind, text = ev.verdict_learning("rm", "uniform", 500, 0.0, 0.0, 0.0)
    assert kind == "warning" and "Start liegt schon im Gleichgewicht" in text
    kind, text = ev.verdict_learning("rm", "random", 2000, 0.0123, 0.4, 0.9)
    assert kind == "info" and "**0,0123**" in text and "0,400" in text and "näher am Gleichgewicht" in text
    assert "nicht schlechter" in ev.verdict_learning("mw", "random", 2000, 0.4, 0.1, 0.9)[1]
    assert "gespielte reine Aktion" in ev.verdict_learning("fp", "random", 100, 0.1, 2.0, 2.0)[1]
    assert ev.verdict_learning("rm", "random", 100, 1e-13, 0.5, 0.5)[0] == "info"           # Durchschnitt schon bei 0, aber die letzte Iteration bewegt sich: kein Stillstand
    assert ev.verdict_learning("rm", "random", 100, 1e-13, 0.0, 0.0)[0] == "warning"


def test_verdict_observer_branches_and_band():
    band = ev.BAND
    kind, text = ev.verdict_observer("rotation", "follow", "low", 0.75, 0.003, 0.997, 1000)
    assert kind == "warning" and "**0,003**" in text and "99,7 %" in text and "Rotation gegen Folgezähler, Gleichstand kleinster Index" in text
    kind, text = ev.verdict_observer("rotation", "freq", "random", 0.75, 0.947, 0.053, 1000)
    assert kind == "success" and "nutzt das Muster nicht aus" in text
    assert ev.verdict_observer("iid_eq", "fixed", "low", 0.75, 0.75 + band, 0.25, 500)[0] == "info"          # genau auf dem Rand: noch im Band
    assert ev.verdict_observer("iid_eq", "fixed", "low", 0.75, 0.75 + band + 1e-9, 0.25, 500)[0] == "success"
    assert ev.verdict_observer("iid_eq", "fixed", "low", 0.75, 0.75 - band - 1e-9, 0.25, 500)[0] == "warning"
    assert "Gleichstand" not in ev.verdict_observer("iid_eq", "fixed", "low", 0.75, 0.75, 0.25, 500)[1]       # die feste Strategie kennt keinen Gleichstand


def test_cache_projections_ignore_unrelated_settings():
    a = Settings(game="rps", learn="mw", rounds=500, pattern="habit", counter="window", bm="shapley")
    b = Settings(game="rps", learn="mw", rounds=500, pattern="rotation", counter="follow", bm="random", obs_rounds=100)
    assert ev.learn_settings(a) == ev.learn_settings(b) and ev.learn_settings(a) != ev.learn_settings(Settings(game="rps", learn="rm", rounds=500))
    c = Settings(routes=(1.0, 0.5), pattern="habit", counter="window", game="rps", learn="fp")
    d = Settings(routes=(1.0, 0.5), pattern="habit", counter="window", game="saddle", learn="rm", rounds=100)
    assert ev.observer_settings(c) == ev.observer_settings(d) and ev.observer_settings(c) != ev.observer_settings(Settings(routes=(1.0, 0.5), pattern="norepeat", counter="window"))


def test_learn_run_reports_the_lp_value_and_the_peak_of_the_last_iterate():
    s = Settings(game="rps", learn="rm", rounds=300, start="random", seed=3)
    out = ev.learn_run(s)
    assert out["value"] == pytest.approx(0.0, abs=1e-9) and out["peak_last"] == pytest.approx(float(out["run"].expl_last.max())) and out["peak_last"] > 0.1
    still = ev.learn_run(Settings(game="rps", learn="rm", rounds=300, start="uniform"))
    assert still["peak_last"] <= 1e-12


def test_observer_run_overview_has_the_seven_counter_combinations_and_matches_the_chosen_one():
    s = Settings(pattern="rotation", counter="follow", tie="low", obs_rounds=300)
    out = ev.observer_run(s)
    assert [(o["counter"], o["tie"]) for o in out["overview"]] == list(ev.COUNTER_COMBOS)
    chosen = next(o for o in out["overview"] if (o["counter"], o["tie"]) == ("follow", "low"))
    assert chosen["mean"] == pytest.approx(out["obs"].mean_delivery) and out["value"] == pytest.approx(0.75)
    assert next(o for o in out["overview"] if (o["counter"], o["tie"]) == ("freq", "low"))["mean"] == pytest.approx(0.75)


def test_bimatrix_games_and_analysis():
    A, B = ev.bimatrix_game("shapley", 7, 1)
    assert A.shape == (3, 3) and A.tolist() == np.eye(3).tolist()
    A, B = ev.bimatrix_game("zero_sum", 4, 9)
    assert np.allclose(A + B, 1.0)
    A, B = ev.bimatrix_game("random", 4, 9)
    assert not np.allclose(A + B, 1.0) and A.shape == (4, 4)
    shap = ev.analyse_bimatrix("shapley", 3, 1)
    assert len(shap.equilibria) == 1 and shap.coverage == 1.0 and shap.pure == 0 and len(shap.lh_rows) == 6 and all(r["found"] == 0 for r in shap.lh_rows)
    zs = ev.analyse_bimatrix("zero_sum", 3, 4)
    assert len(zs.equilibria) == 1 and zs.zero_sum_value == pytest.approx(float(zs.equilibria[0][0] @ zs.A @ zs.equilibria[0][1]), abs=1e-9)
    rnd = ev.analyse_bimatrix("random", 4, 12)
    assert len(rnd.equilibria) % 2 == 1 and 0 < rnd.coverage <= 1 and len(rnd.lh_rows) == 8 and zs.zero_sum_value is not None and rnd.zero_sum_value is None


def test_verdict_bimatrix_branches():
    kind, text = ev.verdict_bimatrix("zero_sum", 1, 0, 1.0, 0.5, 0.5)
    assert kind == "success" and "1 Gleichgewicht**" in text
    assert ev.verdict_bimatrix("zero_sum", 3, 0, 1.0, 0.5, 0.5)[0] == "warning"
    assert ev.verdict_bimatrix("zero_sum", 1, 0, 1.0, 0.5, 0.6)[0] == "warning"
    kind, text = ev.verdict_bimatrix("random", 3, 1, 0.667, None, None)
    assert kind == "info" and "**3 Gleichgewichte** (ungerade Zahl; davon 1 in reinen Strategien)" in text and "67 %" in text
    assert "**1 Gleichgewicht** (ungerade Zahl; davon 0 in reinen Strategien)" in ev.verdict_bimatrix("random", 1, 0, 1.0, None, None)[1] and "100 %" in ev.verdict_bimatrix("random", 1, 0, 1.0, None, None)[1]
    assert "**4 Gleichgewichte** (gerade Zahl" in ev.verdict_bimatrix("random", 4, 0, 0.5, None, None)[1]


def test_window_range_and_shapley_cycle_shapes():
    assert ev.window_range([5, 1, 3, 2, 9], 2, 4) == (1.0, 3.0)
    cyc = ev.shapley_cycle(2000)
    assert len(cyc["dist"]) == 2000 and len(cyc["zero_dist"]) == 2000 and cyc["switches"] >= 1
    lo, hi = ev.window_range(cyc["dist"], 1001, 2000)
    assert 0.05 < lo <= hi < 0.6 and float(cyc["zero_dist"][-1]) < lo


def test_verdict_shapley_names_the_range():
    kind, text = ev.verdict_shapley(20000, 0.17, 0.27, 0.0049)
    assert kind == "warning" and "zwischen **0,170** und **0,270**" in text and "0,005" in text and "20000 Runden" in text


def test_aggregate_e1_on_synthetic_cells():
    cells = [{"m": m, "n": n, "seed": s, "games": 10, "saddle": sad, "value": 0.5, "security": 0.4, "gain": 0.1, "gain_nonsaddle": gn, "support": 2.0, "certificate": 1e-15}
             for (m, n) in ((3, 3), (2, 2), (2, 4)) for s, sad, gn in ((1, 0.5, 0.2), (2, 0.7, None))]
    rows = ev.aggregate_e1(cells)
    assert [(r["m"], r["n"]) for r in rows] == [(2, 2), (3, 3), (2, 4)]
    assert rows[0]["saddle"] == pytest.approx(0.6) and rows[0]["saddle_std"] == pytest.approx(0.1) and rows[0]["games"] == 20 and rows[0]["gain_nonsaddle"] == pytest.approx(0.2)
    assert rows[0]["formula"] == pytest.approx(4 / 6) and rows[1]["formula"] == pytest.approx(36 / 120) and rows[2]["formula"] is None
    assert ev.cells_order((2, 2)) < ev.cells_order((3, 3)) < ev.cells_order((2, 4))


def test_e1_cell_matches_a_direct_computation_on_two_by_two_games():
    """2 × 2: Sattelpunkt, sonst Wert (ad − bc)/(a + d − b − c); der Anteil und der Mittelwert des Wertes der Zelle gleich der direkten Rechnung."""
    cell = ev.e1_cell(2, 2, 11, 40)
    saddles, values = 0, []
    from nsm_scenario import cell_seed
    for g in range(40):
        a, b, c, d = draw_matrix(2, 2, cell_seed(11, 2, 2, g)).ravel()
        if G.saddle_points(np.array([[a, b], [c, d]])):
            saddles += 1
            values.append(max(min(a, b), min(c, d)))
        else:
            values.append((a * d - b * c) / (a + d - b - c))
    assert cell["saddle"] == pytest.approx(saddles / 40) and cell["value"] == pytest.approx(float(np.mean(values)), abs=1e-9)
    assert cell["certificate"] < 1e-9 and cell["games"] == 40
