"""Abbildungen: jede baut sich, alle Achsen sind gesperrt, Nullen stehen auf logarithmischen Achsen über dem Boden, gestrichene Routen sind grau."""

import numpy as np
import plotly.graph_objects as go
import pytest

import nsm_constants as C
import nsm_evaluation as ev
import nsm_visualization as V


def _locked(fig):
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_strategy_chart_marks_dropped_routes_and_shows_ranges_only_when_not_unique():
    p = [0.2, 0.3, 0.5, 0.0]
    fig = V.build_strategy(p, [0.2, 0.3, 0.5, 0.0], [0.2, 0.3, 0.5, 0.0], ["a", "b", "c", "d"], C.COLOR_LOADER, "Wahrscheinlichkeit", True, {3})
    _locked(fig)
    assert list(fig.data[0].x)[3] == "d (gestrichen)" and fig.data[0].marker.color[3] == C.COLOR_DROPPED and fig.data[0].error_y.array is None
    fig = V.build_strategy([0.5, 0.5], [0.0, 0.0], [1.0, 1.0], ["a", "b"], C.COLOR_LOADER, "W", False)
    assert list(fig.data[0].error_y.array) == [0.5, 0.5] and list(fig.data[0].error_y.arrayminus) == [0.5, 0.5]


def test_rules_chart_has_the_value_line_and_handles_negative_values():
    fig = V.build_rules(["a", "b"], [0.2, 0.5], 0.5, "garantiert", C.COLOR_LOADER)
    _locked(fig)
    assert fig.layout.shapes[0].y0 == 0.5
    neg = V.build_rules(["a", "b"], [-1.0, 0.0], 0.0, "garantiert", C.COLOR_LOADER)
    assert neg.layout.yaxis.range[0] < 0


def test_learning_chart_puts_zeros_at_the_floor_on_log_axes():
    fig = V.build_learning([1, 10, 100], [0.5, 0.0, 0.01], [0.0, 0.0, 0.0])
    _locked(fig)
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert min(fig.data[0].y) == V.LOG_FLOOR and set(fig.data[1].y) == {V.LOG_FLOOR}


def test_observer_chart_has_band_and_value_line():
    fig = V.build_observer(np.linspace(0.2, 0.75, 50), 0.75, 0.05)
    _locked(fig)
    assert fig.layout.shapes[0].y0 == pytest.approx(0.70) and fig.layout.shapes[0].y1 == pytest.approx(0.80)


def test_shapley_chart_downsamples_long_runs():
    dist = np.linspace(0.3, 0.2, 200000)
    fig = V.build_shapley(dist, dist * 0.01)
    _locked(fig)
    assert len(fig.data[0].x) <= 300 and fig.data[0].x[-1] == 200000


def test_experiment_charts_build_from_the_measurement_files():
    grid = ev.load_grid()
    rows = ev.aggregate_e1(grid["e1"])
    fig = V.build_e1(rows)
    _locked(fig)
    assert len(fig.data[0].x) == 6 and len(fig.data[1].y) == 6                          # quadratische Größen 2, 3, 4, 5, 7, 10
    est = grid["e2est"]
    scenarios = list(dict.fromkeys(r["opponent"] for r in est))
    fig = V.build_e2_estimate(est, scenarios)
    _locked(fig)
    assert len(fig.data) == 3                                                           # die zwei Gleichgewichtsgegner haben keinen Anteil (voller Gewinn 0)
    summary = ev.e3_summary(grid)
    table = {m: {"horizons": s["horizons"], "avg": s["avg"], "last": s["last"]} for m, s in summary.items()}
    fig = V.build_e3(table)
    _locked(fig)
    assert len(fig.data) == 10
    fig = V.build_e4(ev.e4_summary(grid))
    _locked(fig)
    assert len(fig.data[0].x) == 7
