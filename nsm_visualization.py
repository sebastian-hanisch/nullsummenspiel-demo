"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import nsm_constants as C

LOG_FLOOR = 1e-6          # Nullen werden auf einer logarithmischen Achse hier gezeichnet (Hinweis unter dem Diagramm)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height: int = 320, legend: bool = True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", yanchor="top", y=-0.25) if legend else None)
    return lock_axes(fig)


def build_strategy(p, lo, hi, names, color, ytitle: str, unique: bool, dropped=(), height: int = 300):
    """Balken der optimalen Strategie je Aktion; ist sie nicht eindeutig, zeigen die Fehlerbalken den Bereich über alle optimalen Strategien. Gestrichene Routen sind grau."""
    p, lo, hi = np.asarray(p, float), np.asarray(lo, float), np.asarray(hi, float)
    colors = [C.COLOR_DROPPED if i in dropped else color for i in range(len(p))]
    labels = [f"{n} (gestrichen)" if i in dropped else n for i, n in enumerate(names)]
    error = None if unique else dict(type="data", symmetric=False, array=np.maximum(hi - p, 0.0), arrayminus=np.maximum(p - lo, 0.0), color=C.COLOR_VALUE)
    fig = go.Figure(go.Bar(x=labels, y=p, marker_color=colors, text=[f"{100 * v:.1f} %".replace(".", ",") for v in p], textposition="outside", error_y=error, showlegend=False))
    fig.update_yaxes(title_text=ytitle, range=[0, 1.12], tickformat=".0%")
    return _base(fig, height, legend=False)


def build_rules(names, values, value: float, ytitle: str, color, height: int = 300):
    """Balken: was jede Regel garantiert (Verlader) bzw. höchstens zulässt (Kontrolleur), mit dem Wert des Spiels als Linie."""
    fig = go.Figure(go.Bar(x=list(names), y=list(values), marker_color=color, text=[f"{v:.3f}".replace(".", ",") for v in values], textposition="outside", showlegend=False))
    fig.add_hline(y=value, line=dict(color=C.COLOR_VALUE, dash="dash", width=2), annotation_text="Wert des Spiels", annotation_position="top right")
    top = max(max(values), value, 1e-9)
    fig.update_yaxes(title_text=ytitle, range=[min(0, min(values)) * 1.2 if min(values) < 0 else 0, top * 1.25])
    return _base(fig, height, legend=False)


def build_learning(steps, expl_avg, expl_last, height: int = 340):
    """Ausnutzbarkeit der Durchschnittsstrategie und der letzten Iteration gegen die Rundenzahl (beide Achsen logarithmisch; Nullen stehen bei 10⁻⁶)."""
    avg = np.maximum(np.asarray(expl_avg, float), LOG_FLOOR)
    last = np.maximum(np.asarray(expl_last, float), LOG_FLOOR)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=avg, mode="lines+markers", name="Durchschnittsstrategie", line=dict(color=C.COLOR_AVG, width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=steps, y=last, mode="lines+markers", name="letzte Iteration", line=dict(color=C.COLOR_LAST, width=2), marker=dict(size=4)))
    fig.update_xaxes(title_text="Runden", type="log")
    fig.update_yaxes(title_text="Ausnutzbarkeit", type="log", range=[np.log10(LOG_FLOOR) - 0.1, 0.5])
    return _base(fig, height)


def build_observer(running_mean, value: float, band: float, height: int = 320):
    """Mittlere Lieferung der ersten t Runden gegen den Wert des Spiels (Band ± band)."""
    t = np.arange(1, len(running_mean) + 1)
    fig = go.Figure()
    fig.add_hrect(y0=value - band, y1=value + band, fillcolor="rgba(120,120,120,0.15)", line_width=0)
    fig.add_trace(go.Scatter(x=t, y=running_mean, mode="lines", name="Lieferung je Sendung (Mittel der ersten t Runden)", line=dict(color=C.COLOR_LOADER, width=3)))
    fig.add_hline(y=value, line=dict(color=C.COLOR_VALUE, dash="dash", width=2), annotation_text="Wert des Spiels", annotation_position="top right")
    fig.update_xaxes(title_text="Runde t")
    fig.update_yaxes(title_text="Lieferung je Sendung", range=[0, max(1.0, float(np.max(running_mean)) * 1.05, value * 1.2)])
    return _base(fig, height)


def build_shapley(dist, zero_dist, height: int = 320):
    """Abstand der Häufigkeiten zur Gleichverteilung (Fictitious Play) im Shapley-Spiel und in seinem Nullsummen-Gegenstück, logarithmische Rundenachse."""
    t = np.arange(1, len(dist) + 1)
    idx = np.unique(np.round(np.geomspace(1, len(dist), 300)).astype(int)) - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[idx], y=np.asarray(dist)[idx], mode="lines", name="Shapley-Spiel", line=dict(color=C.COLOR_CONTROL, width=3)))
    fig.add_trace(go.Scatter(x=t[idx], y=np.asarray(zero_dist)[idx], mode="lines", name="Nullsummen-Gegenstück (B = 1 − A)", line=dict(color=C.COLOR_AVG, width=2, dash="dash")))
    fig.update_xaxes(title_text="Runden", type="log")
    fig.update_yaxes(title_text="Abstand zur Gleichverteilung", range=[0, 0.7])
    return _base(fig, height)


def build_e1(rows, height: int = 320):
    """E1: Anteil der Zufallsspiele mit Sattelpunkt (Messung) gegen die Formel n!²/(2n−1)!, quadratische Spiele."""
    sq = [r for r in rows if r["m"] == r["n"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{r['n']} × {r['n']}" for r in sq], y=[r["saddle"] for r in sq], name="gemessen", marker_color=C.COLOR_LOADER))
    fig.add_trace(go.Scatter(x=[f"{r['n']} × {r['n']}" for r in sq], y=[r["formula"] for r in sq], mode="markers", name="Formel n!²/(2n−1)!", marker=dict(color=C.COLOR_CONTROL, size=11, symbol="diamond")))
    fig.update_yaxes(title_text="Anteil mit Sattelpunkt", range=[0, 0.75], tickformat=".0%")
    return _base(fig, height)


def build_e2_estimate(rows, scenarios, height: int = 320):
    """E2 (Schätzen): Anteil des vollen Gewinns, den ein Beobachter nach T Zügen erreicht, je Gegner."""
    palette = [C.COLOR_AVG, C.COLOR_LOADER, C.COLOR_CONTROL, C.COLOR_LAST, C.COLOR_VALUE]
    fig = go.Figure()
    for k, name in enumerate(scenarios):
        pts = sorted([r for r in rows if r["opponent"] == name and r["share"] is not None], key=lambda r: r["T"])
        if pts:
            fig.add_trace(go.Scatter(x=[r["T"] for r in pts], y=[r["share"] for r in pts], mode="lines+markers", name=name, line=dict(color=palette[k % len(palette)], width=2)))
    fig.update_xaxes(title_text="beobachtete Züge T", type="log")
    fig.update_yaxes(title_text="Anteil des vollen Gewinns", range=[-0.6, 1.1], tickformat=".0%")
    return _base(fig, height)


def build_e3(table, height: int = 340):
    """E3: mittlere Ausnutzbarkeit der Durchschnittsstrategie (durchgezogen) und der letzten Iteration (gestrichelt) je Verfahren und Horizont."""
    palette = {"rm": C.COLOR_AVG, "rmplus": C.COLOR_LOADER, "mw": C.COLOR_LAST, "fp": C.COLOR_CONTROL, "rm_sampled": C.COLOR_VALUE}
    fig = go.Figure()
    for method, row in table.items():
        fig.add_trace(go.Scatter(x=row["horizons"], y=row["avg"], mode="lines+markers", name=C.LEARN_METHODS[method], line=dict(color=palette[method], width=2)))
        fig.add_trace(go.Scatter(x=row["horizons"], y=row["last"], mode="lines", name=f"{C.LEARN_METHODS[method]}, letzte Iteration", line=dict(color=palette[method], width=1, dash="dot"), showlegend=False))
    fig.update_xaxes(title_text="Runden T", type="log")
    fig.update_yaxes(title_text="mittlere Ausnutzbarkeit", type="log")
    return _base(fig, height)


def build_e4(rows, height: int = 320):
    """E4: mittlere Zahl der Gleichgewichte und Anteil der Spiele mit genau einem Gleichgewicht, je Größe n."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"n = {r['n']}" for r in rows], y=[r["mean_count"] for r in rows], name="mittlere Zahl der Gleichgewichte", marker_color=C.COLOR_LOADER))
    fig.add_trace(go.Scatter(x=[f"n = {r['n']}" for r in rows], y=[r["only_one"] for r in rows], mode="lines+markers", name="Anteil mit genau einem", yaxis="y2", line=dict(color=C.COLOR_CONTROL, width=2)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], tickformat=".0%", fixedrange=True))
    fig.update_yaxes(title_text="Zahl der Gleichgewichte", rangemode="tozero")
    return _base(fig, height)
