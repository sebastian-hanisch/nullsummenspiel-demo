"""Nullsummenspiele: Zufall mit Absicht - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Verlader schickt eine Sendung über eine von n Routen, ein Kontrolleur prüft eine Route. Die Demo rechnet den Wert des Spiels und die optimalen gemischten Strategien (lineares Programm, Minimax-Satz, geschlossene Form mit Streichen
von Routen), zeigt, welche Faustregeln ausnutzbar sind, lässt Lernverfahren und Zähler gegen Muster spielen und erklärt in einem kleinen Reiter allgemeine Zwei-Personen-Spiele (Aufzählung, Lemke–Howson, Shapley-Beispiel).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import nsm_constants as C
import nsm_evaluation as ev
import nsm_game as G
from nsm_layout import dunkles_design, kopf_css
from nsm_pdf_export import build_pdf
from nsm_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_game_seed,
    randomize_seed,
    settings_from_state,
    sidebar_presets,
    sync_query_params,
)
from nsm_visualization import (
    build_e1,
    build_e2_estimate,
    build_e3,
    build_e4,
    build_learning,
    build_observer,
    build_rules,
    build_shapley,
    build_strategy,
)

st.set_page_config(page_title="Nullsummenspiele – Sebastian Hanisch", layout="wide")

de = ev.de
pct = ev.pct


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _learn(settings):
    return ev.learn_run(settings)


@st.cache_data(show_spinner=False)
def _observer(settings):
    return ev.observer_run(settings)


@st.cache_data(show_spinner=False)
def _bimatrix(kind, n, seed):
    return ev.analyse_bimatrix(kind, n, seed)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


def _show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in nsm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Nullsummenspiele: Zufall mit Absicht")
st.markdown(
    """
**Ein Verlader schickt eine Sendung über eine von n Routen, ein Kontrolleur prüft eine Route.** Kommt die Sendung durch, zählt ihr Wert; wird ihre Route geprüft, ist sie verloren. Wo es keinen Sattelpunkt gibt, wird eine feste Regel von einem
mitdenkenden Gegner ausgenutzt. Die Fragen: **Wie viel kostet das, wann hilft Zufall, wie gewichtet man ihn, und was kann ein Gegner lernen, der nur zusieht?** Die Demo gehört zur Linie zu Verhandlung, Auktion, Vertrag und Kontrolle und ist
dort der Boden für alles, was mit Zufall und Festlegung zu tun hat. Abgrenzung: `nash-demo` rechnet das 2×2-Mini-Spiel und reine Gleichgewichte, `noregret-demo` Hedge und EXP3 im Torwahl-Spiel, `minimax-demo` Baumsuche mit offenen Zügen;
hier geht es um gemischte Strategien in einer Auszahlungsmatrix, ihren Wert und das Lernen darin.

Weiter unten: das Experiment **🔬 Mischen hilft**, drei Reiter (Beobachter mit Experiment 2, Lernen mit Experiment 3, Allgemeine Spiele mit Experiment 4), **🚧 Wo die Annahmen enden**, der Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und
der Expander **📐 Mathematische Formulierung**. **Eigennutz und ein mitdenkender Gegner sind Modellannahmen, keine Aussagen über Betriebe oder echte Kontrollen.**
"""
)

with st.expander("So funktioniert ein Nullsummenspiel mit gemischten Strategien", expanded=True):
    st.markdown(
        r"""
Die Matrix **A** hat eine Zeile je Route des Verladers und eine Spalte je geprüfte Route; der Eintrag ist die Auszahlung an den Verlader (hier: der Anteil der Ladung, der ankommt). Der Kontrolleur will sie klein halten.

| Begriff | Was er hier bedeutet |
|---|---|
| **Sicherheitsniveau** | Was der Verlader mit der besten **reinen** Route mindestens bekommt, wenn der Gegner die dazu passende Antwort kennt: das größte Zeilenminimum. |
| **Gemischte Strategie, garantiert / lässt höchstens zu** | Eine Wahrscheinlichkeitsverteilung über die Routen. Der Verlader mit Verteilung x **garantiert** sich das Minimum über die Spalten; der Kontrolleur mit Verteilung y **lässt höchstens** das Maximum über die Zeilen **zu**. |
| **Minimax-Satz, Wert des Spiels** | Das beste Garantierte des Verladers und das beste Zugelassene des Kontrolleurs sind gleich: der Wert des Spiels. Beides ist ein lineares Programm, und die beiden Programme sind dual zueinander. |
| **Indifferenz** | Im Gleichgewicht zahlt jede Route, die der Verlader benutzt, gegen die Strategie des Kontrolleurs gleich viel; deshalb lohnt sich keine Abweichung. |
| **Ausnutzbarkeit** | Wie viel ein Paar (x, y) von Strategien dem Gegner anbietet: was die beste Antwort auf y dem Verlader bringt, minus was die beste Antwort auf x dem Kontrolleur lässt. Null genau im Gleichgewicht. |
| **Regret Matching, RM+** | Jede Seite spielt die Aktionen mit positivem Bedauern (was ihr gegenüber der gespielten Mischung entgangen ist) im Verhältnis dieses Bedauerns; RM+ schneidet negatives Bedauern auf null ab und gewichtet den Durchschnitt linear. |
| **Multiplicative Weights (Hedge)** | Wahrscheinlichkeit proportional zu exp(η · Summe der bisherigen Auszahlungen) mit einer Lernrate η, die vom Horizont abhängt. |
| **Fictitious Play** | Jede Seite spielt die beste Antwort auf die Häufigkeiten, mit denen der Gegner bisher gespielt hat. |
| **Durchschnitt, letzte Iteration** | Die Durchschnittsstrategie ist das Mittel aller gespielten Strategien; die letzte Iteration ist die Strategie der Runde t selbst. |
| **Lemke–Howson** | Ein Pivotverfahren für Zwei-Personen-Spiele: Es startet bei einem Anfangslabel und folgt einem Pfad bis zu einem Nash-Gleichgewicht. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Spiel")
    st.selectbox("Spiel der Hauptansicht", list(C.GAMES), format_func=C.GAMES.get, key="game_select", help="Welche Matrix die Hauptansicht und der Reiter Lernen benutzen. Die übrigen Regler dieser Gruppe wirken nur bei dem Spiel, das sie nennen.")
    st.text_input("Routenwerte v (2 bis 8, getrennt durch „;“, je 0,05 bis 1)", key="routes_input", help="Wert jeder Route als Anteil der Ladung, z. B. „1; 0,75; 0,5; 0,25“. Wirkt im Spiel Verstecken und Suchen der Hauptansicht und im Reiter Beobachter.")
    st.slider("Zeilen m (Zufallsspiel)", *bounds("m_slider"), step=1, key="m_slider", help="Wirkt im Spiel Zufallsspiel m × n der Hauptansicht und im Reiter Lernen.")
    st.slider("Spalten n (Zufallsspiel)", *bounds("n_slider"), step=1, key="n_slider", help="Wirkt im Spiel Zufallsspiel m × n der Hauptansicht und im Reiter Lernen.")
    st.number_input("Zufalls-Seed des Spiels", *bounds("game_seed_input"), key="game_seed_input", step=1, help="Wirkt im Zufallsspiel m × n.")
    st.button("🎲 Neues Zufallsspiel", width="stretch", on_click=randomize_game_seed)
    st.header("📈 Lernen")
    st.selectbox("Lernverfahren", list(C.LEARN_METHODS), format_func=C.LEARN_METHODS.get, key="learn_select", help="Wirkt im Reiter Lernen. RM+ ist aus dem Gedächtnis umgesetzt (Quelle nicht eingesehen).")
    st.radio("Start", list(C.STARTS), format_func=C.STARTS.get, key="start_radio", help="Zufallsstart: beide Seiten beginnen mit einer zufälligen Mischung. Gleichverteilung: Wer dort schon im Gleichgewicht ist, bewegt sich bei Regret Matching, RM+ und Multiplicative Weights nicht. Wirkt im Reiter Lernen.")
    st.slider("Lernrunden", *bounds("rounds_slider"), step=100, key="rounds_slider", help="Bei Multiplicative Weights ist die Rundenzahl der Horizont der Lernrate. Wirkt im Reiter Lernen.")
    st.header("👀 Beobachter")
    st.selectbox("Muster des Verladers", list(C.PATTERNS), format_func=C.PATTERNS.get, key="pattern_select", help="Wie der Verlader seine Routen wählt. Wirkt im Reiter Beobachter.")
    st.selectbox("Kontrolleur", list(C.COUNTERS), format_func=C.COUNTERS.get, key="counter_select", help="Was der Kontrolleur aus den bisherigen Sendungen macht. Wirkt im Reiter Beobachter.")
    st.radio("Gleichstand beim Zähler", list(C.TIES), format_func=C.TIES.get, key="tie_radio", help="Was der Zähler tut, wenn mehrere Routen gleich gut aussehen. Wirkt im Reiter Beobachter bei den drei Zählern (nicht bei der festen Strategie).")
    st.slider("Runden", *bounds("obs_rounds_slider"), step=100, key="obs_rounds_slider", help="Zahl der Sendungen im Reiter Beobachter.")
    st.header("🧮 Allgemeine Spiele")
    st.selectbox("Spiel", list(C.BM_KINDS), format_func=C.BM_KINDS.get, key="bm_select", help="Wirkt im Reiter Allgemeine Spiele. Das Shapley-Beispiel hat feste Matrizen (3 × 3).")
    st.slider("Aktionen n je Spieler", *bounds("bm_n_slider"), step=1, key="bm_n_slider", help="Wirkt im Reiter Allgemeine Spiele bei Zufallsspiel und Nullsumme; die Aufzählung rechnet alle Paare gleich großer Stützen (bei n = 8: 12 869).")
    st.header("🎲 Zufall")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Zufallsstart des Lernens, gezogene Züge und Muster der Beobachter.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

settings, routes_valid = settings_from_state()
sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})
if not routes_valid:
    st.warning("Die Routenwerte sind keine gültige Liste (2 bis 8 Werte zwischen 0,05 und 1, getrennt durch „;“): es gelten vorläufig die Werte der Voreinstellung.")

a = _analysis(settings)
names_rows, names_cols = ev.label_names(settings, *a.A.shape)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Spiel: {C.GAMES[settings.game]}, Matrix {a.A.shape[0]} × {a.A.shape[1]}. Verlader maximiert, Kontrolleur minimiert.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wert des Spiels", de(a.value), help="Das Garantierte des Verladers bei bester gemischter Strategie; gleich dem, was der Kontrolleur höchstens zulässt (Minimax-Satz).")
m2.metric("Sicherheitsniveau", de(a.security), delta=f"{de(a.value - a.security)} Gewinn durch Mischen", delta_color="off", help="Was die beste reine Zeile garantiert (größtes Zeilenminimum).")
m3.metric("Zertifikat: Lücke", ev.sci(a.sol.upper - a.sol.lower), delta=f"garantiert {de(a.sol.lower, 4)}, zugelassen {de(a.sol.upper, 4)}", delta_color="off",
          help="x garantiert den ersten, y lässt höchstens den zweiten Wert zu; beide aus der Matrix gerechnet, nicht vom Löser übernommen. Die Lücke ist ihr Unterschied (null im Minimax-Satz).")
if a.form is not None:
    m4.metric("Geschlossene Form gegen LP", ev.sci(a.form["deviation"]), delta=f"{len(a.form['active'])} von {len(settings.routes)} Routen im Spiel", delta_color="off",
              help="Größte Abweichung von Wert, x und y der Formel vom linearen Programm (bei nicht eindeutiger Lösung nur der Wert).")
else:
    m4.metric("Sattelpunkte", str(len(a.saddles)), help="Zahl der Paare (Zeile, Spalte) mit reiner optimaler Strategie.")
c1, c2 = st.columns(2)
dropped_x = {i for i in range(len(a.sol.x)) if a.form is not None and i not in a.form["active"]}
with c1:
    st.markdown("**Strategie des Verladers (wählt)**")
    st.plotly_chart(build_strategy(a.sol.x, a.ranges[0], a.ranges[1], names_rows, C.COLOR_LOADER, "Wahrscheinlichkeit", a.unique, dropped_x), width="stretch", key="main_x")
with c2:
    st.markdown("**Strategie des Kontrolleurs (prüft)**")
    st.plotly_chart(build_strategy(a.sol.y, a.ranges[2], a.ranges[3], names_cols, C.COLOR_CONTROL, "Wahrscheinlichkeit", a.unique, dropped_x), width="stretch", key="main_y")
if dropped_x:
    st.caption("Grau: gestrichene Routen. Sie sind zu billig, jede Sendung dorthin ist dem Verlader weniger wert als der Wert des Spiels; beide Seiten lassen sie weg.")
if not a.unique:
    st.caption("Fehlerbalken: Bereich jeder Wahrscheinlichkeit über alle optimalen Strategien (Spiel mit mehreren optimalen Strategien). Die Balken zeigen eine davon.")
    st.dataframe([{"Aktion": f"{n} (Verlader)", "kleinste Wahrscheinlichkeit": pct(lo), "größte Wahrscheinlichkeit": pct(hi)} for n, lo, hi in zip(names_rows, a.ranges[0], a.ranges[1])]
                 + [{"Aktion": f"{n} (Kontrolleur)", "kleinste Wahrscheinlichkeit": pct(lo), "größte Wahrscheinlichkeit": pct(hi)} for n, lo, hi in zip(names_cols, a.ranges[2], a.ranges[3])], hide_index=True, width="stretch")
with st.expander("Matrix A und Strategien in Zahlen"):
    st.dataframe([{"": names_rows[i], **{names_cols[j]: de(a.A[i, j], 3) for j in range(a.A.shape[1])}, "optimale Wahrscheinlichkeit": pct(a.sol.x[i], 2)} for i in range(a.A.shape[0])], hide_index=True, width="stretch")
    st.caption("Zeile: Aktion des Verladers; Spalte: Aktion des Kontrolleurs. " + "Optimale Wahrscheinlichkeiten des Kontrolleurs: " + "; ".join(f"{n} {pct(p, 2)}" for n, p in zip(names_cols, a.sol.y)) + ".")
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="nullsummenspiel-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Hilft Zufall, und wie gewichtet man ihn?")
_show(ev.verdict_main(a.value, a.security, len(a.saddles), a.unique))

st.markdown("### Faustregeln gegen das Gleichgewicht")
st.caption("Was jede Regel dem Verlader garantiert bzw. dem Verlader höchstens zulässt, gegen die beste Antwort des Gegners. Bei Verstecken und Suchen mit den Regeln „nach Wert verteilen“, sonst nur gleichverteilt und beste reine Strategie.")
r1, r2 = st.columns(2)
with r1:
    st.markdown("**Verlader: garantiert**")
    st.plotly_chart(build_rules([n for n, _ in a.loader_rows], [v for _, v in a.loader_rows], a.value, "garantierter Wert", C.COLOR_LOADER), width="stretch", key="rules_loader")
with r2:
    st.markdown("**Kontrolleur: lässt höchstens zu**")
    st.plotly_chart(build_rules([n for n, _ in a.controller_rows], [v for _, v in a.controller_rows], a.value, "zugelassener Wert", C.COLOR_CONTROL), width="stretch", key="rules_controller")
_show(ev.verdict_rules(a.value, a.loader_rows, a.controller_rows))

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Die vorgerechneten Messreihen (data/nullsummen_grid.json) fehlen; sie entstehen mit `python tools/make_grid.py`. Die Experimente zeigen sie.")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Mischen hilft")
if grid is not None:
    rows1 = ev.aggregate_e1(grid["e1"])
    e1b = ev.e1b_summary(grid)
    ints = ev.e1_int_summary(grid)
    st.caption(f"Zufallsspiele (Einträge gleichverteilt in [0, 1)), {rows1[0]['games']} je Größe, 4 Zufalls-Seeds; Mittelwerte über alle Spiele einer Größe, in Klammern die Standardabweichung der Mittel über die Seeds.")
    st.plotly_chart(build_e1(rows1), width="stretch", key="e1_chart")
    st.dataframe([{"Größe": f"{r['m']} × {r['n']}", "Spiele mit Sattelpunkt": f"{pct(r['saddle'])} ({pct(r['saddle_std'])})", "Formel n!²/(2n−1)!": "–" if r["formula"] is None else pct(r["formula"], 2),
                   "mittlerer Wert": de(r["value"]), "Sicherheitsniveau": de(r["security"]), "Gewinn durch Mischen": de(r["gain"]),
                   "Gewinn bei Spielen ohne Sattelpunkt": "–" if r["gain_nonsaddle"] is None else de(r["gain_nonsaddle"]), "Stütze der Optimalstrategie": de(r["support"], 2)} for r in rows1], hide_index=True, width="stretch")
    st.markdown("**Verstecken und Suchen: geschlossene Form gegen das lineare Programm** (Routenwerte gleichverteilt in [untere Grenze, 1], 400 Spiele je Zelle)")
    st.dataframe([{"untere Wertgrenze": de(low, 2), "Routen n": n, "größte Abweichung (Wert, x, y)": ev.sci(v["deviation"]), "Spiele ohne Streichen": pct(v["kept_all"])} for (low, n), v in sorted(e1b.items(), key=lambda kv: (-kv[0][0], kv[0][1]))],
                 hide_index=True, width="stretch")
    st.markdown("**Faustregeln für vier Wertprofile** (garantierter Wert des Verladers; zugelassener Wert des Kontrolleurs; Wert des Spiels in der ersten Spalte)")
    rule_rows = []
    for r in grid["rules"]:
        rule_rows.append({"Werte": "; ".join(de(t, 2) for t in r["v"]), "Wert des Spiels": de(r["value"], 4),
                          **{f"Verlader: {k}": de(v, 4) for k, v in r["loader"].items()}, **{f"Kontrolleur: {k}": de(v, 4) for k, v in r["controller"].items()}})
    st.dataframe(rule_rows, hide_index=True, width="stretch")
    _show(ev.verdict_e1(rows1, e1b, ints))

# --- Reiter -------------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.caption("Jeder Reiter nennt, welche Regler der Seitenleiste in ihm wirken.")
tabs = st.tabs(["👀 Beobachter", "📈 Lernen", "🧮 Allgemeine Spiele"])

# --- Reiter Beobachter ----------------------------------------------------------------------------------------------------------------------

with tabs[0]:
    st.caption("Wirkt: Routenwerte, Muster, Kontrolleur, Gleichstand, Runden und Zufalls-Seed. Der Reiter benutzt immer das Spiel Verstecken und Suchen mit den Routenwerten.")
    res = _observer(ev.observer_settings(settings))
    obs = res["obs"]
    st.plotly_chart(build_observer(obs.running_mean, res["value"], ev.BAND), width="stretch", key="obs_chart")
    st.caption(f"Linie: mittlere Lieferung der ersten t Runden; gestrichelt der Wert des Spiels {de(res['value'])}, grau das Band ±{de(ev.BAND, 2)}.")
    _show(ev.verdict_observer(settings.pattern, settings.counter, settings.tie, res["value"], obs.mean_delivery, obs.caught_share, settings.obs_rounds))
    st.markdown("**Dasselbe Muster gegen alle Zähler** (gleicher Zufalls-Seed und gleiche Runden)")
    st.dataframe([{"Kontrolleur": C.COUNTERS[o["counter"]].split(" (")[0], "Gleichstand": C.TIES[o["tie"]] if o["counter"] != "fixed" else "–", "Lieferung je Sendung": de(o["mean"]), "gefangen": pct(o["caught"]),
                   "gegen den Wert": de(o["mean"] - res["value"])} for o in res["overview"]], hide_index=True, width="stretch")
    st.subheader("🔬 Experiment 2: Beobachtet werden")
    if grid is not None:
        profile_name = st.selectbox("Wertprofil", list(C.E2_PROFILES), key="e2_profile", help="Nur für die Tabelle dieses Experiments (vorgerechnete Messreihe).")
        lookup = ev.e2_lookup(grid, profile_name)
        st.caption(f"Mittlere Lieferung je Sendung, {C.E2_T} Runden, Mittel über 4 Zufalls-Seeds (Standardabweichung über die Seeds in Klammern). Die Spalten nennen den Zähler und die Gleichstandsregel (niedrig = kleinster Index, zufall = zufällige Wahl).")
        cols = [("freq", "low"), ("freq", "random"), ("window", "low"), ("window", "random"), ("follow", "low"), ("follow", "random"), ("fixed", "low")]
        names = {"freq": "Häufigkeit", "window": "Fenster 20", "follow": "Folge", "fixed": "feste y*"}
        table = []
        for pattern in C.PATTERNS:
            row = {"Muster": C.PATTERNS[pattern].split(" (")[0]}
            for counter, tie in cols:
                c = lookup[(pattern, counter, tie)]
                row[f"{names[counter]}{'' if counter == 'fixed' else ', ' + ('niedrig' if tie == 'low' else 'zufall')}"] = f"{de(c['mean'])} ({de(c['std'])})"
            table.append(row)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Wert des Spiels: {de(lookup[('rotation', 'follow', 'low')]['value'])}.")
        st.markdown("**Erste Runde, ab der der Folgezähler jede Sendung einer Rotation abfängt** (gleiche Routen, Gleichstand kleinster Index)")
        st.dataframe([{"Routen n": int(n), "ab Runde": r} for n, r in sorted(grid["capture"].items(), key=lambda kv: int(kv[0]))], hide_index=True, width="stretch")
        _show(ev.verdict_e2(lookup, grid["capture"], profile_name.capitalize(), C.E2_T))
        st.markdown("**Schätzen aus Beobachtungen:** Ein Beobachter sieht T Züge des Gegners, schätzt dessen feste Mischung aus den Häufigkeiten und spielt die beste reine Antwort darauf. Gewinn gegenüber dem Wert des Spiels, Werte (0,9; 0,8; 0,7; 0,6).")
        est = grid["e2est"]
        scenarios = list(dict.fromkeys(r["opponent"] for r in est))
        st.plotly_chart(build_e2_estimate(est, scenarios), width="stretch", key="e2_chart")
        st.dataframe([{"Gegner": s, "Zahl der Züge T": r["T"], "mittlerer Gewinn": de(r["gain"], 4), "voller Gewinn": de(r["full"], 4), "Anteil": "–" if r["share"] is None else pct(r["share"], 0)}
                      for s in scenarios for r in sorted([x for x in est if x["opponent"] == s], key=lambda x: x["T"])], hide_index=True, width="stretch")
        _show(ev.verdict_e2_estimate(est))

# --- Reiter Lernen -----------------------------------------------------------------------------------------------------------------------

with tabs[1]:
    st.caption("Wirkt: Spiel der Hauptansicht (mit Routenwerten bzw. Zeilen, Spalten und Zufalls-Seed des Spiels), Lernverfahren, Start, Lernrunden und Zufalls-Seed.")
    started = time.perf_counter()
    lr = _learn(ev.learn_settings(settings))
    run = lr["run"]
    st.plotly_chart(build_learning(run.steps, run.expl_avg, run.expl_last), width="stretch", key="learn_chart")
    st.caption(f"Logarithmische Achsen; Ausnutzbarkeit 0 ist bei 10⁻⁶ gezeichnet. Wert des Spiels (LP): {de(lr['value'])}. Rechenzeit dieses Aufrufs (mit Zwischenspeicher): {de(time.perf_counter() - started, 2)} s.")
    _show(ev.verdict_learning(settings.learn, settings.start, settings.rounds, float(run.expl_avg[-1]), float(run.expl_last[-1]), lr["peak_last"]))
    st.dataframe([{"Aktion": f"{n} (Verlader)", "Durchschnitt nach T Runden": pct(p, 1), "optimal (LP)": pct(q, 1)} for n, p, q in zip(names_rows, run.avg_x, a.sol.x)]
                 + [{"Aktion": f"{n} (Kontrolleur)", "Durchschnitt nach T Runden": pct(p, 1), "optimal (LP)": pct(q, 1)} for n, p, q in zip(names_cols, run.avg_y, a.sol.y)], hide_index=True, width="stretch")
    st.caption("Bei mehreren optimalen Strategien kann der Durchschnitt zu einer anderen optimalen Strategie laufen als die in der Spalte „optimal (LP)“ gezeigte.")
    st.subheader("🔬 Experiment 3: Der Durchschnitt konvergiert, die letzte Iteration nicht")
    if grid is not None:
        summary = ev.e3_summary(grid)
        start_sum = ev.e3_start_summary(grid)
        h = summary["rm"]["horizons"]
        st.caption(f"Zufallsspiele 5 × 5 (Einträge gleichverteilt in [0, 1)), {summary['rm']['games']} Spiele, ein Lauf je Spiel mit Zufallsstart; mittlere Ausnutzbarkeit. Multiplicative Weights bekommt je Horizont T einen eigenen Lauf (η = √(8 ln N / T)), die übrigen Verfahren einen Lauf bis T = {max(h):,}.".replace(",", " "))
        st.plotly_chart(build_e3(summary), width="stretch", key="e3_chart")
        st.dataframe([{"Verfahren": C.LEARN_METHODS[m], **{f"Durchschnitt, T = {T}": de(s["avg"][i], 4) for i, T in enumerate(h)}, **{f"letzte Iteration, T = {T}": de(s["last"][i], 3) for i, T in enumerate(h)},
                       "Steigung (Log-Log, T = 100 bis 10 000)": de(s["slope"], 2)} for m, s in summary.items()], hide_index=True, width="stretch")
        st.markdown("**Start in der Gleichverteilung gegen Zufallsstart** (Stein-Schere-Papier und vier gleiche Routen, T = 10 000, Mittel über 4 Zufalls-Seeds)")
        st.dataframe([{"Spiel": {"rps": "Stein-Schere-Papier", "four": "vier gleiche Routen"}[game], "Verfahren": C.LEARN_METHODS[m], "Start": C.STARTS[st_],
                       "Ausnutzbarkeit Durchschnitt": de(start_sum[(game, m, st_)]["avg"], 4), "letzte Iteration (Mittel)": de(start_sum[(game, m, st_)]["last"], 3)}
                      for game in ("rps", "four") for m in C.LEARN_METHODS for st_ in ("uniform", "random")], hide_index=True, width="stretch")
        _show(ev.verdict_e3(summary, start_sum))

# --- Reiter Allgemeine Spiele ----------------------------------------------------------------------------------------------------------

with tabs[2]:
    st.caption("Wirkt: Spiel und Aktionen n dieses Reiters sowie der Zufalls-Seed des Spiels (Zufallsspiel und Nullsumme); die Aufzählung rechnet bis n = 8.")
    st.markdown("**Nullsumme hat einen Wert und ein Gleichgewicht, allgemeine Spiele mehrere, und Lernen findet sie nicht von selbst (gemessen am Shapley-Beispiel unten).** Hier A (Zeilenspieler) und B (Spaltenspieler) getrennt; Nullsumme ist der Sonderfall B = 1 − A.")
    bm = _bimatrix(settings.bm, settings.bm_n, settings.game_seed)
    st.dataframe([{"": f"Zeile {i + 1}", **{f"Spalte {j + 1}": f"{de(bm.A[i, j], 2)} | {de(bm.B[i, j], 2)}" for j in range(bm.A.shape[1])}} for i in range(bm.A.shape[0])], hide_index=True, width="stretch")
    st.caption("Jede Zelle: Auszahlung an Zeilenspieler | Auszahlung an Spaltenspieler.")
    payoff = float(bm.equilibria[0][0] @ bm.A @ bm.equilibria[0][1]) if bm.equilibria else None
    _show(ev.verdict_bimatrix(settings.bm, len(bm.equilibria), bm.pure, bm.coverage, bm.zero_sum_value, payoff))
    st.dataframe([{"Nr.": i + 1, "Strategie Zeilenspieler": "; ".join(pct(p, 1) for p in x), "Strategie Spaltenspieler": "; ".join(pct(p, 1) for p in y), "Stützen": f"{G.support(x)} × {G.support(y)}",
                   "Auszahlung Zeile | Spalte": f"{de(x @ bm.A @ y, 3)} | {de(x @ bm.B @ y, 3)}"} for i, (x, y) in enumerate(bm.equilibria)], hide_index=True, width="stretch")
    st.markdown("**Lemke–Howson von jedem Startlabel** (Labels 1 bis m: Zeilen, danach Spalten; der Pfad nennt die eintretenden Labels)")
    st.dataframe([{"Startlabel": r["label"] + 1, "Pfad": " → ".join(str(t + 1) for t in r["path"]), "Pivots": r["pivots"], "findet Gleichgewicht Nr.": "–" if r["found"] is None else r["found"] + 1} for r in bm.lh_rows], hide_index=True, width="stretch")
    st.markdown("**Shapley-Beispiel: Fictitious Play** (A = Einheitsmatrix, B[i, i+1] = 1; einziges Gleichgewicht die Gleichverteilung)")
    if st.button("▶️ Fictitious Play im Shapley-Spiel starten", key="shapley_start"):
        with st.spinner(f"Spiele {C.SHAPLEY_ROUNDS_LIVE} Runden …"):
            t0 = time.perf_counter()
            st.session_state["shapley_result"] = (ev.shapley_cycle(C.SHAPLEY_ROUNDS_LIVE), time.perf_counter() - t0)
    stored = st.session_state.get("shapley_result")
    if stored:
        cyc, secs = stored
        lo, hi = ev.window_range(cyc["dist"], C.SHAPLEY_ROUNDS_LIVE // 2 + 1, C.SHAPLEY_ROUNDS_LIVE)
        st.plotly_chart(build_shapley(cyc["dist"], cyc["zero_dist"]), width="stretch", key="shapley_chart")
        _show(ev.verdict_shapley(C.SHAPLEY_ROUNDS_LIVE, lo, hi, float(cyc["zero_dist"][-1])))
        st.caption(f"Rechenzeit dieses Laufs: {de(secs, 2)} s. Abstand = größerer der beiden halben L1-Abstände der Häufigkeiten zur Gleichverteilung; Bereich der zweiten Hälfte der Runden.")
    else:
        st.info("Noch kein Lauf: „Fictitious Play im Shapley-Spiel starten“ rechnet ihn.")
    st.subheader("🔬 Experiment 4: Eindeutigkeit, Aufzählung und Lemke–Howson")
    if grid is not None:
        rows4 = ev.e4_summary(grid)
        nulls = ev.e4_nullsum_summary(grid)
        big = ev.e4_big_summary(grid)
        shapley = grid["shapley"]
        st.caption("Zufalls-Bimatrixspiele n × n (Einträge von A und B unabhängig gleichverteilt in [0, 1)), 4 Zufalls-Seeds; Lemke–Howson von allen 2n Startlabels.")
        st.plotly_chart(build_e4(rows4), width="stretch", key="e4_chart")
        st.dataframe([{"n": r["n"], "Spiele": r["games"], "mittlere Zahl der Gleichgewichte": de(r["mean_count"], 2), "größte Zahl": r["max_count"], "genau eines": pct(r["only_one"], 0), "gerade Zahl": r["even"],
                       "mit reinem Gleichgewicht": pct(r["with_pure"], 0), "Pivots je Label (Mittel)": de(r["mean_pivots"], 1), "größte Pfadlänge": r["max_pivots"], "Fehler": r["lh_errors"],
                       "gefundener Anteil der Gleichgewichte": pct(r["coverage"], 0), "Stützenpaare C(2n, n) − 1": f"{r['pairs']:,}".replace(",", " ")} for r in rows4], hide_index=True, width="stretch")
        st.markdown("**Kontrolle Nullsumme** (B = 1 − A) und **Lemke–Howson in großen Zufallsspielen** (ohne Aufzählung)")
        st.dataframe([{"n": n, "Spiele": v["games"], "genau ein Gleichgewicht": pct(v["only_one"], 0), "größte Abweichung der Auszahlung vom LP-Wert": ev.sci(v["deviation"])} for n, v in nulls.items()], hide_index=True, width="stretch")
        st.dataframe([{"n": n, "Pfade": v["paths"], "Mittel": de(v["mean"], 1), "Median": de(v["median"], 0), "Maximum": v["max"], "Fehler": v["errors"]} for n, v in big.items()], hide_index=True, width="stretch")
        st.caption(f"Shapley-Spiel, Fictitious Play, {shapley['rounds']:,} Runden: Abstand der Häufigkeiten zur Gleichverteilung je Fenster (kleinster bis größter Wert) ".replace(",", " ")
                   + "; ".join(f"Runden {k.replace('-', ' bis ')}: {de(v[0])} bis {de(v[1])}" for k, v in shapley["windows"].items())
                   + f". Regret Matching (Start (0,5; 0,3; 0,2) gegen (0,2; 0,3; 0,5)), 10 000 Runden: Nash-Lücke des Durchschnitts {de(shapley['rm_gap_avg'])}, der letzten Iteration {de(shapley['rm_gap_last'])}.")
        _show(ev.verdict_e4(rows4, nulls, big, shapley))
    st.info("Literatur, nicht nachgemessen und nicht im Volltext gelesen: Das Finden eines Nash-Gleichgewichts in allgemeinen Zwei-Personen-Spielen gilt als PPAD-vollständig (Chen und Deng; Daskalakis, Goldberg, Papadimitriou), und Lemke–Howson braucht im ungünstigsten Fall exponentiell viele Schritte (Savani und von Stengel 2006). "
            "Die Messreihe oben zeigt das nicht: Sie hat kurze Pfade in Zufallsspielen, und die Familie der schlechten Spiele ist nicht gebaut.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Kontrollstelle, ein Zug** | Mit mehreren Kontrollstellen oder Ressourcen ändert sich das Spiel (Zuordnung von Kontrollen); die Demo rechnet nur den Fall mit einer Prüfung | – |
| **Der Kontrolleur legt sich nicht vorher fest** | Wer sich zuerst festlegt und der andere reagiert, hat ein anderes Spiel (Stackelberg); im Nullsummenspiel ist das Festlegen nicht schädlich, in allgemeinen Spielen schon | `stackelberg-demo` (reine Strategien) |
| **Erwartungswerte statt Tagesstreuung** | Zufall als Aufteilen derselben Sendung liefert im Erwartungswert dasselbe, die Streuung eines Tages ist eine andere; die Demo rechnet Erwartungswerte | – |
| **Volle Kenntnis der Matrix, keine verdeckte Information** | Spiele mit Zügen nacheinander und ungleich verteilter Information (Spielbaum, Informationsmengen) brauchen eigene Verfahren; hier nicht gebaut | – |
| **Nullsumme** | In Spielen mit gemeinsamem Gewinn oder Verlust kann es mehrere Gleichgewichte geben, und Lernen findet sie nicht von selbst (Reiter Allgemeine Spiele, am Shapley-Beispiel gemessen) | – |
| **Die Gleichverteilung als Zufallsmodell der Matrizen** | Die Zahlen der Experimente gelten für Zufallsspiele mit gleichverteilten Einträgen und gleichverteilten Routenwerten, nicht für beliebige Spiele | – |
| **Keine Daten echter Kontrollen** | Die Experimente zeigen das Modell, nicht das Verhalten von Betrieben oder Kontrolleuren | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(C.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
