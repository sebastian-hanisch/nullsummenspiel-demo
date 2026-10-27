"""Auswertung: Kennzahlen je Ansicht, Zeilen der Messreihen (data/nullsummen_grid.json), Verdikte als reine Funktionen.

Auszahlungen sind Anteile der Ladung (Verstecken und Suchen) oder Einheiten der Matrix (die übrigen Spiele). Alle Verdikte sind `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`.
Jede Zahl im Text wird aus der Rechnung gebildet oder steht in tests/test_claims.py (DEMO-PLAYBOOK §1d)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import nsm_bimatrix as Bm
import nsm_constants as C
import nsm_game as G
import nsm_hideseek as H
import nsm_learning as Ln
import nsm_lp as L
import nsm_observers as O
from nsm_rng import SplitMix64
from nsm_scenario import Settings, cell_seed, draw_bimatrix, draw_matrix, game_matrix, uniform01  # noqa: F401  (Settings wird hier wieder ausgegeben)

GRID_PATH = Path(__file__).resolve().parent / "data" / "nullsummen_grid.json"
BAND = 0.05                    # Abstand zum Wert (in Auszahlungseinheiten), ab dem die Texte der Beobachter-Läufe von „Abweichung“ sprechen
EPS = 1e-9


# --- Formatierung (deutsches Dezimalkomma) ---------------------------------------------------------------------------------------------------------

def de(value: float, digits: int = 3) -> str:
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{round(100 * value, digits) + 0.0:.{digits}f}".replace(".", ",") + " %"


def sci(value: float) -> str:
    """Kleine Zahl in Zehnerpotenz-Schreibweise mit Dezimalkomma (1,2·10⁻¹⁴)."""
    if value == 0:
        return "0"
    mant, exp = f"{value:.1e}".split("e")
    sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{sup}"


def label_names(settings: Settings, m: int, n: int) -> tuple[list[str], list[str]]:
    """Namen der Zeilen und Spalten für Tabellen und Diagramme."""
    if settings.game == "hide":
        return [f"Route {i + 1}" for i in range(m)], [f"Route {j + 1}" for j in range(n)]
    if settings.game == "rps":
        return ["Stein", "Schere", "Papier"], ["Stein", "Schere", "Papier"]
    return [f"Zeile {i + 1}" for i in range(m)], [f"Spalte {j + 1}" for j in range(n)]


# --- Hauptansicht: Verladen und Prüfen ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    A: np.ndarray
    sol: L.Solution
    value: float
    security: float             # beste reine Zeile
    pure_ceiling: float         # beste reine Spalte
    saddles: tuple
    ranges: tuple               # (x_lo, x_hi, y_lo, y_hi) über die optimale Menge
    unique: bool                # beide optimalen Strategien eindeutig
    form: dict | None           # nur Verstecken und Suchen: geschlossene Form und größte Abweichung zum LP
    loader_rows: tuple          # Faustregeln des Verladers: (Name, garantierter Wert)
    controller_rows: tuple      # Faustregeln des Kontrolleurs: (Name, höchstens zugelassener Wert)


def analyse(s: Settings) -> Analysis:
    A = game_matrix(s)
    sol = L.solve_zero_sum(A)
    ranges = L.optimal_ranges(A, sol.value)
    unique = L.is_unique(ranges[0], ranges[1]) and L.is_unique(ranges[2], ranges[3])
    form = None
    if s.game == "hide":
        k, fx, fy, active = H.closed_form(s.routes)
        form = {"value": k, "x": fx, "y": fy, "active": active,
                "deviation": max(abs(k - sol.value), float(np.max(np.abs(fx - sol.x))), float(np.max(np.abs(fy - sol.y)))) if unique else abs(k - sol.value)}
        loader = H.loader_rules(s.routes)
        controller = H.controller_rules(s.routes)
    else:
        m, n = A.shape
        loader = {"gleichverteilt": np.full(m, 1.0 / m)}
        controller = {"gleichverteilt": np.full(n, 1.0 / n)}
    best_row = np.zeros(A.shape[0])
    best_row[int(np.argmax(np.min(A, axis=1)))] = 1.0
    best_col = np.zeros(A.shape[1])
    best_col[int(np.argmin(np.max(A, axis=0)))] = 1.0
    loader = {**loader, "beste reine Zeile (Sicherheitsniveau)": best_row, "Gleichgewicht": sol.x}
    controller = {**controller, "beste reine Spalte": best_col, "Gleichgewicht": sol.y}
    return Analysis(
        s, A, sol, sol.value, G.pure_security(A), G.pure_ceiling(A), tuple(G.saddle_points(A)), ranges, unique, form,
        tuple((name, G.guaranteed(A, x)) for name, x in loader.items()), tuple((name, G.ceiling(A, y)) for name, y in controller.items()),
    )


def verdict_main(value: float, security: float, saddles: int, unique: bool) -> tuple[str, str]:
    """Wert gegen Sicherheitsniveau: was bringt Mischen, und ist die optimale Strategie eindeutig?"""
    gain = value - security
    if gain <= EPS:
        kind = "info"
        text = f"Der Wert des Spiels ist **{de(value)}**. Die beste reine Zeile sichert schon {de(security)}: ein Sattelpunkt, **Zufall bringt hier nichts** ({saddles} Sattelpunkt{'e' if saddles != 1 else ''})."
    else:
        kind = "success"
        text = f"Der Wert des Spiels ist **{de(value)}**, die beste reine Zeile sichert nur {de(security)}: **Mischen bringt +{de(gain)}** (Gewinn durch Zufall mit Absicht)."
    if not unique:
        text += " Die optimale Strategie ist **nicht eindeutig**: Es gibt eine ganze Menge optimaler Strategien (Bereiche in der Tabelle unten), keine einzelne „die“ Lösung."
    return kind, text


def rule_shortfalls(value: float, loader_rows, controller_rows) -> tuple[list, list]:
    """Faustregeln, die den Wert verfehlen: Verlader (garantieren weniger als der Wert) und Kontrolleur (lassen mehr zu als der Wert), je (Name, Zahl)."""
    return ([(n, v) for n, v in loader_rows if v < value - EPS], [(n, v) for n, v in controller_rows if v > value + EPS])


def verdict_rules(value: float, loader_rows, controller_rows) -> tuple[str, str]:
    """Welche Faustregeln sind ausnutzbar? Nennt jede Regel mit ihrem garantierten bzw. zugelassenen Wert gegen den Wert des Spiels."""
    low, high = rule_shortfalls(value, loader_rows, controller_rows)
    if not low and not high:
        return "info", f"Alle gezeigten Regeln erreichen den Wert {de(value)}."
    parts = []
    if low:
        parts.append("Verlader, garantiert weniger als der Wert: " + "; ".join(f"{n} {de(v)}" for n, v in low))
    if high:
        parts.append("Kontrolleur, lässt mehr zu als der Wert: " + "; ".join(f"{n} {de(v)}" for n, v in high))
    return "warning", f"Wert des Spiels {de(value)}. " + ". ".join(parts) + "."


# --- Reiter Lernen ----------------------------------------------------------------------------------------------------------------------------

def verdict_learning(method: str, start: str, rounds: int, expl_avg: float, expl_last: float, peak_last: float) -> tuple[str, str]:
    """Durchschnitt gegen letzte Iteration nach `rounds` Runden; `peak_last` ist die größte Ausnutzbarkeit der letzten Iteration über alle Stützstellen (Start im Gleichgewicht: 0)."""
    name = C.LEARN_METHODS[method]
    if expl_avg <= 1e-12 and peak_last <= 1e-12:
        return "warning", (f"{name} aus der {C.STARTS[start]}: Ausnutzbarkeit 0 in jeder Runde. **Der Start liegt schon im Gleichgewicht**, und alle Auszahlungen sind gleich, es ändert sich nichts. "
                           "Mit dem Zufallsstart bewegt sich das Verfahren.")
    if method == "fp":
        last_text = f"die gespielte reine Aktion der letzten Runde ist um {de(expl_last)} ausnutzbar"
    else:
        last_text = f"die Strategie der letzten Runde ist um {de(expl_last)} ausnutzbar"
    closer = "Der Durchschnitt liegt näher am Gleichgewicht als die letzte Iteration." if expl_avg < expl_last else "Hier liegt die letzte Iteration nicht schlechter als der Durchschnitt."
    return "info", f"{name}, {C.STARTS[start]}, nach {rounds} Runden: die **Durchschnittsstrategie** ist um **{de(expl_avg, 4)}** ausnutzbar, {last_text}. {closer}"


# --- Reiter Beobachter ----------------------------------------------------------------------------------------------------------------------------

def verdict_observer(pattern: str, counter: str, tie: str, value: float, mean_delivery: float, caught_share: float, rounds: int) -> tuple[str, str]:
    """Lieferung eines Laufs gegen den Wert des Spiels (Band ±BAND)."""
    head = f"{C.PATTERNS[pattern].split(' (')[0]} gegen {C.COUNTERS[counter].split(' (')[0]}"
    if counter in ("freq", "window", "follow"):
        head += f", Gleichstand {C.TIES[tie]}"
    body = f"Im Mittel kommen **{de(mean_delivery)}** je Sendung an (Wert des Spiels {de(value)}); der Kontrolleur fängt {pct(caught_share)} der {rounds} Sendungen."
    if mean_delivery < value - BAND:
        return "warning", f"{head}: {body} Das Muster **kostet**: Der Zähler nutzt es aus, es kommt mehr als {de(BAND, 2)} weniger an als im Gleichgewicht."
    if mean_delivery > value + BAND:
        return "success", f"{head}: {body} Dieser Zähler **nutzt das Muster nicht aus**: Es kommt mehr als {de(BAND, 2)} mehr an als im Gleichgewicht."
    return "info", f"{head}: {body} Das ist im Band von ±{de(BAND, 2)} um den Wert: nichts zu holen für den Zähler."


# --- Reiter Allgemeine Spiele ---------------------------------------------------------------------------------------------------------------------

def bimatrix_game(kind: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Das Spiel des Reiters: Zufallsspiel (A und B unabhängig), Nullsumme (B = 1 − A) oder das Shapley-Beispiel (3 × 3, feste Matrizen)."""
    if kind == "shapley":
        return Bm.shapley_game()
    A, B = draw_bimatrix(n, n, seed)
    return (A, B) if kind == "random" else Bm.zero_sum_partner(A)


@dataclass(frozen=True)
class BimatrixAnalysis:
    kind: str
    A: np.ndarray
    B: np.ndarray
    equilibria: tuple           # ((x, y), …)
    pure: int
    lh_rows: tuple              # je Startlabel: dict(label, path, pivots, found)  (found = Index in `equilibria`)
    coverage: float             # Anteil der Gleichgewichte, die mindestens ein Label findet
    zero_sum_value: float | None


def analyse_bimatrix(kind: str, n: int, seed: int) -> BimatrixAnalysis:
    A, B = bimatrix_game(kind, n, seed)
    eq = Bm.enumerate_nash(A, B)
    rows = []
    for label in range(A.shape[0] + A.shape[1]):
        x, y, pivots, path = Bm.lemke_howson(A, B, label)
        found = next((i for i, (xe, ye) in enumerate(eq) if Bm.same_profile(x, y, xe, ye)), None)
        rows.append({"label": label, "path": path, "pivots": pivots, "found": found})
    reached = {r["found"] for r in rows if r["found"] is not None}
    value = L.solve_zero_sum(A).value if kind == "zero_sum" else None
    return BimatrixAnalysis(kind, A, B, tuple(eq), Bm.count_pure(eq), tuple(rows), len(reached) / len(eq) if eq else 0.0, value)


def verdict_bimatrix(kind: str, n_equilibria: int, pure: int, coverage: float, value: float | None, payoff: float | None) -> tuple[str, str]:
    """Zahl der Gleichgewichte, Parität und Abdeckung durch Lemke–Howson."""
    parity = "ungerade" if n_equilibria % 2 == 1 else "gerade"
    if kind == "zero_sum":
        ok = n_equilibria == 1 and value is not None and payoff is not None and abs(payoff - value) <= 1e-9
        return ("success" if ok else "warning"), (f"Nullsumme: **{n_equilibria} Gleichgewicht{'e' if n_equilibria != 1 else ''}**, Auszahlung {de(payoff, 4) if payoff is not None else '–'} gegen LP-Wert {de(value, 4) if value is not None else '–'}. "
                                                  "In Nullsummenspielen ist jedes Gleichgewicht ein Paar optimaler Strategien und liefert denselben Wert.")
    return "info", (f"**{n_equilibria} Gleichgewicht{'e' if n_equilibria != 1 else ''}** ({parity} Zahl; davon {pure} in reinen Strategien). Lemke–Howson findet von jedem Startlabel aus ein Gleichgewicht; alle Labels zusammen finden {pct(coverage, 0)} der Gleichgewichte dieses Spiels.")


def shapley_cycle(rounds: int) -> dict:
    """Shapley-Beispiel: Fictitious Play kreist (Abstand der Häufigkeiten zur Gleichverteilung bleibt groß), das Nullsummen-Gegenstück konvergiert (Abstand fällt)."""
    A, B = Bm.shapley_game()
    fp = Bm.fictitious_play_bimatrix(A, B, rounds)
    zA, zB = Bm.zero_sum_partner(np.eye(3))
    zero = Bm.fictitious_play_bimatrix(zA, zB, rounds)
    return {"dist": fp["dist"], "zero_dist": zero["dist"], "switches": fp["switches"]}


def window_range(dist, lo: int, hi: int) -> tuple[float, float]:
    """Kleinster und größter Abstand der Runden lo … hi (1-basiert, hi eingeschlossen)."""
    part = np.asarray(dist)[lo - 1:hi]
    return float(part.min()), float(part.max())


def verdict_shapley(rounds: int, lo: float, hi: float, zero_last: float) -> tuple[str, str]:
    return "warning", (f"Shapley-Beispiel, Fictitious Play, {rounds} Runden: der Abstand der Häufigkeiten zur Gleichverteilung bleibt in der zweiten Hälfte zwischen **{de(lo)}** und **{de(hi)}** "
                       f"(er fällt in dieser Zeit nicht gegen 0). Im Nullsummen-Gegenstück B = 1 − A ist er am Ende {de(zero_last, 3)}.")


# --- Messreihen: Zeilen der Experimente (tools/make_grid.py rechnet sie vor, tests/test_grid.py prüft eine Probe) -----------------------------------------

def mean_std(values) -> tuple[float, float]:
    """Mittel und Standardabweichung (Grundgesamtheit) einer Liste von Zahlen."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std())


def e1_cell(m: int, n: int, seed: int, games: int) -> dict:
    """E1: `games` Zufallsspiele m × n (Einträge gleichverteilt in [0, 1)): Anteil mit Sattelpunkt, Wert, Sicherheitsniveau, Gewinn durch Mischen, Stütze der Optimalstrategie, größtes Zertifikat."""
    rows = []
    worst = 0.0
    for g in range(games):
        A = draw_matrix(m, n, cell_seed(seed, m, n, g))
        sol = L.solve_zero_sum(A)
        worst = max(worst, sol.upper - sol.lower, abs(sol.value - sol.dual_value))
        rows.append((bool(G.saddle_points(A)), sol.value, G.pure_security(A), G.support(sol.x)))
    saddle = np.array([r[0] for r in rows])
    value, security = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    gain = value - security
    return {"m": m, "n": n, "seed": seed, "games": games, "saddle": float(saddle.mean()), "value": float(value.mean()), "security": float(security.mean()), "gain": float(gain.mean()),
            "gain_nonsaddle": float(gain[~saddle].mean()) if (~saddle).any() else None, "support": float(np.mean([r[3] for r in rows])), "certificate": float(worst)}


def e1_integer_cell(m: int, n: int, seed: int, games: int) -> dict:
    """E1: ganzzahlige Zufallsspiele (Einträge 0 bis 9): Anteil der Spiele, in denen die optimale Strategie des Verladers (oder des Kontrolleurs) nicht eindeutig ist."""
    multi_x = multi_y = 0
    for g in range(games):
        rng = SplitMix64(cell_seed(seed, 99, m, n, g))
        A = np.array([[float(rng.below(10)) for _ in range(n)] for _ in range(m)])
        sol = L.solve_zero_sum(A)
        r = L.optimal_ranges(A, sol.value)
        multi_x += not L.is_unique(r[0], r[1])
        multi_y += not L.is_unique(r[2], r[3])
    return {"m": m, "n": n, "seed": seed, "games": games, "multi_x": multi_x / games, "multi_y": multi_y / games}


def e1b_cell(n: int, low: float, seed: int, games: int) -> dict:
    """E1b: Routenwerte gleichverteilt in [low, 1]: größte Abweichung der geschlossenen Form vom LP (Wert, x, y) und Anteil der Spiele, in denen keine Route gestrichen wird."""
    worst = 0.0
    kept_all = 0
    for g in range(games):
        rng = SplitMix64(cell_seed(seed, 77, n, int(low * 100), g))
        v = [round(low + (1.0 - low) * uniform01(rng), 4) for _ in range(n)]
        sol = L.solve_zero_sum(H.hide_seek_matrix(v))
        k, x, y, active = H.closed_form(v)
        worst = max(worst, abs(k - sol.value), float(np.max(np.abs(x - sol.x))), float(np.max(np.abs(y - sol.y))))
        kept_all += len(active) == n
    return {"n": n, "low": low, "seed": seed, "games": games, "deviation": worst, "kept_all": kept_all / games}


def rules_rows(v) -> dict:
    """Faustregeln für Routenwerte v: garantierter Wert des Verladers und zugelassener Wert des Kontrolleurs je Regel, dazu der Wert des Spiels."""
    A = H.hide_seek_matrix(v)
    k, x, y, _active = H.closed_form(v)
    loader = {n: G.guaranteed(A, p) for n, p in H.loader_rules(v).items()}
    loader["Gleichgewicht"] = G.guaranteed(A, x)
    controller = {n: G.ceiling(A, p) for n, p in H.controller_rules(v).items()}
    controller["Gleichgewicht"] = G.ceiling(A, y)
    return {"v": list(v), "value": k, "loader": loader, "controller": controller}


def e2_cell(profile: tuple, pattern: str, counter: str, tie: str, seeds, rounds: int) -> dict:
    """E2: mittlere Lieferung je Sendung eines Musters gegen einen Zähler, über Zufalls-Seeds (Mittel, Standardabweichung über die Seeds)."""
    k, x, y, _a = H.closed_form(profile)
    runs = [O.simulate(pattern, counter, tie, profile, x, y, rounds, cell_seed(seed, 5)) for seed in seeds]
    mean, std = mean_std([r.mean_delivery for r in runs])
    return {"pattern": pattern, "counter": counter, "tie": tie, "value": k, "mean": mean, "std": std, "caught": float(np.mean([r.caught_share for r in runs]))}


def capture_rounds() -> dict:
    """E2: erste Runde, ab der der Folgezähler (Gleichstand kleinster Index) jede Sendung einer Rotation über n gleiche Routen abfängt, n = 2 … 8."""
    out = {}
    for n in range(C.ROUTES_MIN, C.ROUTES_MAX + 1):
        routes = O.loader_sequence("rotation", n, None, 60, SplitMix64(1))
        checks = O.controller_sequence("follow", "low", [1.0] * n, None, routes, SplitMix64(1))
        out[n] = O.first_certain_round(routes, checks)
    return out


def e2_estimate_cell(profile: tuple, side: str, opponent_name: str, opponent, observed: int, seeds, samples: int) -> dict:
    """E2 (Schätzen): Gewinn eines Beobachters, der `observed` Züge der festen Mischung `opponent` sieht, gegen den vollen Gewinn."""
    A = H.hide_seek_matrix(profile)
    k = H.closed_form(profile)[0]
    results = [O.estimate_and_respond(A, k, side, opponent, observed, samples, cell_seed(seed, 6, observed)) for seed in seeds]
    gain, _s = mean_std([r["gain"] for r in results])
    return {"side": side, "opponent": opponent_name, "T": observed, "gain": gain, "full": results[0]["full"], "share": (gain / results[0]["full"]) if results[0]["full"] > EPS else None}


def e3_game_runs(method: str, size: int, seed: int, games: int) -> dict:
    """E3: `games` Zufallsspiele size × size, je ein Lauf des Verfahrens mit Zufallsstart: Ausnutzbarkeit der Durchschnittsstrategie und der letzten Iteration an den Horizonten E3_HORIZONS (Summe über die Spiele).

    Alle Verfahren außer Multiplicative Weights laufen einmal bis zum längsten Horizont und werden an den Stützstellen gelesen; Multiplicative Weights bekommt für jeden Horizont einen eigenen Lauf (die Lernrate hängt vom Horizont ab)."""
    horizons = list(C.E3_HORIZONS)
    avg, last = np.zeros(len(horizons)), np.zeros(len(horizons))
    for g in range(games):
        A = draw_matrix(size, size, cell_seed(seed, 3, size, g))
        start_seed = cell_seed(seed, g)
        if method == "mw":
            runs = [Ln.run_method("mw", A, T, "random", start_seed) for T in horizons]
            avg += [r.expl_avg[-1] for r in runs]
            last += [r.expl_last[-1] for r in runs]
        else:
            run = Ln.run_method(method, A, max(horizons), "random", start_seed)
            idx = [list(run.steps).index(T) for T in horizons]
            avg += run.expl_avg[idx]
            last += run.expl_last[idx]
    return {"method": method, "seed": seed, "games": games, "horizons": horizons, "avg_sum": [float(v) for v in avg], "last_sum": [float(v) for v in last]}


def e3_start_cell(game: str, method: str, start: str, seed: int, rounds: int) -> dict:
    """E3 (Start): Lauf auf Stein-Schere-Papier (rps), Münzwurf (coin) oder vier gleichen Routen (four): Ausnutzbarkeit nach `rounds` Runden und größte Ausnutzbarkeit der letzten Iteration."""
    A = {"rps": np.array(C.RPS_MATRIX), "coin": H.hide_seek_matrix((1.0, 1.0)), "four": H.hide_seek_matrix((1.0,) * 4)}[game]
    run = Ln.run_method(method, A, rounds, start, seed)
    return {"game": game, "method": method, "start": start, "seed": seed, "avg": float(run.expl_avg[-1]), "last": float(run.expl_last[-1]), "last_max": float(run.expl_last.max())}


def e4_cell(n: int, seed: int, games: int) -> dict:
    """E4: Zufalls-Bimatrixspiele n × n: Zahl der Gleichgewichte (Mittel, Anteil genau 1, gerade, reine), Lemke–Howson (Pivots je Label, Fehler, Abdeckung der Gleichgewichte)."""
    counts, pure, pivots, errors, coverage = [], [], [], 0, []
    for g in range(games):
        A, B = draw_bimatrix(n, n, cell_seed(seed, 4, n, g))
        eq = Bm.enumerate_nash(A, B)
        counts.append(len(eq))
        pure.append(Bm.count_pure(eq))
        found = set()
        for label in range(2 * n):
            x, y, p, _path = Bm.lemke_howson(A, B, label)
            pivots.append(p)
            if not Bm.is_nash(A, B, x, y, tol=1e-8):
                errors += 1
                continue
            idx = next((i for i, (xe, ye) in enumerate(eq) if Bm.same_profile(x, y, xe, ye)), None)
            if idx is None:
                errors += 1
            else:
                found.add(idx)
        coverage.append(len(found) / len(eq) if eq else 0.0)
    counts = np.array(counts)
    return {"n": n, "seed": seed, "games": games, "mean_count": float(counts.mean()), "max_count": int(counts.max()), "only_one": float(np.mean(counts == 1)), "even": int(np.sum(counts % 2 == 0)),
            "with_pure": float(np.mean(np.array(pure) > 0)), "mean_pure": float(np.mean(pure)), "mean_pivots": float(np.mean(pivots)), "max_pivots": int(max(pivots)), "lh_errors": errors,
            "coverage": float(np.mean(coverage))}


def e4_nullsum_cell(n: int, seed: int, games: int) -> dict:
    """E4 (Kontrolle): Nullsummen-Zufallsspiele (B = 1 − A): Anteil mit genau einem Gleichgewicht und größte Abweichung der Auszahlung im Gleichgewicht vom LP-Wert."""
    one, worst = 0, 0.0
    for g in range(games):
        A, B = Bm.zero_sum_partner(draw_matrix(n, n, cell_seed(seed, 8, n, g)))
        eq = Bm.enumerate_nash(A, B)
        one += len(eq) == 1
        v = L.solve_zero_sum(A).value
        worst = max([worst] + [abs(x @ A @ y - v) for x, y in eq])
    return {"n": n, "seed": seed, "games": games, "only_one": one / games, "deviation": float(worst)}


def e4_lh_big(n: int, seed: int, games: int) -> dict:
    """E4: Lemke–Howson allein in großen Zufallsspielen (ohne Aufzählung): Pivots je Startlabel (fünf Labels je Spiel), Fehlerzahl."""
    pivots, errors = [], 0
    for g in range(games):
        A, B = draw_bimatrix(n, n, cell_seed(seed, 5, n, g))
        for label in range(0, 2 * n, max(1, n // 5)):
            x, y, p, _path = Bm.lemke_howson(A, B, label)
            pivots.append(p)
            errors += not Bm.is_nash(A, B, x, y, tol=1e-7)
    return {"n": n, "seed": seed, "pivots": pivots, "errors": errors}


def e4_shapley(rounds: int) -> dict:
    """E4 (Shapley): Fenster des Abstands (Fictitious Play), Zahl der Wechsel, Nullsummen-Gegenstück, Nash-Lücke von Regret Matching bei 10 000 Runden."""
    cyc = shapley_cycle(rounds)
    windows = {}
    for lo, hi in ((1000, 9999), (10000, 99999), (rounds // 2 + 1, rounds)):
        if hi <= rounds:
            windows[f"{lo}-{hi}"] = window_range(cyc["dist"], lo, hi)
    A, B = Bm.shapley_game()
    start_x, start_y = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    rm = Bm.regret_matching_bimatrix(A, B, 10000, start_x, start_y)
    return {"rounds": rounds, "windows": windows, "switches": cyc["switches"], "zero_last": float(cyc["zero_dist"][-1]), "zero_max_after_100000": float(np.max(cyc["zero_dist"][99999:])) if rounds > 100000 else None,
            "rm_gap_avg": rm["gap_avg"], "rm_gap_last": rm["gap_last"]}


# --- Messreihen laden und Probe ------------------------------------------------------------------------------------------------------------------

def load_grid():
    """Die vorgerechneten Messreihen; None, wenn die Datei fehlt."""
    if not GRID_PATH.exists():
        return None
    return json.loads(GRID_PATH.read_text(encoding="utf-8"))


def grid_probe() -> dict:
    """Kleine feste Probe aus denselben Erzeugern wie die Messreihen; tests/test_grid.py vergleicht sie mit der gespeicherten Probe (ändert sich Rechnung oder Seed-Ableitung, schlägt der Test an)."""
    return {
        "e1": e1_cell(3, 3, 11, 8),
        "e1b": e1b_cell(4, 0.3, 11, 8),
        "e2": e2_cell((1.0,) * 4, "rotation", "follow", "low", (11,), 200),
        "e2est": e2_estimate_cell((0.9, 0.8, 0.7, 0.6), "loader", "gleichverteilt", [0.25] * 4, 5, (11,), 20),
        "e3": e3_game_runs("rm", 4, 11, 1),
        "e3start": e3_start_cell("rps", "rm", "random", 11, 300),
        "e4": e4_cell(3, 11, 6),
        "e4null": e4_nullsum_cell(3, 11, 6),
    }


# Hilfen für die Texte der Experimente: Auswahl aus den Zeilen -------------------------------------------------------------------------------------------

def aggregate_e1(cells: list[dict]) -> list[dict]:
    """E1: Zeilen je Größe: Mittel über die Zufalls-Seeds (und Standardabweichung über die Seeds) der Kennzahlen."""
    sizes = sorted({(c["m"], c["n"]) for c in cells}, key=lambda t: cells_order(t))
    out = []
    for m, n in sizes:
        cs = [c for c in cells if (c["m"], c["n"]) == (m, n)]
        row = {"m": m, "n": n, "games": sum(c["games"] for c in cs)}
        for key in ("saddle", "value", "security", "gain", "support"):
            row[key], row[key + "_std"] = mean_std([c[key] for c in cs])
        vals = [c["gain_nonsaddle"] for c in cs if c["gain_nonsaddle"] is not None]
        row["gain_nonsaddle"] = float(np.mean(vals)) if vals else None
        row["formula"] = math.factorial(n) ** 2 / math.factorial(2 * n - 1) if m == n else None
        row["certificate"] = max(c["certificate"] for c in cs)
        out.append(row)
    return out


def cells_order(size: tuple) -> tuple:
    """Reihenfolge der Größen in den Tabellen: quadratische Spiele nach n, danach die nicht quadratischen."""
    m, n = size
    return (0, n, m) if m == n else (1, m, n)


# --- Zusammenfassungen der Messreihen (App, README und Tests benutzen dieselben Funktionen) ------------------------------------------------------------

def e1b_summary(grid: dict) -> dict:
    """E1b: je (untere Wertgrenze, n) die größte Abweichung der geschlossenen Form vom LP und der Anteil der Spiele ohne Streichen (Mittel über die Zufalls-Seeds)."""
    out = {}
    for c in grid["e1b"]:
        out.setdefault((c["low"], c["n"]), []).append(c)
    return {k: {"deviation": max(c["deviation"] for c in cs), "kept_all": float(np.mean([c["kept_all"] for c in cs])), "games": sum(c["games"] for c in cs)} for k, cs in out.items()}


def e1_int_summary(grid: dict) -> dict:
    """E1 (ganzzahlig, Einträge 0 bis 9): je n der Anteil der Spiele mit nicht eindeutiger optimaler Strategie des Verladers (Mittel über die Zufalls-Seeds)."""
    out = {}
    for c in grid["e1_int"]:
        out.setdefault(c["n"], []).append(c)
    return {n: {"multi_x": float(np.mean([c["multi_x"] for c in cs])), "multi_y": float(np.mean([c["multi_y"] for c in cs])), "games": sum(c["games"] for c in cs)} for n, cs in out.items()}


def e2_lookup(grid: dict, profile: str) -> dict:
    """E2: {(Muster, Zähler, Gleichstand): Zelle} für ein Wertprofil."""
    return {(c["pattern"], c["counter"], c["tie"]): c for c in grid["e2"] if c["profile"] == profile}


def e3_summary(grid: dict) -> dict:
    """E3: je Verfahren die mittlere Ausnutzbarkeit der Durchschnittsstrategie und der letzten Iteration an den Horizonten und die Steigung im Log-Log-Bild zwischen T = 100 und T = 10 000."""
    out = {}
    for method in C.LEARN_METHODS:
        cs = [c for c in grid["e3"] if c["method"] == method]
        games = sum(c["games"] for c in cs)
        avg = np.sum([c["avg_sum"] for c in cs], axis=0) / games
        last = np.sum([c["last_sum"] for c in cs], axis=0) / games
        h = cs[0]["horizons"]
        slope = (math.log(avg[h.index(10000)]) - math.log(avg[h.index(100)])) / (math.log(10000) - math.log(100))
        out[method] = {"horizons": h, "avg": [float(v) for v in avg], "last": [float(v) for v in last], "slope": float(slope), "games": games}
    return out


def e3_start_summary(grid: dict) -> dict:
    """E3 (Start): {(Spiel, Verfahren, Start): Mittel über die Zufalls-Seeds von Ausnutzbarkeit des Durchschnitts, der letzten Iteration und größter Wert der letzten Iteration}."""
    out = {}
    for c in grid["e3start"]:
        out.setdefault((c["game"], c["method"], c["start"]), []).append(c)
    return {k: {"avg": float(np.mean([c["avg"] for c in cs])), "last": float(np.mean([c["last"] for c in cs])), "last_max": float(np.max([c["last_max"] for c in cs])), "seeds": len(cs)} for k, cs in out.items()}


def e4_summary(grid: dict) -> list[dict]:
    """E4: je n die Kennzahlen der Zufalls-Bimatrixspiele, gemittelt über die Zufalls-Seeds (Fehler und gerade Zahlen summiert)."""
    rows = []
    for n in C.E4_NS:
        cs = [c for c in grid["e4"] if c["n"] == n]
        rows.append({"n": n, "games": sum(c["games"] for c in cs), "mean_count": float(np.mean([c["mean_count"] for c in cs])), "max_count": max(c["max_count"] for c in cs),
                     "only_one": float(np.mean([c["only_one"] for c in cs])), "even": sum(c["even"] for c in cs), "with_pure": float(np.mean([c["with_pure"] for c in cs])),
                     "mean_pure": float(np.mean([c["mean_pure"] for c in cs])), "mean_pivots": float(np.mean([c["mean_pivots"] for c in cs])), "max_pivots": max(c["max_pivots"] for c in cs),
                     "lh_errors": sum(c["lh_errors"] for c in cs), "coverage": float(np.mean([c["coverage"] for c in cs])), "pairs": Bm.support_pairs(n)})
    return rows


def e4_big_summary(grid: dict) -> dict:
    """E4: Lemke–Howson in großen Zufallsspielen: je n Zahl der Pfade, Mittel, Median, Maximum, Fehler."""
    out = {}
    for n in C.E4_LH_BIG:
        cs = [c for c in grid["e4big"] if c["n"] == n]
        p = np.concatenate([c["pivots"] for c in cs])
        out[n] = {"paths": int(len(p)), "mean": float(p.mean()), "median": float(np.median(p)), "max": int(p.max()), "errors": sum(c["errors"] for c in cs)}
    return out


def e4_nullsum_summary(grid: dict) -> dict:
    out = {}
    for n in C.E4_NULLSUM_NS:
        cs = [c for c in grid["e4null"] if c["n"] == n]
        out[n] = {"games": sum(c["games"] for c in cs), "only_one": float(np.mean([c["only_one"] for c in cs])), "deviation": max(c["deviation"] for c in cs)}
    return out


# --- Verdikte der Experimente -----------------------------------------------------------------------------------------------------------------------

def verdict_e1(rows: list[dict], e1b: dict, ints: dict) -> tuple[str, str]:
    """E1: Sattelpunktanteil gegen die Formel, Gewinn durch Mischen, geschlossene Form gegen LP, Mehrdeutigkeit bei ganzzahligen Einträgen."""
    sq = {r["n"]: r for r in rows if r["m"] == r["n"]}
    lo, hi = min(sq), max(sq)
    worst = max(v["deviation"] for v in e1b.values())
    multi = [v["multi_x"] for v in ints.values()]
    text = (f"Zufallsspiele mit Einträgen in [0, 1), {sq[lo]['games']} je Größe: Der Anteil mit Sattelpunkt fällt von **{pct(sq[lo]['saddle'])}** ({lo} × {lo}; Formel n!²/(2n−1)! gibt {pct(sq[lo]['formula'])}) "
            f"auf **{pct(sq[hi]['saddle'])}** ({hi} × {hi}; Formel {pct(sq[hi]['formula'], 2)}); der mittlere Gewinn durch Mischen steigt von {de(sq[lo]['gain'])} auf {de(sq[hi]['gain'])}. "
            f"Verstecken und Suchen: Die geschlossene Form weicht in {len(e1b)} Zellen höchstens um {sci(worst)} vom linearen Programm ab. "
            f"Bei ganzzahligen Einträgen (0 bis 9) ist die optimale Strategie des Verladers in {pct(min(multi), 0)} bis {pct(max(multi), 0)} der Spiele (je nach Größe) **nicht eindeutig**.")
    return ("success" if worst < EPS else "warning"), text


def verdict_e2(lookup: dict, capture: dict, profile_label: str, rounds: int) -> tuple[str, str]:
    """E2: Rotation gegen Folge- und Häufigkeitszähler (beide Gleichstandsregeln), Gleichgewichtszufall gegen alle Kontrolleure."""
    value = lookup[("rotation", "follow", "low")]["value"]
    fol = lookup[("rotation", "follow", "low")]["mean"]
    low = lookup[("rotation", "freq", "low")]["mean"]
    rnd = lookup[("rotation", "freq", "random")]["mean"]
    iid = [c["mean"] for (p, _c, _t), c in lookup.items() if p == "iid_eq"]
    first = {n: capture[str(n)] for n in (2, 8)}
    text = (f"{profile_label}, {rounds} Runden, Mittel über 4 Zufalls-Seeds, Wert des Spiels {de(value)}: Eine **Rotation** liefert gegen den Folgezähler nur **{de(fol)}** je Sendung (Gleichstand kleinster Index: er fängt ab Runde {first[2]} bei zwei, ab Runde {first[8]} bei acht gleichen Routen jede ab), "
            f"gegen den Häufigkeitszähler mit Gleichstand „kleinster Index“ {de(low)}, mit zufälligem Gleichstand **{de(rnd)}**. "
            f"Ob ein Zähler ein Muster ausnutzt, hängt am Zählertyp und an der Gleichstandsregel. Gegen Zufall im Gleichgewicht liegen alle vier Kontrolleure (die Zähler mit beiden Gleichstandsregeln) zwischen {de(min(iid))} und {de(max(iid))}.")
    return "info", text


def verdict_e2_estimate(rows: list[dict]) -> tuple[str, str]:
    """E2 (Schätzen): Gegen das Gleichgewicht ist der Gewinn 0; sonst Anteil am vollen Gewinn bei kleinstem und größtem T."""
    eq = [r for r in rows if "Gleichgewicht" in r["opponent"]]
    zero = all(abs(r["gain"]) < 1e-9 for r in eq)
    ts = sorted({r["T"] for r in rows})
    uni = {r["T"]: r for r in rows if r["opponent"] == "Kontrolleur prüft gleichverteilt"}
    ctl = {r["T"]: r for r in rows if r["opponent"] == "Verlader wählt gleichverteilt"}
    text = (f"Werte (0,9; 0,8; 0,7; 0,6). Gegen einen Gegner **im Gleichgewicht** ist der Gewinn der Schätzung bei jedem gemessenen T ({ts[0]} bis {ts[-1]}) {'0' if zero else 'nicht 0'}: Alle Routen zahlen dort dasselbe. "
            f"Ein Verlader, der einen gleichverteilt prüfenden Kontrolleur beobachtet, erreicht nach {ts[-1]} Zügen {pct(uni[ts[-1]]['share'], 0)} des vollen Gewinns {de(uni[ts[-1]]['full'], 4)} (nach {ts[0]} Zug {pct(uni[ts[0]]['share'], 0)}); "
            f"der Kontrolleur, der einen gleichverteilt wählenden Verlader beobachtet, nach {ts[-1]} Zügen {pct(ctl[ts[-1]]['share'], 0)} des vollen Gewinns {de(ctl[ts[-1]]['full'], 4)}.")
    return "info", text


def verdict_e3(summary: dict, start: dict) -> tuple[str, str]:
    """E3: Durchschnitt konvergiert, letzte Iteration nicht; Start im Gleichgewicht."""
    h = summary["rm"]["horizons"]
    first, last_t = 0, len(h) - 1
    avg_all_fall = all(s["avg"][last_t] < s["avg"][first] for s in summary.values())
    last_lo = min(s["last"][last_t] for s in summary.values())
    last_hi = max(s["last"][last_t] for s in summary.values())
    avg_lo = min(s["avg"][last_t] for s in summary.values())
    avg_hi = max(s["avg"][last_t] for s in summary.values())
    rps_last = [start[("rps", m, "random")]["last"] for m in ("rm", "rmplus", "mw")]
    rps_lo, rps_hi = min(rps_last), max(rps_last)
    stand = [m for m in ("rm", "rmplus", "mw") if start[("rps", m, "uniform")]["last_max"] <= 1e-12 and start[("rps", m, "uniform")]["avg"] <= 1e-12]
    games = summary["rm"]["games"]
    big = f"{h[last_t]:,}".replace(",", " ")
    text = (f"Zufallsspiele 5 × 5, {games} Spiele, ein Lauf je Spiel mit Zufallsstart: Die Ausnutzbarkeit der **Durchschnittsstrategie** sinkt bei {'allen fünf' if avg_all_fall else 'nicht allen'} Verfahren von T = {h[first]} auf T = {big} "
            f"(dort {de(avg_lo, 4)} bis {de(avg_hi, 4)}); die **letzte Iteration** liegt bei T = {big} zwischen {de(last_lo)} und {de(last_hi)}. "
            f"Im Stein-Schere-Papier (4 Zufalls-Seeds, bis T = {big}) bleiben {', '.join(C.LEARN_METHODS[m] for m in stand)} bei Start in der Gleichverteilung an jeder Stützstelle bei Ausnutzbarkeit 0; "
            f"mit Zufallsstart ist die letzte Iteration bei T = {big} im Mittel um {de(rps_lo)} bis {de(rps_hi)} ausnutzbar.")
    return "info", text


def verdict_e4(rows: list[dict], nullsum: dict, big: dict, shapley: dict) -> tuple[str, str]:
    """E4: Zahl der Gleichgewichte, Parität, Nullsumme, Lemke–Howson (Abdeckung, Pivots), Shapley-Spiel."""
    first, last = rows[0], rows[-1]
    even = sum(r["even"] for r in rows)
    games = sum(r["games"] for r in rows)
    errors = sum(r["lh_errors"] for r in rows)
    ns_games = sum(v["games"] for v in nullsum.values())
    ns_one = sum(round(v["only_one"] * v["games"]) for v in nullsum.values())
    lo = min(w[0] for w in shapley["windows"].values())
    hi = max(w[1] for w in shapley["windows"].values())
    big_n = max(big)
    rounds_text = f"{shapley['rounds']:,}".replace(",", " ")
    text = (f"Zufalls-Bimatrixspiele (A und B unabhängig), {games} Spiele für n = {first['n']} bis {last['n']}: mittlere Zahl der Gleichgewichte von **{de(first['mean_count'], 2)}** auf **{de(last['mean_count'], 2)}**, "
            f"darunter {even} Spiele mit gerader Zahl. Nullsumme (B = 1 − A): in {ns_one} von {ns_games} Spielen **genau ein** Gleichgewicht. "
            f"Lemke–Howson liefert von jedem Startlabel ein Gleichgewicht ({errors} Fehler), aber alle Labels zusammen finden {pct(first['coverage'], 0)} (n = {first['n']}) bis {pct(last['coverage'], 0)} (n = {last['n']}) der Gleichgewichte; "
            f"Pfadlängen: Mittel {de(last['mean_pivots'], 1)} Pivots bei n = {last['n']} (Maximum {last['max_pivots']}), bei n = {big_n} Mittel {de(big[big_n]['mean'], 1)}, Median {de(big[big_n]['median'], 0)}, Maximum {big[big_n]['max']}. "
            f"Shapley-Beispiel: Beim Fictitious Play bleibt der Abstand der Häufigkeiten zur Gleichverteilung ab Runde 1 000 bis Runde {rounds_text} zwischen {de(lo)} und {de(hi)} "
            f"({shapley['switches']} Wechsel der Aktion des Zeilenspielers); im Nullsummen-Gegenstück ist er am Ende {de(shapley['zero_last'], 4)}.")
    return "info", text


# --- Rechnungen der Reiter (App) -------------------------------------------------------------------------------------------------------------------

def learn_settings(s: Settings) -> Settings:
    """Nur die Felder, von denen ein Lernlauf abhängt (Schlüssel für den Zwischenspeicher der App)."""
    base = Settings()
    return Settings(game=s.game, routes=s.routes, m=s.m, n=s.n, game_seed=s.game_seed, learn=s.learn, rounds=s.rounds, start=s.start, seed=s.seed,
                    pattern=base.pattern, counter=base.counter, tie=base.tie, obs_rounds=base.obs_rounds, bm=base.bm, bm_n=base.bm_n)


def learn_run(s: Settings) -> dict:
    """Ein Lernlauf auf dem Spiel der Hauptansicht: der Lauf, der Wert des Spiels (LP) und die größte Ausnutzbarkeit der letzten Iteration über alle Stützstellen."""
    A = game_matrix(s)
    run = Ln.run_method(s.learn, A, s.rounds, s.start, s.seed)
    return {"run": run, "value": L.solve_zero_sum(A).value, "peak_last": float(run.expl_last.max())}


def observer_settings(s: Settings) -> Settings:
    """Nur die Felder, von denen ein Beobachter-Lauf abhängt."""
    base = Settings()
    return Settings(routes=s.routes, pattern=s.pattern, counter=s.counter, tie=s.tie, obs_rounds=s.obs_rounds, seed=s.seed,
                    game=base.game, m=base.m, n=base.n, game_seed=base.game_seed, learn=base.learn, rounds=base.rounds, start=base.start, bm=base.bm, bm_n=base.bm_n)


COUNTER_COMBOS = (("freq", "low"), ("freq", "random"), ("window", "low"), ("window", "random"), ("follow", "low"), ("follow", "random"), ("fixed", "low"))


def observer_run(s: Settings) -> dict:
    """Ein Beobachter-Lauf mit den Einstellungen und der Vergleich des gewählten Musters gegen alle Zähler (je Zähler mit beiden Gleichstandsregeln) mit demselben Zufalls-Seed."""
    value, x, y, _active = H.closed_form(s.routes)
    obs = O.simulate(s.pattern, s.counter, s.tie, s.routes, x, y, s.obs_rounds, s.seed)
    overview = []
    for counter, tie in COUNTER_COMBOS:
        other = O.simulate(s.pattern, counter, tie, s.routes, x, y, s.obs_rounds, s.seed)
        overview.append({"counter": counter, "tie": tie, "mean": other.mean_delivery, "caught": other.caught_share})
    return {"obs": obs, "value": value, "x": x, "y": y, "overview": overview}
