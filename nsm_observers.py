"""Beobachter: Was kostet es, ein Muster zu spielen, und was lernt ein Gegner, der nur zusieht?

Der Verlader schickt in jeder Runde eine Sendung über eine Route nach einem Muster (Rotation, Gewohnheit, kein Wiederholen) oder zufällig im Gleichgewicht. Der Kontrolleur sieht vor jeder
Runde nur die bisherigen Routen und prüft eine Route: mit einem Zähler (Häufigkeit aller bisherigen Sendungen, Fenster der letzten 20, Folge: welche Route auf welche folgt) oder mit der festen
Gleichgewichtsstrategie. Ein Zähler prüft die Route mit dem größten (Zählerstand · Wert); Gleichstände löst die Gleichstandsregel (kleinster Index oder zufällig). Geliefert wird v_i, wenn
Route i nicht geprüft wird, sonst 0. Zweiter Teil: Ein Beobachter sieht T Züge einer festen Mischung des Gegners, schätzt sie aus den Häufigkeiten und spielt die reine beste Antwort."""

from dataclasses import dataclass

import numpy as np

import nsm_constants as C
from nsm_game import TOL
from nsm_learning import draw_action, first_best
from nsm_rng import SplitMix64
from nsm_scenario import cell_seed, uniform01


def loader_sequence(pattern: str, n: int, x_star, rounds: int, rng: SplitMix64) -> np.ndarray:
    """Folge der Routen des Verlader nach dem Muster: rotation (0, 1, …, n − 1, 0, …), habit (bleibt mit Wahrscheinlichkeit ½, sonst gleichverteilt neu gezogen, auch dieselbe Route möglich),
    norepeat (gleichverteilt unter den anderen n − 1 Routen), iid_eq (unabhängig nach x_star)."""
    if pattern == "rotation":
        return np.arange(rounds) % n
    seq = np.empty(rounds, int)
    seq[0] = rng.below(n) if pattern != "iid_eq" else draw_action(x_star, rng)
    for t in range(1, rounds):
        if pattern == "habit":
            seq[t] = seq[t - 1] if uniform01(rng) < 0.5 else rng.below(n)
        elif pattern == "norepeat":
            c = rng.below(n - 1)
            seq[t] = c if c < seq[t - 1] else c + 1
        elif pattern == "iid_eq":
            seq[t] = draw_action(x_star, rng)
        else:
            raise ValueError(f"unbekanntes Muster: {pattern}")
    return seq


def choose_check(weights, tie: str, rng: SplitMix64) -> int:
    """Die Route mit dem größten Gewicht; bei Gleichstand (bis 1e-12) der kleinste Index („low“) oder eine zufällige der gleichen („random“)."""
    best = np.flatnonzero(weights >= np.max(weights) - 1e-12)
    return int(best[0]) if tie == "low" else int(best[rng.below(len(best))])


def controller_sequence(counter: str, tie: str, v, y_star, routes, rng: SplitMix64) -> np.ndarray:
    """Folge der Prüfungen: In Runde t kennt der Kontrolleur nur routes[:t]. fixed: unabhängig nach y_star; freq: Zählerstand aller Routen · v; window: Zählerstand der letzten
    WINDOW Routen · v; follow: Zähler der Routen, die auf die letzte Route folgten · v (in Runde 1 gibt es keine letzte Route: dann Gleichstand aller Routen)."""
    v = np.asarray(v, float)
    n, rounds = len(v), len(routes)
    checks = np.empty(rounds, int)
    counts = np.zeros(n)
    follow = np.zeros((n, n))
    for t in range(rounds):
        if counter == "fixed":
            checks[t] = draw_action(y_star, rng)
        elif counter == "freq":
            checks[t] = choose_check(counts * v, tie, rng)
        elif counter == "window":
            window = np.bincount(routes[max(0, t - C.WINDOW):t], minlength=n).astype(float)
            checks[t] = choose_check(window * v, tie, rng)
        elif counter == "follow":
            checks[t] = choose_check((follow[routes[t - 1]] if t > 0 else np.zeros(n)) * v, tie, rng)
        else:
            raise ValueError(f"unbekannter Zähler: {counter}")
        counts[routes[t]] += 1
        if t >= 1:
            follow[routes[t - 1], routes[t]] += 1
    return checks


def delivered(v, routes, checks) -> np.ndarray:
    """Geliefert je Runde: v[Route], wenn nicht geprüft, sonst 0."""
    v = np.asarray(v, float)
    return np.where(routes == checks, 0.0, v[routes])


@dataclass(frozen=True)
class Observation:
    routes: np.ndarray
    checks: np.ndarray
    delivered: np.ndarray
    mean_delivery: float        # Mittel über alle Runden
    caught_share: float         # Anteil der Runden, in denen die Route geprüft wurde
    running_mean: np.ndarray    # Mittel der ersten t Runden, t = 1 … rounds


def simulate(pattern: str, counter: str, tie: str, v, x_star, y_star, rounds: int, seed: int) -> Observation:
    """Ein Lauf: Folge des Verladers, Folge des Kontrolleurs (getrennte Zufallsströme aus dem Zufalls-Seed), Lieferung."""
    n = len(v)
    routes = loader_sequence(pattern, n, x_star, rounds, SplitMix64(cell_seed(seed, 11)))
    checks = controller_sequence(counter, tie, v, y_star, routes, SplitMix64(cell_seed(seed, 12)))
    got = delivered(v, routes, checks)
    return Observation(routes, checks, got, float(got.mean()), float(np.mean(routes == checks)), np.cumsum(got) / np.arange(1, rounds + 1))


def first_certain_round(routes, checks) -> int | None:
    """Erste Runde (ab 1), ab der jede weitere Sendung abgefangen wird (routes == checks bis zum Ende); None, wenn die letzte Runde nicht abgefangen wurde."""
    caught = np.asarray(routes) == np.asarray(checks)
    if not caught[-1]:
        return None
    miss = np.flatnonzero(~caught)
    return int(miss[-1]) + 2 if len(miss) else 1


def estimate_and_respond(A, value: float, side: str, opponent, observed: int, samples: int, seed: int) -> dict:
    """Ein Beobachter sieht `observed` Züge des Gegners (feste Mischung `opponent`), schätzt die Mischung aus den Häufigkeiten und spielt die reine beste Antwort auf die Schätzung.

    side „loader“: der Verlader beobachtet den Kontrolleur (opponent = y); Gewinn = (A y)[gespielte Zeile] − Wert. side „controller“: der Kontrolleur beobachtet den Verlader
    (opponent = x); Gewinn = Wert − (xᵀA)[gespielte Spalte]. Rückgabe: Mittel des Gewinns über `samples` Beobachter, voller Gewinn (beste Antwort auf die wahre Mischung) und Anteil."""
    A = np.asarray(A, float)
    opponent = np.asarray(opponent, float)
    objective = A if side == "loader" else -A.T                              # Zeile i des Beobachters: Auszahlung der Aktion i gegen jede Aktion des Gegners
    payoff = objective @ opponent                                           # Auszahlung der Aktionen des Beobachters gegen die wahre Mischung (er maximiert)
    reference = value if side == "loader" else -value                       # Gleichgewichtswert aus Sicht des Beobachters
    rng = SplitMix64(seed)
    cdf = np.cumsum(opponent)
    total = 0.0
    for _ in range(samples):
        draws = np.minimum(np.searchsorted(cdf, [uniform01(rng) for _ in range(observed)], side="right"), len(opponent) - 1)      # wie `observed` Aufrufe von draw_action
        total += payoff[first_best(objective @ np.bincount(draws, minlength=len(opponent)))]
    gain = total / samples - reference
    full = float(payoff.max()) - reference
    return {"gain": gain, "full": full, "share": (gain / full) if full > TOL else None}
