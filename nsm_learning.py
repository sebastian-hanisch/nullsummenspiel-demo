"""Lernen im Nullsummenspiel: beide Seiten lernen aus den Auszahlungen vergangener Runden. Verfahren: Regret Matching, Regret Matching+ (RM+), Multiplicative Weights (Hedge),
Fictitious Play und Regret Matching mit gezogenen Zügen.

Aufgezeichnet wird die Ausnutzbarkeit zweier Strategiepaare an den Stützstellen: der Durchschnittsstrategie (Mittel aller bisher gespielten Strategien) und der letzten Iteration
(der Strategie der Runde t selbst). Der Zufallsstart ist Pflicht: Wer in der Gleichverteilung beginnt und dort schon im Gleichgewicht ist, bewegt sich bei Regret Matching,
RM+ und Multiplicative Weights nie weg (Auszahlungen aller Aktionen gleich, alle Regrets null)."""

from dataclasses import dataclass

import numpy as np

import nsm_constants as C
from nsm_game import exploitability
from nsm_rng import SplitMix64
from nsm_scenario import cell_seed, random_simplex, uniform01

TIE_TOL = 1e-12
FIRST_KEPT = 4              # so viele erste Strategien hält ein Lauf für die Handrechnung fest


@dataclass(frozen=True)
class Start:
    x1: np.ndarray          # Strategie des Verladers in Runde 1
    y1: np.ndarray          # Strategie des Kontrolleurs in Runde 1
    a0: int                 # erste Aktion beim Fictitious Play (Verlader)
    b0: int                 # erste Aktion beim Fictitious Play (Kontrolleur)


@dataclass(frozen=True)
class Run:
    steps: np.ndarray       # Runden t der Stützstellen
    expl_avg: np.ndarray    # Ausnutzbarkeit der Durchschnittsstrategie bis t
    expl_last: np.ndarray   # Ausnutzbarkeit der Strategie der Runde t
    avg_x: np.ndarray       # Durchschnittsstrategien nach der letzten Runde
    avg_y: np.ndarray
    last_x: np.ndarray      # Strategien der letzten Runde
    last_y: np.ndarray
    first_x: np.ndarray     # die ersten Strategien (zeilenweise), für die Handrechnung
    first_y: np.ndarray


def checkpoints(rounds: int, count: int = C.LEARN_CHECKPOINTS) -> np.ndarray:
    """Stützstellen 1 … rounds, logarithmisch verteilt, ohne Doppelte, mit der letzten Runde und allen Zehnerpotenzen (10, 100, 1000, … bis rounds)."""
    points = set(np.round(np.geomspace(1, rounds, count)).astype(int).tolist()) | {10 ** k for k in range(5) if 10 ** k <= rounds} | {rounds}
    return np.array(sorted(p for p in points if 1 <= p <= rounds))


def make_start(m: int, n: int, kind: str, rng: SplitMix64) -> Start:
    """Startwerte: „random“ zieht beide Strategien gleichverteilt aus der Simplex und die ersten Aktionen zufällig, „uniform“ beginnt in der Gleichverteilung (erste Aktion: Index 0)."""
    if kind == "random":
        return Start(random_simplex(m, rng), random_simplex(n, rng), rng.below(m), rng.below(n))
    if kind == "uniform":
        return Start(np.full(m, 1.0 / m), np.full(n, 1.0 / n), 0, 0)
    raise ValueError(f"unbekannter Start: {kind}")


def positive_strategy(R) -> np.ndarray:
    """Regret Matching: Wahrscheinlichkeit proportional zum positiven Teil der Regrets; sind alle Regrets ≤ 0, die Gleichverteilung."""
    p = np.maximum(np.asarray(R, float), 0.0)
    total = p.sum()
    return p / total if total > 0 else np.full(len(p), 1.0 / len(p))


def regret_update(A, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Zuwachs der Regrets einer Runde bei erwarteten Auszahlungen: Auszahlung jeder Aktion minus die Auszahlung der gespielten Mischung, für den Verlader (Zeilen, maximiert)
    und den Kontrolleur (Spalten, dessen Auszahlung ist −A)."""
    A = np.asarray(A, float)
    u_x = A @ y
    u_y = -(x @ A)
    return u_x - x @ u_x, u_y - y @ u_y


def first_best(u, tol: float = TIE_TOL) -> int:
    """Index der größten Zahl; bei Gleichstand (bis tol) der kleinste Index."""
    return int(np.flatnonzero(u >= np.max(u) - tol)[0])


def _recorder(A, rounds):
    steps = checkpoints(rounds)
    return steps, {int(t): k for k, t in enumerate(steps)}, np.zeros(len(steps)), np.zeros(len(steps))


def _finish(steps, ea, el, sx, sy, wsum, x, y, firsts_x, firsts_y) -> Run:
    return Run(steps, ea, el, sx / wsum, sy / wsum, x, y, np.array(firsts_x), np.array(firsts_y))


def regret_matching(A, rounds: int, start: Start, plus: bool = False) -> Run:
    """Regret Matching (plus=False) bzw. RM+ (plus=True: Regrets auf null abgeschnitten, Durchschnitt mit Gewicht t). Erwartete Auszahlungen (volle Information).
    RM+ ist aus dem Gedächtnis umgesetzt (Quelle nicht eingesehen); gleichzeitige Aktualisierung beider Seiten."""
    A = np.asarray(A, float)
    m, n = A.shape
    steps, index, ea, el = _recorder(A, rounds)
    Rx, Ry = np.zeros(m), np.zeros(n)
    x, y = start.x1.copy(), start.y1.copy()
    sx, sy, wsum = np.zeros(m), np.zeros(n), 0.0
    firsts_x, firsts_y = [], []
    for t in range(1, rounds + 1):
        w = float(t) if plus else 1.0
        sx += w * x
        sy += w * y
        wsum += w
        if t <= FIRST_KEPT:
            firsts_x.append(x.copy())
            firsts_y.append(y.copy())
        if t in index:
            ea[index[t]] = exploitability(A, sx / wsum, sy / wsum)
            el[index[t]] = exploitability(A, x, y)
        gx, gy = regret_update(A, x, y)
        Rx, Ry = Rx + gx, Ry + gy
        if plus:
            Rx, Ry = np.maximum(Rx, 0.0), np.maximum(Ry, 0.0)
        x, y = positive_strategy(Rx), positive_strategy(Ry)
    return _finish(steps, ea, el, sx, sy, wsum, x, y, firsts_x, firsts_y)


def hedge_rate(A, own_actions: int, rounds: int) -> float:
    """Lernrate η = √(8 ln N / T) / (Spannweite der Auszahlungen) für N eigene Aktionen und Horizont T (Standardwahl für Auszahlungen in [0, 1]; bei konstanter Matrix Spannweite 1)."""
    A = np.asarray(A, float)
    spread = float(A.max() - A.min()) or 1.0
    return float(np.sqrt(8.0 * np.log(max(own_actions, 2)) / rounds) / spread)


def hedge(A, rounds: int, start: Start) -> Run:
    """Multiplicative Weights (Hedge): Wahrscheinlichkeit proportional zu exp(η · Summe der bisherigen erwarteten Auszahlungen); η aus dem Horizont `rounds`. Die Anfangssummen
    sind so gewählt, dass die Strategien der Runde 1 die Startstrategien sind."""
    A = np.asarray(A, float)
    m, n = A.shape
    eta_x, eta_y = hedge_rate(A, m, rounds), hedge_rate(A, n, rounds)
    cx, cy = np.log(start.x1) / eta_x, np.log(start.y1) / eta_y
    steps, index, ea, el = _recorder(A, rounds)
    sx, sy, wsum = np.zeros(m), np.zeros(n), 0.0
    firsts_x, firsts_y = [], []
    for t in range(1, rounds + 1):
        x = np.exp(eta_x * (cx - cx.max()))
        x /= x.sum()
        y = np.exp(eta_y * (cy - cy.max()))
        y /= y.sum()
        sx += x
        sy += y
        wsum += 1.0
        if t <= FIRST_KEPT:
            firsts_x.append(x.copy())
            firsts_y.append(y.copy())
        if t in index:
            ea[index[t]] = exploitability(A, sx / wsum, sy / wsum)
            el[index[t]] = exploitability(A, x, y)
        cx = cx + A @ y
        cy = cy - x @ A
    return _finish(steps, ea, el, sx, sy, wsum, x, y, firsts_x, firsts_y)


def fictitious_play(A, rounds: int, start: Start) -> tuple[Run, np.ndarray, np.ndarray]:
    """Fictitious Play: jede Seite spielt die beste Antwort auf die Häufigkeiten der bisherigen Aktionen des Gegners (Gleichstand: kleinster Index), beide gleichzeitig.
    Gibt den Lauf und die Folgen der gespielten Aktionen beider Seiten zurück. Die „Strategie der Runde t“ ist die gespielte reine Aktion."""
    A = np.asarray(A, float)
    m, n = A.shape
    steps, index, ea, el = _recorder(A, rounds)
    cx, cy = np.zeros(m), np.zeros(n)
    a, b = start.a0, start.b0
    acts_a, acts_b = np.zeros(rounds, int), np.zeros(rounds, int)
    firsts_x, firsts_y = [], []
    for t in range(1, rounds + 1):
        cx[a] += 1
        cy[b] += 1
        acts_a[t - 1], acts_b[t - 1] = a, b
        x_pure, y_pure = np.zeros(m), np.zeros(n)
        x_pure[a], y_pure[b] = 1.0, 1.0
        if t <= FIRST_KEPT:
            firsts_x.append(x_pure)
            firsts_y.append(y_pure)
        if t in index:
            ea[index[t]] = exploitability(A, cx / t, cy / t)
            el[index[t]] = exploitability(A, x_pure, y_pure)
        a, b = first_best(A @ cy), first_best(-(cx @ A))
    return _finish(steps, ea, el, cx, cy, float(rounds), x_pure, y_pure, firsts_x, firsts_y), acts_a, acts_b


def draw_action(p, rng: SplitMix64) -> int:
    """Eine Aktion nach der Verteilung p (Inversionsverfahren mit einer gleichverteilten Zahl)."""
    return int(min(np.searchsorted(np.cumsum(p), uniform01(rng), side="right"), len(p) - 1))


def sampled_regret_update(A, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Zuwachs der Regrets, wenn der Verlader Zeile a und der Kontrolleur Spalte b gespielt hat: Auszahlung jeder anderen Aktion gegen die gespielte Gegneraktion minus die erhaltene Auszahlung.
    Im Erwartungswert über (a, b) gleich `regret_update` (unverzerrt)."""
    A = np.asarray(A, float)
    gain = A[a, b]
    return A[:, b] - gain, gain - A[a, :]


def rm_sampled(A, rounds: int, start: Start, rng: SplitMix64) -> Run:
    """Regret Matching mit gezogenen Zügen: beide ziehen eine Aktion aus ihrer Mischung, der Regret wird aus der tatsächlich gespielten Gegneraktion gebildet (unverzerrt)."""
    A = np.asarray(A, float)
    m, n = A.shape
    steps, index, ea, el = _recorder(A, rounds)
    Rx, Ry = np.zeros(m), np.zeros(n)
    x, y = start.x1.copy(), start.y1.copy()
    sx, sy, wsum = np.zeros(m), np.zeros(n), 0.0
    firsts_x, firsts_y = [], []
    for t in range(1, rounds + 1):
        sx += x
        sy += y
        wsum += 1.0
        if t <= FIRST_KEPT:
            firsts_x.append(x.copy())
            firsts_y.append(y.copy())
        if t in index:
            ea[index[t]] = exploitability(A, sx / wsum, sy / wsum)
            el[index[t]] = exploitability(A, x, y)
        a, b = draw_action(x, rng), draw_action(y, rng)
        gx, gy = sampled_regret_update(A, a, b)
        Rx, Ry = Rx + gx, Ry + gy
        x, y = positive_strategy(Rx), positive_strategy(Ry)
    return _finish(steps, ea, el, sx, sy, wsum, x, y, firsts_x, firsts_y)


def run_method(method: str, A, rounds: int, start_kind: str, seed: int) -> Run:
    """Ein Lauf des gewählten Verfahrens; Start und gezogene Züge kommen aus dem Zufalls-Seed (getrennte Ströme)."""
    A = np.asarray(A, float)
    start = make_start(A.shape[0], A.shape[1], start_kind, SplitMix64(cell_seed(seed, 1)))
    if method == "rm":
        return regret_matching(A, rounds, start)
    if method == "rmplus":
        return regret_matching(A, rounds, start, plus=True)
    if method == "mw":
        return hedge(A, rounds, start)
    if method == "fp":
        return fictitious_play(A, rounds, start)[0]
    if method == "rm_sampled":
        return rm_sampled(A, rounds, start, SplitMix64(cell_seed(seed, 2)))
    raise ValueError(f"unbekanntes Verfahren: {method}")
