"""Szenario: Einstellungen, Routenwerte lesen und schreiben, Zufallsspiele mit eigenem Zufallsgenerator (SplitMix64), damit alle Spiele auf jeder Plattform gleich sind.

Der Generator ist die wörtliche Kopie `nsm_rng.py`; hier liegt nur, was darauf aufbaut. Ein Zufallsspiel hat gleichverteilte Einträge in [0, 1) (53 Bit aus dem Strom, Zeile für Zeile)."""

import math
from dataclasses import dataclass

import numpy as np

import nsm_constants as C
from nsm_rng import SplitMix64


@dataclass(frozen=True)
class Settings:
    game: str = C.DEFAULT_GAME
    routes: tuple = (1.0, 1.0, 1.0, 1.0)
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    game_seed: int = C.DEFAULT_GAME_SEED
    learn: str = C.DEFAULT_LEARN
    rounds: int = C.DEFAULT_ROUNDS
    start: str = C.DEFAULT_START
    pattern: str = C.DEFAULT_PATTERN
    counter: str = C.DEFAULT_COUNTER
    tie: str = C.DEFAULT_TIE
    obs_rounds: int = C.DEFAULT_OBS_ROUNDS
    bm: str = C.DEFAULT_BM
    bm_n: int = C.DEFAULT_BM_N
    seed: int = C.DEFAULT_SEED


def uniform01(rng: SplitMix64) -> float:
    """Gleichverteilte Zahl in [0, 1) mit 53 Bit."""
    return (rng.next() >> 11) / float(1 << 53)


def exponential(rng: SplitMix64) -> float:
    """Exponentialverteilte Zahl mit Mittel 1 (für gleichverteilte Punkte der Wahrscheinlichkeitssimplex)."""
    return -math.log(1.0 - uniform01(rng))


def random_simplex(k: int, rng: SplitMix64) -> np.ndarray:
    """Gleichverteilter Punkt der Wahrscheinlichkeitssimplex mit k Koordinaten (normierte Exponentialzahlen)."""
    p = np.array([exponential(rng) for _ in range(k)])
    return p / p.sum()


def draw_matrix(m: int, n: int, seed: int) -> np.ndarray:
    """Zufallsspiel m × n: gleichverteilte Einträge in [0, 1), deterministisch aus (m, n, seed)."""
    rng = SplitMix64(seed)
    return np.array([[uniform01(rng) for _ in range(n)] for _ in range(m)])


def draw_bimatrix(m: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Zwei unabhängige Zufallsmatrizen (A: Zeilenspieler, B: Spaltenspieler) aus einem Strom: erst A, dann B."""
    rng = SplitMix64(seed)
    A = np.array([[uniform01(rng) for _ in range(n)] for _ in range(m)])
    B = np.array([[uniform01(rng) for _ in range(n)] for _ in range(m)])
    return A, B


def cell_seed(base: int, *key: int) -> int:
    """Zufalls-Seed einer Zelle der Messreihen aus dem Basis-Seed und einem Schlüssel (ganze Zahlen); kein Python-hash()."""
    seed = int(base)
    for k in key:
        seed = (seed * 1_000_003 + int(k) + 1) % (1 << 62)
    return seed


def parse_routes(text) -> tuple | None:
    """Routenwerte aus dem Text „0,75; 0,5“ (Trennzeichen Semikolon, Dezimalkomma oder -punkt): 2 bis 8 Werte in [0,05; 1]; sonst None."""
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    try:
        values = tuple(float(p.replace(",", ".")) for p in parts)
    except ValueError:
        return None
    if not C.ROUTES_MIN <= len(values) <= C.ROUTES_MAX:
        return None
    if any(not (C.VALUE_MIN <= v <= C.VALUE_MAX) for v in values):          # auch nan fällt hier heraus
        return None
    return values


def format_routes(values) -> str:
    """Kanonischer Text der Routenwerte: 0,75; 0,5 (Dezimalkomma, Semikolon, Leerzeichen)."""
    return "; ".join(f"{v:g}".replace(".", ",") for v in values)


def game_matrix(s: Settings) -> np.ndarray:
    """Die Auszahlungsmatrix des Spiels der Hauptansicht."""
    from nsm_hideseek import hide_seek_matrix

    if s.game == "hide":
        return hide_seek_matrix(s.routes)
    if s.game == "random":
        return draw_matrix(s.m, s.n, s.game_seed)
    if s.game == "rps":
        return np.array(C.RPS_MATRIX)
    if s.game == "saddle":
        return np.array(C.SADDLE_MATRIX)
    raise ValueError(f"unbekanntes Spiel: {s.game}")
