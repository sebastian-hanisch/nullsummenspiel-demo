"""Verstecken und Suchen: der Verlader schickt eine Sendung über Route i (Wert v_i), der Kontrolleur prüft eine Route j. Geliefert wird v_i, wenn i ≠ j, sonst 0.

Geschlossene Form (eigene Herleitung, im Test gegen das lineare Programm und die Aufzählung geprüft): mit c = 1 / Σ 1/v_i ist der Wert k = (n − 1) c, der Verlader wählt x_i = c / v_i,
der Kontrolleur prüft y_i = 1 − k / v_i. Wird ein y_i negativ, ist die Route zu billig: beide lassen sie weg, und die Formel gilt für die übrigen (Streichen, kleinster Wert zuerst)."""

import numpy as np


def hide_seek_matrix(v) -> np.ndarray:
    """Matrix des Spiels: Zeile = Route des Verladers, Spalte = geprüfte Route; A[i, j] = v_i für i ≠ j, 0 auf der Diagonalen."""
    v = np.asarray(v, float)
    A = np.tile(v[:, None], (1, len(v)))
    A[np.arange(len(v)), np.arange(len(v))] = 0.0
    return A


def _all_routes_form(v):
    """Formel für alle Routen im Spiel: (Wert, x, y); y kann negative Einträge haben (dann muss gestrichen werden)."""
    c = 1.0 / np.sum(1.0 / v)
    k = (len(v) - 1) * c
    return k, c / v, 1.0 - k / v


def closed_form(v):
    """Wert, Strategie des Verladers, Strategie des Kontrolleurs und die Liste der benutzten Routen (Indizes, aufsteigend).

    Routen mit negativem y_i werden gestrichen, die mit dem kleinsten Wert zuerst, bis alle y_i ≥ 0; es bleiben immer mindestens zwei Routen."""
    v = np.asarray(v, float)
    active = list(range(len(v)))
    while True:
        k, x_a, y_a = _all_routes_form(v[active])
        if np.all(y_a >= -1e-12):
            break
        active.remove(active[int(np.argmin(v[active]))])
    x, y = np.zeros(len(v)), np.zeros(len(v))
    x[active], y[active] = x_a, np.clip(y_a, 0.0, None)
    return float(k), x, y, active


def loader_rules(v) -> dict[str, np.ndarray]:
    """Faustregeln des Verladers als Strategien: gleichverteilt, proportional zum Wert, umgekehrt proportional zum Wert (über alle Routen) und die wertvollste Route allein."""
    v = np.asarray(v, float)
    best = np.zeros(len(v))
    best[int(np.argmax(v))] = 1.0
    return {
        "gleichverteilt": np.full(len(v), 1.0 / len(v)),
        "proportional zum Wert": v / v.sum(),
        "umgekehrt proportional zum Wert (alle Routen)": (1.0 / v) / np.sum(1.0 / v),
        "nur die wertvollste Route": best,
    }


def controller_rules(v) -> dict[str, np.ndarray]:
    """Faustregeln des Kontrolleurs als Strategien: gleichverteilt, proportional zum Wert und nur die wertvollste Route prüfen."""
    v = np.asarray(v, float)
    best = np.zeros(len(v))
    best[int(np.argmax(v))] = 1.0
    return {
        "gleichverteilt": np.full(len(v), 1.0 / len(v)),
        "proportional zum Wert": v / v.sum(),
        "nur die wertvollste Route": best,
    }
