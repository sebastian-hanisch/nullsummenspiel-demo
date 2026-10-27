"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Werte sind von Hand gerechnet, die Rechnung steht daneben."""

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mixed2():
    """A = [[3, 1], [2, 4]]. Kein Sattelpunkt (Zeilenminima 1 und 2, Spaltenmaxima 3 und 4: max-min = 2, min-max = 3). Gleichsetzen der Spalten: 3x + 2(1 − x) = x + 4(1 − x), also 2 + x = 4 − 3x,
    x = 1/2, Wert 5/2. Gleichsetzen der Zeilen: 3y + (1 − y) = 2y + 4(1 − y), also 1 + 2y = 4 − 2y, y = 3/4 (Probe: Zeile 1: 3 · 3/4 + 1 · 1/4 = 10/4 = 5/2; Zeile 2: 2 · 3/4 + 4 · 1/4 = 6/4 + 1 = 5/2)."""
    return {"A": np.array([[3.0, 1.0], [2.0, 4.0]]), "value": F(5, 2), "x": (F(1, 2), F(1, 2)), "y": (F(3, 4), F(1, 4)), "security": 2.0, "ceiling": 3.0}


@pytest.fixture
def saddle():
    """A = [[4, 3], [2, 1]]. Zeilenminima 3 und 1, Spaltenmaxima 4 und 3: max-min = min-max = 3, Sattelpunkt an (0, 1) = Zeile 1, Spalte 2. Eindeutig: x = (1, 0), y = (0, 1)
    (jedes x₁ < 1 sichert nur 3x₁ + x₂ = 1 + 2x₁ < 3; jedes y₁ > 0 lässt 4y₁ + 3y₂ = 3 + y₁ > 3 zu)."""
    return {"A": np.array([[4.0, 3.0], [2.0, 1.0]]), "value": F(3), "point": (0, 1), "x": (1.0, 0.0), "y": (0.0, 1.0)}


@pytest.fixture
def flat2():
    """A = [[1, 1], [1, 1]]: Wert 1, jede Strategie beider Seiten ist optimal; der Bereich jeder Koordinate ist [0, 1]; vier Sattelpunkte."""
    return {"A": np.array([[1.0, 1.0], [1.0, 1.0]]), "value": 1.0}


@pytest.fixture
def rps():
    """Stein-Schere-Papier: Wert 0, beide Seiten Gleichverteilung; die reine Zeile garantiert −1 (Sicherheitsniveau), der reine Kontrolleur lässt +1 zu."""
    return {"A": np.array([[0.0, -1, 1], [1, 0, -1], [-1, 1, 0]]), "value": 0.0, "uniform": np.full(3, 1 / 3)}


@pytest.fixture
def hide_10_1():
    """Werte (10, 1). c = 1/(1/10 + 1) = 10/11, Wert k = (2 − 1) c = 10/11, x = c/v = (1/11, 10/11), y = 1 − k/v = (1 − 1/11, 1 − 10/11) = (10/11, 1/11).
    Probe: Verlader auf Route 1: 10 · (1 − 10/11) = 10/11; auf Route 2: 1 · (1 − 1/11) = 10/11. Gleich, also im Gleichgewicht."""
    return {"v": (10.0, 1.0), "value": F(10, 11), "x": (F(1, 11), F(10, 11)), "y": (F(10, 11), F(1, 11))}


@pytest.fixture
def hide_4321():
    """Werte (4, 3, 2, 1). Alle vier Routen: c = 1/(1/4 + 1/3 + 1/2 + 1) = 12/25, k = 3 · 12/25 = 36/25, y₄ = 1 − 36/25 = −11/25 < 0: Route 4 wird gestrichen.
    Rest (4, 3, 2): c = 1/(13/12) = 12/13, k = 2 · 12/13 = 24/13, x = (3/13, 4/13, 6/13), y = 1 − k/v = (1 − 6/13, 1 − 8/13, 1 − 12/13) = (7/13, 5/13, 1/13). Probe der Route 4: 1 · (1 − 0) = 1 ≤ 24/13."""
    return {"v": (4.0, 3.0, 2.0, 1.0), "value": F(24, 13), "x": (F(3, 13), F(4, 13), F(6, 13), F(0)), "y": (F(7, 13), F(5, 13), F(1, 13), F(0)), "active": [0, 1, 2]}
