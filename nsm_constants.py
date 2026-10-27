"""Konstanten: Regler-Grenzen, Wahlmöglichkeiten, Raster der Messreihen, Farben und Presets.

Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): Das lineare Programm (HiGHS) kostet bei 10 × 10 etwa eine Millisekunde; die Bereiche der Optimalmenge sind 2 (m + n) weitere
kleine Programme, also O(m + n) Aufrufe. Ein Lernlauf mit 10 000 Runden ist O(Runden · m · n) (Zehntelsekunden bei 10 × 10), ein Beobachter-Lauf mit 2 000 Runden
O(Runden · Routen), die Stützenaufzählung der Bimatrixspiele O(C(2n, n)) Gleichungssysteme (n = 8: 12 869 Stützenpaare, Zehntelsekunden). Die Messreihen der Experimente stehen
vorgerechnet in data/nullsummen_grid.json (tools/make_grid.py)."""

from nsm_rng import (
    MASK,  # noqa: F401  (Wiederverwendung der Kopie; nsm_rng.py bleibt wörtlich)
)

# --- das Spiel der Hauptansicht ---------------------------------------------------------------------------------------------------------------
GAMES = {
    "hide": "Verstecken und Suchen (Routenwerte)",
    "random": "Zufallsspiel m × n",
    "rps": "Stein-Schere-Papier",
    "saddle": "Sattelpunkt ([[4, 3], [2, 1]])",
}
DEFAULT_GAME = "hide"
ROUTES_MIN, ROUTES_MAX = 2, 8                 # Zahl der Routen
VALUE_MIN, VALUE_MAX = 0.05, 1.0              # Wert einer Route (Anteil der Ladung)
DEFAULT_ROUTES = "1; 1; 1; 1"
M_MIN, M_MAX, DEFAULT_M = 2, 10, 5            # Zeilen des Zufallsspiels
N_MIN, N_MAX, DEFAULT_N = 2, 10, 5            # Spalten des Zufallsspiels
SEED_MAX, DEFAULT_SEED = 999999, 7
DEFAULT_GAME_SEED = 35
SADDLE_MATRIX = ((4.0, 3.0), (2.0, 1.0))
RPS_MATRIX = ((0.0, -1.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 1.0, 0.0))     # Zeile: Stein, Schere, Papier; Spalte ebenso; +1 für Sieg der Zeile

# --- Lernen ------------------------------------------------------------------------------------------------------------------------------------
LEARN_METHODS = {
    "rm": "Regret Matching",
    "rmplus": "Regret Matching+ (RM+)",
    "mw": "Multiplicative Weights (Hedge)",
    "fp": "Fictitious Play",
    "rm_sampled": "Regret Matching mit gezogenen Zügen",
}
DEFAULT_LEARN = "rm"
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 100, 10000, 2000
STARTS = {"random": "Zufallsstart", "uniform": "Gleichverteilung"}
DEFAULT_START = "random"
LEARN_CHECKPOINTS = 60                        # so viele Stützstellen (logarithmisch) zeichnet der Verlauf

# --- Beobachter -------------------------------------------------------------------------------------------------------------------------------
PATTERNS = {
    "rotation": "Rotation (Route 1, 2, 3, …)",
    "habit": "Gewohnheit (bleibt mit Wahrscheinlichkeit ½, sonst zufällig)",
    "norepeat": "kein Wiederholen (nie dieselbe Route zweimal)",
    "iid_eq": "Zufall im Gleichgewicht (unabhängig, Strategie x*)",
}
DEFAULT_PATTERN = "rotation"
COUNTERS = {
    "freq": "Häufigkeitszähler (alle bisherigen Sendungen)",
    "window": "Fensterzähler (letzte 20 Sendungen)",
    "follow": "Folgezähler (welche Route folgt auf welche)",
    "fixed": "feste Gleichgewichtsstrategie y* (kein Zähler)",
}
DEFAULT_COUNTER = "freq"
TIES = {"low": "kleinster Index", "random": "zufällige Wahl"}
DEFAULT_TIE = "low"
WINDOW = 20                                   # Länge des Fensterzählers
OBS_ROUNDS_MIN, OBS_ROUNDS_MAX, DEFAULT_OBS_ROUNDS = 100, 2000, 500

# --- allgemeine Spiele (Reiter) ----------------------------------------------------------------------------------------------------------------
BM_KINDS = {
    "random": "Zufallsspiel (A und B unabhängig)",
    "zero_sum": "Nullsumme (B = 1 − A)",
    "shapley": "Shapley-Beispiel 3 × 3",
}
DEFAULT_BM = "random"
BM_N_MIN, BM_N_MAX, DEFAULT_BM_N = 2, 8, 4    # Aufzählung bis n = 8: 12 869 Stützenpaare, Zehntelsekunden
SHAPLEY_ROUNDS_LIVE = 20000                   # Fictitious Play im Shapley-Spiel, live (Rechenzeit linear)

# --- Messreihen (tools/make_grid.py) -----------------------------------------------------------------------------------------------------------
GRID_SEEDS = (11, 12, 13, 14)
E1_SIZES = ((2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (10, 10), (3, 6), (6, 3))
E1_GAMES = 1000                               # Zufallsspiele je Größe und Zufalls-Seed
E1B_NS = (2, 3, 4, 6, 8)
E1B_RANGES = ((0.3, 1.0), (0.05, 1.0))
E1B_GAMES = 100                               # Spiele je Zelle und Zufalls-Seed
E2_T = 1000
E2_PROFILES = {"vier gleiche Routen": (1.0, 1.0, 1.0, 1.0), "ungleich (0,9; 0,8; 0,7; 0,6)": (0.9, 0.8, 0.7, 0.6)}
E2_EST_TS = (1, 5, 10, 50, 100, 500, 1000)
E2_EST_SAMPLES = 400                          # Beobachter je Zelle und Zufalls-Seed
E3_HORIZONS = (10, 100, 1000, 10000)
E3_GAMES = 5                                  # Zufallsspiele 5 × 5 je Zufalls-Seed
E3_SIZE = 5
E4_NS = (2, 3, 4, 5, 6, 7, 8)
E4_GAMES = {2: 300, 3: 300, 4: 200, 5: 150, 6: 100, 7: 40, 8: 20}      # Spiele je n und Zufalls-Seed
E4_NULLSUM_NS = (2, 3, 4, 5)
E4_LH_BIG = (20, 40)
SHAPLEY_ROUNDS_GRID = 200000

# --- Farben -----------------------------------------------------------------------------------------------------------------------------------
COLOR_LOADER = "#2F6B65"
COLOR_CONTROL = "#C0504D"
COLOR_DROPPED = "rgba(150,150,150,0.55)"
COLOR_AVG = "#4c78a8"
COLOR_LAST = "#e68a00"
COLOR_VALUE = "#333333"

# --- Presets ----------------------------------------------------------------------------------------------------------------------------------
_DEFAULT = {"game": "hide", "routes": DEFAULT_ROUTES}
PRESETS = {
    "Vier gleiche Routen (Voreinstellung)": dict(_DEFAULT),
    "Münzwurf (zwei gleiche Routen)": {"game": "hide", "routes": "1; 1"},
    "Ungleiche Werte (10 : 1)": {"game": "hide", "routes": "1; 0,1"},
    "Vier Routen mit Streichen": {"game": "hide", "routes": "1; 0,75; 0,5; 0,25"},
    "Stein-Schere-Papier": {"game": "rps"},
    "Sattelpunkt": {"game": "saddle"},
    "Zufallsspiel 5 × 5": {"game": "random", "m": 5, "n": 5},
    "Rotation gegen Folgezähler": {"pattern": "rotation", "counter": "follow", "obs_rounds": 1000},
    "Rotation gegen Häufigkeitszähler": {"pattern": "rotation", "counter": "freq", "tie": "random", "obs_rounds": 1000},
}
PRESET_HELP = {
    "Vier gleiche Routen (Voreinstellung)": "Vier Routen mit dem Wert 1 (ganze Ladung): das Gleichgewicht wählt und prüft jede Route mit Wahrscheinlichkeit 1/4, der Wert des Spiels ist 3/4.",
    "Münzwurf (zwei gleiche Routen)": "Zwei Routen mit dem Wert 1: wählen und prüfen wie ein Münzwurf, der Wert des Spiels ist 1/2.",
    "Ungleiche Werte (10 : 1)": "Eine Route zehnmal so wertvoll wie die andere: das Gleichgewicht wählt die billige Route 10-mal so oft.",
    "Vier Routen mit Streichen": "Werte 1; 0,75; 0,5; 0,25: die vierte Route wird im Gleichgewicht von beiden Seiten mit Wahrscheinlichkeit 0 benutzt.",
    "Stein-Schere-Papier": "Das klassische symmetrische Spiel ohne Sattelpunkt: Gleichgewicht ist die Gleichverteilung, Wert 0.",
    "Sattelpunkt": "Hier ist die reine Strategie schon optimal (Wert 3): Zufall bringt nichts.",
    "Zufallsspiel 5 × 5": "Ein Spiel mit gleichverteilten Zufallseinträgen; Zeilen- und Spaltenzahl und Zufalls-Seed sind frei.",
    "Rotation gegen Folgezähler": "Der Verlader rotiert durch die Routen, der Kontrolleur zählt, welche Route auf welche folgt; 1 000 Runden (Reiter Beobachter).",
    "Rotation gegen Häufigkeitszähler": "Der Kontrolleur prüft die bisher häufigste Route und löst Gleichstände zufällig; 1 000 Runden (Reiter Beobachter).",
}


# --- Texte (die Fassung in der README muss mit ASSUMPTIONS_MARKDOWN übereinstimmen; tests/test_claims.py vergleicht beide) ----------------------------------------------------------

ASSUMPTIONS_MARKDOWN = """
**Annahmen**
- **Nullsumme:** Was der Verlader bekommt, verliert der Kontrolleur (Auszahlung = Anteil der Ladung, der ankommt). Eigennutz und ein mitdenkender Gegner sind Modellannahmen, keine Aussagen über Betriebe oder echte Kontrollen.
- **Eine Kontrollstelle, ein Zug:** Der Kontrolleur prüft genau eine Route; mehrere Stellen, Ressourcen und Festlegungen vorab sind nicht modelliert.
- **Erwartungswerte:** Die Auszahlung ist der Erwartungswert. Zufall als Aufteilen derselben Sendung liefert im Erwartungswert dasselbe, die Streuung eines einzelnen Tages ist eine andere; die Demo rechnet sie nicht.
- **Volle Kenntnis der Matrix, gleichzeitige Züge:** Beide Seiten kennen alle Auszahlungen; verdeckte Information und Züge nacheinander (Spielbaum) sind nicht gebaut.
- **Lernen mit erwarteten Auszahlungen:** Regret Matching, RM+, Multiplicative Weights und Fictitious Play sehen in jeder Runde die erwartete Auszahlung jeder Aktion gegen die Mischung des Gegners; nur Regret Matching mit gezogenen Zügen sieht allein die tatsächlich gespielte Aktion.

**Grenzen**
- Die Zahlen der Experimente gelten für die genannten Zufallsfamilien: Einträge gleichverteilt in [0, 1) bzw. Routenwerte gleichverteilt in einem Bereich, jeweils mit festen Zufalls-Seeds. Für andere Familien sind sie nicht gemessen.
- Es sind drei Zähler (Häufigkeit, Fenster mit 20 Sendungen, Folge) und vier Muster des Verladers gemessen. Dass es vom Zählertyp und von der Gleichstandsregel abhängt, ob ein Muster ausgenutzt wird, gilt für diese Zähler und diese Muster.
- Die Aufzählung der Gleichgewichte allgemeiner Spiele geht bis n = 8 Aktionen je Spieler; sie ist für nicht entartete Spiele vollständig. Eine Familie von Spielen, in denen Lemke–Howson lange Pfade braucht, ist nicht gebaut, und die Aussagen „PPAD-vollständig“ und „im ungünstigsten Fall exponentiell“ stehen nur als Literaturzeile.
- Regret Matching+ (RM+) ist aus dem Gedächtnis umgesetzt und seine Quelle nicht eingesehen. Alle Literaturangaben (Minimax-Satz, Fictitious Play, Regret Matching, Lemke–Howson, Komplexität) sind nicht im Volltext gelesen.
- Keine Daten echter Verlader, Kontrollen oder Menschenexperimente.

**Bekannte Probleme und Unsicherheiten**
- **Optimale Menge:** Ob eine optimale Strategie eindeutig ist, entscheidet die Demo aus den Bereichen jeder Wahrscheinlichkeit über die optimale Menge (Toleranz 10⁻⁹, Schwelle 10⁻⁶); bei fast entarteten Spielen kann ein sehr schmaler Bereich als eindeutig gelten.
- **Gleichstände in Lernverfahren und Zählern:** Fictitious Play und die Zähler lösen exakte Gleichstände nach einer Regel (kleinster Index oder Zufall); bei den Zählern zeigt der Reiter Beobachter, dass das Ergebnis daran hängen kann.
- **Multiplicative Weights:** Die Lernrate hängt vom Horizont ab (Lernrunden); die Zahlen bei kleineren T in Experiment 3 stammen aus eigenen Läufen mit diesem Horizont.
- **Test:** Die Oberfläche ist mit Streamlits AppTest geprüft (jeder Knopf, jeder Regler an beiden Grenzen); ein echter Mausklick im Browser ist nicht Teil der Tests.
"""

MATH_MARKDOWN = r"""
**Lesehilfe.** Die Matrix $A$ hat $m$ Zeilen (Routen des Verladers) und $n$ Spalten (geprüfte Routen); $A_{ij}$ ist die Auszahlung an den Verlader, wenn er Zeile $i$ wählt und der Kontrolleur Spalte $j$. Der Verlader mischt mit $x$, der Kontrolleur mit $y$ (Wahrscheinlichkeitsvektoren).

| Symbol | Bedeutung |
|---|---|
| $A_{ij}$, $x$, $y$ | Auszahlung, Strategie des Verladers, Strategie des Kontrolleurs |
| $v$ | Wert des Spiels |
| $v_i$ | Wert der Route $i$ (Verstecken und Suchen); $c$, $k$ Hilfsgrößen der geschlossenen Form |
| $\varepsilon(x, y)$ | Ausnutzbarkeit eines Strategiepaares |
| $R_t(i)$, $p_t$ | Bedauern der Aktion $i$ nach Runde $t$, gespielte Mischung der Runde $t$ |
| $\eta$, $\Delta$, $N$, $T$ | Lernrate, Spannweite der Auszahlungen, Zahl der eigenen Aktionen, Horizont |
| $B$ | Auszahlungsmatrix des Spaltenspielers (allgemeine Spiele) |

**Garantiert und zugelassen:** was $x$ gegen die beste Antwort sichert und was $y$ höchstens zulässt:

$$ \underline{v}(x) = \min_j\, (x^\top A)_j, \qquad \overline{v}(y) = \max_i\, (A\,y)_i $$

**Minimax-Satz:** das Beste, was der Verlader garantieren kann, ist das Beste, was der Kontrolleur zulässt:

$$ \max_x \min_j\, (x^\top A)_j \;=\; \min_y \max_i\, (A\,y)_i \;=\; v $$

**Lineares Programm** des Verladers und sein duales Programm (Kontrolleur):

$$ \max\, v \quad \text{u. d. N.} \quad \sum_i x_i A_{ij} \ge v \;\; (j = 1, \dots, n), \quad \sum_i x_i = 1, \quad x \ge 0 $$

$$ \min\, w \quad \text{u. d. N.} \quad \sum_j A_{ij}\, y_j \le w \;\; (i = 1, \dots, m), \quad \sum_j y_j = 1, \quad y \ge 0 $$

**Ausnutzbarkeit** eines Paares (null genau im Gleichgewicht):

$$ \varepsilon(x, y) = \max_i\, (A\,y)_i - \min_j\, (x^\top A)_j \;\ge\; 0 $$

**Sattelpunkt** in einem Zufallsspiel $n \times n$ mit stetigen Einträgen (eigene Herleitung, in Messung und Test bestätigt):

$$ \Pr[\text{Sattelpunkt}] = \frac{(n!)^2}{(2n-1)!} $$

**Verstecken und Suchen:** $A_{ij} = v_i$ für $i \ne j$ und $A_{ii} = 0$. Mit $c = \big(\sum_i 1/v_i\big)^{-1}$:

$$ v = (n-1)\,c, \qquad x_i = \frac{c}{v_i}, \qquad y_i = 1 - \frac{v}{v_i} $$

Wird ein $y_i$ negativ, ist die Route zu billig: sie wird weggelassen, kleinster Wert zuerst, und die Formel gilt für die übrigen Routen (**Streichen**). Die beste reine Route garantiert 0, denn der Kontrolleur prüft sie.

**Regret Matching** mit dem Bedauern $R_t(i) = \sum_{s \le t}\big(u_s(i) - \langle p_s, u_s\rangle\big)$ und **RM+** (abgeschnittenes Bedauern, Durchschnitt mit Gewicht $t$):

$$ p_{t+1}(i) = \frac{\max\big(R_t(i),\, 0\big)}{\sum_k \max\big(R_t(k),\, 0\big)}, \qquad R_t^{+}(i) = \max\big(R_{t-1}^{+}(i) + u_t(i) - \langle p_t, u_t\rangle,\; 0\big) $$

**Multiplicative Weights (Hedge)** mit Lernrate vom Horizont, und die (aus dem Gedächtnis angesetzte, im Test als Band geprüfte) Schranke für die Ausnutzbarkeit des Durchschnitts bei Start in der Gleichverteilung:

$$ p_t(i) \propto \exp\Big(\eta \sum_{s<t} u_s(i)\Big), \qquad \eta = \frac{1}{\Delta}\sqrt{\frac{8 \ln N}{T}}, \qquad \varepsilon(\bar x_T, \bar y_T) \le \Delta \Big(\sqrt{\tfrac{\ln m}{2T}} + \sqrt{\tfrac{\ln n}{2T}}\Big) $$

**Fictitious Play:** beste Antwort auf die Häufigkeiten $\hat y_t$ der bisherigen Aktionen des Gegners (Gleichstand: kleinster Index):

$$ a_{t+1} \in \arg\max_i\, (A\,\hat y_t)_i, \qquad \hat y_t = \frac{1}{t}\sum_{s \le t} e_{b_s} $$

**Allgemeine Zwei-Personen-Spiele:** ein Paar $(x, y)$ ist ein Nash-Gleichgewicht, wenn jede Aktion mit positiver Wahrscheinlichkeit eine beste Antwort ist. Die Aufzählung prüft alle Paare gleich großer Stützen (bei $n \times n$ sind das $\sum_{k \ge 1}\binom{n}{k}^2 = \binom{2n}{n} - 1$ Paare):

$$ x_i > 0 \;\Rightarrow\; (A\,y)_i = \max_k (A\,y)_k, \qquad y_j > 0 \;\Rightarrow\; (x^\top B)_j = \max_k (x^\top B)_k $$

**Schätzen aus Beobachtungen:** der Beobachter sieht $T$ Züge der festen Mischung $y$ des Gegners, bildet die Häufigkeiten $\hat y$ und spielt $\arg\max_i (A\,\hat y)_i$; sein Gewinn gegenüber dem Wert ist $(A\,y)_i - v$.

**Quellen.** Die Formeln sind durch die unabhängigen Rechenwege der Tests (exakte Brüche, Zertifikat, Aufzählung aller Stützenpaare, Schleifenfassungen der Lernverfahren) und die Messreihen gestützt. Literaturangaben (von Neumann 1928, Minimax-Satz; Dantzig, LP-Dualität; Robinson 1951, Fictitious Play im Nullsummenspiel; Shapley 1964, Gegenbeispiel; Hart und Mas-Colell 2000, Regret Matching; Tammelin, RM+; Lemke und Howson 1964; Savani und von Stengel 2006; Chen und Deng; Daskalakis, Goldberg und Papadimitriou) wurden für diese Demo nicht im Volltext gelesen und stammen aus dem Gedächtnis.
"""
