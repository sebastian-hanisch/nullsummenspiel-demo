# Nullsummenspiele: Zufall mit Absicht – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-nullsummenspiel-demo.streamlit.app/)** -->

Konzepte-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Teil der Linie zu Verhandlung, Auktion, Vertrag und Kontrolle. Ein **Verlader** schickt eine Sendung über eine von n Routen, ein **Kontrolleur** prüft eine Route:
Wird die Route der Sendung geprüft, ist sie verloren, sonst kommt ihr Wert an (Anteil der Ladung). Wo es keinen Sattelpunkt gibt, wird eine feste Regel von einem mitdenkenden Gegner ausgenutzt. Die Demo rechnet den Wert des Spiels und die optimalen gemischten Strategien
(lineares Programm, Primal und Dual getrennt gelöst, Zertifikat aus der Matrix), eine geschlossene Form für Verstecken und Suchen samt Streichen von Routen, zeigt, welche Faustregeln ausnutzbar sind, lässt fünf Lernverfahren und drei Zähler gegen Muster
spielen und bietet einen kleinen Reiter zu allgemeinen Zwei-Personen-Spielen (Stützenaufzählung, Lemke–Howson, Shapley-Beispiel). Abgrenzung im Portfolio: `nash-demo` rechnet das 2×2-Mini-Spiel mit der Indifferenzformel und reine Gleichgewichte im Torwahl-Spiel,
`noregret-demo` laut seiner README Hedge und EXP3 im Torwahl-Überlastungsspiel, `minimax-demo` eine Baumsuche mit offenen Zügen, `stackelberg-demo` reine Strategien. Hier geht es um **gemischte Strategien in einer Auszahlungsmatrix**: Wert, Eindeutigkeit, Lernen darin.
**Eigennutz und ein mitdenkender Gegner sind Modellannahmen, keine Aussagen über Betriebe oder echte Kontrollen.**

## Warum dieses Problem

Eine Kontrolle, die jeden Tag dieselbe Route prüft, ist berechenbar; eine, die nur die wertvollste Route prüft, ebenso. Wie viel das kostet und wie man gegensteuert, ist das Thema: Mischen mit den richtigen Gewichten macht die Strategie unausnutzbar, und gegen einen Gegner,
der im Gleichgewicht spielt, bringt das Beobachten seiner Züge keinen Gewinn (Werte (0,9; 0,8; 0,7; 0,6), 1 bis 1 000 beobachtete Züge: Gewinn 0). Befund (Verstecken und Suchen, vier Routen mit dem Wert 1, Vorgabe der App): Der Wert des Spiels ist 3/4, die beste reine Route garantiert 0 (der Kontrolleur prüft sie), das Gleichgewicht wählt und prüft jede Route mit
Wahrscheinlichkeit 1/4. Eine **Rotation** über die vier Routen liefert gegen einen Kontrolleur, der zählt, welche Route auf welche folgt, im Mittel nur **0,003** je Sendung (1 000 Runden, Mittel über 4 Zufalls-Seeds), gegen einen, der die bisher häufigste Route prüft,
**0,750** bei Gleichstand „kleinster Index“ und **0,947** bei zufälligem Gleichstand: Ob ein Zähler ein Muster ausnutzt, hängt am Zählertyp und an der Gleichstandsregel.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`verhandeln-planung/STUECK-A-nullsummenspiel.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte; alle Zahlen dieser README sind **neu gemessen** (andere Zufalls-Seeds und Generator SplitMix64 statt NumPy). Wo die Vormessung dasselbe maß, stimmen die Befunde überein, mit diesen Ausnahmen und Ergänzungen:

- **Der Hook „leicht gegen schwer“ trägt nicht.** In Zufalls-Bimatrixspielen sind die Lemke–Howson-Pfade kurz (Mittel 8,3 Pivots bei n = 8; bei n = 40 Mittel 97,2, Median 40, Maximum 777), und die Zahl der Gleichgewichte wächst langsam (Mittel 1,26 bei n = 2, 6,05 bei n = 8). Gemessen ist stattdessen: Nullsumme hat genau ein Gleichgewicht, allgemeine Spiele mehrere,
  und Lernverfahren finden sie nicht von selbst (Shapley-Beispiel). „PPAD-vollständig“ und „im ungünstigsten Fall exponentiell“ stehen nur als Literaturzeile (nicht nachgemessen, nicht im Volltext gelesen); die Familie schlechter Spiele ist nicht gebaut, die Brücke zur Linie Komplexitätsklassen entfällt.
- **`noregret-demo`** ist laut README Hedge und EXP3 im Torwahl-Spiel, nicht Regret Matching in Normalform (so stand es im Auftrag); Lernen in einer Auszahlungsmatrix mit Durchschnittsstrategie gegen den LP-Wert gibt es in den geprüften READMEs sonst nicht.
- **Routenwerte (1; 0,75; 0,5; 0,25)** statt (4; 3; 2; 1): die Regler gehen nur bis 1. Wert des Spiels 6/13 = 0,4615 statt 24/13; es gilt dasselbe Streichen (die vierte Route).
- **Experiment 2** legt „Muster gegen Zufall“ und „Schätzen aus Beobachtungen“ der Vormessung zusammen. **Verdeckte Information** ist nicht in A gebaut (Absatz unten).
- **Multiplicative Weights** bekommt in Experiment 3 für jeden Horizont T einen eigenen Lauf (η hängt vom Horizont ab); die Vormessung rechnete ebenfalls mit einer Lernrate je Horizont; die Zahlen bei kleinem T sind eigene Läufe mit diesem Horizont und keine Zwischenstände eines Laufs bis T = 10 000.
- **Der Abstand im Shapley-Beispiel** ist hier der größere der beiden halben L1-Abstände der Häufigkeiten zur Gleichverteilung; die Zahlen der Vormessung (andere Abstandsdefinition) gelten deshalb nicht.
- **`lokale-optima-demo`** verweist für Verfahren zu PPAD auf `nash-demo`; dort gibt es weder Stützenaufzählung noch Lemke–Howson. Dieser Verweis ist **nicht geändert**, nur gemeldet.
- **Gleichwertige Mutanten:** siehe Verifikation.

## Modell

Die Matrix **A** (m × n) hat eine Zeile je Route des Verladers und eine Spalte je geprüfte Route; A[i, j] ist die Auszahlung an den Verlader. Beide Seiten ziehen gleichzeitig; der Verlader maximiert, der Kontrolleur minimiert. Eine **gemischte Strategie** ist ein Wahrscheinlichkeitsvektor.
Der Verlader mit Strategie x **garantiert** min_j (xᵀA)_j, der Kontrolleur mit Strategie y **lässt höchstens** max_i (A y)_i **zu**; der **Minimax-Satz** sagt, dass das Beste aus beiden gleich ist (Wert v), und beides sind duale lineare Programme (`nsm_lp.py`, HiGHS).

- **Verstecken und Suchen** (`nsm_hideseek.py`): A[i, j] = v_i für i ≠ j, 0 auf der Diagonalen (Route i hat den Wert v_i, geprüft ist sie verloren). Mit c = 1/Σ(1/v_i) gilt Wert (n − 1)·c, x_i = c/v_i, y_i = 1 − Wert/v_i; ein negatives y_i heißt, dass die Route zu billig ist und beide Seiten sie weglassen (**Streichen**, kleinster Wert zuerst).
  Eigene Herleitung, gegen das lineare Programm und die exakte Aufzählung geprüft.
- **Zufallsspiele:** Einträge gleichverteilt in [0, 1) (SplitMix64, wörtliche Kopie `nsm_rng.py` aus `pearson-demo`).
- **Lernen** (`nsm_learning.py`): Regret Matching, **RM+** (aus dem Gedächtnis umgesetzt, Quelle nicht eingesehen), Multiplicative Weights (Hedge, Lernrate aus dem Horizont), Fictitious Play, Regret Matching mit gezogenen Zügen; erwartete Auszahlungen gegen die Mischung des Gegners, außer beim letzten Verfahren.
  Aufgezeichnet wird die **Ausnutzbarkeit** der Durchschnittsstrategie und der letzten Iteration. Der Zufallsstart ist Pflicht: der Start in der Gleichverteilung kann im Gleichgewicht liegen.
- **Beobachter** (`nsm_observers.py`): Muster des Verladers (Rotation, Gewohnheit, kein Wiederholen, Zufall im Gleichgewicht) gegen Zähler des Kontrolleurs (Häufigkeit aller Sendungen, Fenster der letzten 20, Folge: welche Route auf welche folgt, feste Gleichgewichtsstrategie); der Zähler prüft die Route mit dem größten (Zählerstand · Wert), Gleichstand: kleinster Index oder zufällig.
  Dazu das **Schätzen**: ein Beobachter sieht T Züge einer festen Mischung, schätzt sie aus den Häufigkeiten und spielt die reine beste Antwort.
- **Allgemeine Spiele** (`nsm_bimatrix.py`): Stützenaufzählung (alle Paare gleich großer Stützen; für nicht entartete Spiele vollständig), Lemke–Howson als Tableauumsetzung mit Pfad, das Shapley-Beispiel (3 × 3), Fictitious Play und Regret Matching im Bimatrixspiel.

## Methodik

- **Kleine Einheiten mit Mini-Instanzen von Hand** (DEMO-PLAYBOOK §1b, `tests/conftest.py`): Sicherheitsniveau, Sattelpunkte, garantiert/zugelassen, Ausnutzbarkeit, LP, Bereiche der optimalen Menge, geschlossene Form mit Streichen, jedes Lernverfahren (Handschritte), Zähler und Gleichstandsregel, Lemke–Howson (Pfade von Hand im Kampf der Geschlechter).
- **Orakel** (`tests/test_oracle_*.py`, anderer Rechenweg): das lineare Programm gegen das **Zertifikat** (aus der Matrix, ohne den Löser), gegen die **exakte Stützenaufzählung mit Brüchen** auf ganzzahligen Spielen (Wert und Bereich jeder Koordinate der optimalen Menge), Primal gegen Dual getrennt und gegen die Sattelpunkt-Formel n!²/(2n−1)!;
  die Lernverfahren gegen **Schleifenfassungen ohne NumPy** (Verlauf der Ausnutzbarkeit an jeder Stützstelle) und die Regret-Schranke von Hedge; `enumerate_nash` gegen die Aufzählung in Brüchen, die Parität (in den Stichproben nie gerade), die Best-Response-Prüfung und die Nullsumme (genau ein Gleichgewicht, gleich dem LP-Wert); Lemke–Howson gegen die Aufzählung und `is_nash`; die Beobachter (Zähler, Lieferung, Schätzen mit demselben Zufallsstrom) gegen Fassungen mit reinen Listen.
- **Eigenschaftstests:** Umbenennen der Aktionen, positiv-affine Umrechnung der Auszahlungen (dieselben Strategien bleiben optimal), Rollentausch durch Transponieren und Negieren, zusätzliche Zeilen und Spalten, dominierte Zeile, Monotonie und Skalierung der Routenwerte.
- **Experimente** (Hauptansicht und Reiter der App, jeweils mit vorgerechneter Messreihe): (1) Mischen hilft, (2) Beobachtet werden, (3) Durchschnitt gegen letzte Iteration, (4) Eindeutigkeit, Aufzählung und Lemke–Howson im Reiter Allgemeine Spiele.
- **Vorgerechnete Messreihen** (`tools/make_grid.py`, `data/nullsummen_grid.json`): alles deterministisch (SplitMix64, feste Zufalls-Seeds 11 bis 14). `tests/test_grid.py` rechnet eine Probe nach.
- **Gegenprobe der Zahlen mit anderen Zufalls-Seeds** (21 bis 24, Scratch-Lauf, nicht im Repo): E1 Sattelpunktanteil 67,4 % (2 × 2), 4,1 % (5 × 5); E2 Rotation gegen Häufigkeitszähler mit zufälligem Gleichstand 0,936, gegen den Folgezähler 0,003; E3 Steigung von RM+ −0,75 statt −0,64 und letzte Iteration von RM+ 0,063 und MW 0,185 statt 0,084 und 0,251 bei T = 10 000;
  E4 Anteil mit genau einem Gleichgewicht 6 % (n = 8) statt 9 %, mittlere Zahl 6,40 statt 6,05 (n = 8). Zahlen mit wenigen Spielen (n = 7 und 8 im Reiter, 20 Spiele in Experiment 3) tragen deshalb nur ein bis zwei Stellen. Diese Gegenprobe-Zahlen stehen nicht als Test im Repo; `tests/test_claims.py` prüft die Aussagen mit anderen Zufalls-Seeds in kleinerem Umfang (Durchschnitt, letzte Iteration, Rotation, Sattelpunktanteil).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Tabellen Zeile für Zeile in der Rundung der App). Jede Zahl nennt ihre Bedingung (Spiel, Größe, Verfahren, Start, Gleichstandsregel, Zahl der Spiele und Läufe).

| Hypothese | Ergebnis | Befund (Bedingung) |
|---|---|---|
| **H1** Minimax-Satz: max-min = min-max (Primal = Dual), Zertifikat und exakte Aufzählung stimmen; die optimale Strategie ist nicht immer eindeutig | Ja | Zufallsspiele 2×2 bis 10×10 (je 4 000, Einträge gleichverteilt in [0, 1)): größte Lücke zwischen dem Garantierten von x und dem Zugelassenen von y 3,2·10⁻¹⁴; 40 ganzzahlige Spiele (2×2 bis 4×5, Einträge 0 bis 9) in der Suite: LP-Wert gleich dem exakten Bruch (bis 10⁻⁹); bei ganzzahligen Einträgen 0 bis 9 (je 400 Spiele, 2×2 bis 5×5) ist die optimale Strategie des Verladers in 10 % bis 15 % der Spiele nicht eindeutig |
| **H2** Mischen hilft, der Sattelpunkt wird mit n selten | Ja | Anteil mit Sattelpunkt 65,5 % (2×2; Formel n!²/(2n−1)! gibt 66,7 %), 4,0 % (5×5; Formel 4,0 %), 0,0 % (10×10; Formel 0,01 %); Gewinn durch Mischen im Mittel 0,035 (2×2) bis 0,253 (10×10); gilt für Einträge gleichverteilt in [0, 1) |
| **H3** Verstecken und Suchen hat eine geschlossene Form mit Streichen von Routen; die beste reine Route garantiert 0 | Ja | größte Abweichung von Wert, x und y gegen das LP 7,5·10⁻¹⁶ in 10 Zellen (Wertebereiche [0,3; 1] und [0,05; 1], n = 2, 3, 4, 6, 8; je 400 Spiele). Ohne Streichen gilt die einfache Form bei Werten in [0,3; 1] in 85,5 % (n = 3), 42,5 % (n = 4) und 0,0 % (n = 8) der Spiele: ab n = 4 ist Streichen der häufigere Fall; bei Werten in [0,05; 1] ohne Streichen 42,2 % (n = 3) und 13,2 % (n = 4) |
| **H4** Die Faustregel „nach Wert verteilen“ ist ausnutzbar | Ja, mit einer Ausnahme | Werte (1; 0,1), Wert 0,0909: „proportional zum Wert“ garantiert dem Verlader nur 0,0091. Werte (1; 0,75; 0,5; 0,25), Wert 0,4615: Verlader proportional 0,3500, Kontrolleur proportional lässt 0,6000 zu. **Ausnahme:** bei zwei Routen ist „proportional zum Wert“ beim Kontrolleur das Gleichgewicht (Werte (1; 0,1): zugelassen 0,0909) |
| **H5** Ein Muster ist ausnutzbar, wenn der Gegner das passende Modell hat | Ja, mit Gegenbefund | vier Routen mit Wert 1 (Wert des Spiels 0,750), 1 000 Runden, Mittel über 4 Zufalls-Seeds: Rotation gegen Folgezähler 0,003 (er fängt ab Runde 3 bei zwei, ab Runde 9 bei acht Routen jede Sendung), gegen Häufigkeitszähler 0,750 (Gleichstand kleinster Index) bzw. 0,947 (zufällig). Gegen Zufall im Gleichgewicht liegen alle Kontrolleure (Zähler mit beiden Gleichstandsregeln, feste y*) zwischen 0,744 und 0,766 (Rauschen der 4 000 Sendungen je Zelle) (Werte (0,9; 0,8; 0,7; 0,6), Wert 0,550: zwischen 0,538 und 0,551) |
| **H6** Der Durchschnitt konvergiert, die letzte Iteration nicht; Rate etwa T^(−1/2) | Ja für den Durchschnitt, die letzte Iteration nicht; Rate nur ungefähr | Zufallsspiele 5 × 5, 20 Spiele je Verfahren: Durchschnitt bei T = 10 000 zwischen 0,0011 und 0,0092, letzte Iteration zwischen 0,084 und 0,456. Steigung (Log-Log, T = 100 bis 10 000): RM -0,52, MW -0,51, Fictitious Play -0,53, RM gezogen -0,52, RM+ -0,64 (mit anderen Zufalls-Seeds −0,75) |
| **H7** Gegen das Gleichgewicht lernt der Beobachter nichts; sonst brauchen 90 % des Gewinns T von 5 bis 1 000 | Ja | Werte (0,9; 0,8; 0,7; 0,6): Gewinn gegen y* und x* bei jedem T von 1 bis 1 000 gleich 0. Verlader gegen gleichverteilt prüfenden Kontrolleur: 85 % des vollen Gewinns 0,1252 nach 1 Zug, 62 % nach 5, 100 % nach 1 000 (nicht monoton). Kontrolleur gegen gleichverteilt wählenden Verlader: 89 % des vollen Gewinns 0,0248 erst nach 1 000 Zügen; gegen einen Verlader, der Route 1 mit 0,7 wählt: 94 % des Gewinns 0,3398 schon nach 5 |
| **H8** Reiter: Nullsumme hat genau ein Gleichgewicht, Bimatrixspiele mehrere, Lernen findet sie nicht von selbst (gemessen am Shapley-Beispiel mit Fictitious Play und Regret Matching) | Ja; der Teil „schwer“ trägt nicht | Nullsumme B = 1 − A: 960 Spiele (n = 2 bis 5), jedes mit genau einem Gleichgewicht. Zufalls-Bimatrixspiele (4440 Spiele, n = 2 bis 8): mittlere Zahl 1,26 bis 6,05, größte 17, in keinem Spiel eine gerade Zahl; Lemke–Howson findet von jedem Label ein Gleichgewicht, alle Labels zusammen 96 % (n = 2) bis 56 % (n = 8) der Gleichgewichte. Shapley-Beispiel, Fictitious Play, 200 000 Runden: Abstand der Häufigkeiten zur Gleichverteilung ab Runde 1 000 zwischen 0,173 und 0,267 (13 Wechsel der Aktion des Zeilenspielers), im Nullsummen-Gegenstück am Ende 0,0015; Regret Matching (Start (0,5; 0,3; 0,2) gegen (0,2; 0,3; 0,5)), 10 000 Runden: Nash-Lücke des Durchschnitts 0,206, der letzten Iteration 1,000. Pfade kurz: Teilbefund „schwer“ nicht gesehen |
| **B1** (Befund, kein Plan-Punkt) Start im Gleichgewicht: der tote Zweig | Ja | Stein-Schere-Papier und vier gleiche Routen, Start in der Gleichverteilung: Regret Matching, RM+ und Multiplicative Weights haben an jeder Stützstelle bis T = 10 000 Ausnutzbarkeit 0 (4 Zufalls-Seeds); mit Zufallsstart ist die letzte Iteration im Stein-Schere-Papier bei T = 10 000 im Mittel um 0,799 (MW), 0,969 (RM+) und 1,673 (RM) ausnutzbar |

**Experiment 1: Mischen hilft** (Zufallsspiele, Einträge gleichverteilt in [0, 1); 4 Zufalls-Seeds; Mittel über alle Spiele, in Klammern die Standardabweichung der Seed-Mittel; Gewinn = Wert − Sicherheitsniveau):

| Größe | Spiele | Sattelpunkt (Std) | Formel n!²/(2n−1)! | Wert | Sicherheitsniveau | Gewinn durch Mischen | Gewinn bei Spielen ohne Sattelpunkt | Stütze von x |
|---|---|---|---|---|---|---|---|---|
| 2 × 2 | 4000 | 65,5 % (1,1 %) | 66,67 % | 0,511 | 0,477 | 0,035 | 0,100 | 1,35 |
| 3 × 3 | 4000 | 31,0 % (0,8 %) | 30,00 % | 0,496 | 0,420 | 0,076 | 0,111 | 1,79 |
| 4 × 4 | 4000 | 11,1 % (0,8 %) | 11,43 % | 0,499 | 0,380 | 0,119 | 0,134 | 2,29 |
| 5 × 5 | 4000 | 4,0 % (0,5 %) | 3,97 % | 0,502 | 0,354 | 0,148 | 0,154 | 2,75 |
| 7 × 7 | 4000 | 0,3 % (0,1 %) | 0,41 % | 0,498 | 0,299 | 0,200 | 0,200 | 3,75 |
| 10 × 10 | 4000 | 0,0 % (0,0 %) | 0,01 % | 0,501 | 0,248 | 0,253 | 0,253 | 5,24 |
| 3 × 6 | 4000 | 10,0 % (0,4 %) | – | 0,380 | 0,250 | 0,130 | 0,144 | 2,24 |
| 6 × 3 | 4000 | 10,8 % (1,0 %) | – | 0,620 | 0,526 | 0,094 | 0,106 | 2,24 |

Ganzzahlige Einträge (0 bis 9), Anteil der Spiele, in denen die optimale Strategie nicht eindeutig ist (4 Zufalls-Seeds, 100 Spiele je Seed und Größe):

| Größe | Spiele | Strategie des Verladers nicht eindeutig | Strategie des Kontrolleurs nicht eindeutig |
|---|---|---|---|
| 2 × 2 | 400 | 15,0 % | 9,2 % |
| 3 × 3 | 400 | 13,5 % | 14,3 % |
| 4 × 4 | 400 | 14,8 % | 13,8 % |
| 5 × 5 | 400 | 10,2 % | 10,5 % |

Verstecken und Suchen, geschlossene Form gegen das lineare Programm (Routenwerte gleichverteilt in [untere Grenze, 1], 4 Zufalls-Seeds, 400 Spiele je Zelle):

| untere Wertgrenze | Routen n | größte Abweichung (Wert, x, y) | Spiele ohne Streichen |
|---|---|---|---|
| 0,30 | 2 | 2,2·10⁻¹⁶ | 100,0 % |
| 0,30 | 3 | 2,8·10⁻¹⁶ | 85,5 % |
| 0,30 | 4 | 3,3·10⁻¹⁶ | 42,5 % |
| 0,30 | 6 | 2,8·10⁻¹⁶ | 4,5 % |
| 0,30 | 8 | 3,7·10⁻¹⁶ | 0,0 % |
| 0,05 | 2 | 2,2·10⁻¹⁶ | 100,0 % |
| 0,05 | 3 | 3,3·10⁻¹⁶ | 42,2 % |
| 0,05 | 4 | 2,8·10⁻¹⁶ | 13,2 % |
| 0,05 | 6 | 7,1·10⁻¹⁶ | 0,5 % |
| 0,05 | 8 | 7,5·10⁻¹⁶ | 0,0 % |

Faustregeln (garantierter Wert des Verladers, V; zugelassener Wert des Kontrolleurs, K; das Gleichgewicht sichert bzw. lässt den Wert des Spiels):

| Werte | Wert des Spiels | V: gleichverteilt | V: proportional | V: umgekehrt proportional (alle Routen) | V: nur wertvollste Route | K: gleichverteilt | K: proportional | K: nur wertvollste Route |
|---|---|---|---|---|---|---|---|---|
| 1,00; 1,00; 1,00; 1,00 | 0,7500 | 0,7500 | 0,7500 | 0,7500 | 0,0000 | 0,7500 | 0,7500 | 1,0000 |
| 1,00; 0,10 | 0,0909 | 0,0500 | 0,0091 | 0,0909 | 0,0000 | 0,5000 | 0,0909 | 0,1000 |
| 1,00; 0,75; 0,50; 0,25 | 0,4615 | 0,3750 | 0,3500 | 0,3600 | 0,0000 | 0,7500 | 0,6000 | 0,7500 |
| 0,90; 0,80; 0,70; 0,60 | 0,5498 | 0,5250 | 0,4967 | 0,5498 | 0,0000 | 0,6750 | 0,6300 | 0,8000 |

**Experiment 2: Beobachtet werden.** Mittlere Lieferung je Sendung (1 000 Runden, Mittel über 4 Zufalls-Seeds, Standardabweichung über die Seeds in Klammern), vier Routen mit Wert 1 (Wert des Spiels 0,750); „niedrig“ heißt Gleichstand kleinster Index, „zufall“ zufällige Wahl:

| Muster | Häufigkeit, niedrig | Häufigkeit, zufall | Fenster 20, niedrig | Fenster 20, zufall | Folge, niedrig | Folge, zufall | feste y* |
|---|---|---|---|---|---|---|---|
| Rotation | 0,750 (0,000) | 0,947 (0,005) | 0,750 (0,000) | 0,771 (0,013) | 0,003 (0,000) | 0,004 (0,000) | 0,755 (0,016) |
| Gewohnheit | 0,736 (0,012) | 0,735 (0,012) | 0,688 (0,015) | 0,690 (0,013) | 0,377 (0,006) | 0,375 (0,006) | 0,745 (0,007) |
| kein Wiederholen | 0,771 (0,015) | 0,773 (0,012) | 0,791 (0,006) | 0,786 (0,004) | 0,680 (0,008) | 0,680 (0,008) | 0,756 (0,007) |
| Zufall im Gleichgewicht | 0,754 (0,010) | 0,749 (0,009) | 0,766 (0,017) | 0,760 (0,011) | 0,756 (0,015) | 0,755 (0,015) | 0,744 (0,016) |

Werte (0,9; 0,8; 0,7; 0,6), Wert des Spiels 0,550:

| Muster | Häufigkeit, niedrig | Häufigkeit, zufall | Fenster 20, niedrig | Fenster 20, zufall | Folge, niedrig | Folge, zufall | feste y* |
|---|---|---|---|---|---|---|---|
| Rotation | 0,525 (0,000) | 0,526 (0,000) | 0,525 (0,000) | 0,526 (0,000) | 0,002 (0,000) | 0,003 (0,000) | 0,551 (0,011) |
| Gewohnheit | 0,526 (0,014) | 0,526 (0,014) | 0,505 (0,012) | 0,507 (0,011) | 0,282 (0,005) | 0,283 (0,004) | 0,549 (0,008) |
| kein Wiederholen | 0,531 (0,011) | 0,530 (0,010) | 0,585 (0,007) | 0,584 (0,007) | 0,468 (0,008) | 0,468 (0,007) | 0,547 (0,014) |
| Zufall im Gleichgewicht | 0,551 (0,006) | 0,551 (0,006) | 0,550 (0,008) | 0,550 (0,007) | 0,545 (0,003) | 0,545 (0,006) | 0,538 (0,004) |

Erste Runde, ab der der Folgezähler (Gleichstand kleinster Index) jede Sendung einer Rotation über n gleiche Routen abfängt:

| Routen n | ab Runde |
|---|---|
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |
| 5 | 6 |
| 6 | 7 |
| 7 | 8 |
| 8 | 9 |

Schätzen aus Beobachtungen (Werte (0,9; 0,8; 0,7; 0,6); Anteil des vollen Gewinns nach T = 1, 5, 10, 50, 100, 500, 1 000 beobachteten Zügen; 4 Zufalls-Seeds mit je 300 Beobachtern; „–“: voller Gewinn 0):

| Gegner | voller Gewinn | T = 1 | 5 | 10 | 50 | 100 | 500 | 1 000 |
|---|---|---|---|---|---|---|---|---|
| Kontrolleur prüft gleichverteilt | 0,1252 | 85 % | 62 % | 72 % | 87 % | 93 % | 100 % | 100 % |
| Verlader wählt gleichverteilt | 0,0248 | -51 % | -17 % | -10 % | 30 % | 45 % | 80 % | 89 % |
| Verlader wählt meist Route 1 (0,7) | 0,3398 | 51 % | 94 % | 99 % | 100 % | 100 % | 100 % | 100 % |
| Kontrolleur im Gleichgewicht y* | 0,0000 | – | – | – | – | – | – | – |
| Verlader im Gleichgewicht x* | 0,0000 | – | – | – | – | – | – | – |

**Experiment 3: Durchschnitt gegen letzte Iteration.** Zufallsspiele 5 × 5, 20 Spiele, ein Lauf je Spiel mit Zufallsstart, mittlere Ausnutzbarkeit. Multiplicative Weights mit eigenem Lauf und eigener Lernrate je Horizont T; die übrigen Verfahren ein Lauf bis T = 10 000 (T = 10, 100, 1 000, 10 000):

| Verfahren | Durchschnitt, T = 10 | 100 | 1000 | 10000 | Steigung (Log-Log, T = 100 bis 10 000) | letzte Iteration, T = 10 | 100 | 1000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|
| Regret Matching | 0,1308 | 0,0370 | 0,0111 | 0,0034 | -0,52 | 0,367 | 0,294 | 0,333 | 0,423 |
| Regret Matching+ (RM+) | 0,1242 | 0,0208 | 0,0048 | 0,0011 | -0,64 | 0,262 | 0,167 | 0,107 | 0,084 |
| Multiplicative Weights (Hedge) | 0,2704 | 0,0962 | 0,0302 | 0,0092 | -0,51 | 0,272 | 0,268 | 0,258 | 0,251 |
| Fictitious Play | 0,1841 | 0,0417 | 0,0119 | 0,0036 | -0,53 | 0,449 | 0,485 | 0,397 | 0,456 |
| Regret Matching mit gezogenen Zügen | 0,1974 | 0,0484 | 0,0162 | 0,0045 | -0,52 | 0,260 | 0,335 | 0,399 | 0,317 |

Start in der Gleichverteilung gegen Zufallsstart (T = 10 000, Mittel über 4 Zufalls-Seeds; Fictitious Play beginnt mit einer reinen Aktion und Regret Matching mit gezogenen Zügen zieht Züge; beide bewegen sich in diesen Läufen auch bei Start in der Gleichverteilung):

| Spiel | Verfahren | Gleichverteilung: Durchschnitt | Gleichverteilung: letzte Iteration | Zufallsstart: Durchschnitt | Zufallsstart: letzte Iteration |
|---|---|---|---|---|---|
| Stein-Schere-Papier | Regret Matching | 0,0000 | 0,000 | 0,0198 | 1,673 |
| Stein-Schere-Papier | Regret Matching+ (RM+) | 0,0000 | 0,000 | 0,0113 | 0,969 |
| Stein-Schere-Papier | Multiplicative Weights (Hedge) | 0,0000 | 0,000 | 0,0236 | 0,799 |
| Stein-Schere-Papier | Fictitious Play | 0,0134 | 2,000 | 0,0151 | 2,000 |
| Stein-Schere-Papier | Regret Matching mit gezogenen Zügen | 0,0309 | 1,827 | 0,0304 | 1,668 |
| vier gleiche Routen | Regret Matching | 0,0000 | 0,000 | 0,0076 | 0,759 |
| vier gleiche Routen | Regret Matching+ (RM+) | 0,0000 | 0,000 | 0,0029 | 0,273 |
| vier gleiche Routen | Multiplicative Weights (Hedge) | 0,0000 | 0,000 | 0,0095 | 0,541 |
| vier gleiche Routen | Fictitious Play | 0,0130 | 1,000 | 0,0106 | 1,000 |
| vier gleiche Routen | Regret Matching mit gezogenen Zügen | 0,0142 | 0,662 | 0,0119 | 0,564 |

**Experiment 4: Eindeutigkeit, Aufzählung und Lemke–Howson** (Zufalls-Bimatrixspiele n × n, Einträge von A und B unabhängig gleichverteilt in [0, 1), 4 Zufalls-Seeds; Spiele für n = 7 und 8: 160 und 80; Stützenpaare = C(2n, n) − 1, eine Rechnung, keine Messung):

| n | Spiele | mittlere Zahl der Gleichgewichte | größte Zahl | genau eines | gerade Zahl | mit reinem Gleichgewicht | Pivots je Label (Mittel) | größte Pfadlänge | Fehler | gefundener Anteil der Gleichgewichte | Stützenpaare |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1200 | 1,26 | 3 | 87 % | 0 | 87 % | 2,9 | 5 | 0 | 96 % | 5 |
| 3 | 1200 | 1,57 | 7 | 73 % | 0 | 80 % | 3,7 | 11 | 0 | 91 % | 19 |
| 4 | 800 | 2,12 | 9 | 55 % | 0 | 75 % | 4,4 | 17 | 0 | 83 % | 69 |
| 5 | 600 | 2,86 | 9 | 35 % | 0 | 73 % | 5,3 | 25 | 0 | 75 % | 251 |
| 6 | 400 | 3,58 | 15 | 27 % | 0 | 70 % | 6,2 | 39 | 0 | 69 % | 923 |
| 7 | 160 | 4,88 | 17 | 16 % | 0 | 69 % | 7,2 | 37 | 0 | 62 % | 3 431 |
| 8 | 80 | 6,05 | 15 | 9 % | 0 | 65 % | 8,3 | 79 | 0 | 56 % | 12 869 |

Kontrolle Nullsumme und Lemke–Howson in großen Spielen (ohne Aufzählung; fünf Startlabels je Spiel):

| Fall | Spiele bzw. Pfade | Ergebnis | Abweichung bzw. Fehler |
|---|---|---|---|
| Nullsumme B = 1 − A, n = 2 | 240 | 100 % | 2,2·10⁻¹⁶ |
| Nullsumme B = 1 − A, n = 3 | 240 | 100 % | 5,6·10⁻¹⁶ |
| Nullsumme B = 1 − A, n = 4 | 240 | 100 % | 7,8·10⁻¹⁶ |
| Nullsumme B = 1 − A, n = 5 | 240 | 100 % | 7,8·10⁻¹⁶ |
| Lemke–Howson allein, n = 20 | 400 Pfade | Mittel 23,6, Median 13, Maximum 301 | 0 Fehler |
| Lemke–Howson allein, n = 40 | 240 Pfade | Mittel 97,2, Median 40, Maximum 777 | 0 Fehler |

Für n = 10 sind es 184 755 Stützenpaare, für n = 8 12 869; die Aufzählung im Reiter geht bis n = 8.

## Ehrliche Grenzen

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

Negative und gemischte Befunde stehen oben in der Tabelle: der Hook „leicht gegen schwer“ trägt nicht (H8); die Rate T^(−1/2) gilt nur ungefähr und für RM+ nicht; die Zahlen mit wenigen Spielen (Experiment 3, n = 7 und 8 im Reiter) schwanken mit dem Zufalls-Seed; die Lernergebnisse der letzten Iteration hängen an der Spielmenge.

**Einordnung: verdeckte Information.** Spiele mit Zügen nacheinander und ungleich verteilter Information (Spielbaum, Informationsmengen; Verfahren wie Sequenzform-LP und Counterfactual Regret Minimization) brauchen nach Literaturwissen (hier nicht gemessen) andere Darstellungen und andere Verfahren als eine Auszahlungsmatrix. Sie sind in dieser Demo **nicht gebaut** und haben hier keinen Rechenkern; die Annahme „volle Kenntnis der Matrix“ nennt die Grenze.

## Verifikation

- **Kern:** jede Einheit gegen Handrechnung auf den Mini-Instanzen von `tests/conftest.py`; Randfälle (zwei Routen, Streichen genau an der Grenze, 1 × n und n × 1, Gleichstände, Spiel mit lauter Nullen), Reglermaxima (Zufallsspiel 10 × 10, 10 000 Lernrunden, 2 000 Sendungen, acht Routen, Aufzählung bei n = 8).
- **Orakel und Eigenschaften:** siehe Methodik (`tests/test_oracle_zero_sum.py`, `test_oracle_learning.py`, `test_oracle_bimatrix.py`, `test_oracle_observers.py`, `test_hideseek.py`, `test_eigenschaften_nullsumme.py`); exakte Orakel in `tests/exact_games.py`; die Kopie `nsm_rng.py` hat einen Kopie-Test (`tests/test_copies.py`).
- **Aussagen:** `tests/test_claims.py` hält die Zahlen dieser README fest (Tabellen Zeile für Zeile, Zahlen im Fließtext als Zeichenketten in der Rundung der App, die Bänder); `tests/test_verdicts.py` prüft die Zahlen im Text der Verdikte an von Hand gesetzten Werten; `tests/test_grid.py` prüft Aufbau, Zählungen und eine Probe der vorgerechneten Messreihen.
- **Oberfläche:** AppTest für jedes Preset, den Permalink, jeden Knopf (auch den ▶️-Lauf im Shapley-Beispiel), jeden Regler am Minimum und Maximum, jede Kombination aus Muster, Zähler und Gleichstand, jedes Lernverfahren mit beiden Starts auf jedem Spiel; Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
- **Tests:** 286 Tests, die Suite läuft lokal im Test-venv in rund 3 Minuten.
- **Mutationsprüfung** des numerischen Kerns (`nsm_game.py`, `nsm_lp.py`, `nsm_hideseek.py`, `nsm_learning.py`, `nsm_observers.py`, `nsm_bimatrix.py`; ohne die wörtliche Kopie `nsm_rng.py`): 231 Stellen, davon 214 gefangen, 13 gleichwertig festgehalten und 4 durch wortgleiche Zeilen in Nachbar-Demos gesichert, 0 offen (Zustand in `tests/mutation_state.json`; nicht mutiert: Auswertung und Verdikte in `nsm_evaluation.py`, Darstellung, App). Gleichwertig sind: der Vergleich `> SUPPORT_EPS` in der Stützenaufzählung (zwei Stellen, Rauschgrenze 10⁻¹⁰) und der Zweig `len(regular) == 0` (eine leere Auswahl rechnet auch ohne den Zweig nichts); in `checkpoints` drei Stellen (die letzte Runde steht ohnehin in der Menge, Werte über der Rundenzahl filtert die letzte Zeile, eine 0 kommt nicht vor); das Vorzeichen in `cx − cx.max()` und `cy − cy.max()` bei Hedge (ein konstanter Faktor kürzt sich in der Normierung); das Vorzeichen der Toleranz 10⁻⁹ in `optimal_ranges` (sie liegt unter der Zulässigkeitstoleranz des Lösers, nach der SciPy-Dokumentation 10⁻⁷, hier nicht nachgeprüft); drei Stellen des Rückfalls in Runde 1 im Folgezähler (dort ist die Folgetabelle leer) und `<` gegen `<=` bei der Wahrscheinlichkeit ½ der Gewohnheit (die Zufallszahl ist nie genau ½). Lücken, die der Lauf fand, schlossen Tests in `test_oracle_bimatrix.py`, `test_learning.py` und `test_oracle_observers.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht mit Experiment 1, drei Reiter mit den Experimenten 2 bis 4, Annahmen, Mathematik |
| `data/nullsummen_grid.json` | vorgerechnete Messreihen (Experimente 1 bis 4) |
| `nsm_bimatrix.py` | Allgemeine Spiele: Stützenaufzählung, Lemke–Howson, Shapley-Beispiel, Fictitious Play und Regret Matching im Bimatrixspiel |
| `nsm_constants.py` | Regler-Grenzen, Auswahlmöglichkeiten, Raster der Messreihen, Farben, Presets, Annahmen- und Mathematiktext |
| `nsm_evaluation.py` | Kennzahlen je Ansicht, Erzeuger der Messreihen, Zusammenfassungen, Verdikte (reine Funktionen) |
| `nsm_game.py` | Sicherheitsniveau, Sattelpunkte, garantiert/zugelassen, Ausnutzbarkeit, beste Antworten |
| `nsm_hideseek.py` | Verstecken und Suchen: Matrix, geschlossene Form mit Streichen, Faustregeln |
| `nsm_layout.py` | Fester Kopf (Stylesheet) |
| `nsm_learning.py` | Lernverfahren: Regret Matching, RM+, Multiplicative Weights, Fictitious Play, Regret Matching mit gezogenen Zügen |
| `nsm_lp.py` | Lineares Programm (HiGHS), Zertifikat, Bereiche der optimalen Menge |
| `nsm_observers.py` | Muster, Zähler, Gleichstandsregel, Schätzen aus Beobachtungen |
| `nsm_pdf_export.py` | PDF-Export (fpdf2) |
| `nsm_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Buttons |
| `nsm_rng.py` | wörtliche Kopie von `pearson-demo/prs_rng.py` (SplitMix64) |
| `nsm_scenario.py` | Einstellungen (Settings), Routenwerte lesen und schreiben, Zufallsspiele |
| `nsm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, Plotly, fpdf2, SciPy (lineares Programm) |
| `tools/make_grid.py` | rechnet die Messreihen vor |
| `tests/` | Mini-Instanzen von Hand, exakte Orakel, Orakel, Eigenschaften, Aussagen, Messreihen, Randfälle, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopie-Test |

## Bewusst nicht umgesetzt

- **Verdeckte Information und Spielbäume** (Sequenzform, Counterfactual Regret Minimization): nicht gebaut, nur als Einordnung unter „Ehrliche Grenzen“.
- **Mehrere Kontrollstellen und Festlegung vorab** (Kontrollstrategie als Zusage, Ressourcen): nicht gebaut; dort ändert sich das Spiel.
- **Familie schlechter Spiele für Lemke–Howson** und jede Messung zu PPAD oder dem ungünstigsten Fall: nicht gebaut; die Aussagen stehen nur als Literaturzeile.
- **Menschenexperimente und Daten echter Kontrollen:** keine.
- **Streuung eines einzelnen Tages:** die Demo rechnet Erwartungswerte.

## Literatur (nicht im Volltext gelesen, aus dem Gedächtnis, ungeprüft)

Minimax-Satz (von Neumann 1928), lineare Programmierung und Dualität (Dantzig), Konvergenz von Fictitious Play im Nullsummenspiel (Robinson 1951), Gegenbeispiel (Shapley 1964), Regret Matching (Hart und Mas-Colell 2000), RM+ (Tammelin, im Zusammenhang mit CFR+), Lemke und Howson 1964, Savani und von Stengel 2006 (ungünstigster Fall von Lemke–Howson), Komplexität von Nash-Gleichgewichten (Chen und Deng; Daskalakis, Goldberg und Papadimitriou).
Autoren, Jahre und Zuordnungen sind **nicht** anhand der Quellen geprüft; die Formeln stützen sich auf die Messungen und die unabhängigen Rechenwege der Tests. RM+ ist aus dem Gedächtnis umgesetzt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Suite hat 286 Tests. Die Messreihen entstehen mit `python tools/make_grid.py` (rund 40 s auf acht Prozessen); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Rechnung oder Konstanten die Messreihen überholt haben. Lokal läuft Python 3.14, die CI Python 3.12; unter Python 3.12 läuft die Suite erst in der CI (lokal nicht geprüft).

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
