"""PDF-Export (fpdf2, Kernschriften): Spiel, Wert, optimale Strategien und die Verdikte der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import nsm_constants as C
import nsm_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, mathematische Zeichen, hochgestellte Ziffern)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "η": "eta", "δ": "delta", "ε": "eps",
             "₁": "1", "₂": "2", "²": "^2", "⁻": "-", "⁵": "5", "⁶": "6", "⁸": "8", "⁹": "9", "¹": "1", "⁰": "0", "⁴": "4", "⁷": "7", "³": "3", "½": "1/2", "…": "...", "Σ": "Summe"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    names_rows, names_cols = ev.label_names(s, *a.A.shape)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Nullsummenspiele: Zufall mit Absicht"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    detail = f" Routenwerte: {'; '.join(ev.de(v, 2) for v in s.routes)}." if s.game == "hide" else ""
    _para(pdf, 5.5, f"Spiel: {C.GAMES[s.game]}, Matrix {a.A.shape[0]} x {a.A.shape[1]}.{detail}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Ergebnis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Wert des Spiels {ev.de(a.value)}; Sicherheitsniveau der besten reinen Zeile {ev.de(a.security)}; Zertifikat {ev.de(a.sol.lower, 6)} bis {ev.de(a.sol.upper, 6)}.")
    _para(pdf, 5.5, "Verlader: " + "; ".join(f"{n} {ev.pct(p, 1)}" for n, p in zip(names_rows, a.sol.x)) + ". Kontrolleur: " + "; ".join(f"{n} {ev.pct(p, 1)}" for n, p in zip(names_cols, a.sol.y)) + ".")
    _para(pdf, 5.5, ev.verdict_main(a.value, a.security, len(a.saddles), a.unique)[1])
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Faustregeln", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, ev.verdict_rules(a.value, a.loader_rows, a.controller_rows)[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Modellrechnung mit Erwartungswerten; Eigennutz und ein mitdenkender Gegner sind Annahmen, keine Aussagen über echte Kontrollen. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
