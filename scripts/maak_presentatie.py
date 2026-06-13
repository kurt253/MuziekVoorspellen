"""Genereer presentatie.pptx — Muziekvoorspellen."""
from __future__ import annotations
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

# ── Kleuren ────────────────────────────────────────────────────────────────────
DARK    = RGBColor(0x0f, 0x17, 0x2a)
MID     = RGBColor(0x1e, 0x29, 0x3b)
SLATE   = RGBColor(0x33, 0x41, 0x55)
ACCENT  = RGBColor(0x38, 0xbd, 0xf8)   # blauw
ACCENT2 = RGBColor(0xf5, 0x9e, 0x0b)   # geel/amber
ACCENT3 = RGBColor(0xa7, 0x8b, 0xfa)   # paars
GREEN   = RGBColor(0x10, 0xb9, 0x81)
WHITE   = RGBColor(0xe2, 0xe8, 0xf0)
MUTED   = RGBColor(0x94, 0xa3, 0xb8)
DIM     = RGBColor(0x47, 0x55, 0x69)

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)

OUT = Path(__file__).parent.parent / "presentatie.pptx"


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank(prs: Presentation):
    layout = prs.slide_layouts[6]   # volledig leeg
    slide  = prs.slides.add_slide(layout)
    fill_bg(slide, DARK)
    return slide

def fill_bg(slide, color: RGBColor):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, x, y, w, h, fill: RGBColor, alpha: int = 255,
             line_color: RGBColor | None = None, line_pt: float = 1.0):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(line_pt)
    else:
        shape.line.fill.background()
    return shape

def add_text(slide, text: str, x, y, w, h,
             size: int = 20,
             bold: bool = False,
             color: RGBColor = WHITE,
             align=PP_ALIGN.LEFT,
             italic: bool = False,
             wrap: bool = True) -> None:
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    run.font.name = "Segoe UI"

def add_para(tf, text: str, size: int = 16, bold: bool = False,
             color: RGBColor = WHITE, align=PP_ALIGN.LEFT,
             space_before: int = 0, italic: bool = False):
    from pptx.util import Pt as PPt
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    run = p.add_run()
    run.text = text
    run.font.size  = PPt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    run.font.name  = "Segoe UI"
    return p

def heading(slide, text: str, y=Inches(0.35)):
    add_text(slide, text,
             x=Inches(0.55), y=y,
             w=Inches(12.2), h=Inches(0.65),
             size=30, bold=True, color=ACCENT)

def subtext(slide, text: str, y=Inches(1.05)):
    add_text(slide, text,
             x=Inches(0.55), y=y,
             w=Inches(12.2), h=Inches(0.4),
             size=14, color=MUTED)

def card(slide, x, y, w, h,
         title: str, body: str,
         title_color: RGBColor = ACCENT):
    add_rect(slide, x, y, w, h, MID, line_color=SLATE, line_pt=0.75)
    add_text(slide, title,
             x+Inches(0.18), y+Inches(0.12),
             w-Inches(0.36), Inches(0.38),
             size=14, bold=True, color=title_color)
    add_text(slide, body,
             x+Inches(0.18), y+Inches(0.52),
             w-Inches(0.36), h-Inches(0.64),
             size=11, color=MUTED)

def stat_box(slide, x, y, w, h, value: str, label: str):
    add_rect(slide, x, y, w, h, MID, line_color=SLATE, line_pt=0.75)
    add_text(slide, value,
             x, y+Inches(0.1), w, Inches(0.55),
             size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(slide, label,
             x, y+Inches(0.65), w, Inches(0.35),
             size=11, color=MUTED, align=PP_ALIGN.CENTER)

def hbar(slide, x, y, w, h, frac_blue: float):
    # blauw gedeelte
    add_rect(slide, x, y, int(w * frac_blue), h, ACCENT)
    # amber gedeelte
    add_rect(slide, x + int(w * frac_blue), y,
             w - int(w * frac_blue), h, ACCENT2)

def accentline(slide, y=Inches(1.0)):
    add_rect(slide, Inches(0.55), y, Inches(0.06), Inches(0.38), ACCENT)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ══════════════════════════════════════════════════════════════════════════════

def slide_titel(prs):
    s = blank(prs)
    # gradient-achtig via twee rechthoeken
    add_rect(s, 0, 0, W, H, DARK)
    add_rect(s, 0, 0, Inches(6), H, RGBColor(0x0c, 0x1a, 0x3a))

    add_text(s, "🎵", Inches(5.5), Inches(0.6), Inches(2), Inches(1.2),
             size=60, align=PP_ALIGN.CENTER)
    add_text(s, "Muziekvoorspellen",
             Inches(1), Inches(1.8), Inches(11.3), Inches(1.4),
             size=52, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(s, "Bach-muziek aanvullen met een LSTM-taalmodel",
             Inches(2), Inches(3.2), Inches(9.3), Inches(0.6),
             size=18, color=MUTED, align=PP_ALIGN.CENTER)

    badges = [("PyTorch", ACCENT), ("music21", ACCENT2),
              ("Streamlit", ACCENT3), ("Bach-corpus", GREEN)]
    bx = Inches(2.8)
    for lbl, col in badges:
        add_rect(s, bx, Inches(4.1), Inches(1.8), Inches(0.42),
                 col, line_color=None)
        add_text(s, lbl, bx, Inches(4.12), Inches(1.8), Inches(0.4),
                 size=12, bold=True, color=DARK, align=PP_ALIGN.CENTER)
        bx += Inches(2.0)

    add_text(s, "Kurt Maekelberghe  ·  2025–2026",
             Inches(3), Inches(6.8), Inches(7.3), Inches(0.4),
             size=11, color=DIM, align=PP_ALIGN.CENTER)


def slide_vraag(prs):
    s = blank(prs)
    heading(s, "De vraag")
    subtext(s, "Kan een neuraal netwerk de stijl van Bach leren?")

    add_rect(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(1.0),
             RGBColor(0x0e, 0x28, 0x3a), line_color=ACCENT, line_pt=1.5)
    add_text(s,
             "Kan een neuraal netwerk de stijl van Bach leren — en een onafgemaakt werk "
             "muzikaal geloofwaardig aanvullen?",
             Inches(0.8), Inches(1.6), Inches(11.7), Inches(0.8),
             size=15, color=WHITE, italic=True)

    card_w = Inches(3.9)
    card_h = Inches(3.5)
    cards = [
        ("🎼  Aanvulling",
         "Eerste helft als seed, model vult de tweede helft aan met behoud van ritme en toonaard.",
         ACCENT),
        ("🤖  Bach-stijl",
         "Eerste 4 maten als seed, model genereert daarna volledig vrij — ritme én melodie.",
         ACCENT3),
        ("🎹  Meerstemmig",
         "Elke stem (Soprano, Alto, Tenor, Bass) wordt onafhankelijk gegenereerd.",
         ACCENT2),
    ]
    for i, (t, b, c) in enumerate(cards):
        card(s, Inches(0.55) + i * (card_w + Inches(0.18)),
             Inches(2.65), card_w, card_h, t, b, title_color=c)


def slide_pipeline(prs):
    s = blank(prs)
    heading(s, "De pipeline")
    subtext(s, "Vier stappen van ruwe data naar gegenereerde muziek")

    steps = [
        ("1", "📥 Download",  "download_bach.py"),
        ("2", "🔍 Analyseer", "analyse_midi.py"),
        ("3", "🎓 Train",     "Training.py"),
        ("4", "🎵 Genereer",  "Genereer.py"),
    ]
    sw = Inches(2.6)
    sy = Inches(1.7)
    sh = Inches(1.6)
    for i, (num, lbl, fil) in enumerate(steps):
        sx = Inches(0.55) + i * (sw + Inches(0.52))
        add_rect(s, sx, sy, sw, sh, MID, line_color=SLATE, line_pt=0.75)
        add_text(s, f"STAP {num}", sx, sy+Inches(0.12), sw, Inches(0.3),
                 size=9, bold=True, color=DIM, align=PP_ALIGN.CENTER)
        add_text(s, lbl, sx, sy+Inches(0.38), sw, Inches(0.55),
                 size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(s, fil, sx, sy+Inches(0.95), sw, Inches(0.4),
                 size=11, color=ACCENT, align=PP_ALIGN.CENTER)
        if i < 3:
            add_text(s, "→",
                     sx + sw, sy + Inches(0.55), Inches(0.52), Inches(0.5),
                     size=22, color=DIM, align=PP_ALIGN.CENTER)

    stats = [("400+", "Bach-werken"), ("8", "genres"),
             (".mid + .xml", "per werk"), ("LSTM", "model-type")]
    bw = Inches(2.9)
    bh = Inches(1.1)
    by = Inches(5.1)
    for i, (v, l) in enumerate(stats):
        stat_box(s, Inches(0.55) + i * (bw + Inches(0.18)), by, bw, bh, v, l)


def slide_data(prs):
    s = blank(prs)
    heading(s, "Data — het Bach-corpus")
    subtext(s, "Genres op basis van BWV-nummer · twee bestandsformaten per werk")
    accentline(s)

    rows = [
        ("1 – 224",    "Cantate / Koraal (cantate)", False),
        ("225 – 231",  "Motet",                      False),
        ("232 – 236",  "Mis",                        False),
        ("237 – 243",  "Sanctus",                    False),
        ("244 / 245",  "Matthäus- / Johannes-Passion", False),
        ("248",        "Kerstoratorium",             False),
        ("250 – 438",  "Koralen  ← meest gebruikt",  True),
    ]
    tx = Inches(0.55); ty = Inches(1.55)
    row_h = Inches(0.47)
    add_rect(s, tx, ty, Inches(6.5), Inches(0.38), SLATE)
    add_text(s, "BWV-reeks", tx+Inches(0.12), ty+Inches(0.04),
             Inches(2), Inches(0.3), size=11, bold=True, color=MUTED)
    add_text(s, "Genre", tx+Inches(2.3), ty+Inches(0.04),
             Inches(4), Inches(0.3), size=11, bold=True, color=MUTED)

    for i, (bwv, genre, hl) in enumerate(rows):
        ry = ty + Inches(0.38) + i * row_h
        bg = RGBColor(0x0e, 0x28, 0x3a) if hl else MID
        bc = ACCENT if hl else SLATE
        add_rect(s, tx, ry, Inches(6.5), row_h - Inches(0.04), bg,
                 line_color=bc, line_pt=0.5)
        col = ACCENT if hl else WHITE
        add_text(s, bwv,   tx+Inches(0.12), ry+Inches(0.08),
                 Inches(2), row_h, size=12, color=col, bold=hl)
        add_text(s, genre, tx+Inches(2.3),  ry+Inches(0.08),
                 Inches(4), row_h, size=12, color=col, bold=hl)

    card(s, Inches(7.3), Inches(1.55), Inches(5.45), Inches(1.65),
         "MIDI (.mid)",
         "Noten, ritme, stemindeling, tempo.\nBasis voor training én afspelen.",
         ACCENT)
    card(s, Inches(7.3), Inches(3.35), Inches(5.45), Inches(1.65),
         "MusicXML (.xml)",
         "Sleutels, maatsoorten, dynamiek.\nGebruikt voor bladmuziekweergave via abcjs.",
         ACCENT3)
    add_text(s, "Beide formats worden vanuit hetzelfde\nmusic21-object geschreven — niet het\nene van het andere afgeleid.",
             Inches(7.3), Inches(5.2), Inches(5.45), Inches(0.9),
             size=11, color=MUTED)


def slide_tokenisatie(prs):
    s = blank(prs)
    heading(s, "Tokenisatie")
    subtext(s, "Elke noot of rust wordt een VoiceToken: (MIDI-pitch, duur)")
    accentline(s)

    add_text(s, "Voorbeeld: BWV 227 · Soprano-stem",
             Inches(0.55), Inches(1.55), Inches(12), Inches(0.35),
             size=12, color=DIM)

    tokens = ["(72, 1.0)", "(74, 0.5)", "(76, 0.5)", "(0, 1.0)  ← rust", "(71, 2.0)", "…"]
    colors = [ACCENT, ACCENT, ACCENT, DIM, ACCENT, DIM]
    tx = Inches(0.55)
    for tok, col in zip(tokens, colors):
        add_rect(s, tx, Inches(1.9), Inches(1.85), Inches(0.5),
                 RGBColor(0x0f, 0x17, 0x2a), line_color=col, line_pt=1.0)
        add_text(s, tok, tx+Inches(0.08), Inches(1.95),
                 Inches(1.7), Inches(0.4), size=11, color=col, align=PP_ALIGN.CENTER)
        tx += Inches(1.95)
        if tok != "…":
            add_text(s, "→", tx - Inches(0.15), Inches(1.95),
                     Inches(0.22), Inches(0.4), size=14, color=DIM)

    card(s, Inches(0.55), Inches(2.75), Inches(5.9), Inches(2.0),
         "Duratierooster",
         "0.25 · 0.5 · 0.75 · 1.0 · 1.5 · 2.0 · 3.0 · 4.0\n\n"
         "Micro-timings worden afgerond naar de dichtstbijzijnde waarde.",
         ACCENT)
    card(s, Inches(6.65), Inches(2.75), Inches(6.1), Inches(2.0),
         "Vocabulaire",
         "(pitch, duur) → integer index\n\n"
         "Alle unieke tokens in het trainingskorpus worden genummerd en "
         "opgeslagen als .pkl-bestand naast het model.",
         ACCENT2)

    add_rect(s, Inches(0.55), Inches(5.1), Inches(12.2), Inches(0.85),
             RGBColor(0x0e, 0x28, 0x3a), line_color=ACCENT3, line_pt=1.0)
    add_text(s,
             "Bij akkoorden: alleen de hoogste noot wordt gebruikt → compacter vocabulaire, "
             "melodielijn behouden.",
             Inches(0.8), Inches(5.2), Inches(11.7), Inches(0.65),
             size=13, color=WHITE)


def slide_architectuur(prs):
    s = blank(prs)
    heading(s, "LSTM-architectuur")
    subtext(s, "Het model leert: gegeven de laatste N tokens, wat komt er hierna?")
    accentline(s)

    code = (
        "class LSTMModel(nn.Module):\n"
        "    def __init__(self, vocab,\n"
        "                 embed_dim=64,\n"
        "                 hidden=256,\n"
        "                 lagen=2,\n"
        "                 dropout=0.3):\n"
        "        self.embedding = nn.Embedding(vocab, embed_dim)\n"
        "        self.lstm      = nn.LSTM(embed_dim, hidden,\n"
        "                                num_layers=lagen, ...)\n"
        "        self.dropout   = nn.Dropout(dropout)\n"
        "        self.fc        = nn.Linear(hidden, vocab)\n\n"
        "    def forward(self, x):\n"
        "        x = self.embedding(x)       # tokens → vectors\n"
        "        out, _ = self.lstm(x)        # contextuele verwerking\n"
        "        return self.fc(self.dropout( # logits per token\n"
        "            out[:, -1, :]))"
    )
    add_rect(s, Inches(0.55), Inches(1.55), Inches(7.5), Inches(5.55),
             RGBColor(0x02, 0x06, 0x17), line_color=SLATE, line_pt=0.75)
    add_text(s, code, Inches(0.75), Inches(1.65),
             Inches(7.1), Inches(5.3),
             size=11, color=RGBColor(0xa8, 0xb4, 0xc8))

    blocks = [
        ("Embedding", "vocab → 64", ACCENT),
        ("LSTM × 2 lagen", "64 → 256 hidden", ACCENT3),
        ("Dropout (0.3)", "", DIM),
        ("Linear", "256 → vocab", ACCENT2),
        ("Logits (kansen)", "", GREEN),
    ]
    bx = Inches(8.4); bw = Inches(4.5); bh = Inches(0.7)
    by = Inches(1.55)
    for i, (name, sub, col) in enumerate(blocks):
        add_rect(s, bx, by, bw, bh,
                 RGBColor(0x0e, 0x20, 0x35), line_color=col, line_pt=1.2)
        label = f"{name}  —  {sub}" if sub else name
        add_text(s, label, bx+Inches(0.12), by+Inches(0.12),
                 bw-Inches(0.24), bh-Inches(0.24),
                 size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
        by += bh + Inches(0.1)
        if i < len(blocks) - 1:
            add_text(s, "↓", bx, by - Inches(0.12), bw, Inches(0.22),
                     size=14, color=DIM, align=PP_ALIGN.CENTER)


def slide_training(prs):
    s = blank(prs)
    heading(s, "Training")
    subtext(s, "Sliding-window sequenties · early stopping · GPU-ondersteuning")
    accentline(s)

    add_rect(s, Inches(0.55), Inches(1.55), Inches(6.2), Inches(2.4),
             RGBColor(0x02, 0x06, 0x17), line_color=SLATE)
    code = (
        "# tokens: [t0, t1, t2, ... t20]\n"
        "# venster = 16:\n"
        "X[0] = [t0 … t15],  y[0] = t16\n"
        "X[1] = [t1 … t16],  y[1] = t17\n"
        "..."
    )
    add_text(s, code, Inches(0.75), Inches(1.65),
             Inches(5.8), Inches(2.2),
             size=12, color=RGBColor(0xa8, 0xb4, 0xc8))

    params = [
        ("Venster",    "context-lengte  (standaard 16)"),
        ("Hidden",     "64 – 512 units per LSTM-laag"),
        ("Lagen",      "1 – 4 gestapelde LSTM-lagen"),
        ("Epochs",     "max 5 – 200  (early stopping)"),
        ("Geduld",     "stop na N epochs zonder verbetering"),
        ("Validatie",  "80 / 20 train-validatie split"),
    ]
    ty = Inches(4.2)
    for p, v in params:
        add_text(s, f"•  {p}",  Inches(0.55), ty, Inches(2.2), Inches(0.38),
                 size=12, bold=True, color=ACCENT)
        add_text(s, v, Inches(2.75), ty, Inches(4), Inches(0.38),
                 size=12, color=MUTED)
        ty += Inches(0.38)

    card(s, Inches(7.0), Inches(1.55), Inches(5.75), Inches(1.45),
         "🔄  Early stopping",
         "Bewaard het model met het laagste validatieverlies — niet het laatste epoch.",
         ACCENT)
    card(s, Inches(7.0), Inches(3.1), Inches(5.75), Inches(1.45),
         "⚡  GPU-ondersteuning",
         "Automatische detectie:\nIntel Arc (XPU) → NVIDIA (CUDA) → CPU.",
         ACCENT3)
    card(s, Inches(7.0), Inches(4.65), Inches(5.75), Inches(1.45),
         "✂️  Gradient clipping",
         "Max norm 1.0 voorkomt exploderende gradiënten bij lange sequenties.",
         ACCENT2)


def slide_modus_a(prs):
    s = blank(prs)
    heading(s, "Modus A — Aanvulling")
    subtext(s, "Eerste helft als seed · origineel ritme behouden · toonaard-filter actief")
    accentline(s)

    hbar(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.38), 0.5)
    add_text(s, "🔵 Maat 1 – N/2  ·  origineel",
             Inches(0.55), Inches(1.97), Inches(6), Inches(0.32),
             size=11, color=ACCENT)
    add_text(s, "🟡 Maat N/2+1 – N  ·  AI gegenereerd",
             Inches(6.75), Inches(1.97), Inches(6), Inches(0.32),
             size=11, color=ACCENT2, align=PP_ALIGN.RIGHT)

    card(s, Inches(0.55), Inches(2.45), Inches(5.9), Inches(2.0),
         "FASE 1 — Ritme overnemen",
         "Het originele ritme van de tweede helft wordt exact gekopieerd. "
         "Alleen de pitches worden vervangen door AI-gegenereerde waarden.",
         ACCENT)
    card(s, Inches(6.65), Inches(2.45), Inches(6.1), Inches(2.0),
         "FASE 2 — Melodie genereren",
         "Per duurslot kiest het model een pitch.\n\n"
         "Toonaard-filter (Krumhansl-Schmuckler): alleen noten uit de "
         "gedetecteerde toonaard zijn toegestaan.",
         ACCENT2)

    add_rect(s, Inches(0.55), Inches(4.75), Inches(12.2), Inches(0.9),
             RGBColor(0x0e, 0x28, 0x3a), line_color=ACCENT, line_pt=1.5)
    add_text(s,
             "P(pitch | context, duur)  —  sample uit tokens met de juiste duur, "
             "gefilterd op pitchklassen 0–11 van de toonaard.",
             Inches(0.8), Inches(4.87), Inches(11.7), Inches(0.65),
             size=13, color=WHITE, italic=True)

    card(s, Inches(0.55), Inches(5.85), Inches(5.9), Inches(1.3),
         "Toonaard detectie",
         "music21 analyseert de eerste helft via Krumhansl-Schmuckler\n"
         "→ set van toegestane MIDI-pitchklassen (0–11).",
         ACCENT3)
    card(s, Inches(6.65), Inches(5.85), Inches(6.1), Inches(1.3),
         "Resultaat",
         "Originele ritme bewaard · AI-melodie in de juiste toonaard · "
         "elke stem als aparte MIDI-Part.",
         GREEN)


def slide_modus_b(prs):
    s = blank(prs)
    heading(s, "Modus B — Bach-stijl")
    subtext(s, "Eerste 4 maten als seed · model genereert ritme én melodie volledig vrij")
    accentline(s)

    hbar(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.38), 0.15)
    add_text(s, "🔵  4 maten seed",
             Inches(0.55), Inches(1.97), Inches(3), Inches(0.32),
             size=11, color=ACCENT)
    add_text(s, "🟡  rest volledig AI gegenereerd",
             Inches(3.7), Inches(1.97), Inches(9), Inches(0.32),
             size=11, color=ACCENT2)

    add_rect(s, Inches(0.55), Inches(2.45), Inches(5.9), Inches(2.6),
             RGBColor(0x02, 0x06, 0x17), line_color=ACCENT, line_pt=1.0)
    add_text(s, "FASE 1 — Ritme genereren",
             Inches(0.75), Inches(2.55), Inches(5.5), Inches(0.4),
             size=13, bold=True, color=ACCENT)
    code = (
        "# Marginaliseer over pitches:\n"
        "P(duur | context) =\n"
        "    Σ_pitch P(pitch, duur | context)\n\n"
        "# → duur-gewichten via softmax\n"
        "# → sample één duur via multinomial"
    )
    add_text(s, code, Inches(0.75), Inches(2.98),
             Inches(5.5), Inches(1.9),
             size=11, color=RGBColor(0xa8, 0xb4, 0xc8))

    card(s, Inches(6.65), Inches(2.45), Inches(6.1), Inches(2.6),
         "FASE 2 — Melodie genereren",
         "Geen toonaard-filter.\n\n"
         "Het model bepaalt zelf de toonaard op basis van de geleerde "
         "Bach-stijl. Meer creatieve vrijheid dan modus A.",
         ACCENT2)

    card(s, Inches(0.55), Inches(5.3), Inches(5.9), Inches(1.85),
         "Verschil met modus A",
         "• Ritme: model genereert vrij  (A: origineel overnemen)\n"
         "• Toonaard: geen filter  (A: Krumhansl-Schmuckler)\n"
         "• Seed: 4 maten  (A: helft van het werk)",
         ACCENT3)
    card(s, Inches(6.65), Inches(5.3), Inches(6.1), Inches(1.85),
         "Veiligheidsnet",
         "Onvoldoende seed-data → origineel ritme als fallback.\n"
         "Max stappen = max(500, doelduur/0.25 × 10) als stop-criterium.",
         DIM)


def slide_sampling(prs):
    s = blank(prs)
    heading(s, "Sampling-parameters")
    subtext(s, "Temperatuur en top-k bepalen de creatieve vrijheid van het model")
    accentline(s)

    add_rect(s, Inches(0.55), Inches(1.55), Inches(5.9), Inches(3.0),
             RGBColor(0x02, 0x06, 0x17), line_color=ACCENT, line_pt=1.0)
    add_text(s, "🌡️  Temperatuur",
             Inches(0.75), Inches(1.65), Inches(5.5), Inches(0.42),
             size=14, bold=True, color=ACCENT)
    code_t = (
        "probs = softmax(logits / T)\n\n"
        "T = 0.3  →  scherpe verdeling\n"
        "          Bach-achtig, voorspelbaar\n\n"
        "T = 1.5  →  vlakke verdeling\n"
        "          creatief, willekeurig"
    )
    add_text(s, code_t, Inches(0.75), Inches(2.1),
             Inches(5.5), Inches(2.3),
             size=12, color=RGBColor(0xa8, 0xb4, 0xc8))

    add_rect(s, Inches(6.65), Inches(1.55), Inches(6.1), Inches(3.0),
             RGBColor(0x02, 0x06, 0x17), line_color=ACCENT3, line_pt=1.0)
    add_text(s, "🎯  Top-k filtering",
             Inches(6.85), Inches(1.65), Inches(5.7), Inches(0.42),
             size=14, bold=True, color=ACCENT3)
    code_k = (
        "# Maskeer alles buiten de top-k\n"
        "grens = topk(logits, k).values[-1]\n"
        "logits[logits < grens] = -inf\n"
        "# → softmax over k kandidaten\n\n"
        "k = 0   →  alle tokens\n"
        "k = 10  →  alleen 10 beste"
    )
    add_text(s, code_k, Inches(6.85), Inches(2.1),
             Inches(5.7), Inches(2.3),
             size=12, color=RGBColor(0xa8, 0xb4, 0xc8))

    effects = [
        ("Lage T + lage k", "Meest voorspelbaar, Bach-achtig",      ACCENT),
        ("Lage T + hoge k", "Melodieus maar gevarieerd",             ACCENT3),
        ("Hoge T + lage k", "Ritmisch spannend, stabiele toonaard",  ACCENT2),
        ("Hoge T + hoge k", "Maximale creativiteit / willekeur",     GREEN),
    ]
    ew = Inches(2.95)
    ex = Inches(0.55)
    ey = Inches(4.8)
    for i, (lbl, desc, col) in enumerate(effects):
        card(s, ex + i*(ew+Inches(0.18)), ey, ew, Inches(1.7),
             lbl, desc, title_color=col)


def slide_midi_uitvoer(prs):
    s = blank(prs)
    heading(s, "Van tokens naar geluid")
    subtext(s, "De gegenereerde tokens worden omgezet naar een afspeelbaar MIDI-bestand")
    accentline(s)

    steps = [
        ("🔨", "music21 bouwen",
         "Elke AI-stem als aparte Part\ningevoegd, startend op de\nsplitsmaat.", ACCENT),
        ("💾", "In RAM schrijven",
         "mf.writestr() schrijft MIDI naar\nbytes in geheugen.\nGeen schijfbestand.", ACCENT3),
        ("🌐", "data-URI in HTML",
         "Base64-encoded bytes als\ndata:audio/midi;base64,…\ndirect in de HTML pagina.", ACCENT2),
        ("🎹", "Browser speelt af",
         "html-midi-player + Magenta.js:\npiano-roll + geluid.\nDownload ingebouwd.", GREEN),
    ]
    sw = Inches(2.9)
    sy = Inches(1.7)
    sh = Inches(3.0)
    for i, (icon, title, body, col) in enumerate(steps):
        sx = Inches(0.55) + i * (sw + Inches(0.18))
        add_rect(s, sx, sy, sw, sh, MID, line_color=col, line_pt=1.2)
        add_text(s, icon,  sx, sy+Inches(0.15), sw, Inches(0.7),
                 size=28, align=PP_ALIGN.CENTER)
        add_text(s, title, sx+Inches(0.12), sy+Inches(0.8), sw-Inches(0.24), Inches(0.45),
                 size=13, bold=True, color=col)
        add_text(s, body,  sx+Inches(0.12), sy+Inches(1.3), sw-Inches(0.24), Inches(1.5),
                 size=11, color=MUTED)
        if i < 3:
            add_text(s, "→",
                     sx + sw + Inches(0.02), sy + Inches(1.1),
                     Inches(0.16), Inches(0.5),
                     size=18, color=DIM, align=PP_ALIGN.CENTER)

    add_rect(s, Inches(0.55), Inches(5.0), Inches(12.2), Inches(0.85),
             RGBColor(0x0e, 0x28, 0x3a), line_color=ACCENT, line_pt=1.5)
    add_text(s,
             "Geen enkel serververzoek na het laden van de pagina — "
             "alles draait client-side in de browser.",
             Inches(0.8), Inches(5.12), Inches(11.7), Inches(0.6),
             size=14, color=WHITE, italic=True)


def slide_app(prs):
    s = blank(prs)
    heading(s, "De Streamlit-app")
    subtext(s, "Drie pagina's — overzicht, training, generatie")
    accentline(s)

    card(s, Inches(0.55), Inches(1.55), Inches(3.9), Inches(2.3),
         "🏠  app.py — Overzicht",
         "Filterbare tabel met alle werken.\nPer werk: bladmuziek (abcjs)\nen piano-roll.\nStemkeuze via multiselect.",
         ACCENT)
    card(s, Inches(4.65), Inches(1.55), Inches(3.9), Inches(2.3),
         "🎓  Training.py",
         "Model trainen met live voortgang\nen verliesplot.\nModelbeheer (verwijderen).\nGPU auto-detectie.",
         ACCENT3)
    card(s, Inches(8.75), Inches(1.55), Inches(3.9), Inches(2.3),
         "🎵  Genereer.py",
         "Aanvulling of Bach-stijl.\nTemperatuur- en top-k sliders.\nTwee piano-rolls naast\nelkaar met maat-enteller.",
         ACCENT2)

    add_text(s, "Maat-enteller (JavaScript · elke 100 ms)",
             Inches(0.55), Inches(4.1), Inches(6), Inches(0.38),
             size=12, color=DIM)

    for lbl, col, y in [
        ("Maat 12 / 24   (origineel)", ACCENT,  Inches(4.55)),
        ("Maat 16 / 24   (AI)",        ACCENT2, Inches(5.15)),
    ]:
        add_rect(s, Inches(0.55), y, Inches(5.5), Inches(0.5),
                 MID, line_color=SLATE, line_pt=0.75)
        add_text(s, lbl, Inches(0.75), y+Inches(0.08),
                 Inches(5.1), Inches(0.36),
                 size=13, bold=True, color=col)

    add_text(s, "Splitsbalk",
             Inches(7.0), Inches(4.1), Inches(6), Inches(0.38),
             size=12, color=DIM)
    hbar(s, Inches(7.0), Inches(4.55), Inches(6.0), Inches(0.38), 0.5)
    add_text(s, "🔵 eerste helft · origineel",
             Inches(7.0), Inches(5.0), Inches(3), Inches(0.35),
             size=11, color=ACCENT)
    add_text(s, "🟡 tweede helft · AI",
             Inches(10.0), Inches(5.0), Inches(3), Inches(0.35),
             size=11, color=ACCENT2, align=PP_ALIGN.RIGHT)

    add_text(s,
             "Beide formaten (MIDI als data-URI) worden in dezelfde HTML ingebed — "
             "geen extra serververzoeken.",
             Inches(0.55), Inches(6.1), Inches(12.2), Inches(0.5),
             size=12, color=MUTED)


def slide_techstack(prs):
    s = blank(prs)
    heading(s, "Tech-stack")
    subtext(s, "Alle tools en bibliotheken die het project gebruikt")
    accentline(s)

    items = [
        ("🔥", "PyTorch",          "LSTM-model & training",     ACCENT),
        ("🎵", "music21",          "MIDI/XML parsing & export", ACCENT2),
        ("🌊", "Streamlit",        "Interactieve UI",           ACCENT3),
        ("🐼", "pandas",           "Data & CSV-beheer",         GREEN),
        ("🎹", "html-midi-player", "Piano-roll in browser",     ACCENT),
        ("🎼", "abcjs",            "Bladmuziek → SVG",          ACCENT2),
        ("🧮", "NumPy",            "Sequentie-verwerking",      ACCENT3),
        ("⚡", "Intel IPEX",       "Arc GPU-optimalisatie",     GREEN),
    ]
    iw = Inches(2.9); ih = Inches(1.85)
    ix0 = Inches(0.55); iy0 = Inches(1.55)
    cols = 4
    for i, (icon, name, role, col) in enumerate(items):
        row = i // cols
        c   = i %  cols
        ix  = ix0 + c * (iw + Inches(0.18))
        iy  = iy0 + row * (ih + Inches(0.2))
        add_rect(s, ix, iy, iw, ih, MID, line_color=col, line_pt=0.75)
        add_text(s, icon, ix, iy+Inches(0.15), iw, Inches(0.55),
                 size=24, align=PP_ALIGN.CENTER)
        add_text(s, name, ix+Inches(0.12), iy+Inches(0.7),
                 iw-Inches(0.24), Inches(0.45),
                 size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(s, role, ix+Inches(0.12), iy+Inches(1.15),
                 iw-Inches(0.24), Inches(0.55),
                 size=10, color=MUTED, align=PP_ALIGN.CENTER)


def slide_samenvatting(prs):
    s = blank(prs)
    heading(s, "Wat hebben we gebouwd?")
    subtext(s, "Een volledig end-to-end muziekgeneratie-systeem voor Bach-koralen")
    accentline(s)

    items = [
        ("📥", "Volledige datapipeline",
         "400+ Bach-werken automatisch gedownload, geanalyseerd en geclassificeerd in 8 genres.",
         ACCENT),
        ("🧠", "Configureerbaar LSTM",
         "Trainbaar op elk muziekgenre. Hyperparameters instelbaar via de UI. GPU-ondersteuning.",
         ACCENT3),
        ("🎼", "Twee generatiemodi",
         "Aanvulling met ritme- en toonaard-behoud, of vrije Bach-stijl generatie per stem.",
         ACCENT2),
        ("🌐", "Interactieve app",
         "Bladmuziek, piano-roll, maat-enteller en live download — alles client-side in de browser.",
         GREEN),
    ]
    sw = Inches(5.9)
    sy = Inches(1.7)
    sh = Inches(2.3)
    for i, (icon, title, body, col) in enumerate(items):
        row = i // 2
        col_idx = i % 2
        sx = Inches(0.55) + col_idx * (sw + Inches(0.3))
        iy = sy + row * (sh + Inches(0.25))
        add_rect(s, sx, iy, sw, sh, MID, line_color=col, line_pt=1.2)
        add_text(s, icon,  sx+Inches(0.18), iy+Inches(0.12),
                 Inches(0.6), Inches(0.6), size=24)
        add_text(s, title, sx+Inches(0.85), iy+Inches(0.18),
                 sw-Inches(1.0), Inches(0.45),
                 size=14, bold=True, color=col)
        add_text(s, body,  sx+Inches(0.18), iy+Inches(0.75),
                 sw-Inches(0.36), sh-Inches(0.9),
                 size=11, color=MUTED)


def slide_einde(prs):
    s = blank(prs)
    add_rect(s, 0, 0, W, H, DARK)
    add_rect(s, 0, 0, W, Inches(0.6), ACCENT)

    add_text(s, "🎵", Inches(5.5), Inches(0.8), Inches(2.3), Inches(1.3),
             size=64, align=PP_ALIGN.CENTER)
    add_text(s, "Live demo",
             Inches(2), Inches(2.1), Inches(9.3), Inches(1.2),
             size=52, bold=True, color=ACCENT2, align=PP_ALIGN.CENTER)
    add_text(s, "Streamlit openen en zelf genereren",
             Inches(2.5), Inches(3.35), Inches(8.3), Inches(0.6),
             size=18, color=MUTED, align=PP_ALIGN.CENTER)

    add_rect(s, Inches(3.5), Inches(4.2), Inches(6.3), Inches(0.75),
             RGBColor(0x02, 0x06, 0x17), line_color=ACCENT, line_pt=1.5)
    add_text(s, "streamlit run app.py",
             Inches(3.5), Inches(4.28), Inches(6.3), Inches(0.6),
             size=18, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    add_rect(s, 0, H - Inches(0.55), W, Inches(0.55), MID)
    add_text(s, "Kurt Maekelberghe  ·  Muziekvoorspellen  ·  2025–2026",
             Inches(1), H - Inches(0.48), Inches(11.3), Inches(0.42),
             size=11, color=DIM, align=PP_ALIGN.CENTER)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    prs = new_prs()

    slide_titel(prs)
    slide_vraag(prs)
    slide_pipeline(prs)
    slide_data(prs)
    slide_tokenisatie(prs)
    slide_architectuur(prs)
    slide_training(prs)
    slide_modus_a(prs)
    slide_modus_b(prs)
    slide_sampling(prs)
    slide_midi_uitvoer(prs)
    slide_app(prs)
    slide_techstack(prs)
    slide_samenvatting(prs)
    slide_einde(prs)

    prs.save(str(OUT))
    print(f"✓ Opgeslagen → {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
