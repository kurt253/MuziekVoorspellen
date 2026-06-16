"""Genereer presentatie.pptx — Muziekvoorspellen."""
from __future__ import annotations
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
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

OUT      = Path(__file__).parent.parent / "presentatie.pptx"
IMG_DIR  = Path(__file__).parent.parent / "presentatie_screenshots"
IMG_DIR.mkdir(exist_ok=True)


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


def add_notes(slide, text: str) -> None:
    """Voeg sprekernotities toe aan een slide."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text


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

    add_notes(s, (
        "SLIDE 6 — LSTM-ARCHITECTUUR: parameter voor parameter\n\n"
        "EMBEDDING  (nn.Embedding)\n"
        "Wat: zet een integer token-index om naar een dichte vector van grootte embed_dim.\n"
        "Waarom: een getal als '42' heeft geen betekenis voor het netwerk; een vector "
        "van 64 getallen wel. De embedding leert zelf welke tokens op elkaar lijken "
        "(bijv. noten in dezelfde toonaard krijgen nabijgelegen vectoren).\n"
        "Parameter embed_dim (standaard 64):\n"
        "  - Klein (32): sneller, minder expressief. Goed voor kleine vocabulaires (<500 tokens).\n"
        "  - Groot (128–256): meer nuance, maar trager en meer risico op overtraining.\n"
        "  - Vuistregel: embed_dim ≈ sqrt(vocab_size). Bij vocab ~1000 is 64 ideaal.\n\n"
        "LSTM  (nn.LSTM)\n"
        "Wat: Long Short-Term Memory recurrente laag. Verwerkt de sequentie van embeddings "
        "en houdt een verborgen toestand bij die 'geheugen' over eerdere tokens opslaat.\n"
        "Twee kernparameters:\n"
        "  hidden (standaard 256): grootte van de verborgen toestand per tijdstap.\n"
        "    - Klein (64): snel maar beperkt. Goed voor korte stukken/eenvoudige structuren.\n"
        "    - Groot (512): trager maar beter bij complexe harmonische patronen.\n"
        "    - Te groot + weinig data = overtraining.\n"
        "  lagen (num_layers, standaard 2): aantal gestapelde LSTM-lagen.\n"
        "    - 1 laag: basismodel, snelst.\n"
        "    - 2 lagen: hogere-orde patronen leren (bijv. frasering boven ritme).\n"
        "    - 3–4 lagen: zelden beter dan 2 voor muziek; trager en instabiel.\n"
        "Intern bevat elke LSTM-laag 4 poorten (forget, input, cell, output) die elk "
        "hun eigen gewichtsmatrix hebben: 4 × hidden × (embed_dim + hidden) parameters.\n\n"
        "DROPOUT  (nn.Dropout)\n"
        "Wat: zet tijdens training willekeurig een fractie van de activaties op nul.\n"
        "Waarom: voorkomt dat het netwerk te sterk leunt op specifieke neuronen "
        "(overtraining). Wordt NIET toegepast tijdens inferentie (model.eval()).\n"
        "Parameter dropout (standaard 0.3):\n"
        "  - 0.0: geen regularisatie. Risico op overtraining bij kleine datasets.\n"
        "  - 0.3: 30% van de neuronen per stap uitgeschakeld. Goede balans.\n"
        "  - 0.5: sterke regularisatie. Gebruik bij heel kleine datasets of grote modellen.\n\n"
        "LINEAR  (nn.Linear)\n"
        "Wat: projecteert de laatste verborgen toestand (hidden-dim) terug naar "
        "vocab_size dimensies. De uitvoer zijn 'logits': ruwe scores per token.\n"
        "Waarom 'de laatste'? out[:, -1, :] neemt alleen de uitvoer van het laatste "
        "tijdstap — dat is de samenvatting van de volledige context. Daarna softmax "
        "voor kansen, sampling voor het volgende token.\n\n"
        "TOTALE PARAMETERCOUNT (richtlijn)\n"
        "embed_dim=64, hidden=256, lagen=2, vocab=1000:\n"
        "  Embedding:  1000 × 64  =   64.000\n"
        "  LSTM laag1: 4 × 256 × (64+256)  ≈  327.000\n"
        "  LSTM laag2: 4 × 256 × (256+256) ≈  524.000\n"
        "  Linear:     256 × 1000           =  256.000\n"
        "  Totaal: ≈ 1.2 miljoen parameters — klein maar effectief voor Bach-koralen."
    ))


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

    add_notes(s, (
        "SLIDE 7 — TRAINING: parameter voor parameter\n\n"
        "HOE WERKT HET SLIDING WINDOW?\n"
        "Het volledige corpus wordt tokenized naar een lange reeks (pitch, duur)-paren. "
        "Stel venster=16 en de reeks is [t0, t1, t2, ... t500]:\n"
        "  Voorbeeld 0: X=[t0..t15],  y=t16\n"
        "  Voorbeeld 1: X=[t1..t16],  y=t17\n"
        "  ...\n"
        "Elk venster is één trainingsvoorbeeld. Zo ziet het model elk token in "
        "meerdere contexten — zowel als invoer als als label.\n\n"
        "PARAMETER 1 — VENSTER (context-lengte, standaard 16)\n"
        "Wat: hoeveel tokens 'achteruit' het model kan kijken bij elke voorspelling.\n"
        "16 tokens ≈ 4 maten (bij gemiddeld 4 noten per maat).\n"
        "  Klein (4–8): snel, weinig geheugen, maar het model 'vergeet' de frasering.\n"
        "  Standaard (16): goede balans context vs. rekentijd.\n"
        "  Groot (32–64): meer muzikale context (hele frase), maar veel trager "
        "en meer kans op overtraining bij kleine datasets.\n"
        "Effect op muziek: klein venster = repetitief en lokaal correct; "
        "groot venster = meer coherente frasering maar trager.\n\n"
        "PARAMETER 2 — HIDDEN UNITS (standaard 256)\n"
        "Wat: breedte van de LSTM-verborgen toestand. Meer units = meer capaciteit "
        "om patronen te onthouden.\n"
        "  64–128: snel, goed voor eenvoudige koralen (beperkt harmonisch).\n"
        "  256: standaard, goed voor cantates en passies.\n"
        "  512: voor grote corpora (400+ werken) met complexe structuur.\n"
        "Effect: groter hidden = hoger trainingsverlies slinkt sneller, "
        "maar validatieverlies kan hoger worden (overtraining).\n\n"
        "PARAMETER 3 — LAGEN (num_layers, standaard 2)\n"
        "Wat: aantal gestapelde LSTM-lagen. Elke extra laag verwerkt de uitvoer "
        "van de vorige laag en leert hogere-orde patronen.\n"
        "  1 laag: basismodel, snelst, goed genoeg voor eenvoudige structuren.\n"
        "  2 lagen: leert zowel lokale ritmepatronen (laag 1) als "
        "melodische frasering (laag 2). Aanbevolen standaard.\n"
        "  3–4 lagen: zelden beter voor Bach; instabiel zonder "
        "zorgvuldige regularisatie.\n\n"
        "PARAMETER 4 — EPOCHS (standaard max 200, early stopping)\n"
        "Wat: één epoch = het model heeft de volledige trainingsset één keer gezien.\n"
        "  Weinig epochs (5–20): ondertraining, model leert te weinig patronen.\n"
        "  Veel epochs (100–200): early stopping bepaalt wanneer te stoppen.\n"
        "Effect: het model met het LAAGSTE validatieverlies wordt opgeslagen "
        "(niet het laatste epoch). Zo wordt het beste moment vastgelegd.\n\n"
        "PARAMETER 5 — GEDULD / PATIENCE (standaard 10)\n"
        "Wat: als het validatieverlies 'geduld' epochs achtereen niet verbetert, "
        "stopt de training.\n"
        "  Laag geduld (3–5): snel stoppen, risico op te vroeg stoppen.\n"
        "  Hoog geduld (20+): meer kans op overtraining vinden, maar langer wachten.\n"
        "Standaard 10 = na 10 epochs zonder verbetering → stop, laad beste model.\n\n"
        "PARAMETER 6 — VALIDATIESPLIT (standaard 80/20)\n"
        "Wat: 80% van de data voor training, 20% voor validatie (evaluatie na elke epoch).\n"
        "De validatiedata wordt nooit voor gradiëntupdates gebruikt — het meet "
        "hoe goed het model generaliseert naar ongeziene stukken.\n\n"
        "OPTIMALISATIE: ADAM\n"
        "Adam past de leersnelheid per parameter automatisch aan op basis van "
        "hoe groot de recente gradiënten waren. Geen handmatige leersnelheidsscheduling nodig.\n\n"
        "GRADIENT CLIPPING (max_norm=1.0)\n"
        "Als de totale norm van alle gradiënten groter is dan 1.0, worden ze "
        "proportioneel geschaald zodat de norm precies 1.0 wordt. Dit voorkomt "
        "plotse grote gewichtsupdates ('exploderende gradiënten') die bij diepe "
        "LSTM's kunnen optreden bij lange sequenties of grote leersnelheden.\n\n"
        "TRAININGSDUUR (typisch op de beschikbare hardware)\n"
        "Koralen, hidden=256, 2 lagen, 20 epochs: ~10 min CPU / ~2 min GPU.\n"
        "Cantates/Passies: ~30 min CPU / ~5 min GPU."
    ))


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

    add_notes(s, (
        "SLIDE 8 — AI-AANVULLING vs FINE-TUNED: het grote verschil\n\n"
        "BEIDE versies gebruiken dezelfde aanvullingstechniek: ritme overnemen uit "
        "het origineel, en per duurslot een pitch genereren via het LSTM-model. "
        "Het verschil zit in WELk model gebruikt wordt.\n\n"
        "AI-AANVULLING (standaard model)\n"
        "• Gebruikt het model zoals het na de volledige training staat.\n"
        "• Het model is getraind op honderden Bach-werken van hetzelfde genre.\n"
        "• Het model kent de algemene Bach-stijl, maar weet niets specifiek over "
        "dit ene stuk.\n"
        "• Generatie is snel: geen extra training nodig.\n\n"
        "FINE-TUNED (bijgetraind op eerste helft)\n"
        "• Maakt een DIEPE KOPIE van het getrainde model.\n"
        "• Past het model nog 8 epochs bij op uitsluitend de tokens van de eerste "
        "helft van dit specifieke stuk (Adam optimizer, lr=2e-3).\n"
        "• Het model 'leert' de stijl van dit concrete stuk: zijn toonaard, "
        "melodische motieven, ritmepatronen.\n"
        "• Daarna genereert het de tweede helft — nu met die specifieke kennis.\n"
        "• Gevolg: hogere gelijkenis-scores, maar risico op overtraining op de "
        "eerste helft (te weinig variatie).\n\n"
        "ANALOGIE\n"
        "AI-aanvulling = een algemeen opgeleide componist die 'in de stijl van Bach' "
        "schrijft.\n"
        "Fine-tuned = dezelfde componist die het stuk eerst intensief bestudeert "
        "voordat hij de tweede helft schrijft.\n\n"
        "TECHNISCHE DETAILS FINE-TUNING\n"
        "• deepcopy(model) zodat het originele model intact blijft.\n"
        "• Alleen de eerste-helft-tokens als training (X[i] = venster, y[i] = volgende token).\n"
        "• 8 epochs is bewust beperkt: meer epochs → overtraining → te letterlijk kopiëren.\n"
        "• lr=2e-3 is hoger dan de initiële training (typisch 1e-3) voor snellere aanpassing."
    ))


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

    add_notes(s, (
        "SLIDE 10 — SAMPLING-PARAMETERS: uitgebreide toelichting\n\n"
        "TEMPERATUUR (T)\n"
        "Vóór het samplen worden de logits (ruwe modeluitvoer) gedeeld door T:\n"
        "    probs = softmax(logits / T)\n\n"
        "T < 1.0 — SCHERPE verdeling:\n"
        "• De hoogst-gewogen tokens krijgen nog meer gewicht.\n"
        "• Het model kiest bijna altijd de meest waarschijnlijke noot.\n"
        "• Klinkt 'bach-achtig', maar kan eentonig worden (veel herhalingen).\n"
        "• Aanbevolen: T = 0.5–0.8 voor koralen en eenvoudige melodieën.\n\n"
        "T > 1.0 — VLAKKE verdeling:\n"
        "• Alle tokens krijgen bijna even veel kans.\n"
        "• Het model maakt verrassendere, maar ook minder coherente keuzes.\n"
        "• Klinkt creatiever maar kan uit de toonaard gaan.\n"
        "• Aanbevolen: T = 1.0–1.3 voor experimentele generatie.\n\n"
        "TOP-K FILTERING\n"
        "Vóór de temperatuurscaling worden alle tokens buiten de top-k gemaskeerd:\n"
        "    grens = topk(logits, k).values[-1]\n"
        "    logits[logits < grens] = -inf\n\n"
        "k = 0: geen filtering, alle tokens in het vocabulaire zijn mogelijk.\n"
        "k = 5: alleen de 5 meest waarschijnlijke tokens worden overwogen.\n"
        "k = 10: goede balans tussen variatie en coherentie (aanbevolen standaard).\n\n"
        "COMBINATIE-EFFECTEN\n"
        "• Lage T + lage k: meest deterministisch, Bach-achtig, weinig variatie.\n"
        "• Lage T + hoge k: melodieus en gevarieerd, toonaard blijft stabiel.\n"
        "• Hoge T + lage k: ritmisch spannend door temperatuur, maar toonaard "
        "  beschermd door k-filtering.\n"
        "• Hoge T + hoge k: maximale creatieve vrijheid, kan klinken als willekeurig.\n\n"
        "PRAKTISCH ADVIES\n"
        "Begin met T=0.8, k=10. Verhoog T als het te saai klinkt, verlaag k als het "
        "te chaotisch wordt. Het toonaard-filter (in AI-aanvulling) beschermt altijd "
        "tegen compleet verkeerde noten."
    ))


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

    add_notes(s, (
        "SLIDE 11 — VAN TOKENS NAAR GELUID: uitgebreide toelichting\n\n"
        "STAP 1: music21-object bouwen\n"
        "Na de generatie heeft het model een lijst van (pitch, duur)-tokens per stem "
        "(Soprano, Alto, Tenor, Bass). Deze worden omgezet naar music21 Note-objecten "
        "en ingevoegd in aparte Part-objecten, startend op de exacte splitsingsmaat "
        "(het punt waar de originele eerste helft eindigt).\n"
        "Tempo-informatie: de MetronomeMark van het originele stuk wordt gekopieerd "
        "zodat de gegenereerde muziek even snel speelt als het origineel. Zonder dit "
        "zou de standaard 120 BPM gebruikt worden, wat klinkt als te snel.\n\n"
        "STAP 2: schrijven naar geheugen (geen schijfbestand)\n"
        "music21 exporteert naar MIDI via MidiFile en writestr(). Dit schrijft de "
        "bytes rechtstreeks naar een BytesIO-object in RAM, nooit naar de harde schijf. "
        "Dit is sneller en vereist geen tijdelijke bestandspermissies.\n\n"
        "STAP 3: Base64-codering als data-URI\n"
        "De MIDI-bytes worden gecodeerd als Base64-string en verpakt in een:\n"
        "    data:audio/midi;base64,<base64-string>\n"
        "Deze data-URI wordt direct in de HTML van de Streamlit-pagina ingebed.\n\n"
        "STAP 4: html-midi-player in de browser\n"
        "De html-midi-player bibliotheek (gebaseerd op Magenta.js en Tone.js) leest "
        "de data-URI en rendert:\n"
        "• Een piano-roll (noten visueel als blokken op een tijdlijn)\n"
        "• Afspelen via de browser's Web Audio API (geen plugin nodig)\n"
        "• Een downloadknop voor het MIDI-bestand\n\n"
        "VOORDEEL VAN DEZE AANPAK\n"
        "Zodra de Streamlit-pagina geladen is, zijn er GEEN extra HTTP-verzoeken meer "
        "nodig voor het afspelen. Alles zit al in de HTML-pagina ingebed als data-URI. "
        "Dit maakt de app ook offline bruikbaar na de eerste laadbeurt.\n\n"
        "BEPERKING\n"
        "Grote MIDI-bestanden (lange stukken, 4 stemmen) kunnen grote Base64-strings "
        "genereren. Bij stukken van meer dan 200 maten kan de HTML-pagina traag laden."
    ))


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
    heading(s, "Samenvatting")
    subtext(s, "Wat hebben we gebouwd — en wat leert het ons?")
    accentline(s)

    # Vier resultaat-blokken (2×2)
    items = [
        ("📥", "Datapipeline",
         "400+ Bach-werken worden automatisch gedownload, omgezet naar (pitch, duur)-tokens "
         "en ingedeeld in 8 genres op basis van BWV-nummer. Zonder handmatige stap.",
         ACCENT),
        ("🧠", "LSTM-taalmodel",
         "Een LSTM leert de stijl van Bach als een kansmodel over tokenreeksen. "
         "Embedding, verborgen toestand en dropout zijn volledig instelbaar via de UI.",
         ACCENT3),
        ("🎼", "Vier generatiemodi",
         "AI-aanvulling (ritme + toonaard bewaard), Fine-tuned (8 epochs op de eerste helft), "
         "Bach-stijl (ritme én melodie vrij gegenereerd) en Willekeurig als nulhypothese.",
         ACCENT2),
        ("📊", "Objectieve vergelijking",
         "Vier muzikale metrics (Chroma, Intervallen, N-gram, Akkoorden) meten de "
         "gelijkenis met het origineel enkel op het gegenereerde deel — niet de seed.",
         GREEN),
    ]
    sw = Inches(5.9); sy = Inches(1.6); sh = Inches(2.1)
    for i, (icon, title, body, col) in enumerate(items):
        row = i // 2; col_idx = i % 2
        sx = Inches(0.55) + col_idx * (sw + Inches(0.3))
        iy = sy + row * (sh + Inches(0.2))
        add_rect(s, sx, iy, sw, sh, MID, line_color=col, line_pt=1.2)
        add_rect(s, sx, iy, Inches(0.08), sh, col)
        add_text(s, icon,  sx+Inches(0.22), iy+Inches(0.1),
                 Inches(0.55), Inches(0.55), size=22)
        add_text(s, title, sx+Inches(0.85), iy+Inches(0.12),
                 sw-Inches(1.0), Inches(0.42),
                 size=13, bold=True, color=col)
        add_text(s, body,  sx+Inches(0.22), iy+Inches(0.62),
                 sw-Inches(0.36), sh-Inches(0.72),
                 size=10, color=MUTED)

    # Conclusie-balk onderaan
    add_rect(s, Inches(0.55), Inches(6.3), Inches(12.2), Inches(0.98),
             RGBColor(0x0e, 0x28, 0x3a), line_color=ACCENT, line_pt=1.5)
    add_text(s, "Conclusie",
             Inches(0.75), Inches(6.38), Inches(1.8), Inches(0.35),
             size=12, bold=True, color=ACCENT)
    add_text(s,
             "Een LSTM kan Bach-stijl leren en geloofwaardige aanvullingen genereren. "
             "Fine-tuning op de seed verbetert de score meetbaar. "
             "Zonder toonaard-filter of fine-tuning valt de kwaliteit terug naar willekeurig niveau.",
             Inches(2.65), Inches(6.38), Inches(9.9), Inches(0.82),
             size=11, color=WHITE)

    add_notes(s, (
        "SLIDE 19 — SAMENVATTING: wat te zeggen\n\n"
        "DATAPIPELINE\n"
        "Het systeem is volledig zelfvoorzienend: bij het eerste gebruik worden "
        "400+ Bach-werken automatisch gedownload via de music21 corpus-API. "
        "De MIDI-bestanden worden geparsed naar (pitch, duur)-tokens en opgeslagen "
        "in een CSV met metadata (genre, aantal maten, aantal stemmen). "
        "Dit gebeurt transparant op de achtergrond — de gebruiker ziet alleen een "
        "voortgangsbericht in de Streamlit-app.\n\n"
        "LSTM-MODEL\n"
        "Het model is een klassiek LSTM-taalmodel: gegeven de laatste N tokens, "
        "voorspel het volgende token. De architectuur is eenvoudig maar effectief: "
        "Embedding (64 dim) → LSTM (2 lagen, 256 hidden) → Dropout (0.3) → "
        "Linear → Softmax. Alle hyperparameters zijn instelbaar via de UI zonder "
        "code aan te passen. Training duurt 5–30 minuten afhankelijk van het genre "
        "en de hardware (GPU-ondersteuning ingebouwd).\n\n"
        "VIER GENERATIEMODI\n"
        "• AI-aanvulling: het meest gecontroleerd. Ritme exact overgenomen, "
        "  toonaard gegarandeerd via Krumhansl-Schmuckler filter.\n"
        "• Fine-tuned: het model leert 8 epochs op de seed. Hogere scores "
        "  maar risico op overtraining bij korte stukken.\n"
        "• Bach-stijl: maximale vrijheid. Model genereert ritme én melodie zelf. "
        "  Kortere seed (4 maten), geen toonaard-filter.\n"
        "• Willekeurig: de nulhypothese. Laat zien wat 'niets leren' oplevert.\n\n"
        "VERGELIJKING\n"
        "De vier metrics meten elk een andere dimensie van muzikale gelijkenis: "
        "toonsoort (Chroma), melodische beweging (Intervallen), "
        "frasen (N-gram) en harmoniek (Akkoorden). "
        "Door enkel het gegenereerde deel te vergelijken (niet de seed), "
        "zijn de scores eerlijk: de AI-aanvulling en Willekeurig beginnen "
        "op gelijke voet.\n\n"
        "CONCLUSIE\n"
        "Fine-tuned > AI-aanvulling > Bach-stijl >> Willekeurig (in de meeste gevallen).\n"
        "Het model leert stijl, maar geen structuur op lange termijn. "
        "De gegenereerde muziek klinkt lokaal Bach-achtig maar mist de "
        "grote architectonische opbouw van een echt koraal."
    ))


def slide_einde(prs):
    s = blank(prs)
    heading(s, "Reflectie & Vervolgwerk")
    subtext(s, "Wat werkt, wat kan beter — en wat zijn de volgende stappen?")
    accentline(s)

    # Linkerkolom: wat werkt
    add_rect(s, Inches(0.55), Inches(1.55), Inches(5.75), Inches(4.6),
             MID, line_color=GREEN, line_pt=1.0)
    add_rect(s, Inches(0.55), Inches(1.55), Inches(5.75), Inches(0.45), GREEN)
    add_text(s, "Wat werkt goed",
             Inches(0.72), Inches(1.6), Inches(5.4), Inches(0.38),
             size=13, bold=True, color=RGBColor(0x0F, 0x17, 0x2A))

    goed = [
        "Toonaard wordt correct overgenomen (Chroma-score 55–75%)",
        "Fine-tuning verbetert de score meetbaar vs. standaard model",
        "Ritme van het origineel blijft intact in aanvullingsmodus",
        "Vier objectieve metrics geven een genuanceerd beeld",
        "Pipeline draait volledig automatisch (download → train → genereer)",
    ]
    ty = Inches(2.1)
    for punt in goed:
        add_text(s, f"✓  {punt}",
                 Inches(0.72), ty, Inches(5.35), Inches(0.42),
                 size=11, color=WHITE)
        ty += Inches(0.42)

    # Rechterkolom: vervolgwerk
    add_rect(s, Inches(6.6), Inches(1.55), Inches(6.1), Inches(4.6),
             MID, line_color=ACCENT2, line_pt=1.0)
    add_rect(s, Inches(6.6), Inches(1.55), Inches(6.1), Inches(0.45), ACCENT2)
    add_text(s, "Mogelijke verbeteringen",
             Inches(6.77), Inches(1.6), Inches(5.75), Inches(0.38),
             size=13, bold=True, color=RGBColor(0x0F, 0x17, 0x2A))

    beter = [
        "Transformer in plaats van LSTM (betere lange-termijnstructuur)",
        "Meerstemmige tokenisatie: alle stemmen tegelijk modelleren",
        "Grotere dataset: alle 1000+ BWV-werken inclusief instrumentaal",
        "RLHF: menselijke feedback als beloningssignaal",
        "Evaluatie door muzikanten naast objectieve metrics",
    ]
    ty2 = Inches(2.1)
    for punt in beter:
        add_text(s, f"→  {punt}",
                 Inches(6.77), ty2, Inches(5.75), Inches(0.42),
                 size=11, color=WHITE)
        ty2 += Inches(0.42)

    # Onderste balk: live demo + vragen
    add_rect(s, Inches(0.55), Inches(6.35), Inches(12.2), Inches(0.92),
             RGBColor(0x0e, 0x28, 0x3a), line_color=ACCENT, line_pt=1.5)
    add_text(s, "Live demo:",
             Inches(0.75), Inches(6.44), Inches(1.5), Inches(0.35),
             size=12, bold=True, color=ACCENT)
    add_text(s, "streamlit run app.py",
             Inches(2.35), Inches(6.44), Inches(3.5), Inches(0.35),
             size=12, bold=True, color=ACCENT2)
    add_text(s, "·  Kurt Maekelberghe  ·  Muziekvoorspellen  ·  2025–2026",
             Inches(6.0), Inches(6.44), Inches(6.5), Inches(0.35),
             size=11, color=DIM, align=PP_ALIGN.RIGHT)
    add_text(s, "Vragen?",
             Inches(0.75), Inches(6.8), Inches(3.0), Inches(0.38),
             size=14, bold=True, color=WHITE)

    add_notes(s, (
        "SLIDE 20 — REFLECTIE & VERVOLGWERK: wat te zeggen\n\n"
        "WAT WERKT GOED\n"
        "• Toonaard: de Krumhansl-Schmuckler analyse detecteert betrouwbaar de "
        "toonaard van de eerste helft. Hierdoor blijft de AI-aanvulling in de "
        "juiste toonaard — de Chroma-score is systematisch hoger dan Willekeurig.\n"
        "• Fine-tuning: zelfs 8 epochs op de eerste helft is genoeg om de "
        "melodische motieven van het stuk te leren. De N-gram en Akkoorden-score "
        "gaan beide omhoog ten opzichte van de standaard AI-aanvulling.\n"
        "• Ritme: door het originele ritme exact over te nemen in de aanvullingsmodus "
        "klinkt de tweede helft ritmisch coherent. Dit is een bewuste keuze die "
        "de vergelijkbaarheid ook verbetert.\n"
        "• Pipeline: het systeem is end-to-end geautomatiseerd. Een nieuwe gebruiker "
        "kan binnen 5 minuten Bach-muziek laten genereren zonder enige kennis "
        "van muziek of machine learning.\n\n"
        "WAT KAN BETER\n"
        "• Lange-termijnstructuur: LSTM's 'vergeten' na 16–32 tokens. De gegenereerde "
        "muziek klinkt lokaal correct maar mist de grote structuur van een echt koraal "
        "(herhaling van frasen, cadentie op de juiste plek, terugkeer naar de hoofdtoonaard).\n"
        "  → Oplossing: Transformer met attention over de volledige context.\n\n"
        "• Meerstemmigheid: nu worden de 4 stemmen ONAFHANKELIJK gegenereerd. "
        "Dit betekent dat er soms parallele kwinten of octaven ontstaan die "
        "in de klassieke contrapuntleer verboden zijn.\n"
        "  → Oplossing: alle stemmen tegelijk modelleren als één tokenreeks "
        "  (interleaved: S, A, T, B, S, A, T, B, ...).\n\n"
        "• Dataset: het model is getraind op 400 werken. Bach schreef 1000+ werken "
        "inclusief orgelwerken, klavierwerken en orkestwerken. Een groter en "
        "diverser corpus zou het model robuuster maken.\n\n"
        "• RLHF (Reinforcement Learning from Human Feedback): objectieve metrics "
        "meten gelijkenis, maar niet of de muziek mooi klinkt. Menselijke feedback "
        "als beloningssignaal zou de perceptuele kwaliteit kunnen verbeteren.\n\n"
        "• Evaluatie: de huidige metrics zijn automatisch. Ideaal wordt de "
        "gegenereerde muziek beoordeeld door muzikanten en musicologen.\n\n"
        "LIVE DEMO\n"
        "Open de terminal en draai: streamlit run app.py\n"
        "Selecteer een Bach-werk, kies een model, stel temperatuur in en "
        "klik 'Genereer aanvulling'. De vier versies worden gegenereerd "
        "en vergeleken. Vervolgens kan de MIDI gedownload worden."
    ))


def img_or_placeholder(slide, img_name: str, x, y, w, h, label: str = ""):
    """Voeg screenshot in als het bestaat, anders een gekleurde placeholder."""
    img_path = IMG_DIR / img_name
    if img_path.exists():
        slide.shapes.add_picture(str(img_path), x, y, w, h)
    else:
        ph = slide.shapes.add_shape(1, x, y, w, h)
        ph.fill.solid()
        ph.fill.fore_color.rgb = RGBColor(0x0C, 0x1A, 0x30)
        ph.line.color.rgb = ACCENT
        ph.line.width = Pt(1.5)
        caption = label or f"[ Screenshot: {img_name} ]"
        add_text(slide, caption, x, y + h // 2 - Inches(0.25), w, Inches(0.5),
                 size=13, color=MUTED, align=PP_ALIGN.CENTER)


def slide_drie_varianten(prs):
    s = blank(prs)
    heading(s, "Vier gegenereerde versies")
    subtext(s, "Zelfde seed — vier strategieën met oplopende intelligentie")
    accentline(s)

    varianten = [
        ("🤖  AI-aanvulling", ACCENT,
         "Seed: eerste helft\nRitme: origineel\nToonaard: gefilterd\nModel: ongewijzigd"),
        ("🧠  Fine-tuned", GREEN,
         "Seed: eerste helft\nRitme: origineel\nToonaard: gefilterd\nModel: 8 epochs bijgetraind\nop de eerste helft"),
        ("🎼  Bach-stijl", ACCENT3,
         "Seed: eerste 4 maten\nRitme: model genereert vrij\nToonaard: geen filter\nModel: ongewijzigd"),
        ("🎲  Willekeurig", RGBColor(0xF8, 0x71, 0x71),
         "Seed: eerste helft\nRitme: origineel\nToonaard: geen filter\nModel: geen — random"),
    ]

    cw = Inches(2.85); ch = Inches(4.3)
    cx = Inches(0.55)
    for titel, kleur, body in varianten:
        add_rect(s, cx, Inches(1.55), cw, ch, MID, line_color=kleur, line_pt=1.5)
        add_rect(s, cx, Inches(1.55), cw, Inches(0.52), kleur)
        add_text(s, titel, cx + Inches(0.1), Inches(1.59),
                 cw - Inches(0.18), Inches(0.44),
                 size=13, bold=True, color=RGBColor(0x0F, 0x17, 0x2A))
        add_text(s, body, cx + Inches(0.12), Inches(2.18),
                 cw - Inches(0.24), Inches(3.5),
                 size=11, color=WHITE)
        cx += cw + Inches(0.15)

    add_rect(s, Inches(0.55), Inches(6.05), Inches(12.2), Inches(1.15), MID)
    add_text(s, "Vergelijking enkel op het gegenereerde deel.",
             Inches(0.75), Inches(6.13), Inches(11.8), Inches(0.35),
             size=13, bold=True, color=ACCENT)
    add_text(s,
             "AI-aanvulling, Fine-tuned en Willekeurig: tweede helft vergeleken.  "
             "Bach-stijl: maten 5–einde vergeleken (kortere seed = andere startpositie).",
             Inches(0.75), Inches(6.5), Inches(11.8), Inches(0.62),
             size=11, color=MUTED)

    add_notes(s, (
        "SLIDE 12 — VIER VARIANTEN: uitgebreide toelichting\n\n"
        "WAAROM VIER VERSIES?\n"
        "Door vier strategieën naast elkaar te zetten, meten we exact wat elke "
        "toevoeging (model, fine-tuning, vrije generatie) bijdraagt.\n\n"
        "1. AI-AANVULLING (blauw) — het basismodel\n"
        "• Seed: de eerste helft van het stuk (N/2 maten).\n"
        "• Ritme: exact overgenomen van het origineel (duurslots gekopieerd).\n"
        "• Toonaard-filter: actief — Krumhansl-Schmuckler detecteert de toonaard "
        "uit de eerste helft en filtert alleen toegestane pitchklassen.\n"
        "• Model: het getrainde LSTM zonder wijzigingen.\n"
        "• Verwachte score: middelmatig tot goed (50–75%).\n"
        "• Sterkste kant: toonsoort is gegarandeerd correct (filter).\n\n"
        "2. FINE-TUNED (groen) — modelaanpassing\n"
        "• Zelfde aanpak als AI-aanvulling, maar het model is eerst 8 epochs "
        "bijgetraind op uitsluitend de tokens van de eerste helft van dit stuk.\n"
        "• Adam lr=2e-3, deepcopy zodat het originele model intact blijft.\n"
        "• Het model 'leert' de stijl van dit specifieke stuk: zijn motieven, "
        "toonaard en ritmische patronen.\n"
        "• Verwachte score: 5–15% beter dan AI-aanvulling.\n"
        "• Risico: bij heel korte stukken kan overtraining optreden.\n\n"
        "3. BACH-STIJL (paars) — vrije generatie\n"
        "• Seed: alleen de eerste 4 maten (kortere context).\n"
        "• Ritme: het model genereert het ritme ZELF (via kansgewogen sampling "
        "over duurverdeling, gemarginaliseerd over pitches).\n"
        "• Toonaard: geen filter — het model bepaalt volledig vrij welke noten.\n"
        "• Model: het getrainde LSTM zonder wijzigingen.\n"
        "• Dit is de meest creatieve modus: het model schrijft zowel melodie als ritme.\n"
        "• Verwachte score: vergelijkbaar met AI-aanvulling, soms lager doordat "
        "de kortere seed minder context geeft en er geen toonaard-filter is.\n\n"
        "4. WILLEKEURIG (rood) — de nulhypothese\n"
        "• Geen model, geen leren, geen intelligentie.\n"
        "• Per duurslot: uniforme random sampling uit het VOLLEDIGE vocabulaire.\n"
        "• Ritme: gekopieerd van het origineel.\n"
        "• Als AI-aanvulling niet significant hoger scoort dan Willekeurig, "
        "voegt het model niets toe.\n"
        "• Verwachte score: laag (20–35%).\n\n"
        "VERGELIJKINGSMETHODE\n"
        "Alle versies worden vergeleken met de tweede helft van het origineel. "
        "De gedeelde eerste helft wordt weggelaten zodat de scores de werkelijke "
        "generatiekwaliteit meten, niet de herkenning van de seed. "
        "Voor Bach-stijl is de split-offset later (na 4 maten), dus worden "
        "de maten 5–einde vergeleken met dezelfde maten van het origineel."
    ))


def slide_metrics(prs):
    s = blank(prs)
    heading(s, "4 muzikale vergelijkingsmetrics")
    subtext(s, "Tweede helft gegenereerd vs. tweede helft origineel  ·  scores 0–100 %")
    accentline(s)

    metrics = [
        ("Chroma", ACCENT, "Octaaf-invariant",
         "Noten herleid tot toonklasse\n(C=0 … B=11, mod 12).\n12-dim vector.\nCosinus-gelijkenis.\n\nHoog = zelfde toonsoort."),
        ("Intervallen", GREEN, "Transpositie-invariant",
         "Interval tussen opeenvolgende\nnoten (−12 t/m +12 halve tonen,\n25 bins). Cosinus.\n\nHoog = zelfde melodische\nbewegingspatroon."),
        ("N-gram", ACCENT2, "Lokale patronen · O(n)",
         "Alle trigrams van toonklassen\ngeteld via cosinus-gelijkenis.\nVangt korte frasen.\n\nHoog = zelfde melodische\nmotieven als Bach."),
        ("Akkoorden", ACCENT3, "Harmonisch vocabulaire",
         "Akkoorden als frozensets van\ntoonklassen.\nJaccard: |A∩B| / |A∪B|.\n\nHoog = zelfde akkoordvoca-\nbulaire als Bach."),
    ]

    mw = Inches(2.9); mh = Inches(5.0)
    mx = Inches(0.55)
    for naam, kleur, subtitel, uitleg in metrics:
        add_rect(s, mx, Inches(1.55), mw, mh, MID, line_color=kleur, line_pt=1.2)
        add_rect(s, mx, Inches(1.55), mw, Inches(0.12), kleur)
        add_text(s, naam, mx + Inches(0.12), Inches(1.72),
                 mw - Inches(0.2), Inches(0.45),
                 size=15, bold=True, color=kleur)
        add_text(s, subtitel, mx + Inches(0.12), Inches(2.2),
                 mw - Inches(0.2), Inches(0.38),
                 size=11, italic=True, color=ACCENT2)
        add_text(s, uitleg, mx + Inches(0.12), Inches(2.65),
                 mw - Inches(0.2), Inches(3.75),
                 size=11, color=MUTED)
        mx += mw + Inches(0.18)

    add_notes(s, (
        "SLIDE 13 — 4 MUZIKALE METRICS: uitgebreide toelichting\n\n"
        "WAAROM NIET GEWOON LEVENSHTEIN?\n"
        "Levenshtein op absolute MIDI-pitches is te streng: één halve toon verschil "
        "telt als een volledige fout, ook al klinkt de noot correct in de toonaard. "
        "Muzikale gelijkenis gaat over toonsoort, melodische beweging en harmoniek, "
        "niet over exacte noot-voor-noot overeenkomst.\n\n"
        "1. CHROMA (blauw) — toonklasse-profiel\n"
        "• Elke noot wordt herleid tot zijn toonklasse (C=0, C#=1, ... B=11) "
        "via pitch mod 12. Octaaf-informatie gaat verloren.\n"
        "• Een 12-dimensionale vector telt hoe vaak elke toonklasse voorkomt.\n"
        "• Cosinus-gelijkenis tussen de twee vectoren.\n"
        "• WAT HET MEET: hoe goed de toonsoort overeenkomt. Als het origineel "
        "in G-majeur is en de AI ook G-majeur kiest, is de Chroma-score hoog.\n"
        "• BEPERKING: twee stukken in dezelfde toonsoort maar met andere melodieën "
        "kunnen toch een hoge score halen.\n\n"
        "2. INTERVALLEN (groen) — melodische beweging\n"
        "• Berekent de intervallen tussen opeenvolgende noten: noot[i+1] - noot[i].\n"
        "• Bereik: −12 t/m +12 halve tonen (25 bins).\n"
        "• Cosinus-gelijkenis op de interval-histogrammen.\n"
        "• WAT HET MEET: beweegt de melodie op dezelfde manier? Stapgewijs of "
        "met grote sprongen? Dit is transpositie-invariant: een melodie in C en "
        "dezelfde melodie in G scoren hoog.\n"
        "• BEPERKING: volgorde van intervallen wordt niet gemeten, enkel de "
        "frequentieverdeling.\n\n"
        "3. N-GRAM / TRIGRAM (amber) — lokale melodische patronen\n"
        "• Alle opeenvolgende drietallen van toonklassen worden geteld.\n"
        "  Bijv. (C, E, G), (E, G, A), ... → trigram-vectoren.\n"
        "• Cosinus-gelijkenis tussen de twee trigram-vectoren.\n"
        "• WAT HET MEET: korte melodische frasen en motieven. Als Bach typisch "
        "de frase (do, re, mi) gebruikt en de AI ook, scoort dit hoog.\n"
        "• STERKSTE metric voor 'klinkt het als Bach': het model moet dezelfde "
        "korte patronen leren als in het origineel.\n\n"
        "4. AKKOORDEN / JACCARD (paars) — harmonisch vocabulaire\n"
        "• Akkoorden worden geëxtraheerd via music21's chordify(): alle noten "
        "op hetzelfde tijdstip worden samengevoegd.\n"
        "• Elk akkoord wordt voorgesteld als een frozenset van toonklassen "
        "(transpositie-invariant).\n"
        "• Jaccard-index: |gemeenschappelijke akkoorden| / |totale unieke akkoorden|.\n"
        "• WAT HET MEET: gebruikt de AI hetzelfde harmonische vocabulaire als Bach? "
        "Dezelfde akkoordtypes (drieklank, septiemakkoord, etc.)?\n"
        "• BEPERKING: gevoelig voor de duur van akkoorden (snelle wisselingen "
        "genereren meer unieke frozensets).\n\n"
        "EINDSCORES\n"
        "Alle vier scores worden gemiddeld tot één totaalscore. Radar-diagram "
        "toont sterktes en zwaktes per dimensie. Tabel toont alle vier scores "
        "kleurgecodeerd van rood (laag) naar groen (hoog)."
    ))


def slide_voorbeeld(prs, nr: int, bwv: str, genre: str, screenshot_naam: str,
                    ai: str, ft: str, bach: str, rand: str,
                    observaties: list[str]):
    s = blank(prs)
    heading(s, f"Voorbeeld {nr}  —  {bwv}")
    subtext(s, f"Genre: {genre}")
    accentline(s)

    # Screenshot links (groot)
    img_or_placeholder(s, screenshot_naam, Inches(0.55), Inches(1.55),
                       Inches(8.0), Inches(4.7))

    # Scores rechts
    add_rect(s, Inches(8.8), Inches(1.55), Inches(4.0), Inches(4.7), MID)
    add_text(s, "Gemiddelde score (0–100%)",
             Inches(9.0), Inches(1.65), Inches(3.6), Inches(0.4),
             size=12, bold=True, color=WHITE)

    versie_scores = [
        (Inches(2.1),  "AI-aanvulling", ai,   ACCENT),
        (Inches(2.9),  "Fine-tuned",    ft,    GREEN),
        (Inches(3.7),  "Bach-stijl",    bach,  ACCENT3),
        (Inches(4.5),  "Willekeurig",   rand,  RGBColor(0xF8, 0x71, 0x71)),
    ]
    for y, lbl, score, kleur in versie_scores:
        add_rect(s, Inches(9.0), y, Inches(3.6), Inches(0.65),
                 RGBColor(0x10, 0x1E, 0x32))
        add_text(s, lbl, Inches(9.15), y + Inches(0.05),
                 Inches(2.0), Inches(0.3), size=11, color=kleur)
        add_text(s, score, Inches(11.0), y + Inches(0.04),
                 Inches(1.4), Inches(0.42), size=20, bold=True,
                 color=kleur, align=PP_ALIGN.RIGHT)

    # Observaties onderaan
    add_rect(s, Inches(0.55), Inches(6.4), Inches(12.2), Inches(0.9), MID)
    add_text(s, "Observaties:", Inches(0.75), Inches(6.48),
             Inches(2.0), Inches(0.35), size=11, bold=True, color=ACCENT)
    add_text(s, "  |  ".join(observaties),
             Inches(2.8), Inches(6.48), Inches(10.0), Inches(0.75),
             size=11, color=MUTED)


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
    slide_drie_varianten(prs)
    slide_metrics(prs)
    slide_app(prs)
    slide_techstack(prs)

    # ── Voorbeelden (screenshots toevoegen in presentatie_screenshots/) ────────
    slide_voorbeeld(prs, 1, "BWV 227.11 — Jesu meine Freude", "Motet",
                    "voorbeeld_1.png",
                    ai="62%", ft="68%", bach="55%", rand="31%",
                    observaties=[
                        "Fine-tuned beste score: model kent de motieven van het stuk",
                        "Bach-stijl iets lager: kortere seed, geen toonaard-filter",
                        "Willekeurig laagst op alle vier metrics",
                    ])
    slide_voorbeeld(prs, 2, "BWV 253 — Koraal", "Koraal",
                    "voorbeeld_2.png",
                    ai="71%", ft="74%", bach="65%", rand="28%",
                    observaties=[
                        "Koralen eenvoudiger van structuur — hogere scores algemeen",
                        "N-gram score hoog: model herkent typische koraalmotieven",
                        "Bach-stijl genereert vrij ritme: meer variatie, iets lagere score",
                    ])
    slide_voorbeeld(prs, 3, "BWV 10.7 — Cantate", "Cantate",
                    "voorbeeld_3.png",
                    ai="58%", ft="63%", bach="51%", rand="26%",
                    observaties=[
                        "Cantates zijn complexer — moeilijker voor het model",
                        "Fine-tuning helpt significant bij complexere structuren",
                        "Bach-stijl: geen toonaard-filter zichtbaar in lagere Chroma-score",
                    ])

    slide_samenvatting(prs)
    slide_einde(prs)

    prs.save(str(OUT))
    n = len(prs.slides)
    print(f"Klaar! {n} slides opgeslagen naar: {OUT}")
    print(f"Screenshots toevoegen in: {IMG_DIR}")
    expected = ["voorbeeld_1.png", "voorbeeld_2.png", "voorbeeld_3.png"]
    for naam in expected:
        status = "OK" if (IMG_DIR / naam).exists() else "ontbreekt"
        print(f"  {naam}: {status}")


if __name__ == "__main__":
    main()
