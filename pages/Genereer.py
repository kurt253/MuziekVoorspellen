"""Muziek genereren — kies model + song, AI vult de tweede helft van de maten aan."""
from __future__ import annotations

import base64
import glob
import importlib.metadata
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ── Paden ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
ANALYSE_CSV  = DATA_DIR / "processed" / "bach_analyse.csv"
METADATA_CSV = DATA_DIR / "processed" / "bach_metadata.csv"
MODEL_DIR    = DATA_DIR / "processed"
RAW_DIR      = DATA_DIR / "raw" / "bach"

DUREN = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# VoiceToken = (midi_pitch, duur)  pitch=0 → rust
VoiceToken  = tuple[int, float]
# ChordToken = (frozenset van MIDI-noten, duur)  — alleen voor MIDI-uitvoer
ChordToken  = tuple[frozenset[int], float]

_MIDI_CDN = (
    "https://cdn.jsdelivr.net/combine/"
    "npm/tone@14,"
    "npm/@magenta/music@1.23.1/es6/core.js,"
    "npm/html-midi-player@1.5.0"
)


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, vocab: int, embed_dim: int = 64, hidden: int = 256,
                 lagen: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden, num_layers=lagen,
                                  batch_first=True,
                                  dropout=dropout if lagen > 1 else 0.0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ── Data helpers ──────────────────────────────────────────────────────────────
def _kwantiseer_duur(duur: float) -> float:
    return min(DUREN, key=lambda d: abs(d - duur))


@st.cache_data
def _laad_data() -> pd.DataFrame:
    analyse  = pd.read_csv(ANALYSE_CSV)
    metadata = pd.read_csv(METADATA_CSV)
    metadata["midi_pad"] = metadata["midi_pad"].apply(lambda p: str(DATA_DIR / p))
    return (
        analyse.merge(metadata, on="naam", how="inner")
        .drop(columns=["bestandsnaam"], errors="ignore")
    )


def _beschikbare_modellen() -> list[Path]:
    return sorted(MODEL_DIR.glob("lstm_*.pt"))


def _model_label(pad: Path) -> str:
    m = re.match(r"lstm_(.+)_(\d+)\.pt$", pad.name)
    if m:
        soort = m.group(1).replace("_", " ").strip()
        return f"{soort}  —  {m.group(2)} werken  ({pad.name})"
    return pad.name


@st.cache_resource
def _laad_model_en_vocab(model_pad_str: str) -> tuple[LSTMModel, dict[VoiceToken, int]]:
    model_pad = Path(model_pad_str)
    vocab_pad = model_pad.with_name(model_pad.stem + "_vocab.pkl")

    with open(vocab_pad, "rb") as f:
        vocab: dict[VoiceToken, int] = pickle.load(f)

    state     = torch.load(str(model_pad), map_location="cpu")
    embed_dim = state["embedding.weight"].shape[1]
    hidden    = state["fc.weight"].shape[1]
    lagen     = sum(1 for k in state if re.match(r"lstm\.weight_ih_l\d+$", k))

    model = LSTMModel(len(vocab), embed_dim=embed_dim, hidden=hidden, lagen=lagen)
    model.load_state_dict(state)
    model.eval()
    return model, vocab


# ── MIDI / token conversie ────────────────────────────────────────────────────
def _midi_naar_stemmen_per_maat(pad: str) -> list[list[list[VoiceToken]]]:
    """Geef per maat, per stem een lijst VoiceTokens.

    Returns
    -------
    list[maat_index] → list[stem_index] → list[VoiceToken]
    """
    from music21 import converter, note, chord

    score = converter.parse(pad)
    n_maten = max(
        len(list(part.getElementsByClass("Measure"))) for part in score.parts
    )

    per_maat: list[list[list[VoiceToken]]] = [[] for _ in range(n_maten)]

    for part in score.parts:
        maten = list(part.getElementsByClass("Measure"))
        for maat_i, maat in enumerate(maten):
            stem_tokens: list[VoiceToken] = []
            for el in maat.notesAndRests:
                duur = _kwantiseer_duur(float(el.duration.quarterLength))
                if duur == 0:
                    continue
                if isinstance(el, note.Rest):
                    stem_tokens.append((0, duur))
                elif isinstance(el, note.Note):
                    stem_tokens.append((el.pitch.midi, duur))
                elif isinstance(el, chord.Chord):
                    stem_tokens.append((max(p.midi for p in el.pitches), duur))
            if stem_tokens:
                per_maat[maat_i].append(stem_tokens)

    return [m for m in per_maat if m]   # lege maten weggooien


def _combineer_stemmen_naar_akkoorden(
    stemmen: list[list[VoiceToken]],
) -> list[ChordToken]:
    """Combineer per-stem VoiceTokens naar ChordTokens via tijdsuitlijning.

    Stap 1: bereken (start, einde, pitch) per noot in elke stem.
    Stap 2: verzamel alle unieke tijdstippen.
    Stap 3: voor elk interval, verzamel actieve noten → ChordToken.
    """
    if not stemmen:
        return []

    events: list[tuple[float, float, int]] = []
    for stem in stemmen:
        t = 0.0
        for pitch, duur in stem:
            events.append((t, t + duur, pitch))
            t += duur

    tijdstippen = sorted({e[0] for e in events} | {e[1] for e in events})

    resultaat: list[ChordToken] = []
    for i in range(len(tijdstippen) - 1):
        t0  = tijdstippen[i]
        t1  = tijdstippen[i + 1]
        dur = _kwantiseer_duur(t1 - t0)
        if dur == 0:
            continue
        actief = frozenset(
            p for (s, e, p) in events if s <= t0 and t0 < e and p != 0
        )
        resultaat.append((actief, dur))

    return resultaat


def _chord_tokens_naar_midi_b64(tokens: list[ChordToken]) -> str:
    from music21 import stream, note, chord
    from music21.midi import translate as midi_translate

    part   = stream.Part()
    offset = 0.0
    for noten, duur in tokens:
        if not noten:
            el = note.Rest(quarterLength=duur)
        elif len(noten) == 1:
            el = note.Note(next(iter(noten)), quarterLength=duur)
        else:
            el = chord.Chord(sorted(noten), quarterLength=duur)
        part.insert(offset, el)
        offset += duur

    mf = midi_translate.music21ObjectToMidiFile(stream.Score([part]))
    return base64.b64encode(mf.writestr()).decode()


def _tempo_uit_midi(midi_pad: str) -> float:
    """Geeft het eerste tempomarkering in BPM, standaard 120."""
    from music21 import converter
    try:
        score  = converter.parse(midi_pad)
        tempos = list(score.flatten().getElementsByClass("MetronomeMark"))
        if tempos:
            return float(tempos[0].number)
    except Exception:
        pass
    return 120.0


def _eerste_helft_plus_ai_b64(
    midi_pad: str,
    helft_maten: int,
    ai_stemmen: list[list[VoiceToken]],
) -> str:
    """Originele eerste helft (alle stemmen) + elke AI-stem als aparte part.

    Elke stem behoudt zijn onafhankelijke ritmische lijn; ze worden NIET
    samengeperst tot één akkoordstroom.
    """
    import copy
    from music21 import converter, stream, note
    from music21.midi import translate as midi_translate

    score    = converter.parse(midi_pad)
    ch_maten = list(score.chordify().getElementsByClass("Measure"))

    split_offset = (
        float(ch_maten[helft_maten].offset)
        if helft_maten < len(ch_maten)
        else float(ch_maten[-1].offset) + float(ch_maten[-1].barDuration.quarterLength)
    )

    gecombineerd = stream.Score()

    # Originele stemmen: eerste helft bewaard
    for part in score.parts:
        nieuwe_part = stream.Part()
        for el in part.flatten().notesAndRests:
            if float(el.offset) < split_offset:
                nieuwe_part.insert(float(el.offset), copy.deepcopy(el))
        gecombineerd.append(nieuwe_part)

    # AI-stemmen: elke stem als eigen part met eigen ritme
    for stem_tokens in ai_stemmen:
        ai_part    = stream.Part()
        cur_offset = split_offset
        for pitch, duur in stem_tokens:
            if pitch == 0:
                el = note.Rest(quarterLength=duur)
            else:
                el = note.Note(pitch, quarterLength=duur)
            ai_part.insert(cur_offset, el)
            cur_offset += duur
        gecombineerd.append(ai_part)

    mf = midi_translate.music21ObjectToMidiFile(gecombineerd)
    return base64.b64encode(mf.writestr()).decode()


# ── Generatie — fase 1: ritme ─────────────────────────────────────────────────
def _genereer_ritme(
    model: LSTMModel,
    seed_codes: list[int],
    doelduur: float,
    duur_naar_codes: dict[float, list[int]],
    duuren: list[float],
    venster: int,
    temperatuur: float,
    top_k: int = 0,
) -> list[float]:
    """Genereer een onafhankelijke duurvolgorde voor één stem.

    P(duur | context) = Σ_pitch  P(pitch, duur | context)
    Elke stem krijgt zo zijn eigen ritmisch karakter.
    """
    ctx         = list(seed_codes)
    duur_seq:   list[float] = []
    totaal_duur = 0.0
    max_stappen = max(500, int(doelduur / 0.25) * 10)

    model.eval()
    with torch.no_grad():
        for _ in range(max_stappen):
            if totaal_duur >= doelduur:
                break

            inp = ctx[-venster:]
            if len(inp) < venster:
                inp = [0] * (venster - len(inp)) + inp
            logits = model(torch.tensor([inp], dtype=torch.long))[0]
            if top_k > 0:
                grens  = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < grens, float("-inf"))
            probs = torch.softmax(logits / max(temperatuur, 1e-6), dim=0)

            # Marginaliseer: P(duur) = som over alle pitches
            duur_w = torch.tensor([
                sum(probs[c].item() for c in duur_naar_codes[d])
                for d in duuren
            ])
            duur_w = torch.clamp(duur_w, min=1e-9)
            duur_w /= duur_w.sum()
            gekozen = duuren[torch.multinomial(duur_w, 1).item()]

            duur_seq.append(gekozen)
            totaal_duur += gekozen
            # Context bijwerken met meest waarschijnlijke pitch voor die duur
            codes = duur_naar_codes[gekozen]
            ctx.append(max(codes, key=lambda c: probs[c].item()))

    return duur_seq


# ── Generatie — fase 2: melodie op vastgelegd ritme ───────────────────────────
def _genereer_melodie(
    model: LSTMModel,
    seed_codes: list[int],
    duur_seq: list[float],
    duur_naar_codes: dict[float, list[int]],
    inv_vocab: dict[int, VoiceToken],
    venster: int,
    temperatuur: float,
    top_k: int = 0,
) -> list[VoiceToken]:
    """Genereer pitches voor één stem op een vastgelegd ritmeschema.

    Per duurslot worden alleen tokens met die exacte duur gesampled.
    """
    ctx       = list(seed_codes)
    resultaat: list[VoiceToken] = []

    model.eval()
    with torch.no_grad():
        for duur_slot in duur_seq:
            inp = ctx[-venster:]
            if len(inp) < venster:
                inp = [0] * (venster - len(inp)) + inp
            logits = model(torch.tensor([inp], dtype=torch.long))[0]
            if top_k > 0:
                grens  = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < grens, float("-inf"))
            probs = torch.softmax(logits / max(temperatuur, 1e-6), dim=0)

            codes   = duur_naar_codes[duur_slot]
            pitch_w = torch.clamp(
                torch.tensor([probs[c].item() for c in codes]), min=1e-9
            )
            pitch_w /= pitch_w.sum()
            gekozen = codes[torch.multinomial(pitch_w, 1).item()]

            ctx.append(gekozen)
            resultaat.append(inv_vocab[gekozen])

    return resultaat


# ── Piano-roll HTML ───────────────────────────────────────────────────────────
def _piano_roll_html(
    midi_b64: str,
    hoogte: int = 280,
    uid: str = "a",
    maat_starts_s: list[float] | None = None,
    totaal_maten: int = 0,
    helft_maten: int = 0,
    is_ai: bool = False,
) -> str:
    import json as _json
    pid = f"rol_{uid}"

    # Maatenteller-blok (alleen als maatinfo beschikbaar is)
    if maat_starts_s:
        starts_js    = _json.dumps(maat_starts_s)
        kleur_orig   = "#38bdf8"   # blauw  = originele maten
        kleur_ai     = "#f59e0b"   # geel   = AI-maten
        label_orig   = "origineel"
        label_ai     = "AI"
        maat_block = f"""
  <div id="maat-{uid}" style="
    font-family:monospace; font-size:14px; font-weight:600;
    color:{kleur_orig}; background:#0f172a;
    padding:6px 12px; border-radius:6px 6px 0 0;
    min-height:28px; display:flex; align-items:center; gap:8px;
  ">
    <span id="maat-nr-{uid}">♩</span>
    <span id="maat-label-{uid}" style="font-weight:400;font-size:12px;opacity:.7"></span>
  </div>
  <script>
  (function(){{
    const starts  = {starts_js};
    const totaal  = {totaal_maten};
    const helft   = {helft_maten};
    const isAi    = {'true' if is_ai else 'false'};
    const nr      = document.getElementById('maat-nr-{uid}');
    const lbl     = document.getElementById('maat-label-{uid}');
    const wrapper = document.getElementById('maat-{uid}');

    function updateMaat() {{
      const player = document.querySelector('midi-player');
      if (!player) return;
      const t = player.currentTime || 0;
      let m = 0;
      for (let i = starts.length - 1; i >= 0; i--) {{
        if (t >= starts[i]) {{ m = i + 1; break; }}
      }}
      if (m === 0) {{ nr.textContent = '♩'; lbl.textContent = ''; return; }}
      nr.textContent = `Maat ${{m}} / ${{totaal}}`;
      if (isAi) {{
        const isOrigHalf = m <= helft;
        wrapper.style.color  = isOrigHalf ? '{kleur_orig}' : '{kleur_ai}';
        lbl.textContent      = isOrigHalf ? '({label_orig})' : '({label_ai})';
      }} else {{
        wrapper.style.color = '{kleur_orig}';
        lbl.textContent     = '({label_orig})';
      }}
    }}
    setInterval(updateMaat, 100);
  }})();
  </script>"""
    else:
        maat_block = ""

    return f"""<!DOCTYPE html><html><head>
<script src="{_MIDI_CDN}"></script>
<style>
  body {{ margin:0; padding:0; background:transparent; }}
  midi-player {{
    display:block; width:100%; margin-bottom:6px;
    --player-background-color:#1e293b;
    --player-button-color:#38bdf8;
    --player-color:#e2e8f0;
  }}
  midi-visualizer {{
    display:block; width:100%; height:{hoogte}px;
    border-radius:0 0 8px 8px; overflow:hidden; background:#0f172a;
  }}
  midi-visualizer svg rect.note {{ rx:2; ry:2; opacity:0.85; }}
</style>
</head><body>
  <midi-player src="data:audio/midi;base64,{midi_b64}"
    sound-font visualizer="#{pid}" style="width:100%;"></midi-player>
  {maat_block}
  <midi-visualizer type="piano-roll" id="{pid}"
    src="data:audio/midi;base64,{midi_b64}"></midi-visualizer>
</body></html>"""


# ── Scherm ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Genereren", page_icon="🎵", layout="wide")
    st.title("🎵 Muziek genereren")

    if not ANALYSE_CSV.exists():
        st.error(f"Analyse-CSV niet gevonden: `{ANALYSE_CSV}`")
        return

    # ── 1. Model kiezen ───────────────────────────────────────────────────────
    st.subheader("1 · Model")
    modellen = _beschikbare_modellen()
    if not modellen:
        st.warning(
            "Geen modellen gevonden in `data/processed/`.  "
            "Train eerst een model via de **Training**-pagina."
        )
        return

    col_mod, col_ven = st.columns([3, 1])
    with col_mod:
        model_keuze = st.selectbox(
            "Kies model",
            options=modellen,
            format_func=_model_label,
        )
    with col_ven:
        venster = st.number_input(
            "Venster",
            min_value=4, max_value=64, value=16, step=4,
            help="Moet overeenkomen met het venster gebruikt tijdens training (standaard 16)",
        )

    # ── 2. Song kiezen ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("2 · Song kiezen")

    df       = _laad_data()
    tellers  = df["classificatie"].value_counts()
    all_types = ["Allemaal"] + sorted(tellers.index.tolist())

    type_filter = st.selectbox(
        "Filter",
        all_types,
        format_func=lambda t: (
            f"Allemaal  ({len(df)})" if t == "Allemaal"
            else f"{t}  ({tellers.get(t, 0)})"
        ),
        label_visibility="collapsed",
    )

    df_toon = (df if type_filter == "Allemaal"
               else df[df["classificatie"] == type_filter]).reset_index(drop=True)

    event = st.dataframe(
        df_toon[["naam", "classificatie", "maten", "aantal_stemmen"]],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "naam":           st.column_config.TextColumn("BWV",       width="small"),
            "classificatie":  st.column_config.TextColumn("Categorie", width="medium"),
            "maten":          st.column_config.NumberColumn("Maten",   width="small"),
            "aantal_stemmen": st.column_config.NumberColumn("St.",     width="small"),
        },
    )

    if not event.selection.rows:
        st.info("Klik op een rij om een song te selecteren.")
        return

    rij       = df_toon.iloc[event.selection.rows[0]]
    midi_pad  = str(rij["midi_pad"])
    song_naam = str(rij["naam"])
    song_maten = int(rij["maten"]) if pd.notna(rij.get("maten")) else "?"

    st.caption(
        f"Geselecteerd: **{song_naam}**  ·  {rij['classificatie']}  ·  {song_maten} maten"
    )

    # ── 3. Instellingen ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("3 · Instellingen")

    col_t, col_k = st.columns(2)
    with col_t:
        temperatuur = st.slider(
            "Temperatuur",
            min_value=0.1, max_value=2.0, value=0.9, step=0.1,
            help="Lager = voorspelbaarder/Bach-achtiger · Hoger = creatiever/willekeuriger",
        )
    with col_k:
        top_k = st.slider(
            "Top-k  (beste akkoorden)",
            min_value=0, max_value=50, value=10, step=5,
            help="0 = uit, anders alleen de k meest waarschijnlijke tokens. "
                 "Hogere waarde = meer variatie, lagere waarde = betere akkoorden",
        )

    # ── Genereer-knop ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("🎵 Genereer aanvulling", type="primary", use_container_width=True):
        _voer_generatie_uit(
            model_pad_str=str(model_keuze),
            midi_pad=midi_pad,
            song_naam=song_naam,
            venster=int(venster),
            temperatuur=float(temperatuur),
            top_k=int(top_k),
        )

    # ── Resultaat ─────────────────────────────────────────────────────────────
    if (
        "gen_origineel_b64" in st.session_state
        and st.session_state.get("gen_song") == song_naam
    ):
        _toon_resultaat()


def _voer_generatie_uit(
    model_pad_str: str,
    midi_pad: str,
    song_naam: str,
    venster: int,
    temperatuur: float,
    top_k: int = 0,
) -> None:
    with st.status("Muziek genereren…", expanded=True) as status:

        st.write("Model en vocabulaire laden…")
        try:
            model, vocab = _laad_model_en_vocab(model_pad_str)
        except FileNotFoundError as e:
            st.error(f"Vocab-bestand niet gevonden: {e}")
            return

        inv_vocab = {i: t for t, i in vocab.items()}

        st.write("Song inlezen en per maat + per stem splitsen…")
        try:
            per_maat = _midi_naar_stemmen_per_maat(midi_pad)
        except Exception as e:
            st.error(f"MIDI-bestand kon niet worden geladen: {e}")
            return

        if len(per_maat) < 4:
            st.error(f"Song heeft te weinig maten ({len(per_maat)}) om te splitsen.")
            return

        helft_maten = len(per_maat) // 2

        # Gedeelde opzoektabel voor beide fasen
        duur_naar_codes: dict[float, list[int]] = {}
        for (_, d), idx in vocab.items():
            duur_naar_codes.setdefault(d, []).append(idx)
        duuren = sorted(duur_naar_codes.keys())

        n_stemmen = max(len(m) for m in per_maat)

        # Bouw per-stem seeds en doelduren op
        stem_seeds:  list[list[int]]   = []
        stem_duuren: list[float]        = []
        stem_tweede: list[list[VoiceToken]] = []
        for stem_i in range(n_stemmen):
            eerste = [t for maat in per_maat[:helft_maten]
                      for t in (maat[stem_i] if stem_i < len(maat) else [])]
            tweede = [t for maat in per_maat[helft_maten:]
                      for t in (maat[stem_i] if stem_i < len(maat) else [])]
            stem_seeds.append([vocab[t] for t in eerste if t in vocab])
            stem_duuren.append(sum(d for _, d in tweede))
            stem_tweede.append(tweede)

        # ── FASE 1: ritme vastleggen per stem ─────────────────────────────────
        st.write("**Fase 1 — Ritme per stem vastleggen**")
        ritmes: list[list[float]] = []
        for stem_i in range(n_stemmen):
            seeds      = stem_seeds[stem_i]
            doelduur_s = stem_duuren[stem_i]
            if len(seeds) < venster or doelduur_s == 0:
                # Origineel ritme overnemen als fallback
                ritmes.append([d for _, d in stem_tweede[stem_i]])
                st.write(f"  Stem {stem_i + 1}: origineel ritme overgenomen")
                continue

            duur_seq = _genereer_ritme(
                model, seeds, doelduur_s,
                duur_naar_codes, duuren,
                venster, temperatuur, top_k,
            )
            ritmes.append(duur_seq)

            # Toon ritmeschema zodat afwijkingen per stem zichtbaar zijn
            schema = "  ".join(str(d) for d in duur_seq[:12])
            suffix = "…" if len(duur_seq) > 12 else ""
            st.write(
                f"  Stem {stem_i + 1}: `{schema}{suffix}`  "
                f"({len(duur_seq)} noten, {sum(duur_seq):.1f} kwartnootheden)"
            )

        # ── FASE 2: melodie per stem op het vastgelegde ritme ─────────────────
        st.write("**Fase 2 — Melodie per stem genereren**")
        ai_stemmen: list[list[VoiceToken]] = []
        for stem_i in range(n_stemmen):
            seeds = stem_seeds[stem_i]
            if len(seeds) < venster or not ritmes[stem_i]:
                ai_stemmen.append(stem_tweede[stem_i])
                continue

            st.write(f"  Stem {stem_i + 1}…")
            melodie = _genereer_melodie(
                model, seeds, ritmes[stem_i],
                duur_naar_codes, inv_vocab,
                venster, temperatuur, top_k,
            )
            ai_stemmen.append(melodie)

        st.write("MIDI bouwen — elke stem als aparte part…")
        with open(midi_pad, "rb") as f:
            origineel_b64 = base64.b64encode(f.read()).decode()

        ai_b64 = _eerste_helft_plus_ai_b64(midi_pad, helft_maten, ai_stemmen)

        # Maatstarttijden (voor maatenteller) op basis van origineel tempo
        bpm = _tempo_uit_midi(midi_pad)
        SPQ = 60.0 / bpm
        maat_starts_s: list[float] = []
        cumulatief = 0.0
        for maat in per_maat:
            maat_starts_s.append(round(cumulatief * SPQ, 4))
            if maat:
                for _, duur in maat[0]:    # gebruik stem 0 voor duurinfo
                    cumulatief += duur

        totaal_nieuwe = sum(len(s) for s in ai_stemmen)
        totaal_orig_2e = sum(
            len(maat[si] if si < len(maat) else [])
            for maat in per_maat[helft_maten:]
            for si in range(n_stemmen)
        )

        st.session_state.update({
            "gen_origineel_b64":   origineel_b64,
            "gen_ai_b64":          ai_b64,
            "gen_song":            song_naam,
            "gen_helft_maten":     helft_maten,
            "gen_totaal_maten":    len(per_maat),
            "gen_seed_tokens":     sum(len(maat[si] if si < len(maat) else [])
                                       for maat in per_maat[:helft_maten]
                                       for si in range(n_stemmen)),
            "gen_nieuwe_tokens":   totaal_nieuwe,
            "gen_orig_2e_tokens":  totaal_orig_2e,
            "gen_maat_starts_s":   maat_starts_s,
        })

        status.update(label="Klaar!", state="complete")


def _toon_resultaat() -> None:
    st.divider()

    hm   = st.session_state["gen_helft_maten"]
    tm   = st.session_state["gen_totaal_maten"]
    st_  = st.session_state["gen_seed_tokens"]
    nt   = st.session_state["gen_nieuwe_tokens"]
    ot   = st.session_state["gen_orig_2e_tokens"]

    # ── Maatenteller ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Totaal maten",          tm)
    c2.metric("Eerste helft (origineel)", f"maat 1 – {hm}")
    c3.metric("Tweede helft (AI)",     f"maat {hm + 1} – {tm}")
    c4.metric("Tokens gegenereerd",    f"{nt}  (orig: {ot})")

    # Visuele splitsbalk
    frac = hm / tm
    st.markdown(
        f"""
        <div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin:6px 0 14px">
          <div style="flex:{frac};background:#38bdf8;"></div>
          <div style="flex:{1-frac};background:#f59e0b;"></div>
        </div>
        <div style="display:flex;font-size:12px;color:#94a3b8;margin-bottom:10px">
          <div style="flex:{frac}">🔵 Origineel (maat 1–{hm})</div>
          <div style="flex:{1-frac};text-align:right">🟡 AI (maat {hm+1}–{tm})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Piano rolls ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    maat_starts = st.session_state.get("gen_maat_starts_s", [])

    with col1:
        st.subheader("🎼 Origineel")
        st.caption(f"Maten 1–{hm} · origineel  +  maten {hm+1}–{tm} · origineel")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["gen_origineel_b64"],
                hoogte=300, uid="orig",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=hm,
                is_ai=False,
            ),
            height=420, scrolling=False,
        )

    with col2:
        st.subheader("🤖 AI-aanvulling")
        st.caption(f"Maten 1–{hm} · origineel  +  maten {hm+1}–{tm} · door AI gegenereerd")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["gen_ai_b64"],
                hoogte=300, uid="ai",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=hm,
                is_ai=True,
            ),
            height=420, scrolling=False,
        )


main()
