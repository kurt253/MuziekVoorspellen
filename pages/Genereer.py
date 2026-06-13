"""Muziek genereren — kies model + song, AI vult de tweede helft van de maten aan."""
from __future__ import annotations

import base64
import copy
import glob
import importlib.metadata
import os
import pickle
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ── Paden ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
DATA_DIR     = PROJECT_ROOT / "data"
ANALYSE_CSV  = DATA_DIR / "processed" / "bach_analyse.csv"
METADATA_CSV = DATA_DIR / "processed" / "bach_metadata.csv"
MODEL_DIR    = DATA_DIR / "processed"
RAW_DIR      = DATA_DIR / "raw" / "bach"

# Toegestane nootdurations in kwartnootheden (zelfde rooster als tijdens training)
DUREN = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# Aantal seed-maten voor de Bach-stijl generatiemodus
N_SEED_MATEN = 4

# Type alias: een stem-token = (MIDI-pitch, duur); pitch 0 = rust
VoiceToken  = tuple[int, float]
# Type alias: een akkoord-token = (frozenset van pitches, duur) — alleen voor MIDI-uitvoer
ChordToken  = tuple[frozenset[int], float]

# CDN-bundel: Tone.js + Magenta + html-midi-player in één request
_MIDI_CDN = (
    "https://cdn.jsdelivr.net/combine/"
    "npm/tone@14,"
    "npm/@magenta/music@1.23.1/es6/core.js,"
    "npm/html-midi-player@1.5.0"
)


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """LSTM-taalmodel voor muziekgeneratie op token-niveau.

    Identieke architectuur als in Training.py zodat opgeslagen gewichten
    direct geladen kunnen worden zonder aanpassingen.

    Architectuur:
      Embedding(vocab, embed_dim) → LSTM(lagen, hidden) → Dropout → Linear(vocab)

    Parameters
    ----------
    vocab:
        Vocabulairegrootte (aantal unieke tokens).
    embed_dim:
        Dimensie van de tokenembedding (standaard 64).
    hidden:
        Aantal hidden units per LSTM-laag (standaard 256).
    lagen:
        Aantal gestapelde LSTM-lagen (standaard 2).
    dropout:
        Dropout-kans na LSTM en tussen lagen (bij lagen > 1).
    """

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
        """Voorwaartse pass: verwerk tokenreeksen naar logits.

        Parameters
        ----------
        x:
            Tensor van shape (batch, venster) met token-indices.

        Returns
        -------
        Logits-tensor van shape (batch, vocab_grootte).
        """
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ── Data helpers ──────────────────────────────────────────────────────────────

def _kwantiseer_duur(duur: float) -> float:
    """Rond een duur af naar de dichtstbijzijnde waarde uit het DUREN-rooster.

    Normaliseert micro-timings uit MIDI naar het vaste duratierooster zodat
    tokens overeenkomen met het vocabulaire dat tijdens training gebruikt werd.

    Parameters
    ----------
    duur:
        Originele duur in kwartnootheden.

    Returns
    -------
    Dichtstbijzijnde waarde uit DUREN.
    """
    return min(DUREN, key=lambda d: abs(d - duur))


@st.cache_data
def _laad_data() -> pd.DataFrame:
    """Laad en merge de analyse- en metadata-CSV's tot één werkend DataFrame.

    Combineert:
      - bach_analyse.csv: classificatie, maten, stemnamen
      - bach_metadata.csv: absoluut MIDI-pad

    Gecached zodat de CSV niet bij elke Streamlit-rerun opnieuw geladen wordt.
    """
    analyse  = pd.read_csv(ANALYSE_CSV)
    metadata = pd.read_csv(METADATA_CSV)
    metadata["midi_pad"] = metadata["midi_pad"].apply(lambda p: str(DATA_DIR / p))
    metadata = metadata.drop(columns=["maten", "stemmen"], errors="ignore")
    return (
        analyse.merge(metadata, on="naam", how="inner")
        .drop(columns=["bestandsnaam"], errors="ignore")
    )


def _beschikbare_modellen() -> list[Path]:
    """Geef een gesorteerde lijst van alle getrainde modelbestanden in MODEL_DIR.

    Zoekt naar bestanden die overeenkomen met het patroon lstm_*.pt.
    """
    return sorted(MODEL_DIR.glob("lstm_*.pt"))


def _model_label(pad: Path) -> str:
    """Zet een modelbestandsnaam om naar een leesbaar label voor de selectbox.

    Verwacht bestandsnamen van de vorm: lstm_<soort>_<n>.pt
    Geeft terug: "<soort>  —  <n> werken  (<bestandsnaam>)"

    Parameters
    ----------
    pad:
        Pad naar het .pt-modelbestand.
    """
    m = re.match(r"lstm_(.+)_(\d+)\.pt$", pad.name)
    if m:
        soort = m.group(1).replace("_", " ").strip()
        return f"{soort}  —  {m.group(2)} werken  ({pad.name})"
    return pad.name


@st.cache_resource
def _laad_model_en_vocab(model_pad_str: str) -> tuple[LSTMModel, dict[VoiceToken, int]]:
    """Laad een getraind LSTM-model en het bijbehorende vocabulaire van disk.

    Leidt de modelarchitectuur automatisch af uit de state-dict zodat er geen
    aparte configuratie opgeslagen hoeft te worden:
      - embed_dim:  uit de vorm van embedding.weight
      - hidden:     uit de vorm van fc.weight
      - lagen:      door te tellen hoeveel lstm.weight_ih_l<N> sleutels er zijn

    Gecached via st.cache_resource zodat het model niet bij elke generatie
    opnieuw van disk geladen wordt (is zwaar voor grote modellen).

    Parameters
    ----------
    model_pad_str:
        String-pad naar het .pt-modelbestand (string i.p.v. Path voor caching).

    Returns
    -------
    Tuple van (geïnitialiseerd LSTMModel in eval-modus, vocab-dict).

    Raises
    ------
    FileNotFoundError:
        Als het bijbehorende vocab-bestand (<stem>_vocab.pkl) niet bestaat.
    """
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
    """Parseer een MIDI-bestand naar een hiërarchische structuur per maat per stem.

    Resultaat: list[maat_index] → list[stem_index] → list[VoiceToken]

    Alle Parts worden doorlopen. Per maat worden alle noten en rusten als
    VoiceToken opgeslagen. Akkoorden worden gereduceerd tot de hoogste noot.
    Lege maten (geen tokens) worden weggegooid.

    Deze structuur maakt het mogelijk om:
      - De eerste helft als seed te gebruiken
      - Per stem het ritme of de melodie te genereren
      - De split op maatniveau precies te bepalen

    Parameters
    ----------
    pad:
        Pad naar het .mid-bestand.

    Returns
    -------
    Lijst van niet-lege maten, elk een lijst van stemmen,
    elk een lijst van VoiceTokens.
    """
    from music21 import converter, note, chord

    score  = converter.parse(pad)
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

    return [m for m in per_maat if m]


def _combineer_stemmen_naar_akkoorden(
    stemmen: list[list[VoiceToken]],
) -> list[ChordToken]:
    """Combineer per-stem VoiceTokens naar ChordTokens via tijdsuitlijning.

    Omdat stemmen verschillende ritmes kunnen hebben, worden alle noten eerst
    omgezet naar (start, einde, pitch)-events. Vervolgens worden alle unieke
    tijdstippen bepaald en voor elk interval worden de actieve noten verzameld.

    Stap 1: bereken (start, einde, pitch) per noot in elke stem.
    Stap 2: verzamel alle unieke tijdstippen (begin + einde van elke noot).
    Stap 3: voor elk interval [t0, t1]: verzamel pitches actief op t0 → ChordToken.

    Rusten (pitch 0) worden niet opgenomen in het akkoord.
    Lege akkoorden (alleen rusten) resulteren in een ChordToken met lege frozenset.

    Parameters
    ----------
    stemmen:
        Lijst van stemmen, elk een lijst van VoiceTokens.

    Returns
    -------
    Lijst van ChordTokens in chronologische volgorde.
    """
    if not stemmen:
        return []

    # Bereken absolute tijdstippen voor elke noot
    events: list[tuple[float, float, int]] = []
    for stem in stemmen:
        t = 0.0
        for pitch, duur in stem:
            events.append((t, t + duur, pitch))
            t += duur

    # Alle unieke tijdstippen bepalen voor de uitlijning
    tijdstippen = sorted({e[0] for e in events} | {e[1] for e in events})

    resultaat: list[ChordToken] = []
    for i in range(len(tijdstippen) - 1):
        t0  = tijdstippen[i]
        t1  = tijdstippen[i + 1]
        dur = _kwantiseer_duur(t1 - t0)
        if dur == 0:
            continue
        # Verzamel alle pitches die actief zijn op tijdstip t0 (exclusief rusten)
        actief = frozenset(
            p for (s, e, p) in events if s <= t0 and t0 < e and p != 0
        )
        resultaat.append((actief, dur))

    return resultaat


def _chord_tokens_naar_midi_b64(tokens: list[ChordToken]) -> str:
    """Converteer een lijst ChordTokens naar een base64-gecodeerd MIDI-bestand.

    Bouwt een enkelvoudige Part op basis van de akkoordtokens:
      - Lege frozenset → rust
      - Één pitch → enkelvoudige noot
      - Meerdere pitches → akkoord (gesorteerd van laag naar hoog)

    De MIDI wordt volledig in RAM gebouwd via music21 en als base64 teruggegeven.
    Er wordt niets naar disk geschreven.

    Parameters
    ----------
    tokens:
        Lijst van (frozenset[pitches], duur)-tuples.

    Returns
    -------
    Base64-gecodeerde string van het MIDI-bestand.
    """
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
    """Lees het eerste tempomarkering uit een MIDI-bestand in BPM.

    Wordt gebruikt om maat-starttijden in seconden te berekenen voor de
    JavaScript maatenteller in de piano-roll weergave.

    Parameters
    ----------
    midi_pad:
        Pad naar het .mid-bestand.

    Returns
    -------
    Tempo in BPM. Standaard 120 als geen tempomarkering gevonden wordt of
    bij een parse-fout.
    """
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
    """Combineer de originele eerste helft met AI-gegenereerde stemmen in één MIDI.

    Elke AI-stem krijgt een eigen Part zodat de onafhankelijke ritmische lijnen
    bewaard blijven (niet samengeperst tot één akkoordstroom).

    Verloop:
      1. Parseer het originele MIDI-bestand.
      2. Bepaal de splitoffset in kwartnootheden op basis van helft_maten.
      3. Kopieer alle originele noten vóór de splitoffset naar nieuwe Parts.
      4. Voeg elke AI-stem toe als eigen Part, startend op de splitoffset.
      5. Exporteer naar MIDI-bytes in RAM en codeer als base64.

    Parameters
    ----------
    midi_pad:
        Pad naar het originele .mid-bestand (voor de eerste helft).
    helft_maten:
        Aantal maten dat als originele eerste helft bewaard wordt.
    ai_stemmen:
        Lijst van AI-gegenereerde stems, elk een lijst VoiceTokens.

    Returns
    -------
    Base64-gecodeerde string van het gecombineerde MIDI-bestand.
    """
    import copy
    from music21 import converter, stream, note, tempo as m21_tempo
    from music21.midi import translate as midi_translate

    score    = converter.parse(midi_pad)
    ch_maten = list(score.chordify().getElementsByClass("Measure"))

    # Bepaal de offset (in kwartnootheden) waarop de AI-stemmen beginnen
    split_offset = (
        float(ch_maten[helft_maten].offset)
        if helft_maten < len(ch_maten)
        else float(ch_maten[-1].offset) + float(ch_maten[-1].barDuration.quarterLength)
    )

    gecombineerd = stream.Score()

    # Kopieer tempomarkeringen zodat het gegenereerde stuk even snel afspeelt
    for mm in score.flatten().getElementsByClass(m21_tempo.MetronomeMark):
        gecombineerd.insert(float(mm.offset), copy.deepcopy(mm))

    # Originele stemmen: kopieer alleen noten vóór de splitoffset
    for part in score.parts:
        nieuwe_part = stream.Part()
        for el in part.flatten().notesAndRests:
            if float(el.offset) < split_offset:
                nieuwe_part.insert(float(el.offset), copy.deepcopy(el))
        gecombineerd.append(nieuwe_part)

    # AI-stemmen: elke stem als eigen Part, startend op split_offset
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


# ── Generatie — fase 1: ritme (model-gestuurd) ───────────────────────────────

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
    """Genereer een duurvolgorde voor één stem via het LSTM-model.

    Wordt gebruikt in de Bach-stijl modus waarbij het model ook het ritme
    volledig zelf bepaalt (in tegenstelling tot de aanvullingsmodus waarbij
    het originele ritme overgenomen wordt).

    De kans op een duur wordt berekend als de marginale som over alle
    tokens met die duur:
        P(duur | context) = Σ_pitch  P(pitch, duur | context)

    Generatie stopt zodra de totale duur >= doelduur, of na max_stappen
    als veiligheidsnet (voorkomt eindeloze lussen bij lage temperatuur).

    Parameters
    ----------
    model:
        Getraind LSTMModel in eval-modus.
    seed_codes:
        Lijst van token-indices die als context dienen (eerste helft).
    doelduur:
        Gewenste totale duur in kwartnootheden.
    duur_naar_codes:
        Dict die elke duur mapt naar de lijst van token-indices met die duur.
    duuren:
        Gesorteerde lijst van alle unieke durations in het vocabulaire.
    venster:
        Contextlengte (sliding window over de gegenereerde tokens).
    temperatuur:
        Regelt de willekeur: lager = voorspelbaarder, hoger = creatiever.
        Wordt geclampd op minimaal 1e-6 om deling door nul te voorkomen.
    top_k:
        Als > 0, worden alleen de k meest waarschijnlijke tokens overwogen
        vóór de duur-marginalisatie. 0 = alle tokens gebruiken.

    Returns
    -------
    Lijst van durations (kwartnootheden) voor de gegenereerde stem.
    """
    ctx         = list(seed_codes)
    duur_seq:   list[float] = []
    totaal_duur = 0.0
    # Veiligheidsgrens: bij erg korte noten kan de lus lang duren
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

            # Top-k filtering: maskeer tokens buiten de top-k op -inf
            if top_k > 0:
                grens  = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < grens, float("-inf"))

            probs = torch.softmax(logits / max(temperatuur, 1e-6), dim=0)

            # Marginaliseer over pitches: bereken gewicht per duur
            duur_w = torch.tensor([
                sum(probs[c].item() for c in duur_naar_codes[d])
                for d in duuren
            ])
            duur_w = torch.clamp(duur_w, min=1e-9)  # voorkom nul-gewichten
            duur_w /= duur_w.sum()
            gekozen = duuren[torch.multinomial(duur_w, 1).item()]

            duur_seq.append(gekozen)
            totaal_duur += gekozen
            # Voeg de meest waarschijnlijke pitch voor deze duur toe aan de context
            codes = duur_naar_codes[gekozen]
            ctx.append(max(codes, key=lambda c: probs[c].item()))

    return duur_seq


# ── Toonaard detectie ─────────────────────────────────────────────────────────

def _toonaard_uit_noten(noten: list[VoiceToken]) -> set[int]:
    """Detecteer de toonaard van een reeks noten via music21's key-analyse.

    Bouwt een tijdelijke Part op basis van de opgegeven noten en laat
    music21's Krumhansl-Schmuckler algoritme de toonaard bepalen.
    Geeft de pitchklassen (0–11) terug van de gedetecteerde toonladder.

    Wordt gebruikt om de AI-generatie te beperken tot noten die in de
    toonaard van de eerste helft passen (aanvullingsmodus).

    Parameters
    ----------
    noten:
        Lijst van VoiceTokens uit de eerste helft van het werk.

    Returns
    -------
    Set van MIDI-pitchklassen (0–11) die in de gedetecteerde toonaard passen.
    Geeft alle 12 pitchklassen terug bij een analyse-fout (geen beperking).
    """
    from music21 import stream, note as note21

    part   = stream.Part()
    offset = 0.0
    for pitch_midi, duur in noten:
        if pitch_midi != 0:
            n = note21.Note()
            n.pitch.midi = pitch_midi
            n.duration.quarterLength = duur
            part.insert(offset, n)
        offset += duur
    try:
        key = part.analyze("key")
        return {p.pitchClass for p in key.pitches}
    except Exception:
        return set(range(12))


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
    toegestane_pitchklassen: set[int] | None = None,
) -> list[VoiceToken]:
    """Genereer pitches voor één stem op een vooraf vastgelegd ritmeschema.

    Voor elk duurslot in duur_seq wordt het model gevraagd een pitch te kiezen.
    Alleen tokens met exact die duur worden als kandidaat beschouwd.

    Pitch 0 (rust) is altijd toegestaan ongeacht de toonaard-filter.
    Overige pitches worden gefilterd op de gedetecteerde toonaard als
    toegestane_pitchklassen opgegeven is.

    Als na toonaard-filtering geen tokens overblijven, worden alle tokens
    met de juiste duur gebruikt (veiligheidsnet).

    Parameters
    ----------
    model:
        Getraind LSTMModel in eval-modus.
    seed_codes:
        Token-indices van de eerste helft als startcontext.
    duur_seq:
        Vastgelegde duurvolgorde (output van fase 1 of origineel ritme).
    duur_naar_codes:
        Dict die elke duur mapt naar token-indices met die duur.
    inv_vocab:
        Omgekeerd vocabulaire: integer index → VoiceToken.
    venster:
        Contextlengte voor het model.
    temperatuur:
        Willekeur van de sampling.
    top_k:
        Top-k filtering vóór toonaard-filter. 0 = uit.
    toegestane_pitchklassen:
        Set van MIDI-pitchklassen (0–11) die toegestaan zijn.
        None = geen beperking (Bach-stijl modus).

    Returns
    -------
    Lijst van VoiceTokens (pitch, duur) voor de gegenereerde stem.
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

            # Top-k filtering
            if top_k > 0:
                grens  = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < grens, float("-inf"))

            probs = torch.softmax(logits / max(temperatuur, 1e-6), dim=0)

            # Selecteer kandidaten met de juiste duur
            codes = duur_naar_codes[duur_slot]

            # Toonaard-filter: verwijder pitches die niet in de toonaard passen
            if toegestane_pitchklassen is not None:
                gefilterd = [
                    c for c in codes
                    if inv_vocab[c][0] == 0 or (inv_vocab[c][0] % 12) in toegestane_pitchklassen
                ]
                if gefilterd:
                    codes = gefilterd

            # Sample één token gewogen naar de model-kansen
            pitch_w = torch.clamp(
                torch.tensor([probs[c].item() for c in codes]), min=1e-9
            )
            pitch_w /= pitch_w.sum()
            gekozen = codes[torch.multinomial(pitch_w, 1).item()]

            ctx.append(gekozen)
            resultaat.append(inv_vocab[gekozen])

    return resultaat


# ── Vergelijkingsfuncties (ingebouwd, zonder externe afhankelijkheden) ────────

def _levenshtein(a: list, b: list) -> float:
    """Levenshtein-afstand tussen twee lijsten."""
    la, lb = len(a), len(b)
    # Begrens lengte voor performance bij lange stukken
    a, b = a[:300], b[:300]
    la, lb = len(a), len(b)
    dp = np.zeros((la + 1, lb + 1))
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[-1][-1]


def _extract_pitches(midi_pad: str) -> list[int]:
    from music21 import converter, note as m21_note
    score = converter.parse(midi_pad)
    return [n.pitch.midi for n in score.recurse().notes if isinstance(n, m21_note.Note)]


def _vergelijk_toonhoogte(f1: str, f2: str) -> float:
    """Melodische gelijkenis op basis van MIDI-nootnummers (Levenshtein)."""
    p1, p2 = _extract_pitches(f1), _extract_pitches(f2)
    if not p1 or not p2:
        return 0.0
    return round(max(0.0, (1 - _levenshtein(p1, p2) / max(len(p1), len(p2))) * 100), 1)


def _vergelijk_ritme_toon(f1: str, f2: str) -> float:
    """Gelijkenis op (noot, duur)-paren (Levenshtein)."""
    from music21 import converter, note as m21_note
    def extract(path: str) -> list:
        score = converter.parse(path)
        return [(n.pitch.midi, round(float(n.duration.quarterLength), 2))
                for n in score.recurse().notes if isinstance(n, m21_note.Note)]
    s1, s2 = extract(f1), extract(f2)
    if not s1 or not s2:
        return 0.0
    return round(max(0.0, (1 - _levenshtein(s1, s2) / max(len(s1), len(s2))) * 100), 1)


def _vergelijk_histogram(f1: str, f2: str) -> float:
    """Toonsoortconsistentie via cosinusgelijkenis van de nootverdeling (0–128)."""
    def hist(path: str) -> np.ndarray:
        pitches = _extract_pitches(path)
        v = np.zeros(128)
        for p in pitches:
            v[p] += 1
        s = v.sum()
        return v / s if s > 0 else v
    h1, h2 = hist(f1), hist(f2)
    denom = np.linalg.norm(h1) * np.linalg.norm(h2)
    return round(float(np.dot(h1, h2) / denom * 100) if denom > 0 else 0.0, 1)


def _vergelijk_akkoorden(f1: str, f2: str) -> float:
    """Harmonische gelijkenis via akkoordreeksen (Levenshtein)."""
    from music21 import converter, chord as m21_chord
    def extract(path: str) -> list:
        score = converter.parse(path)
        return [tuple(sorted(p.midi for p in c.pitches))
                for c in score.chordify().recurse() if isinstance(c, m21_chord.Chord)]
    c1, c2 = extract(f1), extract(f2)
    if not c1 or not c2:
        return 0.0
    return round(max(0.0, (1 - _levenshtein(c1, c2) / max(len(c1), len(c2))) * 100), 1)


# ── Willekeurige generatie ────────────────────────────────────────────────────

def _genereer_willekeurig_b64(
    midi_pad: str,
    helft_maten: int,
    n_stemmen: int,
    ritmes: list[list[float]],
    inv_vocab: dict[int, VoiceToken],
    duur_naar_codes: dict[float, list[int]],
) -> str:
    """Genereer tweede helft volledig willekeurig: uniforme sampling uit het vocabulaire."""
    ai_stemmen: list[list[VoiceToken]] = []
    for stem_i in range(n_stemmen):
        tokens: list[VoiceToken] = []
        for duur in ritmes[stem_i]:
            kandidaten = duur_naar_codes.get(duur)
            if kandidaten:
                tokens.append(inv_vocab[random.choice(kandidaten)])
            else:
                tokens.append((0, duur))
        ai_stemmen.append(tokens)
    return _eerste_helft_plus_ai_b64(midi_pad, helft_maten, ai_stemmen)


# ── Fine-tuning + generatie ───────────────────────────────────────────────────

def _finetunen_en_genereer_b64(
    model: LSTMModel,
    vocab: dict[VoiceToken, int],
    per_maat: list,
    helft_maten: int,
    n_stemmen: int,
    ritmes: list[list[float]],
    duur_naar_codes: dict[float, list[int]],
    inv_vocab: dict[int, VoiceToken],
    midi_pad: str,
    venster: int,
    temperatuur: float,
    top_k: int,
    toegestane_pitchklassen: set[int] | None,
    finetune_epochs: int = 8,
) -> str:
    """Fine-tune het model op de eerste helft, genereer daarna de tweede helft.

    Maakt een diepe kopie van het model zodat de originele gewichten bewaard blijven.
    Past Adam-gradient-updates toe op de eerste-helft-tokens (sliding window),
    en genereert vervolgens de tweede helft met het aangepaste model.
    """
    ft_model = copy.deepcopy(model)
    ft_model.train()
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(finetune_epochs):
        for stem_i in range(n_stemmen):
            eerste = [
                t for maat in per_maat[:helft_maten]
                for t in (maat[stem_i] if stem_i < len(maat) else [])
            ]
            codes = [vocab[t] for t in eerste if t in vocab]
            if len(codes) <= venster:
                continue
            for start in range(len(codes) - venster):
                inp = torch.tensor([codes[start:start + venster]], dtype=torch.long)
                tgt = torch.tensor(codes[start + venster], dtype=torch.long)
                optimizer.zero_grad()
                loss = criterion(ft_model(inp)[0], tgt)
                loss.backward()
                optimizer.step()

    ft_model.eval()

    stem_seeds = []
    for stem_i in range(n_stemmen):
        eerste = [
            t for maat in per_maat[:helft_maten]
            for t in (maat[stem_i] if stem_i < len(maat) else [])
        ]
        stem_seeds.append([vocab[t] for t in eerste if t in vocab])

    ai_stemmen: list[list[VoiceToken]] = []
    for stem_i in range(n_stemmen):
        seeds = stem_seeds[stem_i]
        if len(seeds) < venster or not ritmes[stem_i]:
            ai_stemmen.append([])
            continue
        melodie = _genereer_melodie(
            ft_model, seeds, ritmes[stem_i],
            duur_naar_codes, inv_vocab,
            venster, temperatuur, top_k,
            toegestane_pitchklassen=toegestane_pitchklassen,
        )
        ai_stemmen.append(melodie)

    return _eerste_helft_plus_ai_b64(midi_pad, helft_maten, ai_stemmen)


# ── Vergelijking met origineel ────────────────────────────────────────────────

def _bereken_vergelijking(
    orig_pad: str,
    versies: dict[str, str],   # naam → base64-MIDI
) -> dict[str, dict[str, float]]:
    """Vergelijk elke gegenereerde versie met het origineel.

    Schrijft elke versie tijdelijk naar disk, voert de 4 vergelijkingsfuncties
    uit en verwijdert het tijdelijke bestand daarna.
    """
    resultaten: dict[str, dict[str, float]] = {}
    for naam, b64 in versies.items():
        tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        try:
            tmp.write(base64.b64decode(b64))
            tmp.close()
            resultaten[naam] = {
                "Toonhoogte": _vergelijk_toonhoogte(orig_pad, tmp.name),
                "Ritme+toon": _vergelijk_ritme_toon(orig_pad, tmp.name),
                "Histogram":  _vergelijk_histogram(orig_pad, tmp.name),
                "Akkoorden":  _vergelijk_akkoorden(orig_pad, tmp.name),
            }
        except Exception as e:
            resultaten[naam] = {"Toonhoogte": 0.0, "Ritme+toon": 0.0,
                                 "Histogram": 0.0, "Akkoorden": 0.0}
        finally:
            os.unlink(tmp.name)
    return resultaten


_KLEUREN_VERSIES = {
    "AI-aanvulling": "#38bdf8",
    "Fine-tuned":    "#34d399",
    "Willekeurig":   "#f87171",
}
_CATEGORIEEN = ["Toonhoogte", "Ritme+toon", "Histogram", "Akkoorden"]


def _toon_vergelijking(vergelijking: dict[str, dict[str, float]]) -> None:
    """Toon vergelijkingsscores als radardiagram + staafdiagram + tabel."""
    if not vergelijking:
        st.warning("Geen vergelijkingsdata beschikbaar.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("Plotly niet geïnstalleerd — `pip install plotly`")
        _toon_vergelijking_tabel(vergelijking)
        return

    # ── Radardiagram ──────────────────────────────────────────────────────────
    radar = go.Figure()
    for naam, scores in vergelijking.items():
        waarden = [scores.get(c, 0) for c in _CATEGORIEEN]
        radar.add_trace(go.Scatterpolar(
            r=waarden + [waarden[0]],
            theta=_CATEGORIEEN + [_CATEGORIEEN[0]],
            fill="toself",
            name=naam,
            line_color=_KLEUREN_VERSIES.get(naam, "#94a3b8"),
            opacity=0.75,
        ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=10))),
        showlegend=True,
        height=360,
        margin=dict(l=40, r=40, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.05),
    )

    # ── Staafdiagram (groepering per categorie) ───────────────────────────────
    balk = go.Figure()
    for naam, scores in vergelijking.items():
        balk.add_trace(go.Bar(
            name=naam,
            x=_CATEGORIEEN,
            y=[scores.get(c, 0) for c in _CATEGORIEEN],
            marker_color=_KLEUREN_VERSIES.get(naam, "#94a3b8"),
            text=[f"{scores.get(c,0):.0f}%" for c in _CATEGORIEEN],
            textposition="outside",
        ))
    balk.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 110], title="Score (%)"),
        xaxis=dict(title=""),
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    col_r, col_b = st.columns(2)
    with col_r:
        st.caption("Radardiagram — gelijkenis per dimensie")
        st.plotly_chart(radar, use_container_width=True)
    with col_b:
        st.caption("Staafdiagram — vergelijking per categorie")
        st.plotly_chart(balk, use_container_width=True)

    # ── Uitleg per metric ─────────────────────────────────────────────────────
    with st.expander("ℹ️ Uitleg van de metrics"):
        st.markdown("""
**Toonhoogte** — *Melodische gelijkenis*
Vergelijkt de reeks van MIDI-nootnummers (0–127) van het gegenereerde stuk met het origineel via de Levenshtein-afstand. Een hoge score betekent dat het model dezelfde noten speelt als Bach, in een vergelijkbare volgorde. Een lage score wijst op melodische afwijking.

---

**Ritme+toon** — *Melodie én ritme samen*
Elk element is een (noot, duur)-paar, zodat ook het ritme meeweegt. Twee stukken kunnen dezelfde noten hebben maar een ander ritme — dan scoort Toonhoogte hoog maar Ritme+toon lager. Dit is de strengste maat voor naleving van het origineel.

---

**Histogram** — *Toonsoortconsistentie*
Vergelijkt de verdeling van nootnummers als vector via cosinusgelijkenis. Het maakt niet uit in welke volgorde noten voorkomen, alleen hoe vaak. Een hoge score betekent dat het gegenereerde stuk dezelfde toonsoort en nootvoorkeur heeft als het origineel — het "klinkt in dezelfde toonaard".

---

**Akkoorden** — *Harmonische gelijkenis*
Vergelijkt de reeks van akkoorden (gelijktijdige noten) via Levenshtein. Bach heeft typische akkoordprogressies (I–IV–V–I). Een hoge score wijst erop dat het model vergelijkbare harmonieën gebruikt.
""")

    # ── Samenvattingstabel ────────────────────────────────────────────────────
    _toon_vergelijking_tabel(vergelijking)


def _toon_vergelijking_tabel(vergelijking: dict[str, dict[str, float]]) -> None:
    rijen = []
    for naam, scores in vergelijking.items():
        gem = round(sum(scores.values()) / len(scores), 1)
        rijen.append({"Versie": naam, **scores, "Gemiddeld ★": gem})
    df_verg = pd.DataFrame(rijen).set_index("Versie")
    st.dataframe(
        df_verg.style.background_gradient(cmap="Blues", vmin=0, vmax=100),
        use_container_width=True,
    )


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
    """Genereer een HTML-document met piano-roll speler en optionele maatenteller.

    Bevat:
      - html-midi-player webcomponent (afspelen + download via data-URI)
      - midi-visualizer webcomponent (piano-roll met gekleurde noten)
      - Optionele JavaScript maatenteller die elke 100ms de huidige afspeelttijd
        controleert en het maatnummer toont.

    De maatenteller kleurt het maatnummer:
      - Blauw (#38bdf8): originele maten
      - Geel (#f59e0b): AI-gegenereerde maten (alleen als is_ai=True)

    Parameters
    ----------
    midi_b64:
        Base64-gecodeerde MIDI-data.
    hoogte:
        Pixelhoogte van de piano-roll visualisatie.
    uid:
        Unieke identifier voor DOM-elementen (voorkomt conflicten bij
        meerdere spelers op dezelfde pagina).
    maat_starts_s:
        Lijst van starttijden per maat in seconden. Als None, wordt
        geen maatenteller getoond.
    totaal_maten:
        Totaal aantal maten in het werk (voor de "Maat X / Y" weergave).
    helft_maten:
        Maatnummer waarop de AI-aanvulling begint.
    is_ai:
        Als True, wordt kleurwisseling tussen origineel/AI getoond.

    Returns
    -------
    HTML-string voor gebruik in st.components.v1.html().
    """
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
    // Poll elke 100ms: html-midi-player biedt geen native tijdsevent
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
    """Hoofdfunctie van de Streamlit-genereererpagina.

    Opbouw van de pagina:
      1. Model kiezen: selectbox met alle beschikbare lstm_*.pt bestanden.
      2. Song kiezen: filterbare tabel met alle Bach-werken.
      3. Instellingen: temperatuur en top-k sliders.
      4. Twee generatieknoppen:
           - Aanvulling: origineel ritme + toonaard behouden
           - Bach-stijl: model genereert alles vrij
      5. Resultaatweergave: metrics, splitsbalk, twee piano-rolls naast elkaar.
    """
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

    rij        = df_toon.iloc[event.selection.rows[0]]
    midi_pad   = str(rij["midi_pad"])
    song_naam  = str(rij["naam"])
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

    # ── Genereer-knoppen ──────────────────────────────────────────────────────
    st.divider()
    col_k1, col_k2 = st.columns(2)

    with col_k1:
        st.markdown("**Aanvulling** — eerste helft als seed, origineel ritme + toonaard")
        if st.button("🎵 Genereer aanvulling", type="primary", use_container_width=True):
            _voer_generatie_uit(
                model_pad_str=str(model_keuze),
                midi_pad=midi_pad,
                song_naam=song_naam,
                venster=int(venster),
                temperatuur=float(temperatuur),
                top_k=int(top_k),
            )

    with col_k2:
        st.markdown(f"**Bach-stijl** — eerste {N_SEED_MATEN} maten als seed, model genereert vrij")
        if st.button("🎼 Genereer Bach-stijl", type="secondary", use_container_width=True):
            _voer_vrije_generatie_uit(
                model_pad_str=str(model_keuze),
                midi_pad=midi_pad,
                song_naam=song_naam,
                venster=int(venster),
                temperatuur=float(temperatuur),
                top_k=int(top_k),
            )

    # ── Resultaten ────────────────────────────────────────────────────────────
    if (
        "gen_origineel_b64" in st.session_state
        and st.session_state.get("gen_song") == song_naam
    ):
        st.subheader("Resultaat — Aanvulling")
        _toon_resultaat()

    if (
        "vrij_ai_b64" in st.session_state
        and st.session_state.get("vrij_song") == song_naam
    ):
        st.subheader("Resultaat — Bach-stijl")
        _toon_vrij_resultaat()


def _voer_generatie_uit(
    model_pad_str: str,
    midi_pad: str,
    song_naam: str,
    venster: int,
    temperatuur: float,
    top_k: int = 0,
) -> None:
    """Voer de aanvullingsmodus uit: eerste helft als seed, origineel ritme bewaard.

    Verloop:
      1. Model en vocabulaire laden.
      2. MIDI inlezen en per maat + per stem splitsen.
      3. Eerste helft als seed, tweede helft als doelreferentie.
      4. Fase 1: kopieer het originele ritme van de tweede helft.
      5. Detecteer de toonaard uit de eerste helft.
      6. Fase 2: genereer pitches per stem op het gekopieerde ritme,
         gefilterd op de gedetecteerde toonaard.
      7. Bouw het gecombineerde MIDI-bestand (eerste helft + AI-stemmen).
      8. Bereken maat-starttijden in seconden voor de maatenteller.
      9. Sla alles op in st.session_state voor weergave door _toon_resultaat().

    Parameters
    ----------
    model_pad_str:
        Pad naar het .pt-modelbestand.
    midi_pad:
        Pad naar het originele .mid-bestand.
    song_naam:
        BWV-naam van het geselecteerde werk (voor session_state-koppeling).
    venster:
        Contextlengte voor het model.
    temperatuur:
        Sampling-temperatuur.
    top_k:
        Top-k filtering. 0 = uit.
    """
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

        # Gedeelde opzoektabel: duur → lijst van token-indices met die duur
        duur_naar_codes: dict[float, list[int]] = {}
        for (_, d), idx in vocab.items():
            duur_naar_codes.setdefault(d, []).append(idx)
        duuren = sorted(duur_naar_codes.keys())

        n_stemmen = max(len(m) for m in per_maat)

        # Bouw per-stem seeds (eerste helft) en doelduren (tweede helft)
        stem_seeds:  list[list[int]]            = []
        stem_duuren: list[float]                = []
        stem_tweede: list[list[VoiceToken]]     = []
        for stem_i in range(n_stemmen):
            eerste = [t for maat in per_maat[:helft_maten]
                      for t in (maat[stem_i] if stem_i < len(maat) else [])]
            tweede = [t for maat in per_maat[helft_maten:]
                      for t in (maat[stem_i] if stem_i < len(maat) else [])]
            stem_seeds.append([vocab[t] for t in eerste if t in vocab])
            stem_duuren.append(sum(d for _, d in tweede))
            stem_tweede.append(tweede)

        # ── FASE 1: ritme overnemen van het origineel ─────────────────────────
        st.write("**Fase 1 — Origineel ritme overnemen**")
        ritmes: list[list[float]] = []
        for stem_i in range(n_stemmen):
            duur_seq = [d for _, d in stem_tweede[stem_i]]
            ritmes.append(duur_seq)
            schema = "  ".join(str(d) for d in duur_seq[:12])
            suffix = "…" if len(duur_seq) > 12 else ""
            st.write(
                f"  Stem {stem_i + 1}: `{schema}{suffix}`  "
                f"({len(duur_seq)} noten, {sum(duur_seq):.1f} kwartnootheden)"
            )

        # ── Toonaard detecteren uit de eerste helft ───────────────────────────
        st.write("Toonaard detecteren uit aangeboden deel…")
        alle_eerste_helft: list[VoiceToken] = [
            t
            for stem_i in range(n_stemmen)
            for maat in per_maat[:helft_maten]
            for t in (maat[stem_i] if stem_i < len(maat) else [])
        ]
        toegestane_pitchklassen = _toonaard_uit_noten(alle_eerste_helft)
        st.write(f"  Pitchklassen: {sorted(toegestane_pitchklassen)}")

        # ── FASE 2: melodie per stem op het vastgelegde ritme ─────────────────
        st.write("**Fase 2 — Melodie per stem genereren**")
        ai_stemmen: list[list[VoiceToken]] = []
        for stem_i in range(n_stemmen):
            seeds = stem_seeds[stem_i]
            if len(seeds) < venster or not ritmes[stem_i]:
                # Onvoldoende seed-data: gebruik originele tweede helft als fallback
                ai_stemmen.append(stem_tweede[stem_i])
                continue

            st.write(f"  Stem {stem_i + 1}…")
            melodie = _genereer_melodie(
                model, seeds, ritmes[stem_i],
                duur_naar_codes, inv_vocab,
                venster, temperatuur, top_k,
                toegestane_pitchklassen=toegestane_pitchklassen,
            )
            ai_stemmen.append(melodie)

        st.write("MIDI bouwen — AI-aanvulling…")
        with open(midi_pad, "rb") as f:
            origineel_b64 = base64.b64encode(f.read()).decode()

        ai_b64 = _eerste_helft_plus_ai_b64(midi_pad, helft_maten, ai_stemmen)

        st.write("MIDI bouwen — Fine-tuned versie (model leert op eerste helft)…")
        ft_b64 = _finetunen_en_genereer_b64(
            model, vocab, per_maat, helft_maten, n_stemmen,
            ritmes, duur_naar_codes, inv_vocab, midi_pad,
            venster, temperatuur, top_k,
            toegestane_pitchklassen=toegestane_pitchklassen,
        )

        st.write("MIDI bouwen — Willekeurige versie (geen leren)…")
        rand_b64 = _genereer_willekeurig_b64(
            midi_pad, helft_maten, n_stemmen,
            ritmes, inv_vocab, duur_naar_codes,
        )

        # Maatstarttijden in seconden berekenen voor de JavaScript maatenteller
        bpm = _tempo_uit_midi(midi_pad)
        SPQ = 60.0 / bpm   # seconden per kwartnoot
        maat_starts_s: list[float] = []
        cumulatief = 0.0
        for maat in per_maat:
            maat_starts_s.append(round(cumulatief * SPQ, 4))
            if maat:
                for _, duur in maat[0]:
                    cumulatief += duur

        st.write("Vergelijking berekenen…")
        vergelijking = _bereken_vergelijking(midi_pad, {
            "AI-aanvulling": ai_b64,
            "Fine-tuned":    ft_b64,
            "Willekeurig":   rand_b64,
        })

        totaal_nieuwe  = sum(len(s) for s in ai_stemmen)
        totaal_orig_2e = sum(
            len(maat[si] if si < len(maat) else [])
            for maat in per_maat[helft_maten:]
            for si in range(n_stemmen)
        )

        # Sla alles op in session_state; _toon_resultaat() leest hieruit
        st.session_state.update({
            "gen_origineel_b64":   origineel_b64,
            "gen_ai_b64":          ai_b64,
            "gen_ft_b64":          ft_b64,
            "gen_rand_b64":        rand_b64,
            "gen_vergelijking":    vergelijking,
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
    """Toon het resultaat van de aanvullingsmodus: 4 versies + vergelijking."""
    st.divider()

    hm  = st.session_state["gen_helft_maten"]
    tm  = st.session_state["gen_totaal_maten"]
    nt  = st.session_state["gen_nieuwe_tokens"]
    ot  = st.session_state["gen_orig_2e_tokens"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Totaal maten",             tm)
    c2.metric("Eerste helft (origineel)", f"maat 1–{hm}")
    c3.metric("Tweede helft (AI)",        f"maat {hm+1}–{tm}")
    c4.metric("Tokens gegenereerd",       f"{nt}  (orig: {ot})")

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
          <div style="flex:{1-frac};text-align:right">🟡 Gegenereerd (maat {hm+1}–{tm})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    maat_starts = st.session_state.get("gen_maat_starts_s", [])

    # ── Rij 1: Origineel + AI-aanvulling ─────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎼 Origineel")
        st.caption(f"Maten 1–{tm} · volledig origineel")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["gen_origineel_b64"],
                hoogte=260, uid="orig",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=hm, is_ai=False,
            ),
            height=370, scrolling=False,
        )
    with col2:
        st.subheader("🤖 AI-aanvulling")
        st.caption(f"Maten 1–{hm} · origineel  +  maten {hm+1}–{tm} · AI (ritme + toonaard behouden)")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["gen_ai_b64"],
                hoogte=260, uid="ai",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=hm, is_ai=True,
            ),
            height=370, scrolling=False,
        )

    # ── Rij 2: Fine-tuned + Willekeurig ──────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🧠 Fine-tuned op eerste helft")
        st.caption(f"Maten 1–{hm} · origineel  +  maten {hm+1}–{tm} · model bijgetraind op seed")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["gen_ft_b64"],
                hoogte=260, uid="ft",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=hm, is_ai=True,
            ),
            height=370, scrolling=False,
        )
    with col4:
        st.subheader("🎲 Volledig willekeurig")
        st.caption(f"Maten 1–{hm} · origineel  +  maten {hm+1}–{tm} · willekeurig (geen leren)")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["gen_rand_b64"],
                hoogte=260, uid="rand",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=hm, is_ai=True,
            ),
            height=370, scrolling=False,
        )

    # ── Vergelijking met origineel ────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Vergelijking met origineel")
    st.caption("Scores 0–100: hoe dichter bij het origineel, hoe hoger")
    vergelijking = st.session_state.get("gen_vergelijking", {})
    _toon_vergelijking(vergelijking)


def _voer_vrije_generatie_uit(
    model_pad_str: str,
    midi_pad: str,
    song_naam: str,
    venster: int,
    temperatuur: float,
    top_k: int = 0,
) -> None:
    """Voer de Bach-stijl modus uit: eerste N_SEED_MATEN maten als seed, rest vrij.

    In tegenstelling tot de aanvullingsmodus genereert het model hier ook het
    ritme volledig zelf (fase 1). Er is geen toonaard-filter (fase 2).

    Verloop:
      1. Model en vocabulaire laden.
      2. MIDI inlezen.
      3. Eerste N_SEED_MATEN maten als seed per stem.
      4. Fase 1: model genereert ritme vrij via _genereer_ritme().
      5. Fase 2: model genereert pitches vrij via _genereer_melodie()
         (toegestane_pitchklassen=None → geen filter).
      6. MIDI bouwen en opslaan in session_state.

    Parameters
    ----------
    model_pad_str:
        Pad naar het .pt-modelbestand.
    midi_pad:
        Pad naar het originele .mid-bestand (voor seed + originele vergelijking).
    song_naam:
        BWV-naam voor session_state-koppeling.
    venster:
        Contextlengte voor het model.
    temperatuur:
        Sampling-temperatuur.
    top_k:
        Top-k filtering. 0 = uit.
    """
    with st.status("Bach-stijl muziek genereren…", expanded=True) as status:

        st.write("Model en vocabulaire laden…")
        try:
            model, vocab = _laad_model_en_vocab(model_pad_str)
        except FileNotFoundError as e:
            st.error(f"Vocab-bestand niet gevonden: {e}")
            return

        inv_vocab = {i: t for t, i in vocab.items()}

        st.write("Song inlezen…")
        try:
            per_maat = _midi_naar_stemmen_per_maat(midi_pad)
        except Exception as e:
            st.error(f"MIDI-bestand kon niet worden geladen: {e}")
            return

        n_maten = len(per_maat)
        if n_maten <= N_SEED_MATEN:
            st.error(f"Song heeft te weinig maten ({n_maten}); minimum is {N_SEED_MATEN + 1}.")
            return

        duur_naar_codes: dict[float, list[int]] = {}
        for (_, d), idx in vocab.items():
            duur_naar_codes.setdefault(d, []).append(idx)
        duuren = sorted(duur_naar_codes.keys())

        n_stemmen = max(len(m) for m in per_maat)

        # Bouw seed (eerste N_SEED_MATEN maten) en rest per stem
        stem_seeds:     list[list[int]]       = []
        stem_doelduren: list[float]           = []
        stem_rest:      list[list[VoiceToken]] = []
        for stem_i in range(n_stemmen):
            seed_tok = [
                t for maat in per_maat[:N_SEED_MATEN]
                for t in (maat[stem_i] if stem_i < len(maat) else [])
            ]
            rest_tok = [
                t for maat in per_maat[N_SEED_MATEN:]
                for t in (maat[stem_i] if stem_i < len(maat) else [])
            ]
            stem_seeds.append([vocab[t] for t in seed_tok if t in vocab])
            stem_doelduren.append(sum(d for _, d in rest_tok))
            stem_rest.append(rest_tok)

        # ── Fase 1: model genereert ritme vrij ────────────────────────────────
        st.write("**Fase 1 — Ritme genereren (Bach-stijl)**")
        ritmes: list[list[float]] = []
        for stem_i in range(n_stemmen):
            seeds    = stem_seeds[stem_i]
            doelduur = stem_doelduren[stem_i]
            if len(seeds) < venster or doelduur == 0:
                # Onvoldoende seed of geen doelduur: gebruik origineel ritme als fallback
                ritmes.append([d for _, d in stem_rest[stem_i]])
                st.write(f"  Stem {stem_i + 1}: origineel ritme als fallback")
                continue
            duur_seq = _genereer_ritme(
                model, seeds, doelduur,
                duur_naar_codes, duuren,
                venster, temperatuur, top_k,
            )
            ritmes.append(duur_seq)
            schema = "  ".join(str(d) for d in duur_seq[:12])
            suffix = "…" if len(duur_seq) > 12 else ""
            st.write(
                f"  Stem {stem_i + 1}: `{schema}{suffix}`  "
                f"({len(duur_seq)} noten, {sum(duur_seq):.1f} kw.)"
            )

        # ── Fase 2: model genereert pitches vrij (geen toonaard-filter) ───────
        st.write("**Fase 2 — Melodie genereren (model bepaalt toonaard)**")
        ai_stemmen: list[list[VoiceToken]] = []
        for stem_i in range(n_stemmen):
            seeds = stem_seeds[stem_i]
            if len(seeds) < venster or not ritmes[stem_i]:
                ai_stemmen.append(stem_rest[stem_i])
                continue
            st.write(f"  Stem {stem_i + 1}…")
            melodie = _genereer_melodie(
                model, seeds, ritmes[stem_i],
                duur_naar_codes, inv_vocab,
                venster, temperatuur, top_k,
                toegestane_pitchklassen=None,   # geen filter: model bepaalt zelf de toonaard
            )
            ai_stemmen.append(melodie)

        st.write("MIDI bouwen…")
        with open(midi_pad, "rb") as f:
            origineel_b64 = base64.b64encode(f.read()).decode()

        ai_b64 = _eerste_helft_plus_ai_b64(midi_pad, N_SEED_MATEN, ai_stemmen)

        bpm = _tempo_uit_midi(midi_pad)
        SPQ = 60.0 / bpm
        maat_starts_s: list[float] = []
        cumulatief = 0.0
        for maat in per_maat:
            maat_starts_s.append(round(cumulatief * SPQ, 4))
            if maat:
                for _, duur in maat[0]:
                    cumulatief += duur

        st.session_state.update({
            "vrij_origineel_b64": origineel_b64,
            "vrij_ai_b64":        ai_b64,
            "vrij_song":          song_naam,
            "vrij_seed_maten":    N_SEED_MATEN,
            "vrij_totaal_maten":  n_maten,
            "vrij_nieuwe_tokens": sum(len(s) for s in ai_stemmen),
            "vrij_maat_starts_s": maat_starts_s,
        })

        status.update(label="Klaar!", state="complete")


def _toon_vrij_resultaat() -> None:
    """Toon het resultaat van de Bach-stijl modus.

    Leest de gegenereerde MIDI uit st.session_state en toont:
      - Vier metrics: totaal maten, seed-bereik, gegenereerd bereik, token-count
      - Een visuele splitsbalk (blauw = seed, geel = AI)
      - Twee piano-rolls: origineel vs. Bach-stijl aanvulling
    """
    st.divider()

    sm = st.session_state["vrij_seed_maten"]
    tm = st.session_state["vrij_totaal_maten"]
    nt = st.session_state["vrij_nieuwe_tokens"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Totaal maten",       tm)
    c2.metric("Seed (origineel)",   f"maat 1 – {sm}")
    c3.metric("Gegenereerd (AI)",   f"maat {sm + 1} – {tm}")
    c4.metric("Tokens gegenereerd", nt)

    frac = sm / tm
    st.markdown(
        f"""
        <div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin:6px 0 14px">
          <div style="flex:{frac};background:#38bdf8;"></div>
          <div style="flex:{1-frac};background:#f59e0b;"></div>
        </div>
        <div style="display:flex;font-size:12px;color:#94a3b8;margin-bottom:10px">
          <div style="flex:{frac}">🔵 Seed (maat 1–{sm})</div>
          <div style="flex:{1-frac};text-align:right">🟡 Bach-stijl AI (maat {sm+1}–{tm})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    maat_starts = st.session_state.get("vrij_maat_starts_s", [])

    with col1:
        st.subheader("🎼 Origineel")
        st.caption(f"Alle {tm} maten origineel")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["vrij_origineel_b64"],
                hoogte=300, uid="vorig",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=sm,
                is_ai=False,
            ),
            height=420, scrolling=False,
        )

    with col2:
        st.subheader("🤖 Bach-stijl aanvulling")
        st.caption(f"Maat 1–{sm} · origineel  +  maat {sm+1}–{tm} · Bach-stijl AI")
        st.components.v1.html(
            _piano_roll_html(
                st.session_state["vrij_ai_b64"],
                hoogte=300, uid="vai",
                maat_starts_s=maat_starts,
                totaal_maten=tm, helft_maten=sm,
                is_ai=True,
            ),
            height=420, scrolling=False,
        )


main()
