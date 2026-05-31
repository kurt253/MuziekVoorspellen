"""Bach Muziekoverzicht — Streamlit-app.

Starten::

    streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "processed" / "bach_analyse.csv"
RAW_DIR = DATA_DIR / "raw" / "bach"

# ---------------------------------------------------------------------------
# Data & conversies (gecached)
# ---------------------------------------------------------------------------

@st.cache_data
def laad_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    if "bestandsnaam" not in df.columns:
        df["bestandsnaam"] = df["naam"] + ".mid"
    return df


@st.cache_data
def laad_gefilterde_midi_b64(midi_pad: str, geselecteerde_indices: tuple[int, ...]) -> str:
    """MIDI-bestand inlezen, alleen gevraagde tracks behouden, base64 teruggeven."""
    from music21 import converter, stream
    from music21.midi import translate as midi_translate

    partituur = converter.parse(midi_pad)
    delen = list(partituur.parts)

    if not geselecteerde_indices or set(geselecteerde_indices) == set(range(len(delen))):
        with open(midi_pad, "rb") as f:
            return base64.b64encode(f.read()).decode()

    gefilterd = stream.Score()
    for i in geselecteerde_indices:
        if i < len(delen):
            gefilterd.append(delen[i])

    mf = midi_translate.music21ObjectToMidiFile(gefilterd)
    return base64.b64encode(mf.writestr()).decode()


@st.cache_data
def laad_abc(naam: str, geselecteerde_indices: tuple[int, ...]) -> str | None:
    """MusicXML of MIDI omzetten naar ABC-notatie voor abcjs."""
    from music21 import converter, stream

    xml_pad = RAW_DIR / f"{naam}.xml"
    midi_pad = RAW_DIR / f"{naam}.mid"
    bron = str(xml_pad) if xml_pad.exists() else str(midi_pad)

    try:
        partituur = converter.parse(bron)
    except Exception:
        return None

    delen = list(partituur.parts)
    if geselecteerde_indices and set(geselecteerde_indices) != set(range(len(delen))):
        gefilterd = stream.Score()
        for i in geselecteerde_indices:
            if i < len(delen):
                gefilterd.append(delen[i])
        partituur = gefilterd

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".abc", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp_path = tmp.name
        partituur.write("abc", fp=tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# HTML-componenten
# ---------------------------------------------------------------------------

_MIDI_CDN = (
    "https://cdn.jsdelivr.net/combine/"
    "npm/tone@14,"
    "npm/@magenta/music@1.23.1/es6/core.js,"
    "npm/html-midi-player@1.5.0"
)
_ABCJS_CDN = "https://cdn.jsdelivr.net/npm/abcjs@6.4.4/dist/abcjs-min.js"
_ABCJS_CSS = "https://cdn.jsdelivr.net/npm/abcjs@6.4.4/abcjs-audio.css"


def piano_roll_html(midi_b64: str, hoogte: int = 300) -> str:
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
    border-radius:8px; overflow:hidden; background:#0f172a;
  }}
  midi-visualizer svg rect.note {{ rx:2; ry:2; opacity:0.85; }}
</style>
</head><body>
  <midi-player src="data:audio/midi;base64,{midi_b64}"
    sound-font visualizer="#rol" style="width:100%;"></midi-player>
  <midi-visualizer type="piano-roll" id="rol"
    src="data:audio/midi;base64,{midi_b64}"></midi-visualizer>
</body></html>"""


def partituur_html(abc_string: str) -> str:
    # json.dumps escapet alle speciale tekens veilig voor een JS-string-literal
    abc_js = json.dumps(abc_string)

    return f"""<!DOCTYPE html><html><head>
<link rel="stylesheet" href="{_ABCJS_CSS}">
<script src="{_ABCJS_CDN}"></script>
<style>
  body {{
    margin: 0; padding: 8px 14px 16px;
    background: #fff;
    font-family: sans-serif;
  }}
  #score {{ width: 100%; }}
  #speler {{
    margin: 6px 0 10px;
  }}
  /* Gespeelde noten geel markeren */
  .actief path  {{ fill: #f59e0b !important; stroke: #d97706 !important; }}
  .actief rect  {{ fill: #f59e0b !important; }}
  /* abcjs afspeelcursor */
  .abcjs-cursor {{ stroke: #3b82f6; }}
</style>
</head><body>
<div id="speler"></div>
<div id="score"></div>

<script>
const abcTekst = {abc_js};

const visualObjs = ABCJS.renderAbc("score", abcTekst, {{
  responsive: "resize",
  add_classes: true,
  wrap: {{ minSpacing: 1.8, maxSpacing: 2.8, preferredMeasuresPerLine: 4 }},
}});

function wisActief() {{
  document.querySelectorAll(".actief").forEach(el => el.classList.remove("actief"));
}}

const cursorBeheer = {{
  onStart:    wisActief,
  onFinished: wisActief,
  onBeat:     function() {{}},
  onEvent: function(ev) {{
    wisActief();
    if (ev && ev.elements) {{
      ev.elements.forEach(function(stemEls) {{
        if (stemEls) stemEls.forEach(el => el.classList.add("actief"));
      }});
    }}
  }},
}};

if (ABCJS.synth.supportsAudio()) {{
  const synthBeheer = new ABCJS.synth.SynthController();
  synthBeheer.load("#speler", cursorBeheer, {{
    displayRestart:  true,
    displayPlay:     true,
    displayProgress: true,
    displayClock:    true,
  }});
  synthBeheer.setTune(visualObjs[0], false).catch(function(err) {{
    document.getElementById("speler").insertAdjacentHTML(
      "beforeend",
      "<p style='color:#ef4444;font-size:13px'>Audio niet beschikbaar: " + err + "</p>"
    );
  }});
}} else {{
  document.getElementById("speler").innerHTML =
    "<p style='color:#6b7280;font-size:13px'>Web Audio niet ondersteund in deze browser.</p>";
}}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Bach — Muziekoverzicht",
        page_icon="🎵",
        layout="wide",
    )
    st.title("🎵 Bach Muziekoverzicht")

    if not CSV_PATH.exists():
        st.error(
            f"Analyse-CSV niet gevonden: `{CSV_PATH}`\n\n"
            "Voer eerst uit:  `python scripts/analyse_midi.py`"
        )
        return

    df = laad_data()

    # ---- Categorie-selectie ----
    tellers = df["classificatie"].value_counts()
    cat_keys   = ["Allemaal"] + sorted(tellers.index.tolist())
    cat_labels = [f"Allemaal  ({len(df)})"] + [
        f"{cat}  ({tellers[cat]})" for cat in sorted(tellers.index)
    ]
    keuze_label = st.radio(
        "Categorie", cat_labels, horizontal=True, label_visibility="collapsed"
    )
    selected_cat = cat_keys[cat_labels.index(keuze_label)]

    st.divider()

    # ---- Gefilterde tabel ----
    gefilterd = (
        df if selected_cat == "Allemaal"
        else df[df["classificatie"] == selected_cat]
    ).reset_index(drop=True)

    st.caption(f"{len(gefilterd)} werken — klik op een rij voor details en afspelen")

    kolommen = [
        c for c in ["naam", "bestandsnaam", "classificatie", "maten", "aantal_stemmen", "stemmen"]
        if c in gefilterd.columns
    ]
    event = st.dataframe(
        gefilterd[kolommen],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "naam":           st.column_config.TextColumn("BWV",       width="small"),
            "bestandsnaam":   st.column_config.TextColumn("Bestand",   width="small"),
            "classificatie":  st.column_config.TextColumn("Categorie", width="medium"),
            "maten":          st.column_config.NumberColumn("Maten",   width="small"),
            "aantal_stemmen": st.column_config.NumberColumn("St.",     width="small"),
            "stemmen":        st.column_config.TextColumn("Stemnamen", width="large"),
        },
    )

    if not event.selection.rows:
        return

    # ---- Geselecteerde rij ----
    rij = gefilterd.iloc[event.selection.rows[0]]
    midi_pad = RAW_DIR / str(rij["bestandsnaam"])
    naam = str(rij["naam"])

    st.divider()
    st.subheader(f"🎼 {naam}  ·  {rij['classificatie']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Maten",     int(rij["maten"])          if pd.notna(rij.get("maten"))          else "—")
    c2.metric("Stemmen",   int(rij["aantal_stemmen"])  if pd.notna(rij.get("aantal_stemmen")) else "—")
    c3.metric("Categorie", rij["classificatie"])
    c4.metric("Bestand",   rij["bestandsnaam"])

    # Stemnamen bepalen
    stemmen_val = rij.get("stemmen", "")
    if pd.notna(stemmen_val) and str(stemmen_val).strip():
        stem_namen = [s.strip() for s in str(stemmen_val).split("|")]
    else:
        aantal = int(rij["aantal_stemmen"]) if pd.notna(rij.get("aantal_stemmen")) else 0
        stem_namen = [f"Stem {i + 1}" for i in range(aantal)]

    st.write("")

    if not midi_pad.exists():
        st.warning(f"MIDI-bestand niet gevonden: `{midi_pad}`")
        return

    # ---- Stemkeuze ----
    geselecteerde_namen = st.multiselect(
        "Stemmen afspelen",
        options=stem_namen,
        default=stem_namen,
        placeholder="Kies een of meer stemmen…",
    )
    if not geselecteerde_namen:
        st.info("Selecteer minstens één stem om af te spelen.")
        return

    geselecteerde_indices = tuple(
        i for i, n in enumerate(stem_namen) if n in geselecteerde_namen
    )

    # ---- Tabs ----
    tab_partituur, tab_pianorol = st.tabs(["🎼 Partituur", "🎹 Piano Roll"])

    with tab_partituur:
        with st.spinner("Partituur laden…"):
            abc = laad_abc(naam, geselecteerde_indices)
        if abc:
            st.components.v1.html(partituur_html(abc), height=750, scrolling=True)
        else:
            st.warning("Partituur kon niet worden gegenereerd voor dit werk.")

    with tab_pianorol:
        with st.spinner("Piano roll laden…"):
            midi_b64 = laad_gefilterde_midi_b64(str(midi_pad), geselecteerde_indices)
        st.components.v1.html(piano_roll_html(midi_b64), height=430, scrolling=False)


if __name__ == "__main__":
    main()
