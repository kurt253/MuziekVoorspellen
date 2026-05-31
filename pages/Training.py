"""Training — selecteer type en aantal, configureer en train het LSTM-model."""
from __future__ import annotations

import os
import glob
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paden ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
ANALYSE_CSV  = DATA_DIR / "processed" / "bach_analyse.csv"
METADATA_CSV = DATA_DIR / "processed" / "bach_metadata.csv"
MODEL_DIR    = DATA_DIR / "processed"

# ── Constanten (uit notebook) ─────────────────────────────────────────────────
BEKENDE_TYPES = [
    "koraal",
    "koraal (cantate)",
    "cantate",
    "onbekend",
    "Kerstoratorium",
    "Matthäus-Passion",
    "Johannes-Passion",
    "motet",
    "overige",
]

DUREN = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def _laad_alles() -> pd.DataFrame:
    analyse  = pd.read_csv(ANALYSE_CSV)
    metadata = pd.read_csv(METADATA_CSV)
    metadata["midi_pad"] = metadata["midi_pad"].apply(lambda p: str(DATA_DIR / p))
    return (
        analyse.merge(metadata, on="naam", how="inner")
        .drop(columns=["bestandsnaam"], errors="ignore")
    )


# ── MIDI-verwerking ───────────────────────────────────────────────────────────
# VoiceToken = (midi_pitch, duur)  waarbij pitch=0 = rust
VoiceToken = tuple[int, float]

def _kwantiseer_duur(duur: float) -> float:
    return min(DUREN, key=lambda d: abs(d - duur))


def _midi_naar_stemmen(pad: str) -> list[list[VoiceToken]]:
    """Parseer MIDI naar afzonderlijke stemmen als (pitch, duur) tokens."""
    from music21 import converter, note, chord
    score   = converter.parse(pad)
    stemmen: list[list[VoiceToken]] = []
    for part in score.parts:
        tokens: list[VoiceToken] = []
        for el in part.flatten().notesAndRests:
            duur = _kwantiseer_duur(float(el.duration.quarterLength))
            if duur == 0:
                continue
            if isinstance(el, note.Rest):
                tokens.append((0, duur))
            elif isinstance(el, note.Note):
                tokens.append((el.pitch.midi, duur))
            elif isinstance(el, chord.Chord):
                tokens.append((max(p.midi for p in el.pitches), duur))
        if tokens:
            stemmen.append(tokens)
    return stemmen


def _bouw_vocab(df: pd.DataFrame) -> dict[VoiceToken, int]:
    tokens: set[VoiceToken] = set()
    for pad in df["midi_pad"]:
        try:
            for stem in _midi_naar_stemmen(pad):
                tokens.update(stem)
        except Exception:
            pass
    return {t: i for i, t in enumerate(sorted(tokens))}


def _df_naar_reeksen(
    df: pd.DataFrame, vocab: dict[VoiceToken, int], venster: int
) -> tuple[np.ndarray, np.ndarray]:
    """Alle stemmen van alle MIDI-bestanden → trainingssequenties."""
    alle_X, alle_y = [], []
    for _, rij in df.iterrows():
        try:
            for stem in _midi_naar_stemmen(rij["midi_pad"]):
                codes = [vocab[t] for t in stem if t in vocab]
                for i in range(len(codes) - venster):
                    alle_X.append(codes[i : i + venster])
                    alle_y.append(codes[i + venster])
        except Exception:
            pass
    if not alle_X:
        return np.zeros((0, venster), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.array(alle_X, dtype=np.int64), np.array(alle_y, dtype=np.int64)


# ── Model (uit notebook) ──────────────────────────────────────────────────────
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


# ── Apparaat ──────────────────────────────────────────────────────────────────
def _kies_apparaat() -> torch.device:
    for dll_pad in glob.glob(
        r"C:\Windows\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_*\igc64.dll"
    ):
        try:
            os.add_dll_directory(os.path.dirname(dll_pad))
        except Exception:
            pass
    try:
        import importlib.metadata as im
        tv = im.version("torch").split("+")[0]
        iv = im.version("intel_extension_for_pytorch").split("+")[0]
        if ".".join(tv.split(".")[:2]) == ".".join(iv.split(".")[:2]):
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if torch.xpu.is_available():
                return torch.device("xpu")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Modelbeheer ───────────────────────────────────────────────────────────────
def _toon_modelbeheer() -> None:
    import re as _re
    from datetime import datetime

    st.subheader("Bestaande modellen")

    modellen = sorted(MODEL_DIR.glob("lstm_*.pt"))
    if not modellen:
        st.caption("Geen modellen gevonden in `data/processed/`.")
        return

    for model_pad in modellen:
        vocab_pad = model_pad.with_name(model_pad.stem + "_vocab.pkl")

        # Bestandsinfo
        mtime = datetime.fromtimestamp(model_pad.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        mb    = model_pad.stat().st_size / 1024 / 1024

        # Naam uit bestandsnaam afleiden
        m = _re.match(r"lstm_(.+)_(\d+)\.pt$", model_pad.name)
        label = f"{m.group(1).replace('_', ' ')}  —  {m.group(2)} werken" if m else model_pad.name

        col_naam, col_info, col_del = st.columns([3, 2, 1])
        col_naam.markdown(f"**{label}**  \n`{model_pad.name}`")
        col_info.caption(f"{mtime}  ·  {mb:.1f} MB")

        if col_del.button("🗑 Verwijder", key=f"del_{model_pad.name}"):
            model_pad.unlink(missing_ok=True)
            vocab_pad.unlink(missing_ok=True)
            st.success(f"`{model_pad.name}` verwijderd.")
            st.rerun()


# ── Scherm ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Training", page_icon="🎓", layout="wide")
    st.title("🎓 Model trainen")

    if not ANALYSE_CSV.exists():
        st.error(f"Analyse-CSV niet gevonden: `{ANALYSE_CSV}`")
        return

    df_alles = _laad_alles()
    tellers  = df_alles["classificatie"].value_counts()

    # ── Modelbeheer ───────────────────────────────────────────────────────────
    _toon_modelbeheer()

    st.divider()

    # ── Selectie ──────────────────────────────────────────────────────────────
    st.subheader("Nieuw model trainen")

    col_type, col_info = st.columns([2, 1])

    with col_type:
        opties = ["alles"] + BEKENDE_TYPES
        soort = st.selectbox(
            "Type werk",
            options=opties,
            format_func=lambda t: (
                f"Alles  ({len(df_alles)} werken)"
                if t == "alles"
                else f"{t}  ({tellers.get(t, 0)} werken)"
            ),
        )

    df_type = df_alles if soort == "alles" else df_alles[df_alles["classificatie"] == soort]
    max_aantal = len(df_type)

    with col_info:
        st.metric("Beschikbaar", max_aantal)

    if max_aantal == 0:
        st.warning("Geen werken gevonden voor dit type.")
        return

    aantal = st.slider(
        "Aantal werken voor training",
        min_value=1,
        max_value=max_aantal,
        value=min(20, max_aantal),
        help="Meer werken = rijker model, maar langere trainingstijd",
    )

    df_selectie = df_type.sample(n=aantal, random_state=42).reset_index(drop=True)

    st.caption(f"{aantal} van {max_aantal} werken geselecteerd")
    st.dataframe(
        df_selectie[["naam", "classificatie", "maten", "aantal_stemmen", "stemmen"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "naam":           st.column_config.TextColumn("BWV",       width="small"),
            "classificatie":  st.column_config.TextColumn("Categorie", width="medium"),
            "maten":          st.column_config.NumberColumn("Maten",   width="small"),
            "aantal_stemmen": st.column_config.NumberColumn("St.",     width="small"),
            "stemmen":        st.column_config.TextColumn("Stemmen",   width="large"),
        },
    )

    # ── Trainingsparameters ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Trainingsparameters")

    col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(7)
    venster       = col_a.number_input("Venster",        min_value=4,  max_value=64,  value=16, step=4)
    epochs        = col_b.number_input("Epochs",         min_value=5,  max_value=200, value=50, step=5)
    hidden        = col_c.number_input("Hidden units",   min_value=64, max_value=512, value=256, step=64)
    lagen         = col_d.number_input("LSTM-lagen",     min_value=1,  max_value=4,   value=2)
    batch_grootte = col_e.number_input("Batchgrootte",   min_value=16, max_value=512, value=128, step=16)
    val_split     = col_f.number_input("Validatie %",    min_value=5,  max_value=30,  value=20, step=5,
                                       help="Percentage data apart gehouden als validatieset")
    geduld        = col_g.number_input("Geduld",         min_value=3,  max_value=30,  value=7,  step=1,
                                       help="Early stopping: stop na dit aantal epochs zonder verbetering van validatieverlies")

    # ── Modelbestandsnaam ─────────────────────────────────────────────────────
    soort_label = re.sub(r"[^\w]", "_", soort).strip("_")
    model_stam  = f"lstm_{soort_label}_{aantal}"
    model_pad   = MODEL_DIR / f"{model_stam}.pt"
    vocab_pad   = MODEL_DIR / f"{model_stam}_vocab.pkl"

    if model_pad.exists():
        st.info(f"Bestaand model gevonden: `{model_pad.name}` — training overschrijft dit.")

    # ── Start training ────────────────────────────────────────────────────────
    st.divider()

    if st.button("🚀 Start training", type="primary", use_container_width=True):
        _voer_training_uit(
            df_selectie=df_selectie,
            venster=int(venster),
            epochs=int(epochs),
            hidden=int(hidden),
            lagen=int(lagen),
            batch_grootte=int(batch_grootte),
            model_pad=model_pad,
            vocab_pad=vocab_pad,
            val_split=float(val_split) / 100,
            geduld=int(geduld),
        )


def _voer_training_uit(
    df_selectie: pd.DataFrame,
    venster: int,
    epochs: int,
    hidden: int,
    lagen: int,
    batch_grootte: int,
    model_pad: Path,
    vocab_pad: Path,
    val_split: float = 0.2,
    geduld: int = 7,
) -> None:
    status    = st.empty()
    voortgang = st.progress(0)
    log       = st.empty()
    train_verlies: list[float] = []
    val_verlies:   list[float] = []

    # Apparaat
    status.info("Apparaat bepalen…")
    apparaat = _kies_apparaat()
    if apparaat.type == "xpu":
        apparaat_naam = f"Intel Arc GPU (XPU) — {torch.xpu.get_device_name(0)}"
    elif apparaat.type == "cuda":
        apparaat_naam = f"NVIDIA GPU (CUDA) — {torch.cuda.get_device_name(0)}"
    else:
        apparaat_naam = "CPU"
    status.info(f"Apparaat: **{apparaat_naam}**")

    # Vocabulaire
    status.info("MIDI-bestanden inlezen en vocabulaire bouwen…")
    vocab = _bouw_vocab(df_selectie)
    if not vocab:
        st.error("Geen tokens gevonden — controleer de MIDI-paden.")
        return

    # Sequenties
    status.info("Trainingssequenties aanmaken…")
    X, y = _df_naar_reeksen(df_selectie, vocab, venster)
    if len(X) == 0:
        st.error("Te weinig data voor dit venster. Verklein het venster of kies meer werken.")
        return

    # Train/validatie splitsing
    rng      = np.random.default_rng(42)
    idx      = rng.permutation(len(X))
    n_val    = max(1, int(len(X) * val_split))
    idx_val  = idx[:n_val]
    idx_train = idx[n_val:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]

    st.caption(
        f"Vocab: **{len(vocab)}** tokens · "
        f"Training: **{len(X_train):,}** · "
        f"Validatie: **{len(X_val):,}** · "
        f"Apparaat: **{apparaat.type.upper()}**"
    )

    gebruik_pin   = apparaat.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    lader         = DataLoader(train_dataset, batch_size=batch_grootte, shuffle=True,
                               pin_memory=gebruik_pin, num_workers=0)
    val_lader     = DataLoader(val_dataset,   batch_size=batch_grootte, shuffle=False,
                               pin_memory=gebruik_pin, num_workers=0)

    # Model
    model     = LSTMModel(len(vocab), hidden=hidden, lagen=lagen).to(apparaat)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterium = nn.CrossEntropyLoss()

    if apparaat.type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        except Exception:
            pass

    beste_val      = float("inf")
    beste_state    = None
    geen_verbetering = 0
    vroeg_gestopt  = False

    # Trainingslus
    status.info(f"Training: max {epochs} epochs (early stopping na {geduld} zonder verbetering)…")
    for epoch in range(1, epochs + 1):
        model.train()
        totaal = 0.0
        for xb, yb in lader:
            xb, yb = xb.to(apparaat, non_blocking=True), yb.to(apparaat, non_blocking=True)
            optimizer.zero_grad()
            v = criterium(model(xb), yb)
            v.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            totaal += v.item() * len(xb)

        gem_train = totaal / len(train_dataset)
        train_verlies.append(gem_train)

        # Validatiestap
        model.eval()
        val_totaal = 0.0
        with torch.no_grad():
            for xb, yb in val_lader:
                xb, yb = xb.to(apparaat, non_blocking=True), yb.to(apparaat, non_blocking=True)
                val_totaal += criterium(model(xb), yb).item() * len(xb)
        gem_val = val_totaal / len(val_dataset)
        val_verlies.append(gem_val)

        # Beste model bijhouden op basis van validatieverlies
        if gem_val < beste_val:
            beste_val        = gem_val
            beste_state      = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            geen_verbetering = 0
            label_extra      = " ✓ beste"
        else:
            geen_verbetering += 1
            label_extra       = f" ({geen_verbetering}/{geduld})"

        voortgang.progress(epoch / epochs)
        log.markdown(
            f"Epoch **{epoch}/{epochs}** &nbsp;·&nbsp; "
            f"train = `{gem_train:.4f}` &nbsp;·&nbsp; "
            f"val = `{gem_val:.4f}`{label_extra}"
        )

        if geen_verbetering >= geduld:
            vroeg_gestopt = True
            break

    # Beste model opslaan (laagste validatieverlies)
    status.info("Beste model opslaan…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(beste_state, model_pad)
    with open(vocab_pad, "wb") as f:
        pickle.dump(vocab, f)

    voortgang.progress(1.0)
    overfit = val_verlies[-1] - train_verlies[-1]
    overfit_waarsch = "⚠️ mogelijk overfit" if overfit > 0.5 else "✅ geen overfit"
    stop_label = (
        f"Vroeg gestopt na **{len(train_verlies)}** epochs (geen verbetering voor {geduld} epochs)"
        if vroeg_gestopt
        else f"Voltooid na **{len(train_verlies)}** epochs"
    )
    status.success(
        f"{stop_label} · beste validatieverlies: **{beste_val:.4f}** · {overfit_waarsch}\n\n"
        f"Model: `{model_pad.name}` &nbsp;·&nbsp; "
        f"Vocab: `{vocab_pad.name}`"
    )

    # Verliesplot met train én validatie
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    xs = range(1, len(train_verlies) + 1)
    ax.plot(xs, train_verlies, marker="o", markersize=3, label="Training")
    ax.plot(xs, val_verlies,   marker="o", markersize=3, label="Validatie", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Verlies")
    ax.set_title("Trainingsverloop")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


main()
