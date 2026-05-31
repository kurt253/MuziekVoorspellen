# BachAI — Muziekgeneratie met LSTM

Een Streamlit-webapplicatie die Bach-choralen analyseert, visualiseert en de tweede helft van een werk automatisch aanvult met een zelf getraind LSTM-neuraal netwerk.

---

## Wat doet het?

1. **Overzicht** — Blader door het volledige Bach-corpus (choralen, cantates, passions, ...), bekijk partituren en beluister werken via een interactieve piano roll.
2. **Training** — Train een LSTM-model op een selectie van Bach-werken. Het model leert welke noot logisch volgt op een reeks voorgaande noten.
3. **Generatie** — Kies een werk en een getraind model. De app neemt de eerste helft als context en laat de AI de tweede helft opnieuw componeren.

---

## Projectstructuur

```
MuziekVoorspellen/
├── app.py                        # Startpagina: Bach-overzicht + afspelen
├── pages/
│   ├── Training.py               # Model trainen
│   └── Genereer.py               # Muziek genereren
├── scripts/
│   ├── download_bach.py          # Bach-corpus downloaden via music21
│   ├── analyse_midi.py           # MIDI-bestanden analyseren → CSV
│   └── train.py                  # Hulpscript voor dataselectie
├── data/
│   ├── raw/bach/                 # MIDI + MusicXML bestanden
│   └── processed/                # CSV-analyses + getrainde modellen (.pt)
├── requirements.txt
└── README.md
```

---

## Installatie

### 1. Repository klonen

```bash
git clone <url>
cd MuziekVoorspellen
```

### 2. Virtuele omgeving aanmaken

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Afhankelijkheden installeren

```bash
pip install -r requirements.txt
```

**Afhankelijkheden:**

| Pakket | Versie | Gebruik |
|--------|--------|---------|
| `streamlit` | ≥ 1.35 | Webinterface |
| `torch` | ≥ 2.2 | LSTM-model |
| `music21` | ≥ 9.1 | MIDI-analyse + Bach-corpus |
| `numpy` | ≥ 1.26 | Matrixoperaties |
| `pandas` | ≥ 2.2 | Dataverwerking |
| `matplotlib` | ≥ 3.8 | Trainingsgrafieken |

---

## Gebruik — stap voor stap

### Stap 1: Bach-corpus downloaden

```bash
python scripts/download_bach.py
```

Dit haalt alle Bach-werken op uit het ingebouwde `music21`-corpus en exporteert ze als MIDI en MusicXML naar `data/raw/bach/`. Een metadata-CSV wordt opgeslagen in `data/processed/bach_metadata.csv`.

Opties:
```bash
python scripts/download_bach.py --max 10      # alleen eerste 10 werken (testen)
python scripts/download_bach.py --midi-only   # alleen MIDI, geen MusicXML
```

### Stap 2: MIDI-bestanden analyseren

```bash
python scripts/analyse_midi.py
```

Analyseert elk MIDI-bestand: aantal maten, stemmen, stemnamen en genre (op basis van BWV-nummer). Resultaat: `data/processed/bach_analyse.csv`.

```bash
python scripts/analyse_midi.py --max 10       # snelle test op 10 bestanden
```

### Stap 3: App starten

```bash
streamlit run app.py
```

De browser opent automatisch op `http://localhost:8501`.

---

## Pagina's

### Startpagina — Overzicht

- Filter werken op categorie (koraal, cantate, Matthäus-Passion, ...)
- Klik op een rij voor details: maten, stemmen, genre
- Kies welke stemmen je wilt horen via multiselect
- **Partituurweergave** met gesynchroniseerde notenhighlighting (via abcjs)
- **Piano roll** met interactieve speler (via html-midi-player)

### Training

Selecteer een type werk en aantal werken, stel parameters in en train een LSTM-model.

**Parameters:**

| Parameter | Standaard | Uitleg |
|-----------|-----------|--------|
| Venster | 16 | Aantal vorige noten als context. Moet gelijk zijn aan venster bij generatie. |
| Epochs | 50 | Maximum trainingsrondes. Early stopping stopt eerder. |
| Hidden units | 256 | Geheugengrootte per LSTM-laag. Groter = expressiever, maar meer overfit-risico. |
| LSTM-lagen | 2 | Diepte van het netwerk. 2 is een goede balans. |
| Batchgrootte | 128 | Sequenties per gewichtsupdate. |
| Validatie % | 20 | Percentage data apart gehouden als validatieset. |
| Geduld | 7 | Early stopping: stop na dit aantal epochs zonder verbetering van validatieverlies. |

**Trainingsproces:**

1. De MIDI-bestanden worden ingelezen en omgezet naar tokens: `(toonhoogte, duur)`.
2. De tokens worden gesplitst in trainings- en validatiesequenties.
3. Na elke epoch wordt het validatieverlies berekend.
4. Het model met het **laagste validatieverlies** wordt opgeslagen — niet per se de laatste epoch.
5. Als het validatieverlies `geduld` epochs niet verbetert, stopt de training automatisch.

Na afloop toont de app een grafiek met het trainings- én validatieverlies per epoch.

**Overfitting herkennen:**

```
Goed:     train ↘  val ↘  (beide dalen samen)
Overfit:  train ↘  val ↗  (val gaat omhoog terwijl train daalt)
```

Bij overfit: selecteer meer werken, verklein het model (hidden = 128, lagen = 1) of verlaag het venster.

### Genereren

Kies een getraind model en een Bach-werk. De app:

1. Leest het werk in en splitst het in twee helften.
2. Gebruikt de **eerste helft als seed** (context voor het model).
3. Genereert de **tweede helft** per stem in twee fasen:
   - **Fase 1 — Ritme:** bepaalt de nootduren via kansmarginalisatie.
   - **Fase 2 — Melodie:** kiest toonhoogtes passend bij het vastgestelde ritme.
4. Toont het origineel en de AI-aanvulling naast elkaar met gekleurde piano rolls.

**Generatieparameters:**

| Parameter | Uitleg |
|-----------|--------|
| Temperatuur (0.1–2.0) | Lager = voorspelbaarder/Bach-achtiger. Hoger = creatiever/willekeuriger. |
| Top-k (0–50) | Beperkt keuze tot de k meest waarschijnlijke tokens. 0 = uitgeschakeld. |
| Venster | Moet overeenkomen met het venster gebruikt tijdens training. |

**Aanbevolen instellingen:**

| Doel | Temperatuur | Top-k |
|------|-------------|-------|
| Bach-getrouw | 0.7 | 5 |
| Goede balans | 0.9 | 10 |
| Creatief | 1.2 | 20 |

---

## Model — technische details

### Architectuur

```
Invoer (tokenreeks van lengte venster)
    ↓
Embedding-laag  (token → dichte vector, standaard 64 dimensies)
    ↓
LSTM  (standaard 2 lagen, 256 hidden units per laag)
    ↓
Dropout  (standaard 30%)
    ↓
Fully connected laag  (→ kansen over alle tokens in het vocabulaire)
```

### Tokenisatie

Elke noot wordt voorgesteld als een token `(MIDI-toonhoogte, duur)`:
- Toonhoogte: MIDI-waarde 0–127 (0 = rust)
- Duur: gekwantiseerd naar `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]` kwartnootheden

Elke unieke combinatie krijgt een integer-index. Het vocabulaire wordt per model opgeslagen als `.pkl`.

### Opgeslagen bestanden

Getrainde modellen staan in `data/processed/`:

```
lstm_koraal_50.pt           # modelgewichten (PyTorch)
lstm_koraal_50_vocab.pkl    # vocabulaire (Python dict)
```

De bestandsnaam bevat het type werk en het aantal gebruikte werken.

---

## GPU-ondersteuning

De app detecteert automatisch het beste beschikbare apparaat:

| Apparaat | Vereiste |
|----------|----------|
| Intel Arc GPU (XPU) | `intel_extension_for_pytorch` geïnstalleerd |
| NVIDIA GPU (CUDA) | CUDA-versie van PyTorch |
| CPU | Altijd beschikbaar (trager) |

---

## Tips voor betere resultaten

- **Gebruik meer werken** — meer data vermindert overfit sterk. Streef naar 20.000+ trainingssequenties.
- **Venster consistent houden** — gebruik hetzelfde venster bij training én generatie.
- **Validatiecurve volgen** — als validatie al vroeg stijgt, verklein het model of gebruik meer data.
- **Geduld aanpassen** — bij weinig data kan een lager geduld (5) vroegtijdiger stoppen en overfit beperken.
- **Traineer per genre** — een model getraind alleen op choralen presteert beter op choralen dan een model op het volledige corpus.
