# Hoe werkt Muziekvoorspellen?

Dit document legt de volledige werking van het project uit: van het downloaden van de data tot het afspelen van AI-gegenereerde muziek. Per bestand worden alle functies beschreven.

---

## Overzicht

Het project bestaat uit vier stappen die in volgorde worden uitgevoerd:

```
1. Download      →  scripts/download_bach.py
2. Analyseer     →  scripts/analyse_midi.py
3. Train         →  Streamlit: pages/Training.py
4. Genereer      →  Streamlit: pages/Genereer.py
```

De Streamlit-app (`app.py`) dient als overzichtspagina om het corpus te bekijken en te beluisteren.

---

## Mapstructuur

```
Muziekvoorspellen/
├── app.py                          # Streamlit overzichtspagina
├── pages/
│   ├── Training.py                 # LSTM trainen via Streamlit
│   └── Genereer.py                 # Muziek genereren via Streamlit
├── scripts/
│   ├── download_bach.py            # Bach-corpus downloaden
│   ├── analyse_midi.py             # MIDI-bestanden analyseren
│   └── train.py                    # Hulpscript: trainingsdata selecteren
└── data/
    ├── raw/bach/                   # Ruwe MIDI- en XML-bestanden
    └── processed/
        ├── bach_metadata.csv       # Naam, stemmen, BPM, MIDI-pad, …
        ├── bach_analyse.csv        # Classificatie, maten, stemnamen, …
        ├── lstm_<type>_<n>.pt      # Getraind model
        └── lstm_<type>_<n>_vocab.pkl  # Bijbehorend vocabulaire
```

---

## `scripts/download_bach.py`

Downloadt het volledige Bach-corpus uit de ingebouwde bibliotheek van music21 en exporteert elk werk als MIDI en MusicXML naar `data/raw/bach/`.

### Functies

#### `_zorg_voor_music21()`
Controleert of het `music21`-pakket beschikbaar is. Als dat niet het geval is, installeert het automatisch via `pip` zodat het script ook werkt op een verse Python-omgeving.

#### `_maak_mappen()`
Maakt de mappen `data/raw/bach/` en `data/processed/` aan als ze nog niet bestaan. Gebruikt `parents=True` zodat tussenliggende mappen ook aangemaakt worden.

#### `_extraheer_metadata(partituur, naam, midi_pad, xml_pad) → dict`
Extraheert basismetadata uit een music21-partituurobject terwijl het nog in geheugen staat. Geeft de volgende velden terug:

| Veld | Inhoud |
|---|---|
| `naam` | BWV-identifier |
| `stemmen` | Aantal Parts |
| `maten` | Aantal maten (eerste Part) |
| `toonsoort` | Eerste KeySignature als string |
| `maatsoort` | Eerste TimeSignature als breuk |
| `bpm` | Tempo in BPM (None als niet aanwezig) |
| `aantal_noten` | Totaal noten en akkoorden |
| `midi_pad` | Relatief pad t.o.v. `data/` |
| `xml_pad` | Relatief pad t.o.v. `data/` (leeg bij `--midi-only`) |

#### `_voortgangsbalk(huidig, totaal, breedte) → str`
Genereert een ASCII-voortgangsbalk voor terminal-uitvoer, bijv.: `[████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  42/400`.

#### `download_bach_corpus(max_werken, midi_only)`
De hoofdfunctie. Haalt alle Bach-paden op via `corpus.getComposer("bach")`, slaat bestaande bestanden over (hervatbaar), exporteert elk werk als MIDI en optioneel MusicXML, en schrijft alle metadata naar `bach_metadata.csv`. Beide formaten worden vanuit hetzelfde music21-object geschreven — de XML is niet afgeleid van de MIDI.

#### `_schrijf_csv(metadata_lijst)`
Schrijft een lijst van dicts naar `bach_metadata.csv`. Verzamelt alle kolommen die in de lijst voorkomen en vult ontbrekende velden aan met een lege string zodat de CSV altijd rechthoekig is.

#### `_parseer_argumenten() → Namespace`
Parseert `--max N` en `--midi-only` voor de command-line interface.

---

## `scripts/analyse_midi.py`

Leest alle MIDI-bestanden in `data/raw/bach/` in via music21 en schrijft `bach_analyse.csv` met classificatie, maten en stemnamen.

### Functies

#### `_classificeer(naam, aantal_stemmen) → str`
Bepaalt het muzikale genre op basis van het BWV-nummer. De BWV-reeks wordt gebruikt als classificatiecriterium:

| BWV-reeks | Genre |
|---|---|
| 1 – 224 | cantate (of koraal als beweging ≥ 4 én 4 stemmen) |
| 225 – 231 | motet |
| 232 – 236 | mis |
| 237 – 243 | sanctus |
| 244 | Matthäus-Passion |
| 245 | Johannes-Passion |
| 248 | Kerstoratorium |
| 250 – 438 | koraal |

Het bewegingsnummer wordt afgeleid uit het gedeelte na de punt in de bestandsnaam (bijv. `bwv227.11` → beweging 11).

#### `_analyseer_midi(midi_pad) → dict`
Analyseert één MIDI-bestand: parseer via music21, haal stemnamen op (MIDI-tracknaam als beschikbaar, anders `"Stem N"`), tel maten, classificeer. Bij een parse-fout wordt een fout-rij teruggegeven zodat één corrupt bestand het script niet blokkeert.

#### `_voortgangsbalk(huidig, totaal, breedte) → str`
Zie `download_bach.py`.

#### `analyseer(max_bestanden)`
Hoofdfunctie: verwerkt alle `.mid`-bestanden in `RAW_DIR`, toont een categorieoverzicht en schrijft alle resultaten naar `bach_analyse.csv`.

#### `_parseer_args() → Namespace`
Parseert `--max N` voor de command-line interface.

---

## `scripts/train.py`

Hulpscript voor het selecteren van MIDI-bestanden voor training. Wordt ook gebruikt door `pages/Training.py`.

### Functies

#### `maak_trainings_df(aantal, soort) → DataFrame`
Voegt `bach_analyse.csv` en `bach_metadata.csv` samen via een inner join op de BWV-naam. Filtert optioneel op genre (`soort`) en neemt een willekeurige steekproef van grootte `aantal` met vaste random seed (42) voor reproduceerbaarheid. Converteert de relatieve `midi_pad`-kolom naar absolute paden.

#### `_parse_args() → Namespace`
Parseert `--aantal`, `--soort` en `--lijst` voor de command-line interface.

---

## `app.py` — Streamlit overzichtspagina

De startpagina van de Streamlit-app. Toont het Bach-corpus in een filterbare tabel en biedt per werk een partituur- en piano-roll weergave.

### Functies

#### `laad_data() → DataFrame`  *(gecached)*
Leest `bach_analyse.csv` en zorg dat de kolom `bestandsnaam` altijd aanwezig is (compatibiliteit met oudere CSV-versies).

#### `laad_gefilterde_midi_b64(midi_pad, geselecteerde_indices) → str`  *(gecached)*
Laadt een MIDI-bestand en geeft alleen de gevraagde stemmen terug als base64-string. Als alle stemmen geselecteerd zijn, wordt het bestand direct gecodeerd (geen onnodige herverwerking). Bij een subset hercomponeert music21 een nieuw MIDI-bestand in RAM — niets wordt naar disk geschreven.

#### `laad_abc(naam, geselecteerde_indices) → str | None`  *(gecached)*
Converteert een Bach-werk naar ABC-notatie. Probeert eerst het MusicXML-bestand (rijkere info), valt terug op MIDI als XML niet beschikbaar is. Werkt via een tijdelijk `.abc`-bestand dat altijd wordt opgeruimd in de `finally`-clausule.

#### `piano_roll_html(midi_b64, hoogte) → str`
Genereert een volledig HTML-document met de `html-midi-player` webcomponent. De MIDI-data zit als data-URI in de HTML ingebed — geen serververzoek bij afspelen of downloaden.

#### `partituur_html(abc_string) → str`
Genereert een HTML-document dat bladmuziek tekent via de `abcjs`-bibliotheek. Gespeelde noten worden geel gemarkeerd via de `onEvent`-callback van `ABCJS.synth.SynthController`.

#### `main()`
Hoofdfunctie: toont een filterbaar tabeloverzicht, stemkeuze via multiselect, en twee tabs: **Partituur** (ABC/abcjs) en **Piano Roll** (html-midi-player).

---

## `pages/Training.py` — Model trainen

Streamlit-pagina voor het trainen van een LSTM-model op de Bach-MIDI-bestanden.

### Datatypes

| Alias | Type | Betekenis |
|---|---|---|
| `VoiceToken` | `tuple[int, float]` | (MIDI-pitch, duur); pitch 0 = rust |

### Functies

#### `_laad_alles() → DataFrame`  *(gecached)*
Laadt en merget `bach_analyse.csv` en `bach_metadata.csv`. Converteert relatieve MIDI-paden naar absolute paden.

#### `_kwantiseer_duur(duur) → float`
Rondt een willekeurige duur af naar de dichtstbijzijnde waarde uit het DUREN-rooster `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]`. Normaliseert micro-timings uit MIDI.

#### `_midi_naar_stemmen(pad) → list[list[VoiceToken]]`
Parseer één MIDI-bestand naar een lijst van stemmen als `(pitch, duur)`-reeksen. Bij akkoorden wordt alleen de hoogste noot gebruikt. Lege stemmen worden weggelaten.

#### `_bouw_vocab(df) → dict[VoiceToken, int]`
Bouwt het vocabulaire op door alle unieke tokens in alle geselecteerde werken te verzamelen. Geeft een gesorteerde mapping `(pitch, duur) → integer` terug. Werken die niet geparsed kunnen worden, worden stilzwijgend overgeslagen.

#### `_df_naar_reeksen(df, vocab, venster) → tuple[ndarray, ndarray]`
Converteert alle stemmen naar overlappende sliding-window sequenties voor training:
- `X.shape = (N, venster)` — invoercontext
- `y.shape = (N,)` — doeltoken (volgende token)

#### `class LSTMModel(nn.Module)`
LSTM-taalmodel met architectuur: `Embedding → LSTM → Dropout → Linear`.

- **`__init__(vocab, embed_dim, hidden, lagen, dropout)`** — Initialiseert alle lagen. Bij één LSTM-laag wordt inter-laag-dropout uitgeschakeld (PyTorch-vereiste).
- **`forward(x) → Tensor`** — Verwerkt een batch `(batch, venster)` naar logits `(batch, vocab)`. Gebruikt alleen de laatste tijdstap van de LSTM-uitvoer.

#### `_kies_apparaat() → torch.device`
Detecteert het snelste beschikbare rekenplatform. Volgorde: Intel Arc GPU (XPU via IPEX) → NVIDIA GPU (CUDA) → CPU. Laadt op Windows handmatig de Intel GPU DLL's voor XPU-ondersteuning.

#### `_toon_modelbeheer()`
Toont een overzicht van alle `lstm_*.pt`-bestanden in `MODEL_DIR` met bestandsinfo en een verwijderknop. Na verwijdering wordt `st.rerun()` aangeroepen.

#### `main()`
Hoofdfunctie: modelbeheer, werktype- en aantalsselectie, trainingsparameters configureren, startknop.

#### `_voer_training_uit(df_selectie, venster, epochs, hidden, lagen, batch_grootte, model_pad, vocab_pad, val_split, geduld)`
Volledige trainingsloop:
1. Apparaat bepalen
2. Vocabulaire bouwen via `_bouw_vocab()`
3. Sequenties aanmaken via `_df_naar_reeksen()`
4. Train/validatie splitsen (vaste seed)
5. Model + Adam-optimizer + CrossEntropy initialiseren
6. Optioneel IPEX-optimalisatie bij Intel GPU
7. Trainingsloop met gradient clipping (max norm 1.0) en early stopping
8. Beste model opslaan (laagste validatieverlies)
9. Verliesplot tonen (train + validatie)

---

## `pages/Genereer.py` — Muziek genereren

Streamlit-pagina met twee generatiemodi voor het aanvullen van Bach-muziek met een LSTM-model.

### Datatypes

| Alias | Type | Betekenis |
|---|---|---|
| `VoiceToken` | `tuple[int, float]` | (MIDI-pitch, duur); pitch 0 = rust |
| `ChordToken` | `tuple[frozenset[int], float]` | (set pitches, duur) — voor MIDI-uitvoer |

### Data- en modelfuncties

#### `_kwantiseer_duur(duur) → float`
Zie `Training.py`.

#### `_laad_data() → DataFrame`  *(gecached)*
Zie `Training.py`.

#### `_beschikbare_modellen() → list[Path]`
Geeft alle `lstm_*.pt`-bestanden in `MODEL_DIR` gesorteerd op naam.

#### `_model_label(pad) → str`
Zet een modelbestandsnaam om naar een leesbaar selectbox-label: `"<soort>  —  <n> werken  (<bestandsnaam>)"`.

#### `_laad_model_en_vocab(model_pad_str) → tuple[LSTMModel, dict]`  *(gecached)*
Laadt een getraind model en vocabulaire van disk. Leidt de modelarchitectuur automatisch af uit de state-dict (embed_dim, hidden, lagen) zodat geen aparte configuratie opgeslagen hoeft te worden.

### MIDI/token conversiefuncties

#### `_midi_naar_stemmen_per_maat(pad) → list[list[list[VoiceToken]]]`
Parseer een MIDI-bestand naar een driedimensionale structuur: `[maat][stem][token]`. Lege maten worden verwijderd. Maakt het mogelijk om op maatniveau te splitsen tussen seed en generatie.

#### `_combineer_stemmen_naar_akkoorden(stemmen) → list[ChordToken]`
Combineer per-stem VoiceTokens naar ChordTokens via tijdsuitlijning:
1. Bereken (start, einde, pitch) per noot
2. Bepaal alle unieke tijdstippen
3. Voor elk interval: verzamel actieve pitches → ChordToken

#### `_chord_tokens_naar_midi_b64(tokens) → str`
Converteer ChordTokens naar een base64-gecodeerd MIDI-bestand volledig in RAM. Geen bestanden op disk.

#### `_tempo_uit_midi(midi_pad) → float`
Leest de eerste tempomarkering in BPM. Standaard 120 bij geen markering of parse-fout.

#### `_eerste_helft_plus_ai_b64(midi_pad, helft_maten, ai_stemmen) → str`
Combineert de originele eerste helft (exact gekopieerd) met AI-gegenereerde stemmen (elk als aparte Part). Bepaalt de splitoffset op basis van `helft_maten` en bouwt het gecombineerde MIDI-bestand in RAM.

### Generatiefuncties

#### `_genereer_ritme(model, seed_codes, doelduur, duur_naar_codes, duuren, venster, temperatuur, top_k) → list[float]`
**Fase 1 van de Bach-stijl modus.** Het model genereert een duurvolgorde door de kans op een duur te berekenen als de marginale som over alle tokens met die duur:

```
P(duur | context) = Σ_pitch  P(pitch, duur | context)
```

Generatie stopt zodra de totale duur ≥ `doelduur`, met een maximumaantal stappen als veiligheidsnet.

#### `_toonaard_uit_noten(noten) → set[int]`
Detecteert de toonaard via music21's Krumhansl-Schmuckler algoritme. Geeft de MIDI-pitchklassen (0–11) van de gedetecteerde toonladder terug. Wordt gebruikt om de aanvullingsmodus te beperken tot toonaardconforme noten.

#### `_genereer_melodie(model, seed_codes, duur_seq, duur_naar_codes, inv_vocab, venster, temperatuur, top_k, toegestane_pitchklassen) → list[VoiceToken]`
**Fase 2 van beide modi.** Genereert pitches voor een vastgelegd ritmeschema. Per duurslot worden alleen tokens met die exacte duur als kandidaat beschouwd. Optionele toonaard-filtering: pitch 0 (rust) altijd toegestaan; andere pitches gefilterd op `toegestane_pitchklassen % 12`.

### Piano-roll HTML

#### `_piano_roll_html(midi_b64, hoogte, uid, maat_starts_s, totaal_maten, helft_maten, is_ai) → str`
Genereert een HTML-document met:
- `html-midi-player` webcomponent (afspelen + ingebouwde download)
- `midi-visualizer` piano-roll
- Optionele JavaScript maatenteller (poll elke 100ms): toont maatnummer in blauw (origineel) of geel (AI)

### Hoofdfuncties en resultaatweergave

#### `main()`
Opbouw: model kiezen → song kiezen (filterbare tabel) → instellingen (temperatuur, top-k) → twee generatieknoppen → resultaatweergave.

#### `_voer_generatie_uit(model_pad_str, midi_pad, song_naam, venster, temperatuur, top_k)`
**Aanvullingsmodus:**
1. Model + vocab laden
2. MIDI per maat + per stem inlezen
3. Eerste helft als seed, tweede helft als ritme-referentie
4. Fase 1: origineel ritme overnemen
5. Toonaard detecteren uit eerste helft
6. Fase 2: pitches genereren gefilterd op toonaard
7. MIDI bouwen + maat-starttijden berekenen
8. Resultaat opslaan in `st.session_state`

#### `_toon_resultaat()`
Leest resultaat uit `st.session_state` en toont: metrics, splitsbalk (blauw/geel), twee piano-rolls naast elkaar.

#### `_voer_vrije_generatie_uit(model_pad_str, midi_pad, song_naam, venster, temperatuur, top_k)`
**Bach-stijl modus:**
1. Eerste `N_SEED_MATEN` maten als seed
2. Fase 1: ritme genereren via `_genereer_ritme()`
3. Fase 2: pitches genereren zonder toonaard-filter
4. MIDI bouwen + session_state bijwerken

#### `_toon_vrij_resultaat()`
Analoog aan `_toon_resultaat()` maar voor de Bach-stijl modus.

---

## Dataflow samengevat

```
music21 corpus
      │
      ▼
download_bach.py ──► data/raw/bach/*.mid + *.xml
                              │
                ┌─────────────┤
                │             │
                ▼             ▼
       analyse_midi.py    app.py (overzicht)
       bach_analyse.csv   ├─► ABC via XML/MIDI → partituur (abcjs)
                │         └─► Piano roll via MIDI (html-midi-player)
                │
                ▼
         Training.py
         ┌─────────────────────────────────────────┐
         │ MIDI → _midi_naar_stemmen()              │
         │      → _bouw_vocab()                    │
         │      → _df_naar_reeksen()               │
         │      → LSTMModel trainen                │
         │      → lstm_*.pt + vocab.pkl opslaan    │
         └─────────────────────────────────────────┘
                │
                ▼
         Genereer.py
         ┌────────────────────────────────────────────────────┐
         │ Aanvulling:                                        │
         │   seed = eerste helft  │  ritme = origineel        │
         │   toonaard detecteren  │  _genereer_melodie()      │
         │                                                    │
         │ Bach-stijl:                                        │
         │   seed = eerste 4 maten                           │
         │   _genereer_ritme() + _genereer_melodie()         │
         │                                                    │
         │ → _eerste_helft_plus_ai_b64()                     │
         │ → base64 MIDI in HTML → browser (geen serverreq.) │
         └────────────────────────────────────────────────────┘
```

---

## Sleutelconcepten

### VoiceToken
Een `(MIDI-pitch, duur)`-paar. Pitch 0 = rust. Duur in kwartnootheden, gekwantiseerd naar `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]`.

### Vocabulaire
Alle unieke VoiceTokens in het trainingskorpus, gesorteerd en genummerd. Het model leert een kansverdeling over dit vocabulaire. Opgeslagen als `.pkl`-bestand naast het model.

### Temperatuur
Regelt de willekeur van de token-sampling. Lage waarde (bijv. 0.3): voorspelbaar/Bach-achtig. Hoge waarde (bijv. 1.5): creatief/willekeurig. Technisch: `probs = softmax(logits / temperatuur)`.

### Top-k
Vóór de sampling worden alleen de k meest waarschijnlijke tokens overwogen. Hogere k = meer variatie. k=0 = alle tokens.

### Early stopping
De training stopt zodra het validatieverlies `geduld` epochs lang niet verbeterd is. Het model met het laagste validatieverlies wordt opgeslagen, niet het model van de laatste epoch.
