# MuziekVoorspellen

Muziekaanbeveling en -voorspelling op basis van audiofeatures.

## Projectstructuur

```
MuziekVoorspellen/
├── muziekvoorspellen/        # Hoofdpakket
│   ├── data/                 # Data-laden en -verwerking
│   ├── features/             # Feature-extractie
│   ├── models/               # Voorspellingsmodellen
│   └── utils/                # Hulpfuncties
├── tests/                    # Unittests
├── data/
│   ├── raw/                  # Ruwe data (niet ingecheckt)
│   └── processed/            # Verwerkte data (niet ingecheckt)
├── scripts/
│   └── train.py              # Trainingsscript
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installatie

```bash
# Kloon de repository
git clone <url>
cd MuziekVoorspellen

# Maak een virtuele omgeving aan
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Installeer afhankelijkheden (inclusief dev-tools)
pip install -e ".[dev]"
```

## Gebruik

```bash
# Voer het trainingsscript uit
python scripts/train.py
```

## Tests uitvoeren

```bash
pytest
```

## Codekwaliteit

```bash
ruff check .
ruff format .
```
