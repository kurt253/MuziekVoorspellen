"""Analyseer alle Bach MIDI-bestanden: maten, stemmen, stemnamen, classificatie.

Gebruik::

    python scripts/analyse_midi.py              # alles analyseren
    python scripts/analyse_midi.py --max 10     # eerste 10 (voor testen)

Uitvoer: data/processed/bach_analyse.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "bach"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Classificatie op basis van BWV-nummer
# ---------------------------------------------------------------------------

def _classificeer(naam: str, aantal_stemmen: int) -> str:
    """Bepaal het genre op basis van BWV-nummer en stemmencount."""
    kern = naam.lower().replace("bwv", "")
    hoofd = kern.split(".")[0].split("-")[0]
    try:
        bwv = int(hoofd)
    except ValueError:
        return "onbekend"

    if 1 <= bwv <= 224:
        # Slotzang van een cantate (beweging >= 4, 4 stemmen) is vrijwel altijd een koraal
        if "." in kern:
            try:
                mov = int(kern.split(".")[1].split("-")[0])
                if mov >= 4 and aantal_stemmen == 4:
                    return "koraal (cantate)"
            except (ValueError, IndexError):
                pass
        return "cantate"
    elif 225 <= bwv <= 231:
        return "motet"
    elif 232 <= bwv <= 236:
        return "mis"
    elif 237 <= bwv <= 243:
        return "sanctus"
    elif bwv == 244:
        return "Matthäus-Passion"
    elif bwv == 245:
        return "Johannes-Passion"
    elif bwv == 247:
        return "Markus-Passion"
    elif bwv == 248:
        return "Kerstoratorium"
    elif bwv == 249:
        return "Paasoratorium"
    elif 250 <= bwv <= 438:
        return "koraal"
    else:
        return "overige"


# ---------------------------------------------------------------------------
# MIDI-analyse
# ---------------------------------------------------------------------------

def _analyseer_midi(midi_pad: Path) -> dict:
    from music21 import converter

    naam = midi_pad.stem
    try:
        partituur = converter.parse(str(midi_pad))
        delen = partituur.parts
        aantal_stemmen = len(delen)

        # Stemnamen (MIDI-tracknaam, anders fallback)
        stem_namen = []
        for i, deel in enumerate(delen):
            # music21 slaat de MIDI-tracknaam op in partName of id
            naam_deel = (
                getattr(deel, "partName", None)
                or getattr(deel, "id", None)
                or f"Stem {i + 1}"
            )
            stem_namen.append(str(naam_deel).strip())

        # Maten tellen op basis van het eerste deel
        eerste_deel = delen[0] if delen else None
        maten = (
            len(eerste_deel.getElementsByClass("Measure")) if eerste_deel else 0
        )

        classificatie = _classificeer(naam, aantal_stemmen)

        return {
            "naam": naam,
            "bestandsnaam": midi_pad.name,
            "classificatie": classificatie,
            "maten": maten,
            "aantal_stemmen": aantal_stemmen,
            "stemmen": " | ".join(stem_namen),
        }

    except Exception as fout:
        return {
            "naam": naam,
            "bestandsnaam": midi_pad.name,
            "classificatie": "fout",
            "maten": "",
            "aantal_stemmen": "",
            "stemmen": f"FOUT: {fout}",
        }


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def _voortgangsbalk(huidig: int, totaal: int, breedte: int = 40) -> str:
    gevuld = int(breedte * huidig / totaal)
    balk = "█" * gevuld + "░" * (breedte - gevuld)
    return f"[{balk}] {huidig:3d}/{totaal}"


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def analyseer(max_bestanden: int | None = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    midi_bestanden = sorted(RAW_DIR.glob("*.mid"))
    if not midi_bestanden:
        print(f"Geen MIDI-bestanden gevonden in {RAW_DIR}")
        print("Voer eerst scripts/download_bach.py uit.")
        return

    if max_bestanden:
        midi_bestanden = midi_bestanden[:max_bestanden]

    totaal = len(midi_bestanden)
    print(f"🎵 {totaal} MIDI-bestanden gevonden in {RAW_DIR}\n")

    resultaten: list[dict] = []
    fouten: list[str] = []

    for i, pad in enumerate(midi_bestanden, start=1):
        print(f"\r{_voortgangsbalk(i, totaal)}  {pad.stem:<45}", end="", flush=True)
        rij = _analyseer_midi(pad)
        if "FOUT" in rij.get("stemmen", ""):
            fouten.append(rij["naam"])
        resultaten.append(rij)

    print(f"\r{_voortgangsbalk(totaal, totaal)}  {'Klaar!':<45}\n")

    # Samenvatting per classificatie
    from collections import Counter
    tellers = Counter(r["classificatie"] for r in resultaten)
    print("Overzicht per categorie:")
    for cat, n in sorted(tellers.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {n:>4} werken")

    # CSV schrijven
    csv_pad = PROCESSED_DIR / "bach_analyse.csv"
    kolommen = ["naam", "bestandsnaam", "classificatie", "maten", "aantal_stemmen", "stemmen"]
    with open(csv_pad, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kolommen)
        writer.writeheader()
        writer.writerows(resultaten)

    print(f"\n✓ Resultaten opgeslagen → {csv_pad}")
    if fouten:
        print(f"⚠  {len(fouten)} fout(en): {', '.join(fouten)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parseer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyseer Bach MIDI-bestanden op maten, stemmen en genre.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Maximaal aantal bestanden (handig voor testen).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parseer_args()
    analyseer(max_bestanden=args.max)
