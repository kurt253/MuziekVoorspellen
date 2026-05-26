"""Download en exporteer het volledige Bach-corpus via music21.

Werkt als volgt:
  1. Controleert of music21 beschikbaar is (installeert het anders automatisch).
  2. Haalt alle Bach-werken op uit het ingebouwde music21-corpus.
  3. Exporteert elk werk als MIDI en MusicXML naar data/raw/bach/.
  4. Schrijft een metadata-overzicht naar data/processed/bach_metadata.csv.

Gebruik::

    python scripts/download_bach.py              # alles downloaden
    python scripts/download_bach.py --max 10     # eerste 10 werken (testen)
    python scripts/download_bach.py --midi-only  # alleen MIDI exporteren
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Zorg dat de projectroot op het importpad staat.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "bach"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def _zorg_voor_music21() -> None:
    """Installeer music21 als het nog niet beschikbaar is."""
    try:
        import music21  # noqa: F401
    except ImportError:
        print("⏳ music21 niet gevonden — wordt nu geïnstalleerd via pip …")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "music21"],
            stdout=subprocess.DEVNULL,
        )
        print("✓ music21 geïnstalleerd.\n")


def _maak_mappen() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _extraheer_metadata(partituur, naam: str, midi_pad: Path, xml_pad: Path | None) -> dict:
    """Haal basismetadata uit een music21-partituur."""
    from music21 import tempo as m21_tempo

    # Maatgetal (op basis van het eerste deel)
    eerste_deel = partituur.parts[0] if partituur.parts else None
    maten = len(eerste_deel.getElementsByClass("Measure")) if eerste_deel else 0

    # Toonsoort
    toonsoorten = partituur.flatten().getElementsByClass("KeySignature")
    toonsoort = str(toonsoorten[0]) if toonsoorten else "onbekend"

    # Maatsoort
    maatsoorten = partituur.flatten().getElementsByClass("TimeSignature")
    maatsoort = str(maatsoorten[0].ratioString) if maatsoorten else "onbekend"

    # Tempo (BPM)
    tempos = partituur.flatten().getElementsByClass(m21_tempo.MetronomeMark)
    bpm = float(tempos[0].number) if tempos and tempos[0].number else None

    # Tonen tellen
    noten = partituur.flatten().notes
    aantal_noten = len(noten)

    return {
        "naam": naam,
        "stemmen": len(partituur.parts),
        "maten": maten,
        "toonsoort": toonsoort,
        "maatsoort": maatsoort,
        "bpm": bpm,
        "aantal_noten": aantal_noten,
        "midi_pad": str(midi_pad.relative_to(DATA_DIR)),
        "xml_pad": str(xml_pad.relative_to(DATA_DIR)) if xml_pad else "",
    }


def _voortgangsbalk(huidig: int, totaal: int, breedte: int = 40) -> str:
    gevuld = int(breedte * huidig / totaal)
    balk = "█" * gevuld + "░" * (breedte - gevuld)
    return f"[{balk}] {huidig:3d}/{totaal}"


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def download_bach_corpus(max_werken: int | None = None, midi_only: bool = False) -> None:
    _zorg_voor_music21()
    _maak_mappen()

    from music21 import corpus

    print("🎵 Bach-corpus ophalen uit music21 …")
    bach_paden = corpus.getComposer("bach")

    if max_werken:
        bach_paden = bach_paden[:max_werken]

    totaal = len(bach_paden)
    print(f"   {totaal} werken gevonden.\n")

    metadata_lijst: list[dict] = []
    fouten: list[str] = []
    start = time.time()

    for i, pad in enumerate(bach_paden, start=1):
        naam = Path(pad).stem
        voortgang = _voortgangsbalk(i, totaal)
        print(f"\r{voortgang}  {naam:<45}", end="", flush=True)

        midi_pad = RAW_DIR / f"{naam}.mid"
        xml_pad = RAW_DIR / f"{naam}.xml" if not midi_only else None

        # Sla al bestaande bestanden over (hervatbaar script).
        al_klaar = midi_pad.exists() and (midi_only or (xml_pad and xml_pad.exists()))
        if al_klaar:
            # Metadata nog steeds toevoegen via eerder geëxporteerd bestand.
            metadata_lijst.append({"naam": naam, "midi_pad": str(midi_pad.relative_to(DATA_DIR))})
            continue

        try:
            partituur = corpus.parse(pad)

            # MIDI exporteren
            partituur.write("midi", fp=str(midi_pad))

            # MusicXML exporteren (optioneel)
            if not midi_only:
                partituur.write("musicxml", fp=str(xml_pad))

            # Metadata extraheren
            meta = _extraheer_metadata(partituur, naam, midi_pad, xml_pad)
            metadata_lijst.append(meta)

        except Exception as fout:
            fouten.append(f"{naam}: {fout}")

    # Afsluiten van de voortgangsbalk
    print(f"\r{_voortgangsbalk(totaal, totaal)}  {'Klaar!':<45}")

    # Metadata naar CSV schrijven
    _schrijf_csv(metadata_lijst)

    # Samenvatting
    verstreken = time.time() - start
    print(f"\n{'─' * 60}")
    print(f"✓ {len(metadata_lijst)} werken opgeslagen  →  {RAW_DIR}")
    print(f"✓ Metadata opgeslagen  →  {PROCESSED_DIR / 'bach_metadata.csv'}")
    if fouten:
        print(f"⚠  {len(fouten)} fout(en):")
        for f in fouten:
            print(f"    • {f}")
    print(f"⏱  Verstreken tijd: {verstreken:.1f}s")


def _schrijf_csv(metadata_lijst: list[dict]) -> None:
    """Schrijf metadata naar CSV, ontbrekende velden met lege string."""
    if not metadata_lijst:
        return

    # Alle aanwezige kolommen verzamelen (volgorde stabiel houden)
    alle_kolommen: list[str] = []
    gezien: set[str] = set()
    for rij in metadata_lijst:
        for k in rij:
            if k not in gezien:
                alle_kolommen.append(k)
                gezien.add(k)

    csv_pad = PROCESSED_DIR / "bach_metadata.csv"
    with open(csv_pad, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=alle_kolommen, extrasaction="ignore")
        writer.writeheader()
        for rij in metadata_lijst:
            # Vul ontbrekende velden aan met lege string
            writer.writerow({k: rij.get(k, "") for k in alle_kolommen})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parseer_argumenten() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download het volledige Bach-corpus via music21.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Maximaal aantal werken (handig voor testen).",
    )
    parser.add_argument(
        "--midi-only",
        action="store_true",
        help="Exporteer alleen MIDI-bestanden (sneller, minder schijfruimte).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parseer_argumenten()
    download_bach_corpus(max_werken=args.max, midi_only=args.midi_only)
