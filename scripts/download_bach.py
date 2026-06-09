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

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw" / "bach"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def _zorg_voor_music21() -> None:
    """Installeer music21 automatisch als het pakket niet gevonden wordt.

    Wordt aangeroepen vóór elke andere music21-import zodat het script ook
    werkt op een verse Python-omgeving zonder handmatige pip-installatie.
    Installeert via het huidige Python-executable zodat de juiste omgeving
    (venv, conda, ...) gebruikt wordt.
    """
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
    """Maak de benodigde mappen aan als ze nog niet bestaan.

    RAW_DIR      → data/raw/bach/       voor .mid en .xml bestanden
    PROCESSED_DIR → data/processed/     voor CSV-bestanden
    parents=True zorgt dat tussenliggende mappen ook aangemaakt worden.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _extraheer_metadata(partituur, naam: str, midi_pad: Path, xml_pad: Path | None) -> dict:
    """Haal basismetadata op uit een music21-partituurobject.

    Extraheert de volgende velden:
      - naam:          BWV-identifier (bijv. "bwv227.11")
      - stemmen:       aantal Parts in de partituur
      - maten:         aantal maten in het eerste Part
      - toonsoort:     eerste KeySignature als string
      - maatsoort:     eerste TimeSignature als breuk (bijv. "4/4")
      - bpm:           tempo in slagen per minuut (None als niet aanwezig)
      - aantal_noten:  totaal aantal noten en akkoorden (geen rusten)
      - midi_pad:      relatief pad t.o.v. DATA_DIR
      - xml_pad:       relatief pad t.o.v. DATA_DIR (leeg als midi_only)

    Parameters
    ----------
    partituur:
        Een music21 Score-object, reeds geparsed.
    naam:
        BWV-naam zonder extensie.
    midi_pad:
        Pad naar het zojuist geschreven .mid-bestand.
    xml_pad:
        Pad naar het zojuist geschreven .xml-bestand, of None bij --midi-only.
    """
    from music21 import tempo as m21_tempo

    # Maten tellen op basis van het eerste Part
    eerste_deel = partituur.parts[0] if partituur.parts else None
    maten = len(eerste_deel.getElementsByClass("Measure")) if eerste_deel else 0

    # Eerste toonsoort uit de partituur
    toonsoorten = partituur.flatten().getElementsByClass("KeySignature")
    toonsoort = str(toonsoorten[0]) if toonsoorten else "onbekend"

    # Eerste maatsoort, uitgedrukt als breuk (bijv. "4/4")
    maatsoorten = partituur.flatten().getElementsByClass("TimeSignature")
    maatsoort = str(maatsoorten[0].ratioString) if maatsoorten else "onbekend"

    # Eerste tempomarkering in BPM
    tempos = partituur.flatten().getElementsByClass(m21_tempo.MetronomeMark)
    bpm = float(tempos[0].number) if tempos and tempos[0].number else None

    # Totaal aantal noten (akkoorden tellen als één noot)
    noten = partituur.flatten().notes
    aantal_noten = len(noten)

    return {
        "naam":         naam,
        "stemmen":      len(partituur.parts),
        "maten":        maten,
        "toonsoort":    toonsoort,
        "maatsoort":    maatsoort,
        "bpm":          bpm,
        "aantal_noten": aantal_noten,
        "midi_pad":     str(midi_pad.relative_to(DATA_DIR)),
        "xml_pad":      str(xml_pad.relative_to(DATA_DIR)) if xml_pad else "",
    }


def _voortgangsbalk(huidig: int, totaal: int, breedte: int = 40) -> str:
    """Genereer een ASCII-voortgangsbalk voor terminal-uitvoer.

    Geeft een string terug van de vorm:
        [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  42/400

    Parameters
    ----------
    huidig:
        Huidige iteratieteller (1-gebaseerd).
    totaal:
        Totaal aantal iteraties.
    breedte:
        Breedte van de balk in tekens.
    """
    gevuld = int(breedte * huidig / totaal)
    balk = "█" * gevuld + "░" * (breedte - gevuld)
    return f"[{balk}] {huidig:3d}/{totaal}"


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def download_bach_corpus(max_werken: int | None = None, midi_only: bool = False) -> None:
    """Download en exporteer het volledige Bach-corpus uit music21.

    Verloop:
      1. Vraag alle bekende Bach-paden op via corpus.getComposer("bach").
      2. Sla al bestaande bestanden over (hervatbaar: onderbroken runs kunnen
         worden voortgezet zonder dubbel werk).
      3. Exporteer elk werk als MIDI (altijd) en MusicXML (tenzij --midi-only).
      4. Extraheer metadata per werk en verzamel die in een lijst.
      5. Schrijf alle metadata naar bach_metadata.csv.

    Parameters
    ----------
    max_werken:
        Maximaal aantal te verwerken werken. None = alles.
        Handig voor snelle tests zonder het volledige corpus te downloaden.
    midi_only:
        Als True, worden alleen .mid-bestanden geschreven (geen .xml).
        Scheelt schijfruimte en tijd als de partituurweergave niet nodig is.
    """
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
        xml_pad  = RAW_DIR / f"{naam}.xml" if not midi_only else None

        # Sla al bestaande bestanden over (hervatbaar script).
        al_klaar = midi_pad.exists() and (midi_only or (xml_pad and xml_pad.exists()))
        if al_klaar:
            # Metadata nog steeds toevoegen via eerder geëxporteerd bestand.
            metadata_lijst.append({"naam": naam, "midi_pad": str(midi_pad.relative_to(DATA_DIR))})
            continue

        try:
            partituur = corpus.parse(pad)

            # MIDI exporteren vanuit het music21-object (niet vanuit corpus-bestand)
            partituur.write("midi", fp=str(midi_pad))

            # MusicXML exporteren (optioneel); bevat rijkere info dan MIDI
            if not midi_only:
                partituur.write("musicxml", fp=str(xml_pad))

            # Metadata extraheren terwijl het partituurobject nog in geheugen is
            meta = _extraheer_metadata(partituur, naam, midi_pad, xml_pad)
            metadata_lijst.append(meta)

        except Exception as fout:
            fouten.append(f"{naam}: {fout}")

    # Afsluiten van de voortgangsbalk
    print(f"\r{_voortgangsbalk(totaal, totaal)}  {'Klaar!':<45}")

    # Metadata naar CSV schrijven
    _schrijf_csv(metadata_lijst)

    # Samenvatting tonen
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
    """Schrijf de metadata-lijst naar bach_metadata.csv.

    Verzamelt alle kolommen die in de lijst voorkomen (volgorde stabiel, eerste
    occurrence bepaalt volgorde). Ontbrekende velden in een rij worden aangevuld
    met een lege string zodat de CSV altijd rechthoekig is.

    Parameters
    ----------
    metadata_lijst:
        Lijst van dicts, één per werk. Velden mogen per werk verschillen
        (bijv. als een werk al bestond en alleen naam + midi_pad heeft).
    """
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
            writer.writerow({k: rij.get(k, "") for k in alle_kolommen})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parseer_argumenten() -> argparse.Namespace:
    """Parseer command-line argumenten voor het download-script.

    Beschikbare opties:
      --max N        Verwerk maximaal N werken (handig voor testen).
      --midi-only    Exporteer alleen .mid, geen .xml.
    """
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
