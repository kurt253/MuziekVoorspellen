"""Laden van ruwe muziekdata."""

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def laad_ruwe_data(bestandsnaam: str):
    """Laad een ruwe databestand vanuit de raw-map.

    Args:
        bestandsnaam: Naam van het bestand (bijv. 'nummers.csv').

    Returns:
        Pad naar het bestand als Path-object.
    """
    pad = RAW_DIR / bestandsnaam
    if not pad.exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {pad}")
    return pad
