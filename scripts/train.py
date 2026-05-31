"""ML-voorbereiding: selecteer MIDI-bestanden voor training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
ANALYSE_CSV = DATA_DIR / "processed" / "bach_analyse.csv"
METADATA_CSV = DATA_DIR / "processed" / "bach_metadata.csv"

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


def maak_trainings_df(aantal: int | None = None, soort: str = "alles") -> pd.DataFrame:
    """Geef een DataFrame met MIDI-bestanden voor training.

    Parameters
    ----------
    aantal:
        Maximaal aantal rijen. None = alles.
    soort:
        Classificatiefilter (exact, hoofdlettergevoelig).
        Gebruik ``"alles"`` voor alle types.

    Returns
    -------
    pd.DataFrame met kolommen: naam, midi_pad, classificatie,
    maten, aantal_stemmen, stemmen.
    """
    analyse = pd.read_csv(ANALYSE_CSV)
    metadata = pd.read_csv(METADATA_CSV)

    # midi_pad is relatief t.o.v. data/ → maak absoluut
    metadata["midi_pad"] = metadata["midi_pad"].apply(
        lambda p: str(DATA_DIR / p)
    )

    df = analyse.merge(metadata, on="naam", how="inner")
    df = df.drop(columns=["bestandsnaam"])

    if soort != "alles":
        if soort not in BEKENDE_TYPES:
            beschikbaar = ", ".join(f'"{t}"' for t in BEKENDE_TYPES)
            raise ValueError(
                f'Onbekend type "{soort}". Kies uit: {beschikbaar} of "alles".'
            )
        df = df[df["classificatie"] == soort]

    if len(df) == 0:
        raise ValueError(f'Geen bestanden gevonden voor type "{soort}".')

    if aantal is not None:
        if aantal <= 0:
            raise ValueError("aantal moet groter zijn dan 0.")
        df = df.sample(n=min(aantal, len(df)), random_state=42)

    return df.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selecteer MIDI-bestanden voor ML-training.")
    p.add_argument(
        "--aantal", type=int, default=None,
        help="Maximaal aantal bestanden (standaard: alles).",
    )
    p.add_argument(
        "--soort", default="alles",
        help='Classificatiefilter, bv. "koraal" of "alles" (standaard).',
    )
    p.add_argument(
        "--lijst", action="store_true",
        help="Toon beschikbare types en stop.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.lijst:
        print("Beschikbare types:")
        for t in BEKENDE_TYPES:
            print(f"  {t}")
        raise SystemExit(0)

    df = maak_trainings_df(aantal=args.aantal, soort=args.soort)
    print(df.to_string(index=False))
    print(f"\n{len(df)} bestand(en) geselecteerd.")
