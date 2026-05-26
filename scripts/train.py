"""Trainingsscript — voer uit vanuit de projectroot.

Gebruik::

    python scripts/train.py
"""

import sys
from pathlib import Path

# Zorg dat de pakketroot op het pad staat wanneer het script direct wordt uitgevoerd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muziekvoorspellen.models.predictor import MuziekVoorspeller


def main() -> None:
    print("=== MuziekVoorspellen — Training ===")

    # Voorbeelddata — vervang dit door echte data-loading.
    traindata = [
        {"tempo": 120.0, "energie": 0.9, "dansbaar": 0.8, "valentie": 0.7},
        {"tempo": 95.0, "energie": 0.5, "dansbaar": 0.5, "valentie": 0.4},
        {"tempo": 140.0, "energie": 0.85, "dansbaar": 0.9, "valentie": 0.8},
    ]

    model = MuziekVoorspeller()
    model.train(traindata)
    print(f"Model getraind op {len(traindata)} nummers.")

    query = {"tempo": 130.0, "energie": 0.88, "dansbaar": 0.85, "valentie": 0.75}
    aanbevelingen = model.voorspel(query, top_n=2)
    print(f"\nTop-{len(aanbevelingen)} aanbevelingen voor query {query}:")
    for i, nummer in enumerate(aanbevelingen, start=1):
        print(f"  {i}. {nummer}")


if __name__ == "__main__":
    main()
