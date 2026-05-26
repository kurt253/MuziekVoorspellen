"""Voorspellingsmodel voor muziekaanbevelingen."""

from __future__ import annotations

from typing import List


class MuziekVoorspeller:
    """Eenvoudig voorspellingsmodel op basis van feature-gelijkenis.

    Voorbeeld gebruik::

        voorspeller = MuziekVoorspeller()
        voorspeller.train(traindata)
        resultaten = voorspeller.voorspel(query_features)
    """

    def __init__(self) -> None:
        self._traindata: List[dict] = []

    def train(self, data: List[dict]) -> None:
        """Sla trainingsdata op in het model.

        Args:
            data: Lijst van feature-dicts.
        """
        self._traindata = data

    def voorspel(self, features: dict, top_n: int = 5) -> List[dict]:
        """Geef de meest gelijkende nummers terug.

        Args:
            features: Feature-dict van het query-nummer.
            top_n: Aantal terug te geven resultaten.

        Returns:
            Gesorteerde lijst van de meest overeenkomende nummers.
        """
        if not self._traindata:
            raise RuntimeError("Model is nog niet getraind. Roep eerst train() aan.")

        gesorteerd = sorted(
            self._traindata,
            key=lambda item: self._gelijkenis(features, item),
            reverse=True,
        )
        return gesorteerd[:top_n]

    @staticmethod
    def _gelijkenis(a: dict, b: dict) -> float:
        """Berekent een eenvoudige cosinusgelijkenis tussen twee feature-dicts."""
        sleutels = set(a) & set(b)
        if not sleutels:
            return 0.0
        dot = sum(a[k] * b[k] for k in sleutels)
        norm_a = sum(a[k] ** 2 for k in sleutels) ** 0.5
        norm_b = sum(b[k] ** 2 for k in sleutels) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
