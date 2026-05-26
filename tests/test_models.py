"""Tests voor het voorspellingsmodel."""

import pytest
from muziekvoorspellen.models.predictor import MuziekVoorspeller


TRAINDATA = [
    {"tempo": 120.0, "energie": 0.9, "dansbaar": 0.8},
    {"tempo": 80.0, "energie": 0.3, "dansbaar": 0.2},
    {"tempo": 140.0, "energie": 0.7, "dansbaar": 0.9},
]


def test_voorspel_geeft_resultaten():
    model = MuziekVoorspeller()
    model.train(TRAINDATA)
    resultaten = model.voorspel({"tempo": 130.0, "energie": 0.8, "dansbaar": 0.85})
    assert len(resultaten) <= 3


def test_voorspel_zonder_training():
    model = MuziekVoorspeller()
    with pytest.raises(RuntimeError, match="getraind"):
        model.voorspel({"tempo": 100.0})


def test_top_n():
    model = MuziekVoorspeller()
    model.train(TRAINDATA)
    resultaten = model.voorspel({"tempo": 120.0, "energie": 0.9}, top_n=2)
    assert len(resultaten) == 2
