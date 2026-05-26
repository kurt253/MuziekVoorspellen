"""Tests voor feature-extractie."""

import pytest
from muziekvoorspellen.features.extractor import extraheer_features


def test_extraheer_features_volledig():
    nummer = {
        "tempo": 120.0,
        "energie": 0.8,
        "dansbaar": 0.7,
        "valentie": 0.6,
        "populariteit": 85,
    }
    features = extraheer_features(nummer)
    assert features["tempo"] == 120.0
    assert features["energie"] == 0.8
    assert features["populariteit"] == 85


def test_extraheer_features_ontbrekende_velden():
    features = extraheer_features({})
    assert features["tempo"] == 0.0
    assert features["populariteit"] == 0
