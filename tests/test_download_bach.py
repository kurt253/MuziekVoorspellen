"""Tests voor het Bach-downloader script."""

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importeer alleen de pure hulpfuncties (geen music21 nodig voor deze tests).
from scripts.download_bach import _voortgangsbalk, _schrijf_csv


class TestVoortgangsbalk:
    def test_begin(self):
        balk = _voortgangsbalk(0, 10)
        assert "0" in balk
        assert "10" in balk

    def test_einde(self):
        balk = _voortgangsbalk(10, 10)
        assert "█" in balk

    def test_breedte(self):
        balk = _voortgangsbalk(5, 10, breedte=20)
        # Balk bevat blokken en lege plekken
        assert "█" in balk
        assert "░" in balk


class TestSchrijfCsv:
    def test_schrijft_csv(self, tmp_path, monkeypatch):
        # Patch PROCESSED_DIR zodat we naar tmp_path schrijven.
        import scripts.download_bach as module
        monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)

        metadata = [
            {"naam": "bwv001", "stemmen": 4, "maten": 10},
            {"naam": "bwv002", "stemmen": 4, "maten": 12},
        ]
        _schrijf_csv(metadata)

        csv_pad = tmp_path / "bach_metadata.csv"
        assert csv_pad.exists()

        with open(csv_pad, encoding="utf-8") as f:
            rijen = list(csv.DictReader(f))

        assert len(rijen) == 2
        assert rijen[0]["naam"] == "bwv001"
        assert rijen[1]["maten"] == "12"

    def test_lege_lijst_maakt_geen_bestand(self, tmp_path, monkeypatch):
        import scripts.download_bach as module
        monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)

        _schrijf_csv([])

        assert not (tmp_path / "bach_metadata.csv").exists()

    def test_ontbrekende_velden_worden_aangevuld(self, tmp_path, monkeypatch):
        import scripts.download_bach as module
        monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)

        metadata = [
            {"naam": "bwv001", "bpm": 120},
            {"naam": "bwv002"},           # geen bpm
        ]
        _schrijf_csv(metadata)

        csv_pad = tmp_path / "bach_metadata.csv"
        with open(csv_pad, encoding="utf-8") as f:
            rijen = list(csv.DictReader(f))

        assert rijen[1]["bpm"] == ""
