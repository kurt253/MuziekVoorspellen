"""Tests voor hulpfuncties."""

from muziekvoorspellen.utils.helpers import normaliseer, plat_dict


def test_normaliseer_standaard():
    resultaat = normaliseer([0.0, 5.0, 10.0])
    assert resultaat == [0.0, 0.5, 1.0]


def test_normaliseer_leeg():
    assert normaliseer([]) == []


def test_normaliseer_constante_waarden():
    resultaat = normaliseer([3.0, 3.0, 3.0])
    assert resultaat == [0.0, 0.0, 0.0]


def test_plat_dict():
    genest = {"a": 1, "b": {"c": 2, "d": 3}}
    resultaat = plat_dict(genest)
    assert resultaat == {"a": 1, "b_c": 2, "b_d": 3}
