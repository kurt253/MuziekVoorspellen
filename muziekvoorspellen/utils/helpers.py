"""Algemene hulpfuncties."""

from typing import Any, Dict, List


def normaliseer(waarden: List[float]) -> List[float]:
    """Normaliseer een lijst waarden naar het bereik [0, 1].

    Args:
        waarden: Lijst met numerieke waarden.

    Returns:
        Genormaliseerde lijst.
    """
    if not waarden:
        return []
    min_w, max_w = min(waarden), max(waarden)
    bereik = max_w - min_w
    if bereik == 0:
        return [0.0] * len(waarden)
    return [(w - min_w) / bereik for w in waarden]


def plat_dict(genest: Dict[str, Any], scheidingsteken: str = "_") -> Dict[str, Any]:
    """Maak een genest woordenboek plat.

    Args:
        genest: Genest dict.
        scheidingsteken: Scheidingsteken tussen sleutelniveaus.

    Returns:
        Plat dict.
    """
    items: Dict[str, Any] = {}
    for sleutel, waarde in genest.items():
        if isinstance(waarde, dict):
            for sub_sleutel, sub_waarde in plat_dict(waarde, scheidingsteken).items():
                items[f"{sleutel}{scheidingsteken}{sub_sleutel}"] = sub_waarde
        else:
            items[sleutel] = waarde
    return items
