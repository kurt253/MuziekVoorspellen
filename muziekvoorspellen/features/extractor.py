"""Extractie van muziekfeatures."""


def extraheer_features(nummer: dict) -> dict:
    """Extraheer relevante features uit een nummer-record.

    Args:
        nummer: Dict met ruwe nummerattributen.

    Returns:
        Dict met genormaliseerde features.
    """
    features = {
        "tempo": nummer.get("tempo", 0.0),
        "energie": nummer.get("energie", 0.0),
        "dansbaar": nummer.get("dansbaar", 0.0),
        "valentie": nummer.get("valentie", 0.0),
        "populariteit": nummer.get("populariteit", 0),
    }
    return features
