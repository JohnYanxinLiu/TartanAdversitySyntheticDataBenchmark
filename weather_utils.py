"""Canonical weather-condition handling shared across all models.

Every dataset in this benchmark (BDD100K, DAWN, ACDC, Foggy Zurich, ...) labels
its images with its own weather vocabulary. DAWN, for example, splits fog into
``fog``/``haze``/``mist``, while BDD100K uses ``foggy``/``rainy``/``snowy`` and
also carries conditions we do not evaluate (``overcast``, ``partly cloudy``,
``undefined``, ...).

To make ResNet, ConvNeXt, and YOLO all log identical, comparable metrics we
collapse every raw label to a single canonical set and drop everything else from
the per-weather breakdown. Dropped images still contribute to the ``overall``
(dataset-level) metric; they are only excluded from per-weather slices.
"""

# The only weather conditions we report per-weather metrics for.
CANONICAL_WEATHER = ("clear", "rainy", "foggy", "snowy")

# Bucket used for any label that is not one of CANONICAL_WEATHER. Callers should
# skip this bucket when logging per-weather metrics.
OTHER_WEATHER = "other"

# Maps every known raw label (lowercased) to a canonical condition.
_WEATHER_ALIASES = {
    # clear
    "clear": "clear",
    "sunny": "clear",
    # rainy
    "rain": "rainy",
    "rainy": "rainy",
    "rain_storm": "rainy",
    # foggy (DAWN reports haze/mist separately; we accumulate them into foggy)
    "fog": "foggy",
    "foggy": "foggy",
    "haze": "foggy",
    "mist": "foggy",
    # snowy
    "snow": "snowy",
    "snowy": "snowy",
    "snow_storm": "snowy",
}


def normalize_weather(raw):
    """Map a raw dataset weather label to one of ``CANONICAL_WEATHER``.

    Returns ``None`` for anything outside the canonical set (e.g. ``overcast``,
    ``partly cloudy``, ``undefined``, ``night``, ``sand``, ``unknown``), so the
    caller can decide to drop it from per-weather reporting.
    """
    if raw is None:
        return None
    return _WEATHER_ALIASES.get(str(raw).strip().lower())


def canonical_weather_or_other(raw):
    """Like ``normalize_weather`` but returns ``OTHER_WEATHER`` instead of None.

    Convenient when a value is required (e.g. as a dict key) and the ``other``
    bucket will be filtered out at logging time.
    """
    return normalize_weather(raw) or OTHER_WEATHER
