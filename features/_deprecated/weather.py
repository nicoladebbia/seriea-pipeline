"""DEPRECATED: This module has been merged into enhanced_weather.py.

All functionality from this file (basic weather impact scoring, city-based
estimation, and the add_weather_features DataFrame entry point) is now
available in ``features.enhanced_weather``. Use that module instead:

    from features.enhanced_weather import add_weather_features
    from features.enhanced_weather import add_all_weather_features  # recommended
    from features.enhanced_weather import get_weather_impact_score
    from features.enhanced_weather import estimate_weather_from_month_location

This file re-exports key functions for backward compatibility.
It will be removed in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "features.weather is deprecated. Use features.enhanced_weather instead. "
    "All weather features are now in enhanced_weather.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
from features.enhanced_weather import (  # noqa: F401
    add_weather_features,
    estimate_weather_from_month_location,
    get_weather_impact_score,
)

__all__ = [
    "add_weather_features",
    "estimate_weather_from_month_location",
    "get_weather_impact_score",
]
