"""DEPRECATED: This module has been merged into enhanced_momentum.py.

All functionality from this file (basic streak features) is now available in
``features.enhanced_momentum``. Use that module instead:

    from features.enhanced_momentum import add_momentum_features
    from features.enhanced_momentum import add_all_momentum_features  # recommended

This file re-exports ``add_momentum_features`` for backward compatibility.
It will be removed in a future version.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "features.momentum is deprecated. Use features.enhanced_momentum instead. "
    "All basic streak features are now in enhanced_momentum.add_momentum_features().",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
from features.enhanced_momentum import add_momentum_features  # noqa: F401

__all__ = ["add_momentum_features"]
