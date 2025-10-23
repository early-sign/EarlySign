"""Essential calculations for survival analysis schemes.

This module contains core computational utilities for time-to-event tests
that are used in both design planning and runtime analysis.
"""

from earlysign.stats.schemes.survival.essentials.effects import (
    TimeToEventEffectCalculator,
)

__all__ = ["TimeToEventEffectCalculator"]
