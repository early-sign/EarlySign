"""Survival analysis and time-to-event testing schemes.

This module provides utilities for analyzing time-to-event data,
including log-rank tests and Cox proportional hazards models.
"""

from earlysign.stats.schemes.survival.essentials.effects import (
    TimeToEventEffectCalculator,
)

__all__: list[str] = ["TimeToEventEffectCalculator"]
