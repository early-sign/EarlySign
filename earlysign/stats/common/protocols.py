from typing import Protocol


class EffectSizeCalculator(Protocol):
    """Protocol for calculating sample size for a given effect."""

    def calculate_sample_size(
        self, effect_size: float, alpha: float, power: float
    ) -> int: ...
    def get_null_value(self) -> float: ...
