"""
Fundamental data structures for group sequential designs.

This module collects small, pure dataclasses that represent reusable
concepts across group sequential implementations.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ASNDesignSummary:
    """Lightweight summary of a group sequential design evaluation."""

    information_rates: NDArray[np.float64]
    z_boundaries: NDArray[np.float64]
    stagewise_rejection: NDArray[np.float64]
    expected_sample_size: float
    max_sample_size: float
    per_stage_alpha: NDArray[np.float64]
