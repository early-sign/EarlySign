"""Core engines for AVI Operating Characteristics Evaluation.

This module provides the base classes and implementations for evaluating
operating characteristics using Monte Carlo simulation for Anytime Valid Inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from earlysign.builtin.group_sequential.design.operating_characteristics.engines import (
    EvaluationResult,
    SimulationCurve,
)


class AVIOperatingCharacteristicsEvaluator(ABC):
    """Abstract base class for AVI OC evaluators.

    Evaluates the Operating Characteristics (Power, Expected Sample Size) of a given AVI Design.
    """

    @abstractmethod
    def evaluate_point(
        self,
        effect_size: float,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate OC at a single effect size point."""
        pass

    def evaluate_curve(
        self,
        effect_sizes: Sequence[float] | NDArray[np.float64],
        **kwargs: Any,
    ) -> SimulationCurve:
        """Evaluate OC over a range of effect sizes."""
        results = [
            self.evaluate_point(
                d,
                **kwargs,
            )
            for d in effect_sizes
        ]
        return SimulationCurve(
            x_values=np.asarray(effect_sizes),
            results=results,
            metric_type="effect_size",
        )


class AVIMonteCarloSimulator(AVIOperatingCharacteristicsEvaluator):
    """Base Monte Carlo Simulator for Anytime Valid Inference."""

    def __init__(
        self,
        protocol: Any,
        n_sims: int = 2000,
        seed: Optional[int] = None,
        max_n: int = 5000,
    ):
        self.protocol = protocol
        self.n_sims = n_sims
        self.seed = seed
        self.max_n = max_n

        # Try to infer max_n from protocol method if available
        if (
            hasattr(self.protocol.method, "max_n")
            and self.protocol.method.max_n is not None
        ):
            self.max_n_total = int(self.protocol.method.max_n)
        else:
            self.max_n_total = max_n

        from earlysign.builtin.AVI.schema import AVITaskSpec

        task = cast(AVITaskSpec, self.protocol.task)
        import earlysign.schema.ES3.Base as ES3_BASE

        if isinstance(task.arms, ES3_BASE.TwoArmComparison):
            self.n_max_per_arm = {
                task.arms.control_arm_name: self.max_n_total // 2,
                task.arms.treatment_arm_name: self.max_n_total // 2,
            }
        elif isinstance(task.arms, ES3_BASE.SingleArm):
            self.n_max_per_arm = {
                task.arms.arm_name: self.max_n_total,
            }
        else:
            # Fallback for MultiArm or unknown
            self.n_max_per_arm = {}

    # Children must implement evaluate_point to generate paths and compute boundaries
