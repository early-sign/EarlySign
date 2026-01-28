"""Core engines for Operating Characteristics Evaluation.

This module provides the base classes and implementations for evaluating
operating characteristics using different methods (Monte Carlo, Numerical Integration).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess


@dataclass
class EvaluationResult:
    """Result of evaluating a design at a specific drift."""

    drift: float
    power: float  # Probability of rejecting H0 (Approximate if 2-sided)
    asn: float  # Expected Sample Size fraction (Canonical Scale, 0.0-1.0)
    prob_stop_efficacy: NDArray[np.float64]
    prob_stop_futility: NDArray[np.float64]
    prob_stop_total: NDArray[np.float64]
    expected_sample_size: Optional[float] = (
        None  # Actual expected sample size if available
    )

    @property
    def total_prob_stop(self) -> float:
        return float(np.sum(self.prob_stop_total))


@dataclass
class SimulationCurve:
    """A collection of evaluation results over a range of parameter values."""

    x_values: NDArray[np.float64]  # The parameter values (Drift or Effect Size)
    results: List[EvaluationResult]
    metric_type: str = "drift"  # Context label
    n_max: Optional[int] = None
    null_x_value: Optional[float] = None
    target_x_value: Optional[float] = None
    p_control: Optional[float] = None


class StatisticalProcess(ABC):
    """Protocol for statistical processes used in AsymptoticSimulator."""

    @abstractmethod
    def compute_stopping_probabilities(
        self,
        *,
        upper: NDArray[np.float64],
        lower: NDArray[np.float64],
        n_sims: int,
        drift: float = 0.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute stopping probabilities for efficacy and futility."""
        pass


class OperatingCharacteristicsEvaluator(ABC):
    """Abstract base class for OC evaluators.

    Evaluates the Operating Characteristics (Power, ASN) of a given Group Sequential Design.
    """

    @abstractmethod
    def evaluate_point(
        self,
        drift: float,
        info_times: NDArray[np.float64],
        upper_boundaries: NDArray[np.float64],
        lower_boundaries: NDArray[np.float64],
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate OC at a single drift point."""
        pass

    def evaluate_curve(
        self,
        drifts: Sequence[float] | NDArray[np.float64],
        info_times: NDArray[np.float64],
        upper_boundaries: NDArray[np.float64],
        lower_boundaries: NDArray[np.float64],
        **kwargs: Any,
    ) -> SimulationCurve:
        """Evaluate OC over a range of drifts."""
        results = [
            self.evaluate_point(
                d,
                info_times=info_times,
                upper_boundaries=upper_boundaries,
                lower_boundaries=lower_boundaries,
                **kwargs,
            )
            for d in drifts
        ]
        return SimulationCurve(
            x_values=np.asarray(drifts),
            results=results,
            metric_type="drift",
        )


class AsymptoticSimulator(OperatingCharacteristicsEvaluator):
    """AsymptoticModelMonteCarloSimulator: Simulates statistics using asymptotic process models (Gaussian/T).

    Uses a `StatisticalProcess` model (e.g. CanonicalGaussianProcess, CanonicalTProcess) to simulate
    test statistics and evaluate boundary crossing probabilities.
    """

    def __init__(
        self,
        model: Any,  # Should ideally be StatisticalProcess but strict typing might be heavy here
        n_sims: int = 50000,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.n_sims = n_sims
        self.seed = seed

    def evaluate_point(
        self,
        drift: float,
        info_times: NDArray[np.float64],
        upper_boundaries: NDArray[np.float64],
        lower_boundaries: NDArray[np.float64],
        **kwargs: Any,
    ) -> EvaluationResult:
        # Prepare kwargs for the process (e.g. m_counts for TProcess)
        # We pass 't=info_times' explicitly for Gaussian Process.

        process_kwargs = kwargs.copy()

        prob_eff, prob_fut = self.model.compute_stopping_probabilities(
            t=info_times,
            upper=upper_boundaries,
            lower=lower_boundaries,
            n_sims=self.n_sims,
            drift=drift,
            seed=self.seed,
            **process_kwargs,
        )

        prob_total = prob_eff + prob_fut
        power = float(np.sum(prob_eff))

        # ASN Calculation (Fractional)
        # Assuming stops occur at info times
        remainder = float(1.0 - np.sum(prob_total))
        remainder = max(0.0, remainder)

        asn_fraction = float(np.sum(prob_total * info_times)) + float(
            remainder * 1.0  # Assumes max info time is 1.0
        )

        # Force probabilities to sum to 1.0 (remainder goes to last futility/acceptance)
        if remainder > 0:
            prob_fut[-1] += remainder
            prob_total[-1] += remainder

        return EvaluationResult(
            drift=drift,
            power=power,
            asn=asn_fraction,
            prob_stop_efficacy=prob_eff,
            prob_stop_futility=prob_fut,
            prob_stop_total=prob_total,
            expected_sample_size=None,  # Asymptotic doesn't know N
        )


class MonteCarloSimulator(OperatingCharacteristicsEvaluator):
    """DataMonteCarloSimulatorBase: Base class for evaluators that simulate raw trial data."""

    # Concrete implementations (like BinomialAB) should implement evaluate_point
    pass


class NumericalCalculator(OperatingCharacteristicsEvaluator):
    """Evaluator using Numerical Integration."""

    def __init__(self) -> None:
        pass

    def evaluate_point(
        self,
        drift: float,
        info_times: NDArray[np.float64],
        upper_boundaries: NDArray[np.float64],
        lower_boundaries: NDArray[np.float64],
        **kwargs: Any,
    ) -> EvaluationResult:
        # Instantiates generic Gaussian Process for integration
        gp = CanonicalGaussianProcess(drift=drift)

        prob_eff, prob_fut = gp.compute_stopping_probabilities(
            t=info_times,
            upper=upper_boundaries,
            lower=lower_boundaries,
            method="numerical_integration",
        )

        prob_total = prob_eff + prob_fut
        power = float(np.sum(prob_eff))

        remainder = float(1.0 - np.sum(prob_total))
        remainder = max(0.0, remainder)

        asn_fraction = float(np.sum(prob_total * info_times)) + float(remainder * 1.0)

        if remainder > 0:
            prob_fut[-1] += remainder
            prob_total[-1] += remainder

        return EvaluationResult(
            drift=drift,
            power=power,
            asn=asn_fraction,
            prob_stop_efficacy=prob_eff,
            prob_stop_futility=prob_fut,
            prob_stop_total=prob_total,
            expected_sample_size=None,  # Numerical Integration is always fractional
        )
