"""Adapter layer to use optimization_v2 with the old DesignOptimizer interface.

This module provides backward compatibility by wrapping the new optimization_v2
functions and classes within the old DesignOptimizer/DesignObjective interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.optimization_v2 import (
    OBFSpending,
    PocockSpending,
    SpendingStrategy,
    get_optimal_information_rates,
)
from earlysign.stats.design.gst.common.types import (
    InformationSpacing,
    SpendingFunction,
)


class DesignObjective(ABC):
    """Abstract base class for group sequential design optimization objectives.

    This adapter maintains the original interface while delegating to
    optimization_v2 functions internally.
    """

    @abstractmethod
    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        """Evaluate objective function (return value to minimize).

        Args:
            spec: Design specification
            lab: DesignLab instance

        Returns:
            Objective function value (lower is better)
        """
        pass

    @abstractmethod
    def get_constraints(self, spec: DesignSpec) -> Dict[str, Any]:
        """Return constraint conditions.

        Args:
            spec: Design specification

        Returns:
            Dictionary of constraints
        """
        pass


class MinimizeASN(DesignObjective):
    """Minimize expected sample size (ASN - Average Sample Number).

    This adapter uses optimization_v2 internally while maintaining the
    original MinimizeASN interface.
    """

    def __init__(self, planned_max_n: int, target_power: float = 0.90):
        self.planned_max_n = planned_max_n
        self.target_power = target_power

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        """Compute ASN (with penalty for constraint violations)."""
        try:
            lab.compute_boundaries()
            lab.run_simulations()

            assert lab.simulation_results is not None

            # Check power constraint
            power = lab.simulation_results["power"]
            if power < self.target_power:
                # Penalty for insufficient power
                return float(
                    self.planned_max_n * 10 + (self.target_power - power) * 10000
                )

            # Check maximum sample size constraint
            max_sample = lab.simulation_results["max_sample_size"]
            if max_sample > self.planned_max_n:
                return float(
                    self.planned_max_n * 10 + (max_sample - self.planned_max_n) * 100
                )

            # Return ASN
            return float(lab.simulation_results["expected_sample_size"])
        except Exception:
            return self.planned_max_n * 100

    def get_constraints(self, spec: DesignSpec) -> Dict[str, Any]:
        return {"planned_max_n": self.planned_max_n, "target_power": self.target_power}


class MaximizePower(DesignObjective):
    """Maximize statistical power subject to maximum sample size constraint."""

    def __init__(self, planned_max_n: int):
        self.planned_max_n = planned_max_n

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        """Compute power (return negative value to convert maximization to minimization)."""
        try:
            lab.compute_boundaries()
            lab.run_simulations()

            assert lab.simulation_results is not None

            max_sample = lab.simulation_results["max_sample_size"]
            if max_sample > self.planned_max_n:
                return 1.0  # Penalty

            power = lab.simulation_results["power"]
            return float(-power)  # Negative for minimization
        except Exception:
            return 1.0

    def get_constraints(self, spec: DesignSpec) -> Dict[str, Any]:
        return {"planned_max_n": self.planned_max_n}


class BalancedDesign(DesignObjective):
    """Multi-objective optimization balancing power and sample size efficiency."""

    def __init__(
        self,
        power_weight: float = 1.0,
        sample_weight: float = 1.0,
        target_power: float = 0.90,
        target_n: int = 3000,
    ):
        self.power_weight = power_weight
        self.sample_weight = sample_weight
        self.target_power = target_power
        self.target_n = target_n

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        """Composite objective function: weighted score of power and sample size."""
        try:
            lab.compute_boundaries()
            lab.run_simulations()

            assert lab.simulation_results is not None

            power = lab.simulation_results["power"]
            ess = lab.simulation_results["expected_sample_size"]

            # Normalized scores
            power_score = abs(power - self.target_power) / self.target_power
            sample_score = abs(ess - self.target_n) / self.target_n

            return float(
                self.power_weight * power_score + self.sample_weight * sample_score
            )
        except Exception:
            return 100.0

    def get_constraints(self, spec: DesignSpec) -> Dict[str, Any]:
        return {
            "target_power": self.target_power,
            "target_n": self.target_n,
            "power_weight": self.power_weight,
            "sample_weight": self.sample_weight,
        }


def _spec_to_spending_strategy(spec: DesignSpec) -> SpendingStrategy:
    """Convert DesignSpec spending function to SpendingStrategy."""
    spending_func = spec.boundary.spending_function
    alpha = spec.test.alpha
    sided = 2 if spec.test.sided == "two" else 1

    if spending_func == SpendingFunction.OBRIEN_FLEMING:
        return OBFSpending(alpha=alpha, sided=sided)
    elif spending_func == SpendingFunction.POCOCK:
        return PocockSpending(alpha=alpha)
    else:
        # Default to OBF
        return OBFSpending(alpha=alpha, sided=sided)


class DesignOptimizer:
    """Design optimization engine using optimization_v2 internally.

    This adapter maintains the original DesignOptimizer interface while
    delegating to the new optimization_v2 functions.
    """

    def __init__(self, base_spec: DesignSpec, objective: DesignObjective):
        self.base_spec = base_spec
        self.objective = objective
        self.optimization_history: list[Dict[str, Any]] = []

    def _clone_spec(self) -> DesignSpec:
        """Clone the base spec for modifications."""
        import copy

        return copy.deepcopy(self.base_spec)

    def optimize_info_times(self, n_analyses: int, n_samples: int = 100) -> np.ndarray:
        """Optimize information times using optimization_v2.

        Args:
            n_analyses: Number of analyses
            n_samples: Number of random samples to try (ignored, kept for compatibility)

        Returns:
            Optimal information times (monotonically increasing, last is 1.0)
        """
        spec = self.base_spec

        # Extract parameters from spec
        alpha = spec.test.alpha
        beta = 1.0 - spec.test.power
        sided = 2 if spec.test.sided == "two" else 1

        # Get effect size parameters
        # For proportions: alternative = difference in proportions
        if hasattr(spec, "effect"):
            alternative = spec.effect.effect_size
            # Estimate standard deviation for proportions
            # Using pooled estimate: sqrt(p*(1-p) * (1/n1 + 1/n2))
            p_control = getattr(spec.effect, "p_control", 0.5)
            p_treatment = p_control + alternative
            p_pooled = (p_control + p_treatment) / 2
            st_dev = np.sqrt(p_pooled * (1 - p_pooled))
        else:
            # Default values if effect not specified
            alternative = 0.2
            st_dev = 1.0

        # Get allocation ratio (default 1:1)
        allocation_ratio = 1.0

        # Create spending strategy
        spending = _spec_to_spending_strategy(spec)

        # Use optimization_v2 to find optimal information rates
        try:
            info_rates = get_optimal_information_rates(
                alpha=alpha,
                beta=beta,
                sided=sided,
                alternative=alternative,
                st_dev=st_dev,
                allocation_ratio_planned=allocation_ratio,
                spending=spending,
                k_max=n_analyses,
                min_gap=0.02,
                seed=42,
                n_jobs=1,
                n_restarts=12,
                restart_scale=0.2,
            )

            if not info_rates:
                # Fall back to equally spaced
                return np.linspace(0, 1, n_analyses + 1)[1:]

            return np.array(info_rates)
        except Exception:
            # Fall back to equally spaced if optimization fails
            return np.linspace(0, 1, n_analyses + 1)[1:]

    def optimize_info_times_and_n_analyses(
        self, max_k: int, min_k: int = 2, n_samples: int = 20
    ) -> tuple[int, np.ndarray]:
        """Optimize both number of analyses and information times.

        Args:
            max_k: Maximum number of analyses
            min_k: Minimum number of analyses
            n_samples: Number of random splits per k (ignored, kept for compatibility)

        Returns:
            Tuple (best_n_analyses, best_info_times)
        """
        best_score = float("inf")
        best_k = min_k
        best_times = np.linspace(0, 1, min_k + 1)[1:]

        for k in range(min_k, max_k + 1):
            try:
                info_times = self.optimize_info_times(k)
                spec = self._clone_spec()
                spec.sequential.n_analyses = k
                spec.sequential.info_times = info_times.tolist()
                spec.sequential.info_spacing = InformationSpacing.CUSTOM
                lab = DesignLab(spec)
                score = self.objective.evaluate(spec, lab)

                if score < best_score:
                    best_score = score
                    best_k = k
                    best_times = info_times
            except Exception:
                continue

        return best_k, best_times

    def optimize_n_analyses(self, min_k: int = 2, max_k: int = 10) -> int:
        """Search for optimal number of analyses.

        Args:
            min_k: Minimum number of analyses
            max_k: Maximum number of analyses

        Returns:
            Optimal number of analyses
        """
        best_k, _ = self.optimize_info_times_and_n_analyses(max_k, min_k)
        return best_k

    def optimize_comprehensive(self) -> DesignSpec:
        """Comprehensive optimization.

        Returns:
            Optimized design specification
        """
        spec = self._clone_spec()

        # Step 1: Optimize number of analyses
        optimal_k = self.optimize_n_analyses()
        spec.sequential.n_analyses = optimal_k

        # Step 2: Optimize information times
        optimal_times = self.optimize_info_times(optimal_k)
        spec.sequential.info_times = optimal_times.tolist()
        spec.sequential.info_spacing = InformationSpacing.CUSTOM

        return spec

    def get_optimization_summary(self) -> Any:
        """Return optimization history as DataFrame."""
        import pandas as pd

        if not self.optimization_history:
            return pd.DataFrame()

        return pd.DataFrame(self.optimization_history)
