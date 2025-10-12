"""Design optimization objectives and optimizer."""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from earlysign.stats.design.config import DesignSpec
from earlysign.stats.design.lab import DesignLab
from earlysign.stats.design.types import InformationSpacing


class DesignObjective(ABC):
    """Abstract base class representing design objectives."""

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
    """Minimize ASN (Average Sample Number / Expected sample size).

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> objective = MinimizeASN(max_n=3000, target_power=0.90)
    >>> objective.max_n
    3000
    >>> objective.target_power
    0.9
    """

    def __init__(self, max_n: int, target_power: float = 0.90):
        self.max_n = max_n
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
                return float(self.max_n * 10 + (self.target_power - power) * 10000)

            # Check maximum sample size constraint
            max_sample = lab.simulation_results["max_sample_size"]
            if max_sample > self.max_n:
                return float(self.max_n * 10 + (max_sample - self.max_n) * 100)

            # Return ASN
            return float(lab.simulation_results["expected_sample_size"])
        except Exception:
            return self.max_n * 100

    def get_constraints(self, spec: DesignSpec) -> Dict[str, Any]:
        return {"max_n": self.max_n, "target_power": self.target_power}


class MaximizePower(DesignObjective):
    """Maximize power with fixed sample size.

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> objective = MaximizePower(max_n=3000)
    >>> objective.max_n
    3000
    """

    def __init__(self, max_n: int):
        self.max_n = max_n

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        """Compute power (return negative value to convert maximization to minimization)."""
        try:
            lab.compute_boundaries()
            lab.run_simulations()

            assert lab.simulation_results is not None

            max_sample = lab.simulation_results["max_sample_size"]
            if max_sample > self.max_n:
                return 1.0  # Penalty

            power = lab.simulation_results["power"]
            return float(-power)  # Negative for minimization
        except Exception:
            return 1.0

    def get_constraints(self, spec: DesignSpec) -> Dict[str, Any]:
        return {"max_n": self.max_n}


class BalancedDesign(DesignObjective):
    """Optimize balance between power and sample size.

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> objective = BalancedDesign(power_weight=1.0, sample_weight=1.0,
    ...                           target_power=0.90, target_n=3000)
    >>> objective.target_power
    0.9
    """

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


class DesignOptimizer:
    """Design optimization engine.

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> objective = MinimizeASN(max_n=3000)
    >>> optimizer = DesignOptimizer(spec, objective)
    >>> optimizer.base_spec.test.alpha
    0.025
    """

    def __init__(self, base_spec: DesignSpec, objective: DesignObjective):
        self.base_spec = base_spec
        self.objective = objective
        self.optimization_history: list[Dict[str, Any]] = []

    def optimize_info_times(self, n_analyses: int) -> np.ndarray:
        """Optimize information times.

        Args:
            n_analyses: Number of analyses

        Returns:
            Optimal information times
        """

        def objective_func(x: np.ndarray) -> float:
            """Objective function to optimize."""
            # x is [t1, t2, ..., t_{k-1}] (last one fixed at 1.0)
            info_times = np.append(x, 1.0)
            info_times = np.sort(info_times)  # Ensure monotonic increase

            # Update configuration
            spec = self._clone_spec()
            spec.sequential.info_times = info_times.tolist()
            spec.sequential.info_spacing = InformationSpacing.CUSTOM

            # Evaluate
            lab = DesignLab(spec)
            score = self.objective.evaluate(spec, lab)

            self.optimization_history.append(
                {"info_times": info_times.tolist(), "score": score}
            )

            return score

        # Initial guess: equal spacing
        x0 = np.array([(i + 1) / n_analyses for i in range(n_analyses - 1)])

        # Constraints: 0 < t1 < t2 < ... < t_{k-1} < 1
        bounds = [(0.1, 0.99) for _ in range(n_analyses - 1)]

        # Execute optimization
        result = minimize(
            objective_func,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50},
        )

        optimal_times = np.append(result.x, 1.0)
        return np.sort(optimal_times)

    def optimize_n_analyses(self, min_k: int = 2, max_k: int = 10) -> int:
        """Search for optimal number of analyses.

        Args:
            min_k: Minimum number of analyses
            max_k: Maximum number of analyses

        Returns:
            Optimal number of analyses
        """

        best_k = min_k
        best_score = float("inf")

        for k in range(min_k, max_k + 1):
            spec = self._clone_spec()
            spec.sequential.n_analyses = k

            lab = DesignLab(spec)
            score = self.objective.evaluate(spec, lab)

            if score < best_score:
                best_score = score
                best_k = k

        return best_k

    def optimize_alpha(self, min_alpha: float = 0.001, max_alpha: float = 0.1) -> float:
        """Search for optimal alpha level.

        Args:
            min_alpha: Minimum alpha value
            max_alpha: Maximum alpha value

        Returns:
            Optimal alpha value
        """

        def objective_func(alpha: float) -> float:
            spec = self._clone_spec()
            spec.test.alpha = alpha
            lab = DesignLab(spec)
            return self.objective.evaluate(spec, lab)

        result = minimize_scalar(
            objective_func, bounds=(min_alpha, max_alpha), method="bounded"
        )

        return float(result.x)

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

    def _clone_spec(self) -> DesignSpec:
        """Create a clone of the design specification."""
        return copy.deepcopy(self.base_spec)

    def get_optimization_summary(self) -> pd.DataFrame:
        """Return optimization history as DataFrame."""
        if not self.optimization_history:
            return pd.DataFrame()

        return pd.DataFrame(self.optimization_history)
