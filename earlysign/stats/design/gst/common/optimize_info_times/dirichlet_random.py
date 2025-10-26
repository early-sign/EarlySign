"""Design optimization objectives and optimizer."""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.types import InformationSpacing


class DesignObjective(ABC):
    """Abstract base class for group sequential design optimization objectives.

    This class defines the interface for optimization objectives used in
    sequential trial design. Concrete implementations specify what aspect
    of the design to optimize (e.g., minimize sample size, maximize power).

    The optimization framework follows a common pattern:
    1. Evaluate the objective function for a given design
    2. Apply constraints (max sample size, target power, etc.)
    3. Use penalty functions to handle constraint violations
    4. Return a single scalar value to minimize

    All objectives are formulated as minimization problems. For maximization
    objectives (e.g., maximize power), return the negative value.

    Attributes
    ----------
    None (defined by subclasses)

    Methods
    -------
    evaluate(spec, lab) -> float
        Compute the objective function value for a design specification.
        Lower values are better. Constraint violations should return
        large penalty values.

    get_constraints(spec) -> Dict[str, Any]
        Return the constraints applied to this objective as a dictionary.
        Used for reporting and validation purposes.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> from earlysign.stats.design.gst.common.lab import DesignLab
    >>>
    >>> # Define a custom objective
    >>> class CustomObjective(DesignObjective):
    ...     def evaluate(self, spec, lab):
    ...         lab.compute_boundaries()
    ...         lab.run_simulations()
    ...         return lab.simulation_results["expected_sample_size"]
    ...     def get_constraints(self, spec):
    ...         return {}

    See Also
    --------
    MinimizeASN : Minimize expected sample size
    MaximizePower : Maximize statistical power
    BalancedDesign : Balance multiple objectives
    DesignOptimizer : Optimization engine
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

    This objective function minimizes the expected sample size under the
    alternative hypothesis while ensuring that:
    1. Statistical power meets or exceeds the target
    2. Maximum sample size does not exceed the specified limit

    Constraint violations are handled with penalty functions that return
    large values, guiding the optimizer away from infeasible regions.

    Parameters
    ----------
    planned_max_n : int
        Maximum allowable sample size at the final analysis.
        Designs exceeding this will incur penalties.
    target_power : float, default=0.90
        Minimum required statistical power (probability of rejecting
        null when alternative is true). Must be in (0, 1).

    Attributes
    ----------
    planned_max_n : int
        Maximum sample size constraint
    target_power : float
        Minimum power constraint

    Methods
    -------
    evaluate(spec, lab) -> float
        Compute expected sample size with penalties for constraint violations.
    get_constraints(spec) -> Dict[str, Any]
        Return {'planned_max_n': int, 'target_power': float}

    Examples
    --------
    >>> objective = MinimizeASN(planned_max_n=3000, target_power=0.90)
    >>> objective.planned_max_n
    3000
    >>> objective.target_power
    0.9

    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> from earlysign.stats.design.gst.common.lab import DesignLab
    >>> spec = ProportionsDesignSpec()
    >>> lab = DesignLab(spec)
    >>> value = objective.evaluate(spec, lab)  # Returns ASN or penalty

    Notes
    -----
    The penalty structure:
    - Power < target: penalty = planned_max_n * 10 + (target_power - power) * 10000
    - Sample > planned_max_n: penalty = planned_max_n * 10 + (sample - planned_max_n) * 100
    - Other errors: penalty = planned_max_n * 100

    See Also
    --------
    MaximizePower : Alternative objective focusing on power
    BalancedDesign : Multi-objective optimization
    DesignOptimizer : Optimization engine
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
    """Maximize statistical power subject to maximum sample size constraint.

    This objective function maximizes the probability of rejecting the null
    hypothesis when the alternative is true, while ensuring the maximum
    sample size does not exceed a specified limit.

    Since the optimization framework minimizes objectives, this returns
    the negative power value.

    Parameters
    ----------
    planned_max_n : int
        Maximum allowable sample size at the final analysis.
        Designs exceeding this will incur penalties (return 1.0).

    Attributes
    ----------
    planned_max_n : int
        Maximum sample size constraint

    Methods
    -------
    evaluate(spec, lab) -> float
        Returns -power (negative for minimization). Returns 1.0 penalty
        if max_sample_size > planned_max_n or if computation fails.
    get_constraints(spec) -> Dict[str, Any]
        Return {'planned_max_n': int}

    Examples
    --------
    >>> objective = MaximizePower(planned_max_n=3000)
    >>> objective.planned_max_n
    3000

    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> from earlysign.stats.design.gst.common.lab import DesignLab
    >>> spec = ProportionsDesignSpec()
    >>> lab = DesignLab(spec)
    >>> neg_power = objective.evaluate(spec, lab)  # Returns -power or 1.0

    Notes
    -----
    The returned value is -power because scipy.optimize.minimize() minimizes
    objectives. To maximize power, we minimize -power.

    Constraint violations (max_sample_size > planned_max_n) return 1.0, which is
    worse than any feasible solution (power ∈ [0, 1] → -power ∈ [-1, 0]).

    See Also
    --------
    MinimizeASN : Alternative objective focusing on sample size
    BalancedDesign : Multi-objective optimization
    DesignOptimizer : Optimization engine
    """

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
    """Multi-objective optimization balancing power and sample size efficiency.

    This objective function creates a weighted composite score that balances
    statistical power against expected sample size. It allows flexible
    trade-offs through configurable weights and target values.

    The composite score measures:
    1. Power deficit: (target_power - actual_power)² weighted by power_weight
    2. Sample size ratio: (ASN / target_n) weighted by sample_weight

    This formulation allows users to specify their priorities:
    - Higher power_weight → prioritize achieving target power
    - Higher sample_weight → prioritize reducing sample size
    - Equal weights → balanced optimization

    Parameters
    ----------
    power_weight : float, default=1.0
        Relative importance of power deficit in composite score.
        Larger values prioritize achieving target_power.
    sample_weight : float, default=1.0
        Relative importance of sample size in composite score.
        Larger values prioritize reducing expected sample size.
    target_power : float, default=0.90
        Target statistical power. Power deficit is measured relative
        to this value. Must be in (0, 1).
    target_n : int, default=3000
        Reference sample size for normalization. Sample size cost is
        expressed as ASN / target_n.

    Attributes
    ----------
    power_weight : float
        Weight for power component
    sample_weight : float
        Weight for sample size component
    target_power : float
        Target power level
    target_n : int
        Reference sample size

    Methods
    -------
    evaluate(spec, lab) -> float
        Compute weighted composite score:
        power_weight * (target_power - power)² + sample_weight * (ASN / target_n)
    get_constraints(spec) -> Dict[str, Any]
        Return all parameter values as constraints dict

    Examples
    --------
    >>> # Equal weighting
    >>> objective = BalancedDesign(power_weight=1.0, sample_weight=1.0,
    ...                           target_power=0.90, target_n=3000)
    >>> objective.target_power
    0.9

    >>> # Prioritize power
    >>> objective_power = BalancedDesign(power_weight=10.0, sample_weight=1.0,
    ...                                 target_power=0.90, target_n=3000)

    >>> # Prioritize efficiency
    >>> objective_eff = BalancedDesign(power_weight=1.0, sample_weight=10.0,
    ...                               target_power=0.80, target_n=3000)

    Notes
    -----
    The composite score does not use hard constraints or penalties.
    Instead, it smoothly trades off power and sample size according to
    the specified weights. This can result in designs that:
    - Slightly undershoot target_power if sample size benefits are large
    - Use more samples than optimal ASN if power gains are significant

    For hard constraints, use MinimizeASN or MaximizePower instead.

    See Also
    --------
    MinimizeASN : Hard power constraint with ASN minimization
    MaximizePower : Hard sample size constraint with power maximization
    DesignOptimizer : Optimization engine
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

    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> objective = MinimizeASN(planned_max_n=3000)
    >>> optimizer = DesignOptimizer(spec, objective)
    >>> optimizer.base_spec.test.alpha
    0.025
    """

    def __init__(self, base_spec: DesignSpec, objective: DesignObjective):
        self.base_spec = base_spec
        self.objective = objective
        self.optimization_history: list[Dict[str, Any]] = []

    def optimize_info_times(self, n_analyses: int, n_samples: int = 100) -> np.ndarray:
        """Random sampling on the simplex: pick the best info time split.

        Args:
            n_analyses: Number of analyses
            n_samples: Number of random samples to try

        Returns:
            Optimal information times (monotonically increasing, last is 1.0)
        """
        best_score = float("inf")
        best_times: np.ndarray | None = None
        for _ in range(n_samples):
            # Dirichlet sampling for k positive segments summing to 1
            s = np.random.dirichlet([1.0] * n_analyses)
            info_times = np.cumsum(s)
            # Ensure last is exactly 1.0
            info_times[-1] = 1.0

            # Update config
            spec = self._clone_spec()
            spec.sequential.info_times = info_times.tolist()
            spec.sequential.info_spacing = InformationSpacing.CUSTOM
            lab = DesignLab(spec)
            score = self.objective.evaluate(spec, lab)
            if score < best_score:
                best_score = score
                best_times = info_times.copy()

        assert best_times is not None, "No valid info times found"
        return best_times

    def optimize_info_times_and_n_analyses(
        self, max_k: int, min_k: int = 2, n_samples: int = 20
    ) -> tuple[int, np.ndarray]:
        """Randomly sample both number of analyses and info time splits, return best (n_analyses, info_times), with right-skewed splits (late looks).

        Args:
            max_k: Maximum number of analyses (inclusive, required)
            min_k: Minimum number of analyses (inclusive, default=2)
            n_samples: Number of random splits per k

        Returns:
            Tuple (best_n_analyses, best_info_times)
        """
        best_score = float("inf")
        best_k: int | None = None
        best_times: np.ndarray | None = None
        for k in range(min_k, max_k + 1):
            # Right-skewed Dirichlet: last segment has large alpha
            dirichlet_alpha = [1.0] * (k - 1) + [2.0]
            for _ in range(n_samples):
                s = np.random.dirichlet(dirichlet_alpha)
                info_times = np.cumsum(s)
                info_times[-1] = 1.0
                spec = self._clone_spec()
                spec.sequential.n_analyses = k
                spec.sequential.info_times = info_times.tolist()
                spec.sequential.info_spacing = InformationSpacing.CUSTOM
                lab = DesignLab(spec)
                score = self.objective.evaluate(spec, lab)
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_times = info_times.copy()
        assert best_k is not None and best_times is not None, "No valid design found"
        return best_k, best_times

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

        from scipy.optimize import OptimizeResult

        result = minimize_scalar(
            objective_func, bounds=(min_alpha, max_alpha), method="bounded"
        )
        # minimize_scalar returns OptimizeResult, which always has .x
        return float(cast(OptimizeResult, result).x)

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
