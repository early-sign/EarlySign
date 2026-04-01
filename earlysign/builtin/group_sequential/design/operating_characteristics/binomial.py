"""Binomial Operating Characteristics Evaluator.

This module provides the domain-specific adapter for evaluating Binomial A/B tests.
"""

from typing import Any, List, Literal, Optional, cast

import numpy as np

from earlysign.builtin.group_sequential.core.model import CanonicalJointModel
from earlysign.builtin.group_sequential.design.operating_characteristics.engines import (
    AsymptoticSimulator,
    EvaluationResult,
    MonteCarloSimulator,
    NumericalCalculator,
    OperatingCharacteristicsEvaluator,
    SimulationCurve,
)
from earlysign.parts.stats.gaussian_process import CanonicalGaussianProcess
from earlysign.schema.ES3 import (
    GST,
    Base as ES3_BASE,
)


class BinomialOperatingCharacteristicsEvaluator(MonteCarloSimulator):
    """Adapter for operating characteristics of Binomial tests.

    This simulation-based evaluator acts as a DataMonteCarloSimulator (via delegation
    or direct inheritance in the future). Currently delegates to AsymptoticSimulator
    or NumericalCalculator for model-based asymptotic evaluation, but interprets
    the results in the context of the Binomial Protocol (sample size, lift).
    """

    def __init__(
        self,
        protocol: GST.Protocol,
        method: Literal["simulation", "numerical_integration"] = "simulation",
        n_sims: int = 50000,
        seed: Optional[int] = None,
    ):
        """Initialize the Binomial Evaluator.

        Args:
            protocol: The BinomialABProtocol describing the design.
            method: Evaluation method ("simulation" or "numerical_integration").
            n_sims: Number of simulations (for simulation method).
            seed: Random seed.
        """
        self.protocol = protocol
        self.method = method
        self.n_sims = n_sims
        self.seed = seed
        self.evaluator: OperatingCharacteristicsEvaluator

        # 1. Inspect Task to get Baseline/Target Props and Arms
        task = protocol.task
        if not isinstance(task.hypotheses.target_effect, GST.BinaryEffectSize):
            raise ValueError(
                "Evaluator requires BinaryEffectSize in protocol hypotheses."
            )

        props = task.hypotheses.target_effect.proportions
        arms_struct = task.arms
        if not isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            raise ValueError(
                f"BinomialOperatingCharacteristicsEvaluator requires a TwoArmComparison arm structure, but got {type(arms_struct).__name__}."
            )

        self.control_arm_name = arms_struct.control_arm_name
        self.treatment_arm_name = arms_struct.treatment_arm_name
        self.arm_names = [self.control_arm_name, self.treatment_arm_name]
        self.arms = 2

        # Identify control and treatment proportions explicitly
        self.p_control = float(props[self.control_arm_name])
        p_t = float(props[self.treatment_arm_name])

        self.target_delta = float(p_t) - self.p_control

        # 2. Derive statistical parameters for Canonical Model
        timer = protocol.method.stopping_policy.timer
        if not isinstance(timer, GST.SampleSizeTimer):
            raise ValueError(
                f"Protocol timer must be SampleSizeTimer, got {type(timer).__name__}."
            )
        self.n_max_total = sum(timer.max_sample_size.values())

        # Variance per arm (assumed balanced for now)
        sigma2 = self.p_control * (1.0 - self.p_control)
        if self.arms == 1:
            # I = N / sigma^2
            self.i_max = self.n_max_total / sigma2
            self.n_max_per_arm = timer.max_sample_size
        else:
            # I = N_total / (4 * sigma^2) = n_arm / (2 * sigma^2)
            self.i_max = self.n_max_total / (4 * sigma2)
            n_arm = self.n_max_total // 2
            self.n_max_per_arm = {name: n_arm for name in self.arm_names}

        # 3. Solve Design Boundaries (Target Drift)
        target_drift = self.target_delta * np.sqrt(self.i_max)

        self.model = CanonicalJointModel.from_spec(protocol)

        # Solve boundaries for the DESIGN (using default integration for solving)
        self.upper, self.lower = self.model.solve_boundaries(drift=target_drift)
        self.upper = cast(np.ndarray, self.upper)
        self.lower = cast(np.ndarray, self.lower)
        self.info_times = cast(np.ndarray, self.model.info_times)

        # 4. Instantiate Inner Evaluator (Delegation)
        if method == "simulation":
            self.evaluator = AsymptoticSimulator(
                model=CanonicalGaussianProcess(),
                n_sims=n_sims,
                seed=seed,
            )
        else:
            self.evaluator = NumericalCalculator()

    def evaluate_point(
        self,
        drift: float,
        info_times: Optional[np.ndarray] = None,
        upper_boundaries: Optional[np.ndarray] = None,
        lower_boundaries: Optional[np.ndarray] = None,
        sided: int = 1,
        **kwargs: Any,
    ) -> EvaluationResult:
        # Compatibility wrapper
        # Use internal values if not provided
        it = info_times if info_times is not None else self.info_times
        ub = upper_boundaries if upper_boundaries is not None else self.upper
        lb = lower_boundaries if lower_boundaries is not None else self.lower

        assert it is not None
        assert ub is not None
        assert lb is not None

        return self.evaluator.evaluate_point(
            drift,
            info_times=it,
            upper_boundaries=ub,
            lower_boundaries=lb,
            sided=sided,
            **kwargs,
        )

    def evaluate_lift_at(
        self,
        effect_sizes_pct: List[float],
        metric_type: Literal[
            "relative_lift_pct", "absolute_diff_pct"
        ] = "relative_lift_pct",
    ) -> SimulationCurve:
        """Evaluate OC curve at specific relative lift percentages or absolute differences.

        Args:
            effect_sizes_pct: List of values in percent (e.g., [5.0, 10.0]).
            metric_type: "relative_lift_pct" (pt-pc)/pc or "absolute_diff_pct" (pt-pc).

        Returns:
            SimulationCurve with results mapped to the requested metric.
        """
        x_values = np.array(effect_sizes_pct, dtype=float)

        if metric_type == "relative_lift_pct":
            lifts = x_values / 100.0
            p_treatments = self.p_control * (1 + lifts)
            deltas = p_treatments - self.p_control
            null_val = 0.0
            target_val = (
                (self.target_delta / self.p_control) * 100
                if self.p_control > 0
                else 0.0
            )
        elif metric_type == "absolute_diff_pct":
            deltas = x_values / 100.0
            null_val = 0.0
            target_val = self.target_delta * 100
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        return self._evaluate_lift_from_deltas(
            deltas, x_values, metric_type, target_val, null_val
        )

    def _evaluate_lift_from_deltas(
        self,
        deltas: np.ndarray,
        x_values: np.ndarray,
        metric_type: str,
        target_val: float,
        null_val: float,
    ) -> SimulationCurve:
        """Internal helper to evaluate curve from health-scale deltas."""
        from scipy.stats import norm

        # 0. Calculate n_fixed (for equivalent alpha/power at target delta)
        efficacy = self.protocol.task.efficacy
        if efficacy is None:
            raise ValueError("Protocol task missing efficacy requirements.")
        alpha = efficacy.alpha

        futility = self.protocol.task.futility
        if futility is None:
            raise ValueError("Protocol task missing futility requirements.")
        power = futility.power

        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)

        if self.target_delta != 0:
            i_fixed = ((z_alpha + z_beta) ** 2) / (self.target_delta**2)
            var_unit = 4 * self.p_control * (1 - self.p_control)
            n_fixed = i_fixed * var_unit
        else:
            n_fixed = None

        # Map to Drifts
        drifts = deltas * np.sqrt(self.i_max)

        # Evaluate
        curve = self.evaluator.evaluate_curve(
            drifts,
            info_times=self.info_times,
            upper_boundaries=cast(np.ndarray, self.upper),
            lower_boundaries=cast(np.ndarray, self.lower),
            sided=self.model.tails,
        )

        # Inject Domain Context
        curve.x_values = x_values
        curve.metric_type = metric_type
        curve.target_x_value = target_val
        curve.p_control = self.p_control
        curve.null_x_value = null_val
        curve.info_times = self.info_times
        curve.n_max_per_arm = self.n_max_per_arm

        if n_fixed is not None:
            n_fixed_arm = n_fixed / self.arms
            curve.n_fixed_per_arm = {name: n_fixed_arm for name in self.arm_names}

        for res in curve.results:
            res.expected_n_per_arm = {
                name: res.asn * self.n_max_per_arm[name] for name in self.arm_names
            }
            res.n_per_arm_schedule = {
                name: self.info_times * self.n_max_per_arm[name]
                for name in self.arm_names
            }
            res.info_times = self.info_times

        return curve

    def evaluate_lift_curve(
        self,
        range_min: float = -0.5,  # -50%
        range_max: float = 0.5,  # +50%
        n_points: int = 50,
        metric_type: Literal[
            "relative_lift_pct", "absolute_diff_pct"
        ] = "relative_lift_pct",
    ) -> SimulationCurve:
        """Evaluate OC curve over a range of lifts/differences.

        Args:
            range_min: Minimum value of measure (relative lift or absolute diff).
            range_max: Maximum value.
            n_points: Number of points to evaluate.
            metric_type: "relative_lift_pct" or "absolute_diff_pct".

        Returns:
            SimulationCurve with results mapped to the requested metric.
        """
        # Define range in Metric Space
        if metric_type == "relative_lift_pct":
            lifts = np.linspace(range_min, range_max, n_points)
            x_values = lifts * 100

            p_treatments = self.p_control * (1 + lifts)
            deltas = p_treatments - self.p_control

            null_val = 0.0
            target_val = (
                (self.target_delta / self.p_control) * 100
                if self.p_control > 0
                else 0.0
            )

        elif metric_type == "absolute_diff_pct":
            diffs = np.linspace(range_min, range_max, n_points)
            x_values = diffs * 100

            deltas = diffs

            null_val = 0.0
            target_val = self.target_delta * 100
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        return self._evaluate_lift_from_deltas(
            deltas, x_values, metric_type, target_val, null_val
        )

    def evaluate_metric_curve(
        self,
        range_min: float,
        range_max: float,
        n_points: int,
        metric_type: str,
    ) -> SimulationCurve:
        """Alias for evaluate_lift_curve to satisfy ProtocolEvaluator."""
        # Check if metric_type is valid for this evaluator
        if metric_type not in ["relative_lift_pct", "absolute_diff_pct"]:
            # If default generic metric is passed, map to default specific
            if metric_type == "effect_size":
                metric_type = "relative_lift_pct"

        return self.evaluate_lift_curve(
            range_min=range_min,
            range_max=range_max,
            n_points=n_points,
            metric_type=cast(
                Literal["relative_lift_pct", "absolute_diff_pct"], metric_type
            ),
        )

    def evaluate_metric_at(
        self,
        x_values: List[float],
        metric_type: str,
    ) -> SimulationCurve:
        """Alias for evaluate_lift_at to satisfy ProtocolEvaluator."""
        if metric_type not in ["relative_lift_pct", "absolute_diff_pct"]:
            if metric_type == "effect_size":
                metric_type = "relative_lift_pct"

        return self.evaluate_lift_at(
            effect_sizes_pct=x_values,
            metric_type=cast(
                Literal["relative_lift_pct", "absolute_diff_pct"], metric_type
            ),
        )
