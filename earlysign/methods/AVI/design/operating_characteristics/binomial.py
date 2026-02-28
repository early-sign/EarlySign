"""Binomial Operating Characteristics Evaluator for AVI.

This module provides the domain-specific adapter for evaluating Binomial A/B tests
using Anytime Valid Inference (mSPRT and Confidence Sequences).
"""

from typing import Any, List, Literal, Optional, cast

import numpy as np

from earlysign.methods.AVI.design.operating_characteristics.engines import (
    AVIMonteCarloSimulator,
)
from earlysign.methods.group_sequential.design.operating_characteristics.engines import (
    EvaluationResult,
    SimulationCurve,
)
from earlysign.schema.ES3 import Base as ES3_BASE
from earlysign.schema.ES3.AVI import GAVIMethodSpec, MSPRTMethodSpec, Protocol


class BinomialAVIEvaluator(AVIMonteCarloSimulator):
    """Adapter for Operating Characteristics of Binomial tests using AVI."""

    def __init__(
        self,
        protocol: Protocol,
        p_control: float = 0.5,
        n_sims: int = 2000,
        seed: Optional[int] = None,
        max_n: int = 5000,
    ):
        super().__init__(protocol=protocol, n_sims=n_sims, seed=seed, max_n=max_n)

        task = protocol.task
        arms_struct = task.arms
        if not isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            raise ValueError("Evaluator requires a TwoArmComparison arm structure.")

        self.control_arm_name = arms_struct.control_arm_name
        self.treatment_arm_name = arms_struct.treatment_arm_name

        self.p_control = p_control

        self.method_type = (
            "gavi" if isinstance(self.protocol.method, GAVIMethodSpec) else "msprt"
        )

    def evaluate_point(
        self,
        effect_size: float,
        metric_type: Literal[
            "relative_lift_pct", "absolute_diff_pct"
        ] = "relative_lift_pct",
        **kwargs: Any,
    ) -> EvaluationResult:
        if metric_type == "relative_lift_pct":
            p_true_t = self.p_control * (1 + effect_size / 100.0)
        else:
            p_true_t = self.p_control + (effect_size / 100.0)

        # Optional: bound probabilities
        p_true_t = max(min(p_true_t, 1.0), 0.0)

        n_steps = self.max_n_total // 2

        # Simulate Data Paths: 1 success/failure per arm per step -> max_n steps
        paths_c = np.random.binomial(1, self.p_control, size=(self.n_sims, n_steps))
        paths_t = np.random.binomial(1, p_true_t, size=(self.n_sims, n_steps))

        cum_s_c = np.cumsum(paths_c, axis=1)
        cum_s_t = np.cumsum(paths_t, axis=1)

        n_array = np.arange(1, n_steps + 1)
        est_c = cum_s_c / n_array
        est_t = cum_s_t / n_array

        trajectory = est_t - est_c
        boundary = np.full(n_steps, np.inf)

        if self.method_type == "msprt":
            method_msprt = cast(MSPRTMethodSpec, self.protocol.method)
            alpha = (
                method_msprt.alpha * 2
                if method_msprt.sides == "one"
                else method_msprt.alpha
            )
            tau = method_msprt.mde

            var_c = est_c * (1 - est_c)
            var_t = est_t * (1 - est_t)
            var_diff = (var_c + var_t) / n_array
            var_diff = np.maximum(var_diff, 1e-10)

            info = np.where(var_diff > 0, 1.0 / var_diff, 0.0)
            phi = 1.0 / (tau**2)
            ratio = (info + phi) / phi

            term = 2 * var_diff * (1 + phi / info) * np.log(np.sqrt(ratio) / alpha)
            term = np.maximum(term, 0.0)
            boundary = np.sqrt(term)

        elif self.method_type == "gavi":
            method_gavi = cast(GAVIMethodSpec, self.protocol.method)
            alpha = (
                method_gavi.alpha * 2
                if method_gavi.sides == "one"
                else method_gavi.alpha
            )
            sigma2 = method_gavi.variance  # Assuming fixed known variance for design
            if sigma2 is None:
                sigma2 = 0.25  # Fallback to bounded variance max

            V = (sigma2 / n_array) * 2
            phi = float(method_gavi.max_n)
            n_val = (n_array * 2) / 2.0

            denom = np.log(np.log(np.exp(1) * alpha ** (-2))) - 2 * np.log(alpha)
            rho = phi / denom
            term_log = np.log((n_val + rho) / (rho * alpha**2))
            uv = np.sqrt((n_val + rho) * term_log)

            boundary = np.sqrt(V / 2.0) * uv / np.sqrt(n_val)

        # Enforce configurable burn-in period to simulate engine behavior and avoid instability
        burn_in = getattr(self.protocol.method, "burn_in", 100) or 100
        # Now handles both 1D (theoretical) and 2D (empirical) shape boundaries
        if boundary.ndim == 2:
            boundary[:, n_array < burn_in] = np.inf
        else:
            boundary[n_array < burn_in] = np.inf

        sides = str(self.protocol.method.sides)
        if sides == "two":
            crossed = np.abs(trajectory) > boundary
        else:
            crossed = trajectory > boundary

        crossed_with_end = np.hstack([crossed, np.ones((self.n_sims, 1), dtype=bool)])
        stop_idx = np.argmax(crossed_with_end, axis=1)

        is_stopped = stop_idx < n_steps
        power = float(np.mean(is_stopped))

        stop_n_c = np.where(is_stopped, stop_idx + 1, n_steps)
        expected_n_per_arm = float(np.mean(stop_n_c))

        return EvaluationResult(
            drift=effect_size,
            power=power,
            asn=expected_n_per_arm / n_steps,
            prob_stop_efficacy=np.array([power]),
            prob_stop_futility=np.array([1 - power]),
            prob_stop_total=np.array([1.0]),
            expected_n_per_arm={
                self.control_arm_name: expected_n_per_arm,
                self.treatment_arm_name: expected_n_per_arm,
            },
            expected_sample_size=expected_n_per_arm * 2,
        )

    def evaluate_lift_at(
        self,
        effect_sizes_pct: List[float],
        metric_type: Literal[
            "relative_lift_pct", "absolute_diff_pct"
        ] = "relative_lift_pct",
    ) -> SimulationCurve:
        if self.seed is not None:
            np.random.seed(self.seed)

        x_values = np.array(effect_sizes_pct, dtype=float)

        results = [
            self.evaluate_point(eff, metric_type=metric_type) for eff in x_values
        ]

        # Approximation of generic "target_x_value" for visuals
        target_val = 0.0

        curve = SimulationCurve(
            x_values=x_values,
            results=results,
            metric_type=metric_type,
            n_max=self.max_n_total,
            p_control=self.p_control,
            target_x_value=target_val,
            n_max_per_arm=self.n_max_per_arm,
        )
        return curve

    def evaluate_metric_at(
        self,
        x_values: List[float],
        metric_type: str,
    ) -> SimulationCurve:
        """Alias for evaluate_lift_at to satisfy visualization APIs."""
        if metric_type not in ["relative_lift_pct", "absolute_diff_pct"]:
            if metric_type == "effect_size":
                metric_type = "relative_lift_pct"

        return self.evaluate_lift_at(
            effect_sizes_pct=x_values,
            metric_type=cast(
                Literal["relative_lift_pct", "absolute_diff_pct"], metric_type
            ),
        )
