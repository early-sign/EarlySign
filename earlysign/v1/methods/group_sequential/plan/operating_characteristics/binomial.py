"""Binomial Operating Characteristics Evaluator.

This module provides the domain-specific adapter for evaluating Binomial A/B tests.
"""

from typing import Literal, Optional

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.plan.operating_characteristics.engines import (
    AsymptoticSimulator,
    EvaluationResult,
    MonteCarloSimulator,
    NumericalCalculator,
    OperatingCharacteristicsEvaluator,
    SimulationCurve,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
)
from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess


class BinomialABOperatingCharacteristicsEvaluator(MonteCarloSimulator):
    """Adapter for operating characteristics of Binomial A/B tests.

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

        # 1. Inspect Task to get Baseline/Target Props
        self.p_control = 0.5
        self.target_delta = 0.0

        task = protocol.task
        if isinstance(task.hypotheses.target_effect, GST.BinaryEffectSize):
            props = task.hypotheses.target_effect.proportions
            if "control" in props:
                self.p_control = props["control"]
            elif len(props) > 0:
                self.p_control = list(props.values())[0]

            if "treatment" in props:
                p_t = props["treatment"]
                self.target_delta = p_t - self.p_control
            elif len(props) > 1:
                p_t = list(props.values())[1]
                self.target_delta = p_t - self.p_control

        # 2. Derive statistical parameters for Canoncial Model
        timer = protocol.method.stopping_policy.timer
        self.n_max = (
            int(timer.max_sample_size) if hasattr(timer, "max_sample_size") else 1000
        )

        var_diff = 2 * self.p_control * (1 - self.p_control)
        self.i_max = self.n_max / var_diff

        # 3. Solve Design Boundaries (Target Drift)
        target_drift = self.target_delta * np.sqrt(self.i_max)

        self.model = CanonicalJointModel.from_spec(protocol)

        # Solve boundaries for the DESIGN (using default integration for solving)
        self.upper, self.lower = self.model.solve_boundaries(drift=target_drift)
        self.info_times = self.model.info_times

        # 4. Instantiate Inner Evaluator (Delegation)
        if method == "simulation":
            self.evaluator = AsymptoticSimulator(
                model=CanonicalGaussianProcess(),
                n_sims=n_sims,
                seed=seed,
            )
        else:
            self.evaluator = NumericalCalculator()

    def evaluate_point(self, drift: float) -> EvaluationResult:
        # Compatibility wrapper
        return self.evaluator.evaluate_point(
            drift,
            info_times=self.info_times,
            upper_boundaries=self.upper,
            lower_boundaries=self.lower,
        )

    def evaluate_lift_curve(
        self,
        range_min: float = -0.5,  # -50%
        range_max: float = 0.5,  # +50%
        n_points: int = 50,
        metric_type: str = "relative_lift_pct",
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

        # Map to Drifts
        # drift = delta * sqrt(I_max)
        drifts = deltas * np.sqrt(self.i_max)

        # Evaluate
        curve = self.evaluator.evaluate_curve(
            drifts,
            info_times=self.info_times,
            upper_boundaries=self.upper,
            lower_boundaries=self.lower,
        )

        # Inject Domain Context
        curve.x_values = x_values
        curve.metric_type = metric_type
        curve.n_max = self.n_max
        curve.p_control = self.p_control
        curve.null_x_value = null_val
        curve.target_x_value = target_val
        curve.p_control = self.p_control

        # Rescale ASN to Sample Size
        for res in curve.results:
            if res.expected_sample_size is None:
                res.expected_sample_size = res.asn * self.n_max

        return curve
