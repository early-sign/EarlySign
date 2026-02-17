"""Continuous Operating Characteristics Evaluator.

This module provides the domain-specific adapter for evaluating Continuous/Gaussian A/B tests.
"""

from typing import Any, List, Literal, Optional, cast

import numpy as np

from earlysign.methods.group_sequential.plan.operating_characteristics.engines import (
    AsymptoticSimulator,
    EvaluationResult,
    MonteCarloSimulator,
    NumericalCalculator,
    OperatingCharacteristicsEvaluator,
    SimulationCurve,
)
from earlysign.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
)
from earlysign.schema.ES3 import (
    GST,
    Base as ES3_BASE,
)
from earlysign.stats.gaussian_process import CanonicalGaussianProcess


class ContinuousOperatingCharacteristicsEvaluator(MonteCarloSimulator):
    """Adapter for operating characteristics of Continuous/Gaussian tests.

    Assumes Normal distribution with known variance (sigma^2) for planning purposes.

    Examples:
        >>> from earlysign.schema.ES3 import GST, Base as ES3_BASE
        >>> from earlysign.methods.group_sequential.plan.operating_characteristics import (
        ...     ContinuousOperatingCharacteristicsEvaluator,
        ... )

        1. Verify I_max calculation for 1-arm and 2-arm continuous designs

        >>> effect = GST.ContinuousEffectSize(
        ...     means={"control": 0.0, "treatment": 1.0},
        ...     standard_deviation=2.0,
        ... )
        >>> hyp = GST.HypothesisSpec(
        ...     h_null_description="H0",
        ...     h_alt_description="H1",
        ...     test_logic=GST.EqualityHypothesis(),
        ...     target_effect=effect,
        ... )
        >>> timer = GST.SampleSizeTimer(
        ...     unit=GST.Unit.INDIVIDUALS,
        ...     max_sample_size={"control": 50, "treatment": 50}
        ... )
        >>> sched = GST.EquidistantSchedule(n_looks=2)
        >>> strat = GST.OBrienFlemingStrategy(
        ...     alpha=0.05,
        ...     sided=GST.Sided.TWO,
        ...     statistical_model=GST.CanonicalGaussianModel(),
        ... )
        >>> policy = GST.StoppingPolicySpec(
        ...     statistic=GST.TwoArmContinuousZ(
        ...         variance=GST.TwoArmEstimatedVariance(method=GST.MethodModel.POOLED)
        ...     ),
        ...     strategy=strat,
        ...     timer=timer,
        ...     schedule=sched,
        ... )
        >>> method = GST.MethodSpec(kind="group_sequential", stopping_policy=policy)

        --- 2-Arm Check ---

        >>> task_2arm = GST.TaskSpec(
        ...     kind="group_sequential",
        ...     response_type=GST.ResponseType.CONTINUOUS,
        ...     arms=ES3_BASE.TwoArmComparison(
        ...         control_arm_name="control", treatment_arm_name="treatment"
        ...     ),
        ...     hypotheses=hyp,
        ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
        ... )
        >>> protocol_2arm = GST.Protocol(name="Test", task=task_2arm, method=method)
        >>> eval_2arm = ContinuousOperatingCharacteristicsEvaluator(protocol_2arm)
        >>> # I_max = N / (4 * sigma^2) = 100 / (4 * 4) = 100 / 16 = 6.25
        >>> eval_2arm.i_max
        6.25
        >>> eval_2arm.arms
        2

        --- 1-Arm Check ---

        >>> task_1arm = GST.TaskSpec(
        ...     kind="group_sequential",
        ...     response_type=GST.ResponseType.CONTINUOUS,
        ...     arms=ES3_BASE.SingleArm(arm_name="treatment"),
        ...     hypotheses=hyp,
        ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
        ... )
        >>> protocol_1arm = GST.Protocol(name="Test", task=task_1arm, method=method)
        >>> eval_1arm = ContinuousOperatingCharacteristicsEvaluator(protocol_1arm)
        >>> # I_max = N / sigma^2 = 100 / 4 = 25.0
        >>> eval_1arm.i_max
        25.0
        >>> eval_1arm.arms
        1

        2. Verify that power increases with effect size

        >>> effect_power = GST.ContinuousEffectSize(
        ...     means={"control": 0.0, "treatment": 1.0},
        ...     standard_deviation=1.0,
        ... )
        >>> task_power = GST.TaskSpec(
        ...     kind="group_sequential",
        ...     response_type=GST.ResponseType.CONTINUOUS,
        ...     arms=ES3_BASE.TwoArmComparison(
        ...         control_arm_name="control", treatment_arm_name="treatment"
        ...     ),
        ...     hypotheses=GST.HypothesisSpec(
        ...         h_null_description="H0",
        ...         h_alt_description="H1",
        ...         test_logic=GST.EqualityHypothesis(),
        ...         target_effect=effect_power,
        ...     ),
        ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
        ...     futility=GST.FutilityRequirement(power=0.8),
        ... )
        >>> timer_power = GST.SampleSizeTimer(
        ...     unit=GST.Unit.INDIVIDUALS,
        ...     max_sample_size={"control": 100, "treatment": 100}
        ... )
        >>> policy_power = GST.StoppingPolicySpec(
        ...     statistic=GST.TwoArmContinuousZ(
        ...         variance=GST.TwoArmEstimatedVariance(method=GST.MethodModel.POOLED)
        ...     ),
        ...     strategy=GST.OBrienFlemingStrategy(
        ...         alpha=0.05,
        ...         sided=GST.Sided.TWO,
        ...         statistical_model=GST.CanonicalGaussianModel(),
        ...     ),
        ...     timer=timer_power,
        ...     schedule=GST.EquidistantSchedule(n_looks=1),
        ... )
        >>> protocol_power = GST.Protocol(
        ...     name="Test",
        ...     task=task_power,
        ...     method=GST.MethodSpec(kind="group_sequential", stopping_policy=policy_power),
        ... )
        >>> evaluator = ContinuousOperatingCharacteristicsEvaluator(protocol_power, n_sims=5000)
        >>> curve = evaluator.evaluate_metric_at(
        ...     x_values=[0.0, 0.5, 1.0], metric_type="absolute_diff"
        ... )
        >>> # Power at null should be ~ alpha (0.05)
        >>> abs(curve.results[0].power - 0.05) < 0.02
        True
        >>> # Power should be monotonic
        >>> curve.results[1].power > curve.results[0].power
        True
        >>> curve.results[2].power > curve.results[1].power
        True
    """

    def __init__(
        self,
        protocol: GST.Protocol,
        method: Literal["simulation", "numerical_integration"] = "simulation",
        n_sims: int = 50000,
        seed: Optional[int] = None,
    ):
        """Initialize the Continuous Evaluator.

        Args:
            protocol: The Protocol describing the design (must have ContinuousEffectSize).
            method: Evaluation method.
            n_sims: Number of simulations.
            seed: Random seed.
        """
        self.protocol = protocol
        self.method = method
        self.n_sims = n_sims
        self.seed = seed
        self.evaluator: OperatingCharacteristicsEvaluator

        # 1. Inspect Task to get Baseline/Target Means and Variance
        task = protocol.task
        if not isinstance(task.hypotheses.target_effect, GST.ContinuousEffectSize):
            raise ValueError(
                "Evaluator requires ContinuousEffectSize in protocol hypotheses."
            )

        means = task.hypotheses.target_effect.means
        self.sd = task.hypotheses.target_effect.standard_deviation
        self.sigma2 = self.sd**2

        arms_struct = task.arms

        # Determine Arms and Delta
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            self.control_arm_name = arms_struct.control_arm_name
            self.treatment_arm_name = arms_struct.treatment_arm_name
            self.arm_names = [self.control_arm_name, self.treatment_arm_name]
            self.arms = 2

            self.mu_control = float(means[self.control_arm_name])
            mu_t = float(means[self.treatment_arm_name])
            self.target_delta = mu_t - self.mu_control

        elif isinstance(arms_struct, ES3_BASE.SingleArm):
            self.arm_names = [arms_struct.arm_name]
            self.arms = 1
            # For 1-arm, delta is relative to null or fixed value.
            # Usually hypotheses define margins, but EffectSize defines expected mean.
            # We assume target_effect.means contains the single arm's expected mean.
            self.mu_control = 0.0  # Virtual baseline
            mu_t = float(means[arms_struct.arm_name])
            self.target_delta = mu_t  # Absolute value
        else:
            raise ValueError(
                f"ContinuousOperatingCharacteristicsEvaluator supports TwoArmComparison or SingleArm, but got {type(arms_struct).__name__}."
            )

        # 2. Derive statistical parameters for Canonical Model
        timer = protocol.method.stopping_policy.timer
        if not isinstance(timer, GST.SampleSizeTimer):
            raise ValueError(
                f"Protocol timer must be SampleSizeTimer, got {type(timer).__name__}."
            )
        self.n_max_total = sum(timer.max_sample_size.values())

        # Calculate Information I_max
        # I = 1/V_beta
        if self.arms == 1:
            # Var(theta_hat) = sigma^2 / N
            # I = N / sigma^2
            self.i_max = self.n_max_total / self.sigma2
            self.n_max_per_arm = {self.arm_names[0]: self.n_max_total}
        else:
            # Var(theta_hat) = Var(Y_t - Y_c) = sigma^2/n_t + sigma^2/n_c
            # Asssuming n_t = n_c = N/2
            # Var = 4 * sigma^2 / N
            # I = N / (4 * sigma^2)
            self.i_max = self.n_max_total / (4 * self.sigma2)
            self.n_max_per_arm = timer.max_sample_size

        # 3. Solve Design Boundaries (Target Drift)
        # theta = delta
        # drift = theta * sqrt(I)
        target_drift = self.target_delta * np.sqrt(self.i_max)

        self.model = CanonicalJointModel.from_spec(protocol)

        # Solve boundaries
        self.upper, self.lower = self.model.solve_boundaries(drift=target_drift)
        self.upper = cast(np.ndarray, self.upper)
        self.lower = cast(np.ndarray, self.lower)
        self.info_times = cast(np.ndarray, self.model.info_times)

        # 4. Instantiate Inner Evaluator
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

    def evaluate_metric_at(
        self,
        x_values: List[float],
        metric_type: str,
    ) -> SimulationCurve:
        """Evaluate OC curve at specific absolute differences."""
        # Check metric type
        if metric_type not in ["absolute_diff", "absolute_diff_pct"]:
            # Map generic effect size to absolute_diff
            if metric_type == "effect_size":
                metric_type = "absolute_diff"

        # For continuous, usually absolute difference is preferred

        x_vals = np.array(x_values, dtype=float)

        if metric_type == "absolute_diff_pct":
            # For continuous outcomes, 'absolute_diff_pct' is interpreted as a percentage
            # change from the control mean (requires mu_control != 0).
            if abs(self.mu_control) > 1e-9:
                deltas = (x_vals / 100.0) * self.mu_control
            else:
                # Fallback to absolute if baseline is zero.
                deltas = x_vals
        else:
            deltas = x_vals

        return self._evaluate_from_deltas(
            deltas, x_vals, metric_type, self.target_delta, 0.0
        )

    def evaluate_metric_curve(
        self,
        range_min: float,
        range_max: float,
        n_points: int,
        metric_type: str,
    ) -> SimulationCurve:
        """Evaluate OC curve over a range."""
        # Check metric type
        if metric_type not in ["absolute_diff", "absolute_diff_pct"]:
            if metric_type == "effect_size":
                metric_type = "absolute_diff"

        x_vals = np.linspace(range_min, range_max, n_points)

        # Similar logic to evaluate_metric_at
        if metric_type == "absolute_diff_pct":
            if abs(self.mu_control) > 1e-9:
                deltas = (x_vals / 100.0) * self.mu_control
            else:
                deltas = x_vals
        else:
            deltas = x_vals

        return self._evaluate_from_deltas(
            deltas, x_vals, metric_type, self.target_delta, 0.0
        )

    def _evaluate_from_deltas(
        self,
        deltas: np.ndarray,
        x_values: np.ndarray,
        metric_type: str,
        target_val: float,
        null_val: float,
    ) -> SimulationCurve:
        """Internal helper."""
        from scipy.stats import norm

        # Calculate N_fixed for comparison
        efficacy = self.protocol.task.efficacy
        futility = self.protocol.task.futility

        n_fixed = None
        if efficacy and futility and abs(self.target_delta) > 1e-9:
            alpha = efficacy.alpha
            power = futility.power
            z_alpha = norm.ppf(1 - alpha)
            z_beta = norm.ppf(power)

            # I_fixed = (Z_a + Z_b)^2 / theta^2
            i_fixed = ((z_alpha + z_beta) ** 2) / (self.target_delta**2)

            # N_fixed
            if self.arms == 1:
                n_fixed = i_fixed * self.sigma2
            else:
                n_fixed = i_fixed * 4 * self.sigma2

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
