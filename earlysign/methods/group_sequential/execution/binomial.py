from typing import Any, Optional

import numpy as np

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST
from earlysign.methods.group_sequential.execution.stopping_policy import (
    SpendingFunctionStoppingPolicy,
    StoppingPolicy,
    StoppingPolicyFactory,
)
from earlysign.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
)
from earlysign.methods.group_sequential.shared.design_utils import (
    get_standardized_drift,
)
from earlysign.schema.ES3.Binomial import ArmMetrics, ArmStatus, Scoreboard
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult, ScheduleTrigger


class BinomialGSTEngine:
    """
    Orchestrator for Binomial Group Sequential Testing.

    Responsibilities:
    1. Computes Z-statistics from summary data.
    2. Evaluates stopping criteria using StoppingPolicySpec.
    3. Returns LookResult with boundary crossings and status.
    """

    _schedule: GST.FixedSchedule | GST.EquidistantSchedule
    _points: list[float]
    n_max: int
    stopping_policy: StoppingPolicy

    def __init__(self, protocol: GST.Protocol):
        self.protocol = protocol
        method = protocol.method
        schedule = method.stopping_policy.schedule

        # 1. Resolve stopping policy logic
        self.stopping_policy = StoppingPolicyFactory.build_from_spec(
            method.stopping_policy
        )

        # Schedule
        self._schedule = schedule
        schedule_inner = self._schedule
        self._points = []
        if isinstance(schedule_inner, GST.FixedSchedule):
            self._points = schedule_inner.analyses
        elif isinstance(schedule_inner, GST.EquidistantSchedule):
            k = schedule_inner.n_looks
            self._points = list(np.linspace(1 / k, 1.0, k))

        # Max Sample Size
        self.n_max = 0
        timer = method.stopping_policy.timer
        if isinstance(timer, GST.SampleSizeTimer):
            if isinstance(timer.max_sample_size, dict):
                self.n_max = sum(timer.max_sample_size.values())
            else:
                self.n_max = timer.max_sample_size

        # Initialize Canonical Model
        self.canonical_model = CanonicalJointModel.from_spec(protocol)
        self._efficacy_boundaries: Optional[np.ndarray] = None
        self._futility_boundaries: Optional[np.ndarray] = None

    @property
    def info_times(self) -> np.ndarray:
        """Required by BoundarySolver protocol."""
        return np.asarray(self._points, dtype=float)

    @property
    def tails(self) -> int:
        """Required by BoundarySolver protocol."""
        method = self.protocol.method
        strategy = method.stopping_policy.strategy
        if hasattr(strategy, "sided"):
            return 1 if strategy.sided == GST.Sided.ONE else 2
        return 1  # Default to 1-sided if not specified

    def find_critical_value(
        self, shape: np.ndarray, alpha: float, tails: Optional[int] = None
    ) -> float:
        """Required by BoundarySolver protocol. Proxies to the canonical model."""
        return self.canonical_model.find_critical_value(
            shape=shape, alpha=alpha, tails=tails or self.tails
        )

    @property
    def efficacy_boundaries(self) -> Optional[np.ndarray]:
        """Lazy access to pre-calculated efficacy boundaries."""
        if self._efficacy_boundaries is None:
            self.solve_all_boundaries()
        return self._efficacy_boundaries

    @property
    def futility_boundaries(self) -> Optional[np.ndarray]:
        """Lazy access to pre-calculated futility boundaries."""
        if self._futility_boundaries is None:
            self.solve_all_boundaries()
        return self._futility_boundaries

    def solve_all_boundaries(self) -> None:
        """
        Triggers solving for all boundaries at once.
        Useful for fixed-shape designs or to seed the cache.
        """
        # Spending function designs are typically solved step-by-step in run()
        if isinstance(self.stopping_policy, SpendingFunctionStoppingPolicy):
            return

        self._efficacy_boundaries, self._futility_boundaries = (
            self.stopping_policy.solve(self)
        )

    def get_boundary_at_look(
        self, look_index: int, info_time: float, rule_type: str = "efficacy"
    ) -> Optional[float]:
        """
        Public helper to project a boundary for a given look and information time.
        Useful for design and visualization.

        Example:
            >>> import earlysign.schema.ES3.Base as ES3_BASE
            >>> import earlysign.schema.ES3.GST as GST
            >>> from earlysign.methods.group_sequential.execution.binomial import BinomialGSTEngine
            >>> protocol = GST.Protocol(
            ...     name="Example Trial",
            ...     task=GST.TaskSpec(
            ...         kind="group_sequential",
            ...         arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
            ...         response_type=GST.ResponseType.BINARY,
            ...         efficacy=GST.EfficacyRequirement(alpha=0.05),
            ...         hypotheses=GST.HypothesisSpec(
            ...             h_null_description="p_t <= p_c", h_alt_description="p_t > p_c",
            ...             test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
            ...             target_effect=GST.BinaryEffectSize(proportions={"control": 0.2, "treatment": 0.3})
            ...         )
            ...     ),
            ...     method=GST.MethodSpec(
            ...         kind="group_sequential",
            ...         stopping_policy=GST.StoppingPolicySpec(
            ...             statistic=GST.TwoArmBinomialZ(variance_estimation=GST.VarianceEstimation.POOLED),
            ...             strategy=GST.AlphaSpendingStrategy(
            ...                 spending_fn=GST.SpendingFunction(family="obrien_fleming"),
            ...                 budget=0.05,
            ...                 sided=GST.Sided.ONE,
            ...                 statistical_model=GST.CanonicalGaussianModel(),
            ...             ),
            ...             timer=GST.SampleSizeTimer(
            ...                 unit=GST.Unit.INDIVIDUALS,
            ...                 max_sample_size={"control": 50, "treatment": 50}
            ...             ),
            ...             schedule=GST.FixedSchedule(analyses=[0.5, 1.0])
            ...         ),
            ...     )
            ... )
            >>> engine = BinomialGSTEngine(protocol=protocol)
            >>> # Projection of boundary at 50% info time
            >>> engine.get_boundary_at_look(0, 0.5)
            2.3261743106419144
        """
        # Note:
        # - Index-based designs (OBF, Pocock, etc.): Anchored to look_index.
        #   Assumes equidistant looks as per standard software defaults.
        # - Time-based designs (Spending, Whitehead): Follow actual info_time.
        #   Maintains statistical integrity if analysis timing varies.

        if look_index < 0:
            return None

        return self.stopping_policy.get_boundary(
            model=self,
            look_index=look_index,
            info_time=info_time,
            rule_type=rule_type,
        )

    def run(
        self,
        metrics: Scoreboard,
        history: list[tuple[int, LookResult]],
        trigger: Optional[ScheduleTrigger] = None,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result given current summary statistics and trajectory history.

        Zero-State Principle: This method derives the decision state solely from the
        provided metrics, history, and protocol definition.

        Args:
            metrics: Scoreboard containing the aggregated metrics for all arms.
            history: Trajectory of previous LookResult objects from the ledger.
            trigger: The trigger that prompted this analysis (contains look index).
            **kwargs: Additional keyword arguments.

        Returns:
            LookResult containing the test statistic, boundaries, and decision status.
        """
        # Extract arm names from protocol's explicit roles
        arms = self.protocol.task.arms
        if not isinstance(arms, ES3_BASE.TwoArmComparison):
            raise ValueError(
                f"BinomialGSTEngine requires a TwoArmComparison arm structure, but got {type(arms).__name__}."
            )

        control_key = arms.control_arm_name
        treatment_key = arms.treatment_arm_name

        # Default empty metrics if arm not present
        default_arm = ArmStatus(
            metrics=ArmMetrics(total=0, successes=0, p_hat=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        n_c, n_t = summary_c.total, summary_t.total
        cumulative_n = n_c + n_t

        if self.n_max > 0:
            info_frac = min(cumulative_n / self.n_max, 1.0)
        else:
            info_frac = 0.0

        # 1. Calculate Z-statistic
        z_stat = 0.0
        if n_c >= 2 and n_t >= 2:
            p_pool = (summary_c.successes + summary_t.successes) / cumulative_n
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            if se > 0:
                z_stat = (summary_t.p_hat - summary_c.p_hat) / se

        # 1.1 Support Weighted Z-Ratio (Cui-Hung-Wang) if adaptation occurred
        snapshot = self.protocol.method.adaptation_snapshot

        # Check if weighting is enabled in the design spec
        use_weighted = True
        if self.protocol.method.adaptation and hasattr(
            self.protocol.method.adaptation, "use_weighted_statistic"
        ):
            use_weighted = self.protocol.method.adaptation.use_weighted_statistic

        if (
            snapshot
            and use_weighted
            and cumulative_n > snapshot.info_frac * snapshot.original_max_sample_size
        ):
            t = snapshot.info_frac
            n_look_t = t * snapshot.original_max_sample_size
            z_t = snapshot.z_t

            if cumulative_n > n_look_t:
                # Z_rem = (Z_total * sqrt(n_total) - Z_t * sqrt(n_t)) / sqrt(n_total - n_t)
                # Weighted Z = sqrt(t)*Z_t + sqrt(1-t)*Z_rem
                z_rem = (
                    z_stat * np.sqrt(cumulative_n) - z_t * np.sqrt(n_look_t)
                ) / np.sqrt(cumulative_n - n_look_t)
                z_weighted = np.sqrt(t) * z_t + np.sqrt(1 - t) * z_rem
                z_stat = z_weighted

        # 2. Determine Look and Boundaries
        look_num = trigger.index if trigger else None
        look_idx = (look_num - 1) if look_num is not None else -1

        efficacy_boundary = None
        futility_boundary = None
        alpha_spent = None
        beta_spent = None
        is_efficacy_crossed = False
        is_futility_crossed = False
        status = DecisionStatus.CONTINUE_

        if look_num is not None:
            # 3. Resolve boundaries
            (
                efficacy_boundary,
                futility_boundary,
                alpha_spent,
                beta_spent,
            ) = self._resolve_current_boundaries(look_idx, info_frac, history)

            # 4. Evaluate Stopping
            if efficacy_boundary is not None and z_stat > efficacy_boundary:
                is_efficacy_crossed = True
                status = DecisionStatus.STOP_EFFICACY

            if futility_boundary is not None and z_stat < futility_boundary:
                is_futility_crossed = True
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_FUTILITY

            # Final Look check
            if look_idx == len(self._points) - 1:
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            look=look_num,
            trigger=trigger,
            sample_n=int(cumulative_n),
            info_frac=info_frac,
            z_stat=float(z_stat),
            efficacy_boundary=efficacy_boundary,
            is_efficacy_crossed=is_efficacy_crossed,
            futility_boundary=futility_boundary,
            is_futility_crossed=is_futility_crossed,
            alpha_spent=alpha_spent,
            beta_spent=beta_spent,
            status=status,
        )

    def _resolve_current_boundaries(
        self,
        look_idx: int,
        info_frac: float,
        history: list[tuple[int, LookResult]],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Resolves efficacy and futility boundaries for the current look."""
        efficacy_boundary = None
        futility_boundary = None
        alpha_spent = None
        beta_spent = None

        if isinstance(self.stopping_policy, SpendingFunctionStoppingPolicy):
            # Extract realized history
            prev_times = [float(res.info_frac) for idx, res in history]
            prev_eff = [
                (
                    float(res.efficacy_boundary)
                    if res.efficacy_boundary is not None
                    else np.inf
                )
                for idx, res in history
            ]
            prev_fut = [
                (
                    float(res.futility_boundary)
                    if res.futility_boundary is not None
                    else -np.inf
                )
                for idx, res in history
            ]

            # Efficacy
            if self.stopping_policy.efficacy_spending:
                alpha_spent_arr = self.stopping_policy.efficacy_spending.cumulative(
                    np.array([info_frac])
                )
                alpha_spent = float(alpha_spent_arr[0])
                efficacy_boundary = self.canonical_model.solve_next_boundary(
                    previous_times=prev_times,
                    current_t=info_frac,
                    target_cumulative_prob=alpha_spent,
                    previous_efficacy=prev_eff,
                    previous_futility=prev_fut,
                    rule_type="efficacy",
                )

            # Futility
            if self.stopping_policy.futility_spending:
                beta_spent_arr = self.stopping_policy.futility_spending.cumulative(
                    np.array([info_frac])
                )
                beta_spent = float(beta_spent_arr[0])
                # For beta spending, the standardized drift (H1 effect) must be known.
                try:
                    drift = get_standardized_drift(self.protocol)
                except ValueError as e:
                    raise ValueError(
                        f"Cannot compute futility boundary: {str(e)}"
                    ) from e
                futility_boundary = self.canonical_model.solve_next_boundary(
                    previous_times=prev_times,
                    current_t=info_frac,
                    target_cumulative_prob=beta_spent,
                    previous_efficacy=prev_eff,
                    previous_futility=prev_fut,
                    rule_type="futility",
                    drift=drift,
                )
        else:
            # Use pre-calculated or shape-based boundaries
            efficacy_boundary = self.get_boundary_at_look(
                look_idx, info_frac, "efficacy"
            )
            futility_boundary = self.get_boundary_at_look(
                look_idx, info_frac, "futility"
            )

        return efficacy_boundary, futility_boundary, alpha_spent, beta_spent
