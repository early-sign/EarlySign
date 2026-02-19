from typing import Any, List, Optional, Tuple

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.methods.group_sequential.adapters.protocol import (
    get_standardized_drift,
)
from earlysign.methods.group_sequential.core.model import (
    CanonicalJointModel,
)
from earlysign.methods.group_sequential.core.policy import (
    SpendingFunctionStoppingPolicy,
    StoppingPolicy,
    StoppingPolicyFactory,
)
from earlysign.methods.group_sequential.engine.calculators import (
    ZStatisticCalculatorFactory,
)
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult, ScheduleTrigger


class GroupSequentialEngine:
    """
    Architecture-Neutral Orchestrator for Group Sequential Testing.

    Responsibilities:
    1. Orchestrates sequential state (milestones, boundaries, spending) on the universal Z-scale.
    2. Delegates domain-specific statistical calculations (e.g. Binomial Z) to ZStatisticCalculator.
    3. Evaluates stopping criteria for one or more treatment arms (MAMS-ready).
    4. Returns LookResult with explicit handling of missing data (None).
    """

    _schedule: GST.FixedSchedule | GST.EquidistantSchedule
    _points: list[float]
    n_max: int
    stopping_policy: StoppingPolicy

    def __init__(self, protocol: GST.Protocol, rng_seed: Optional[int] = None):
        self.protocol = protocol
        method = protocol.method
        schedule = method.stopping_policy.schedule

        # 1. Resolve stopping policy logic
        self.stopping_policy = StoppingPolicyFactory.build_from_spec(
            method.stopping_policy
        )

        # 2. Build Statistic Calculator
        self.calculator = ZStatisticCalculatorFactory.build(protocol)

        # 3. Resolve Schedule
        self._schedule = schedule
        schedule_inner = self._schedule
        self._points = []
        if isinstance(schedule_inner, GST.FixedSchedule):
            self._points = schedule_inner.analyses
        elif isinstance(schedule_inner, GST.EquidistantSchedule):
            k = schedule_inner.n_looks
            self._points = list(np.linspace(1 / k, 1.0, k))
        else:
            raise ValueError(f"Unsupported schedule type: {type(schedule_inner)}")

        # Max Sample Size
        self.n_max = 0
        timer = method.stopping_policy.timer
        if isinstance(timer, GST.SampleSizeTimer):
            if isinstance(timer.max_sample_size, dict):
                self.n_max = sum(timer.max_sample_size.values())
            else:
                self.n_max = timer.max_sample_size

        # Initialize Canonical Model
        self.canonical_model = CanonicalJointModel.from_spec(
            protocol, rng_seed=rng_seed
        )
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
        return 1

    def find_critical_value(
        self, shape: np.ndarray, alpha: float, tails: Optional[int] = None
    ) -> float:
        """Required by BoundarySolver protocol. Proxies to the canonical model."""
        return self.canonical_model.find_critical_value(
            shape=shape, alpha=alpha, tails=tails or self.tails
        )

    @property
    def efficacy_boundaries(self) -> Optional[np.ndarray]:
        if self._efficacy_boundaries is None:
            self.solve_all_boundaries()
        return self._efficacy_boundaries

    @property
    def futility_boundaries(self) -> Optional[np.ndarray]:
        if self._futility_boundaries is None:
            self.solve_all_boundaries()
        return self._futility_boundaries

    def solve_all_boundaries(self) -> None:
        if isinstance(self.stopping_policy, SpendingFunctionStoppingPolicy):
            return
        self._efficacy_boundaries, self._futility_boundaries = (
            self.stopping_policy.solve(self)
        )

    def get_boundary_at_look(
        self, look_index: int, info_time: float, rule_type: str = "efficacy"
    ) -> Optional[float]:
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
        metrics: Any,
        history: List[Tuple[int, LookResult]],
        trigger: Optional[ScheduleTrigger] = None,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result using the strategy-based calculator and sequential engine.
        """
        # 1. Calculate Z-statistics (Vectorized/Mapped)
        # Returns Dict[arm_name, z_score] or None if critical data missing
        z_scores = self.calculator.calculate(metrics, self.protocol)

        # Determine total sample size
        # We assume total_n is readily available in Scoreboard (aggregated across arms if needed)
        # For Consistency, we calculate it here.
        cumulative_n = sum(arm.metrics.total for arm in metrics.arms.values())

        if self.n_max > 0:
            info_frac = min(cumulative_n / self.n_max, 1.0)
        else:
            info_frac = 0.0

        # If data is missing for required arms, return a CONTINUE result without Z-stat.
        # This prevents STOP decisions based on silent 0.0 defaults.
        if z_scores is None:
            return LookResult(
                look=trigger.index if trigger else None,
                trigger=trigger,
                sample_n=int(cumulative_n),
                info_frac=info_frac,
                z_stat=0.0,  # Schema requires float
                z_stats=None,
                status=DecisionStatus.CONTINUE_,
                is_efficacy_crossed=False,
                is_futility_crossed=False,
            )

        # 2. Determine Look and Boundaries
        look_num = trigger.index if trigger else None
        look_idx = (look_num - 1) if look_num is not None else -1

        efficacy_boundary = None
        futility_boundary = None
        alpha_spent = None
        beta_spent = None

        if look_num is not None:
            (
                efficacy_boundary,
                futility_boundary,
                alpha_spent,
                beta_spent,
            ) = self._resolve_current_boundaries(look_idx, info_frac, history)

        # 3. Evaluate Stopping Logic (MAMS-aware)
        # Representative Z-stat is max across all arms for efficacy
        representative_z = float(max(z_scores.values())) if z_scores else 0.0

        is_efficacy_crossed = False
        is_futility_crossed = False
        status = DecisionStatus.CONTINUE_

        if look_num is not None:
            # Efficacy: STOP if ANY arm crosses boundary
            if efficacy_boundary is not None:
                is_efficacy_crossed = any(
                    z > efficacy_boundary for z in z_scores.values()
                )
                if is_efficacy_crossed:
                    status = DecisionStatus.STOP_EFFICACY

            # Futility: STOP if ALL arms cross futility boundary
            if futility_boundary is not None and status == DecisionStatus.CONTINUE_:
                is_futility_crossed = all(
                    z < futility_boundary for z in z_scores.values()
                )
                if is_futility_crossed:
                    status = DecisionStatus.STOP_FUTILITY

            # Plan End Reached
            if look_idx == len(self._points) - 1:
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            look=look_num,
            trigger=trigger,
            sample_n=int(cumulative_n),
            info_frac=info_frac,
            z_stat=representative_z,
            z_stats=z_scores,
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
        history: List[Tuple[int, LookResult]],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
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


# Alias for backward compatibility during transition if needed
GroupSequentialEngine = GroupSequentialEngine
