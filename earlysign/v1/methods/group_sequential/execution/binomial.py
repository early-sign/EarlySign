from typing import Any, Optional

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.schema.ES3.Binomial import ArmMetrics, ArmStatus, Scoreboard
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult
from earlysign.v1.methods.group_sequential.execution.stopping_policy import (
    StoppingPolicy,
    StoppingPolicyFactory,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
)


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
            self.n_max = timer.max_sample_size

        # Initialize Canonical Model and pre-calculate boundaries
        self.canonical_model = CanonicalJointModel.from_spec(protocol)
        self.efficacy_boundaries, self.futility_boundaries = (
            self.canonical_model.solve_boundaries()
        )

    def get_boundary_at_look(
        self, look_index: int, info_time: float, rule_type: str = "efficacy"
    ) -> Optional[float]:
        """
        Public helper to project a boundary for a given look and information time.
        Useful for design and visualization.
        """
        # Note: This implementation assumes look_index aligns with the model's schedule.
        # If info_time deviates significantly from the schedule, this might be inaccurate
        # for spending designs that depend on exact info time.
        # However, for the Engine execution, we typically look up by index.

        if look_index < 0:
            return None

        if rule_type == "efficacy":
            if self.efficacy_boundaries is not None and look_index < len(
                self.efficacy_boundaries
            ):
                return float(self.efficacy_boundaries[look_index])

        elif rule_type == "futility":
            if self.futility_boundaries is not None and look_index < len(
                self.futility_boundaries
            ):
                return float(self.futility_boundaries[look_index])

        return None

    def run(self, metrics: Scoreboard, **kwargs: Any) -> LookResult:
        """
        Computes the test result given current summary statistics.

        The arm names are retrieved from the protocol's task specification.
        The first arm in protocol.task.arms is treated as control,
        and the second arm as treatment.

        Args:
            metrics: Scoreboard containing the aggregated metrics for all arms.
            **kwargs: Additional keyword arguments (unused, for interface compatibility).

        Returns:
            LookResult containing the test statistic, boundaries, crossing status,
            and decision (CONTINUE, STOP_EFFICACY, STOP_FUTILITY, STOP_PLAN_END_REACHED).
        """
        # Extract arm names from protocol
        arms = self.protocol.task.arms
        if len(arms) < 2:
            raise ValueError(
                "Protocol must define at least 2 arms (control and treatment)."
            )
        control_key = arms[0]
        treatment_key = arms[1]

        # Default empty metrics if arm not present
        default_arm = ArmStatus(
            metrics=ArmMetrics(n=0, successes=0, p_hat=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        n_c, n_t = summary_c.n, summary_t.n
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

        # 2. Determine Look
        look_idx = -1
        for i, pt in enumerate(self._points):
            if info_frac >= pt:
                look_idx = i

        efficacy_boundary = None
        is_efficacy_crossed = False
        futility_boundary = None
        is_futility_crossed = False
        status = DecisionStatus.CONTINUE_

        if look_idx >= 0:
            # Efficacy boundary
            efficacy_boundary = self.get_boundary_at_look(
                look_idx, info_frac, "efficacy"
            )
            if efficacy_boundary is not None and z_stat > efficacy_boundary:
                is_efficacy_crossed = True
                status = DecisionStatus.STOP_EFFICACY

            # Futility boundary
            futility_boundary = self.get_boundary_at_look(
                look_idx, info_frac, "futility"
            )
            if futility_boundary is not None and z_stat < futility_boundary:
                is_futility_crossed = True
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_FUTILITY

            # Final Look check
            if look_idx == len(self._points) - 1:
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            look=look_idx + 1 if look_idx >= 0 else None,
            sample_n=int(cumulative_n),
            info_frac=info_frac,
            z_stat=float(z_stat),
            efficacy_boundary=efficacy_boundary,
            is_efficacy_crossed=is_efficacy_crossed,
            futility_boundary=futility_boundary,
            is_futility_crossed=is_futility_crossed,
            status=status,
        )
