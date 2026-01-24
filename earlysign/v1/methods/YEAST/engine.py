from typing import Any

from earlysign.schema.ES3.Binomial import ArmMetrics, ArmStatus, Scoreboard
from earlysign.schema.ES3.YEAST import Protocol
from earlysign.schema.ES3.YEAST.Log import (
    Boundary as BoundarySchema,
    DecisionStatus,
    LookResult,
)


class BinomialYEASTEngine:
    """
    Orchestrator for Binomial YEAST execution.

    Calculates the trajectory (difference in successes) and compares it against
    the pre-calculated fixed boundary.
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def run(
        self,
        metrics: Scoreboard,
        boundary: BoundarySchema,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result given current summary statistics.

        Args:
            metrics: Scoreboard containing the aggregated metrics for all arms.
            boundary: The current boundary schema containing the threshold value.

        Returns:
            LookResult containing the trajectory, boundaries, and status.
        """
        # Extract arm names
        arms = self.protocol.task.arms
        if len(arms) < 2:
            raise ValueError("Protocol must define at least 2 arms.")

        control_key = arms[0]
        treatment_key = arms[1]

        default_arm = ArmStatus(
            metrics=ArmMetrics(n=0, successes=0, p_hat=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        cumulative_n = summary_c.n + summary_t.n

        # Trajectory: Difference in successes (S_n)
        trajectory = float(summary_t.successes - summary_c.successes)

        # Use the passed boundary value
        boundary_val = boundary.value
        is_crossed = trajectory > boundary_val

        status = DecisionStatus.CONTINUE_
        if is_crossed:
            status = DecisionStatus.STOP_EFFICACY

        # Check for max N (optional but good practice)
        if hasattr(self.protocol.method, "expected_num_observations"):
            if cumulative_n >= self.protocol.method.expected_num_observations:
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            sample_n=cumulative_n,
            info_frac=0.0,  # Placeholder
            trajectory=trajectory,
            efficacy_boundary=boundary_val,
            is_efficacy_crossed=is_crossed,
            status=status,
        )
