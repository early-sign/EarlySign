from typing import Any, Optional

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.schema.ES3.Binomial import ArmMetrics
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult
from earlysign.v1.methods.group_sequential.shared.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)


class BinomialGSTEngine:
    """
    Orchestrator for Binomial Group Sequential Testing.

    Responsibilities:
    1. Manages Efficacy and Futility stopping rule engines (via BoundaryCalculator).
    2. Computes Z-statistics from summary data.
    3. Evaluates stopping criteria.
    """

    def __init__(self, protocol: GST.Protocol):
        self.protocol = protocol
        task = protocol.task
        method = protocol.method

        if not method.efficacy:
            raise ValueError("BinomialGSTEngine requires an efficacy stopping rule.")

        # Efficacy Setup
        alpha = float(task.efficacy.alpha) if task.efficacy else 0.05
        self.efficacy_calc = self._setup_calculator(method.efficacy, "efficacy", alpha)

        # Futility Setup
        self.futility_calc = None
        if method.futility:
            beta = 1.0 - float(task.futility.power) if task.futility else 0.1
            self.futility_calc = self._setup_calculator(
                method.futility, "futility", beta
            )

        # Max Sample Size
        self.n_max = 0
        if method.efficacy.schedule.unit == "sample_size":
            pts = method.efficacy.schedule.interim_points
            if pts:
                self.n_max = int(max(pts))

    def _setup_calculator(
        self, rule: GST.StoppingRule, rule_type: str, budget: float
    ) -> Optional[BoundaryCalculator]:
        """Maps ES3 StoppingRule to BoundaryCalculator."""
        boundary = rule.boundary

        if isinstance(boundary, GST.FixedBoundary):
            return None

        if isinstance(boundary, GST.SpendingBoundary):
            params = boundary.spending_function.params or {}
            gamma = params.get("gamma") or params.get("rho")

            if rule_type == "efficacy":
                eff_spec = EfficacySpec(
                    style="alpha_spending",
                    family=boundary.spending_function.type,
                    gamma=float(gamma) if gamma is not None else None,
                )
                fut_spec = FutilitySpec(mode="none")
                spec = BoundaryCalculatorSpec(
                    alpha=budget,
                    tails=1,  # Assuming 1-sided for now as standard GST engine
                    scale="z",
                    efficacy=eff_spec,
                    futility=fut_spec,
                )
            else:
                # Futility mapping
                eff_spec = EfficacySpec(
                    style="alpha_spending", family="obrien_fleming", alpha_levels=[]
                )
                fut_spec = FutilitySpec(
                    mode="beta_spending",
                    family=boundary.spending_function.type,
                    gamma=float(gamma) if gamma is not None else None,
                    beta=budget,
                )
                spec = BoundaryCalculatorSpec(
                    alpha=0.025,  # Dummy
                    tails=1,
                    scale="z",
                    efficacy=eff_spec,
                    futility=fut_spec,
                )
            return BoundaryCalculator(spec)

        return None

    def get_boundary_at_look(
        self, look_index: int, info_time: float, rule_type: str = "efficacy"
    ) -> Optional[float]:
        """
        Public helper to project a boundary for a given look and information time.
        Useful for design and visualization.
        """
        if rule_type == "efficacy":
            assert self.protocol.method.efficacy is not None
            return self._get_boundary(
                self.protocol.method.efficacy,
                self.efficacy_calc,
                "efficacy",
                look_index,
                info_time,
            )
        elif self.protocol.method.futility:
            assert self.protocol.method.futility is not None
            return self._get_boundary(
                self.protocol.method.futility,
                self.futility_calc,
                "futility",
                look_index,
                info_time,
            )
        return None

    def _get_boundary(
        self,
        rule: GST.StoppingRule,
        calc: Optional[BoundaryCalculator],
        rule_type: str,
        look_index: int,
        info_frac: float,
    ) -> Optional[float]:
        boundary_spec = rule.boundary
        if isinstance(boundary_spec, GST.FixedBoundary):
            return float(boundary_spec.value)

        if calc and isinstance(boundary_spec, GST.SpendingBoundary):
            upper, lower, _ = calc.compute_boundary(
                info_time=info_frac, look=look_index + 1
            )
            return upper if rule_type == "efficacy" else lower

        return None

    def run(
        self, summary_c: ArmMetrics, summary_t: ArmMetrics, **kwargs: Any
    ) -> LookResult:
        """
        Computes the test result given current summary statistics.
        """
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
        assert self.protocol.method.efficacy is not None
        schedule = self.protocol.method.efficacy.schedule
        points = schedule.interim_points or []
        look_idx = -1

        for i, pt in enumerate(points):
            if schedule.unit == GST.Unit.SAMPLE_SIZE:
                if cumulative_n >= pt:
                    look_idx = i
            else:  # INFORMATION_FRACTION
                if info_frac >= pt:
                    look_idx = i

        efficacy_boundary = None
        is_efficacy_crossed = False
        futility_boundary = None
        is_futility_crossed = False
        status = DecisionStatus.CONTINUE_

        if look_idx >= 0:
            # Efficacy
            assert self.protocol.method.efficacy is not None
            efficacy_boundary = self._get_boundary(
                self.protocol.method.efficacy,
                self.efficacy_calc,
                "efficacy",
                look_idx,
                info_frac,
            )
            if efficacy_boundary is not None and z_stat > efficacy_boundary:
                is_efficacy_crossed = True
                status = DecisionStatus.STOP_EFFICACY

            # Futility
            if self.protocol.method.futility:
                futility_boundary = self._get_boundary(
                    self.protocol.method.futility,
                    self.futility_calc,
                    "futility",
                    look_idx,
                    info_frac,
                )
                if futility_boundary is not None and z_stat < futility_boundary:
                    is_futility_crossed = True
                    if status == DecisionStatus.CONTINUE_:
                        status = DecisionStatus.STOP_FUTILITY

            # Final Look check
            if look_idx == len(points) - 1:
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
