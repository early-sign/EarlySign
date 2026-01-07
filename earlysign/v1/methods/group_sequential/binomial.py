from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel

# Import protocol types for type hinting if needed (avoid circular if possible)
import earlysign.schema.ES3.GST as GST
from earlysign.schema.ES3.GST import DecisionStatus
from earlysign.v1.methods.binomial import BinomialSummary
from earlysign.v1.methods.group_sequential.engine import GSTStoppingRuleEngine


class BinomialTestResult(BaseModel):
    """Result of a sequential Binomial Test evaluation."""

    look: Optional[int]
    info_frac: float
    z_stat: float
    # Efficacy
    efficacy_boundary: Optional[float]
    is_efficacy_crossed: bool
    # Futility
    futility_boundary: Optional[float]
    is_futility_crossed: bool

    status: Union[
        DecisionStatus, str
    ]  # "CONTINUE", "STOP_EFFICACY", "STOP_FUTILITY", "STOP_PLAN_END_REACHED"


class BinomialGSTEngine:
    """
    Orchestrator for Binomial Group Sequential Testing.

    Responsibilities:
    1. Manages Efficacy and Futility stopping rule engines.
    2. Computes Z-statistics from summary data.
    3. Evaluates stopping criteria.
    """

    def __init__(self, protocol: GST.Protocol):  # Takes generic or specific protocol
        self.protocol = protocol

        # Extract budget (alpha/beta) from TaskSpec
        # Using getattr to be safe with different TaskSpecs or assume standard structure
        task = protocol.task
        method = protocol.method

        if not method.efficacy:
            raise ValueError("BinomialGSTEngine requires an efficacy stopping rule.")

        # Efficacy Engine
        alpha = task.efficacy.alpha if task.efficacy else 0.05
        self.efficacy_engine = GSTStoppingRuleEngine(
            rule=method.efficacy,
            rule_type="efficacy",
            total_budget=alpha,
            side=1,
        )

        # Futility Engine
        self.futility_engine = None
        if method.futility:
            # GSTStoppingRuleEngine for Futility takes `total_budget` which for beta-spending is `beta`.
            # But FutilityRequirement has `power`. Beta = 1 - power.
            beta_budget = 1.0 - task.futility.power if task.futility else 0.1
            self.futility_engine = GSTStoppingRuleEngine(
                rule=method.futility,
                rule_type="futility",
                total_budget=beta_budget,
                side=1,
            )

        # Determine Max Sample Size for Info Frac calculation
        self.n_max = self.efficacy_engine.get_max_sample_size()

    def run(
        self, summary_c: "BinomialSummary", summary_t: "BinomialSummary", **kwargs: Any
    ) -> BinomialTestResult:
        """
        Computes the test result given current summary statistics.
        """
        n_c, n_t = summary_c.n, summary_t.n
        cumulative_n = n_c + n_t

        if self.n_max > 0:
            info_frac = min(cumulative_n / self.n_max, 1.0)
        else:
            info_frac = 0.0

        # 2. Calculate Z-statistic
        z_stat = 0.0
        if n_c >= 2 and n_t >= 2:
            p_pool = (summary_c.successes + summary_t.successes) / cumulative_n
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            if se > 0:
                z_stat = (summary_t.p_hat - summary_c.p_hat) / se

        # 3. Determine Look

        points = self.efficacy_engine.rule.schedule.interim_points or []
        look_idx = -1

        # Finding the *latest* look passed.
        # We take the highest look index that we have reached.
        for i, pt in enumerate(points):
            if cumulative_n >= pt:
                look_idx = i

        efficacy_boundary = None
        is_efficacy_crossed = False
        futility_boundary = None
        is_futility_crossed = False
        status = DecisionStatus.CONTINUE

        if look_idx >= 0:
            # Get Boundaries
            efficacy_boundary = self.efficacy_engine.get_boundary_at_look(
                look_idx, info_frac
            )

            if efficacy_boundary is not None and z_stat > efficacy_boundary:
                is_efficacy_crossed = True
                status = DecisionStatus.STOP_EFFICACY

            if self.futility_engine:
                futility_boundary = self.futility_engine.get_boundary_at_look(
                    look_idx, info_frac
                )
                if futility_boundary is not None and z_stat < futility_boundary:
                    is_futility_crossed = True
                    if status == DecisionStatus.CONTINUE:
                        status = DecisionStatus.STOP_FUTILITY

            # Check Final Look
            if look_idx == len(points) - 1:
                if status == DecisionStatus.CONTINUE:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return BinomialTestResult(
            look=look_idx + 1 if look_idx >= 0 else None,
            info_frac=info_frac,
            z_stat=float(z_stat),
            efficacy_boundary=efficacy_boundary,
            is_efficacy_crossed=is_efficacy_crossed,
            futility_boundary=futility_boundary,
            is_futility_crossed=is_futility_crossed,
            status=status,
        )
