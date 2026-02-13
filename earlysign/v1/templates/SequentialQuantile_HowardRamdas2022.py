from typing import Any, Dict, List, Mapping, Optional, Tuple

import ibis

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import Protocol, SequentialQuantileMethodSpec, TaskSpec
from earlysign.schema.ES3.AVI.Log import SequentialQuantileLookResult
from earlysign.schema.ES3.SequentialQuantile import ArmMetrics, ArmStatus, Scoreboard
from earlysign.v1.framework.entity import SimpleSequentialEntity
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI.engines.sequential_quantile import (
    SequentialQuantileEngine,
)
from earlysign.v1.templates.base import TemplateBase


class SequentialQuantileMetrics(SimpleSequentialEntity[int, ArmMetrics]):
    """
    Sequential Entity for tracking the trajectory of ArmMetrics for an arm.
    """

    def __init__(self, identity: str):
        super().__init__(state_type=ArmMetrics, identity=identity)


class SequentialQuantileScoreboard:
    """
    Projector that builds the latest Scoreboard from individual ArmMetrics trajectories.
    """

    def __init__(self, arm_ids: List[str]):
        self.arm_ids = arm_ids

    def project(self, table: ibis.Expr):
        from earlysign.v1.framework.projector import ProjectionResult

        arms = {}
        all_trace = []
        for aid in self.arm_ids:
            # We use SequentialQuantileMetrics to get the latest for each arm
            entity = SequentialQuantileMetrics(identity=aid)
            res = entity.project(table)

            # Find latest in trajectory
            if res.data:
                # SequentialEntity.project returns List[Tuple[Index, T]]
                # Sorted by project_trajectory
                latest_metrics = res.data[-1][1]
                arms[aid] = ArmStatus(metrics=latest_metrics, is_active=True)
                all_trace.extend(res.trace)
            else:
                # Initial state for missing arms
                arms[aid] = ArmStatus(
                    metrics=ArmMetrics(
                        n=0, ci_lower=0.0, ci_upper=0.0, quantile_estimate=0.0
                    ),
                    is_active=True,
                )

        return ProjectionResult(data=Scoreboard(arms=arms), trace=all_trace)


class HowardRamdas2022Template(TemplateBase[Protocol]):
    """Template for Howard & Ramdas (2022) Sequential Quantile A/B Testing.

    Based on the method described in:
        Howard, S. R., & Ramdas, A. (2022). Sequential estimation of quantiles
        with applications to A/B testing and best-arm identification.
        Bernoulli, 28(3), 1704–1728. https://doi.org/10.3150/21-BEJ1388

    Example:
    >>> import ibis
    >>> import duckdb
    >>> from earlysign.core.ledger import Ledger
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> from earlysign.v1.templates.SequentialQuantile_HowardRamdas2022 import HowardRamdas2022Template
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events_sq"); ledger.ensure()
    >>> template = HowardRamdas2022Template(ledger)
    >>> protocol = template.design(
    ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="A", treatment_arm_name="B"),
    ...     quantile=0.5,
    ...     alpha=0.05
    ... )
    >>> template.set_protocol(protocol)
    >>> t = con.create_table("raw_data_sq", {"arm": ["A", "A", "B", "B"], "val": [1.0, 2.0, 10.0, 11.0]})
    >>> template.update({"A": t.filter(t.arm == "A"), "B": t.filter(t.arm == "B")})
    >>> res = template.report_result()
    >>> print(f"Est: {res['estimated_quantile']:.2f}, CI: [{res['interval_lower']:.2f}, {res['interval_upper']:.2f}]")
    Est: 11.00, CI: [10.00, 11.00]
    >>> print(f"Status: {res['status']}")
    Status: stop_efficacy
    >>> # Note: The intervals are disjoint (A approx [1, 2], B=[10, 11]), so we stop for efficacy.
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: ES3_BASE.ArmStructure,
        quantile: float = 0.5,
        alpha: float = 0.05,
        max_n: Optional[int] = None,
    ) -> Protocol:
        """Design a Sequential Quantile protocol."""
        method = SequentialQuantileMethodSpec(
            quantile=quantile,
            alpha=alpha,
            max_n=max_n,
        )
        task = TaskSpec(arms=arms, response_type="continuous")
        return Protocol(name="Sequential Quantile A/B Test", task=task, method=method)

    def _calculate_order_statistics(
        self, table: ibis.Table, ranks: Tuple[int, int], q: float = 0.5
    ) -> ArmMetrics:
        """
        Calculates the required order statistics from an Ibis table.
        """
        l_rank, u_rank = ranks

        # 1. Get total n
        n_res = table.count().execute()
        n = int(n_res) if n_res is not None else 0

        if n == 0:
            return ArmMetrics(n=0, ci_lower=0.0, ci_upper=0.0, quantile_estimate=0.0)

        # 2. Get values at ranks
        # Ranks are 1-indexed. Ibis order_by + offset is 0-indexed.
        sorted_table = table.order_by("val")

        # We need to execute to get the actual scalars.
        # Handle cases where ranks might be out of range due to rounding
        l_idx = max(0, min(n - 1, l_rank - 1))
        u_idx = max(0, min(n - 1, u_rank - 1))

        ci_lower = sorted_table.limit(1, offset=l_idx).execute().iloc[0]["val"]
        ci_upper = sorted_table.limit(1, offset=u_idx).execute().iloc[0]["val"]

        # Quantile estimate (point estimate)
        q_idx = max(0, min(n - 1, int(n * q)))
        q_est = sorted_table.limit(1, offset=q_idx).execute().iloc[0]["val"]

        return ArmMetrics(
            n=n,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            quantile_estimate=float(q_est),
        )

    def update(self, arms_data: Mapping[str, ibis.Table]) -> None:
        """
        Update the experiment with new raw data tables for each arm.
        """
        # We must ensure the table exists before creating a Session
        self.ledger.ensure()

        with Session(self.ledger) as sess:
            # 1. Read Protocol
            protocol_traced = sess.read(ProtocolProjector(Protocol))
            protocol = protocol_traced.data
            method = protocol.method

            if not isinstance(method, SequentialQuantileMethodSpec):
                raise TypeError("Ledger protocol is not SequentialQuantile")

            # 2. For each arm, calculate and commit metrics
            arms = protocol.task.arms
            if not isinstance(arms, ES3_BASE.TwoArmComparison):
                raise NotImplementedError(
                    f"SequentialQuantile on {type(arms).__name__} is not yet supported in this template. "
                    "Currently, only TwoArmComparison is supported."
                )
            arm_ids = [arms.control_arm_name, arms.treatment_arm_name]

            for arm_id in arm_ids:
                table = arms_data.get(arm_id)
                if table is None:
                    continue

                # a. Determine n
                n_res = table.count().execute()
                n = int(n_res) if n_res is not None else 0

                # b. Get ranks from Engine
                l_rank, u_rank = SequentialQuantileEngine.get_confidence_interval_ranks(
                    n, method
                )

                # c. Calculate values
                metrics = self._calculate_order_statistics(
                    table, (l_rank, u_rank), q=method.quantile
                )

                # d. Commit to Ledger
                sess.commit(metrics, identity=arm_id)

            # 3. Decision Logic
            # Re-read protocol
            protocol = sess.read(ProtocolProjector(Protocol)).data
            arms = protocol.task.arms
            arm_ids = [arms.control_arm_name, arms.treatment_arm_name]

            scoreboard_projector = SequentialQuantileScoreboard(arm_ids)
            scoreboard_traced = sess.read(scoreboard_projector)

            engine = SequentialQuantileEngine(protocol)

            # Commit LookResult
            sess.call_and_commit(
                SequentialQuantileLookResult, engine.run, metrics=scoreboard_traced.data
            )

    def report_result(self) -> Dict[str, Any]:
        """Report final result."""
        self.ledger.ensure()
        with Session(self.ledger) as sess:
            type_name = SequentialQuantileLookResult.__name__
            # Use sess.table which is already filtered correctly
            res_df = sess.table.filter(sess.table.type == type_name).execute()

            if res_df.empty:
                return {"status": "no_data"}

            # Get latest
            latest_row = res_df.iloc[-1]
            payload = latest_row["payload"]
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            return SequentialQuantileLookResult.model_validate(payload).model_dump()
