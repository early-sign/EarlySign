from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import ibis

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.builtin.AVI.engine import (
    SequentialQuantileEngine,
)
from earlysign.builtin.AVI.schema import (
    ArmMetrics,
    ArmStatus,
    Protocol,
    ResponseType,
    Scoreboard,
    SequentialQuantileLookResult,
    SequentialQuantileMethodSpec,
    TaskSpec,
)
from earlysign.core.ledger import Ledger
from earlysign.framework.controller import Controller
from earlysign.framework.entity import SimpleSequentialEntity
from earlysign.framework.projector import ProtocolProjector
from earlysign.framework.session import Session


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

    def project(self, table: ibis.Expr) -> Any:
        from earlysign.framework.projector import ProjectionResult

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
                        total=0, ci_lower=0.0, ci_upper=0.0, quantile_estimate=0.0
                    ),
                    is_active=True,
                )

        return ProjectionResult(data=Scoreboard(arms=arms), trace=all_trace)


class HowardRamdas2022Controller(Controller[Protocol]):
    """Controller for Howard & Ramdas (2022) Sequential Quantile A/B Testing.

    Based on the method described in:
        Howard, S. R., & Ramdas, A. (2022). Sequential estimation of quantiles
        with applications to A/B testing and best-arm identification.
        Bernoulli, 28(3), 1704–1728. https://doi.org/10.3150/21-BEJ1388

    Example:
    >>> import ibis
    >>> import duckdb
    >>> from earlysign.core.ledger import Ledger
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> from earlysign.builtin.AVI.controllers.SequentialQuantile_HowardRamdas2022 import HowardRamdas2022Controller
    >>> con = ibis.duckdb.connect(":memory:")
    >>> # Scenario:
    >>> # You are an SRE at Acme Corp monitoring the P99 latency of a search service.
    >>> # You need a guarantee that the P99 latency of the new canary (v2) has not degraded
    >>> # significantly compared to stable (v1).
    >>>
    >>> ledger = Ledger(con, "events_sq"); ledger.ensure()
    >>> controller = HowardRamdas2022Controller(ledger)
    >>> protocol = controller.design(
    ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="v1_stable", treatment_arm_name="v2_canary"),
    ...     quantile=0.99,  # Monitoring P99
    ...     alpha=0.05
    ... )
    >>> controller.set_protocol(protocol)
    >>>
    >>> # Simulating Latency Data (ms)
    >>> # v1 is stable around 150ms, v2 has a regression (spikes to 300ms)
    >>> t = con.create_table("raw_latency", {
    ...     "arm": ["v1_stable", "v1_stable", "v2_canary", "v2_canary"],
    ...     "val": [140.0, 160.0, 250.0, 350.0]
    ... })
    >>> controller.update({"v1_stable": t.filter(t.arm == "v1_stable"), "v2_canary": t.filter(t.arm == "v2_canary")})
    >>> res = controller.report_result()
    >>> print(f"Est P99 (Canary): {res['estimated_quantile']:.2f}, CI: [{res['interval_lower']:.2f}, {res['interval_upper']:.2f}]")
    Est P99 (Canary): 350.00, CI: [250.00, 350.00]
    >>> print(f"Status: {res['status']}")
    Status: stop_detected
    >>> # Note: The intervals are disjoint (A approx [1, 2], B=[10, 11]), so we stop for efficacy (flag raised).
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
        task = TaskSpec(arms=arms, response_type=ResponseType.CONTINUOUS)
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
            return ArmMetrics(
                total=0, ci_lower=0.0, ci_upper=0.0, quantile_estimate=0.0
            )

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
            total=n,
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
                    f"SequentialQuantile on {type(arms).__name__} is not yet supported in this controller. "
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

                from earlysign.builtin.AVI.core import QuantileModel

                # b. Get ranks from Engine
                l_rank, u_rank = QuantileModel.get_confidence_interval_ranks(
                    n, method.quantile, method.alpha
                )

                # c. Calculate values
                metrics = self._calculate_order_statistics(
                    table, (l_rank, u_rank), q=method.quantile
                )

                # d. Commit to Ledger
                sess.commit(metrics, identity=arm_id)

            # 3. Decision Logic
            arms_struct = protocol.task.arms
            if not isinstance(arms_struct, ES3_BASE.TwoArmComparison):
                raise TypeError("Expected TwoArmComparison")
            arm_ids = [arms_struct.control_arm_name, arms_struct.treatment_arm_name]

            scoreboard_projector = SequentialQuantileScoreboard(arm_ids)
            from earlysign.framework.projector import ProjectionResult
            from earlysign.parts.trackers.binomial import Scoreboard

            scoreboard_traced_raw = sess.read(cast(Any, scoreboard_projector))
            scoreboard_traced: ProjectionResult[Scoreboard] = cast(
                ProjectionResult[Scoreboard], scoreboard_traced_raw
            )

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
