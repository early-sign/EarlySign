from dataclasses import dataclass
from typing import Any, Dict, cast

import ibis
import pandas as pd
from ibis import BaseBackend
from matplotlib.figure import Figure

from earlysign.core.ledger import Ledger
from earlysign.framework import templates as tpl
from earlysign.reporting.group_sequential import plot_design_boundaries
from earlysign.stats.common.group_sequential.essentials.boundaries import (
    resolve_boundary_from_design,
)
from earlysign.stats.common.group_sequential.operators.boundary_op import (
    BoundaryFromDesign,
)
from earlysign.stats.common.group_sequential.operators.info_op import (
    InformationTime,
)
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialDecisionSignalRecord,
    GroupSequentialDesignRecord,
)
from earlysign.stats.schemes.two_proportions.group_sequential import (
    GSDecisionFromWaldZ,
)
from earlysign.stats.schemes.two_proportions.operators import (
    BinomialCountsSnapshot,
    WaldZStatistic,
)
from earlysign.stats.schemes.two_proportions.records import (
    BinomialCountsRecord,
    BinomialCountsSnapshotRecord,
    WaldZStatisticRecord,
)


@dataclass
class State:
    stop_recommended: bool = False


class BinomialABTest(tpl.TemplateBase):
    """
    Example:
    >>> import ibis
    >>> BinomialABTest(ibis.connect("duckdb://:memory:"), "my_exp")
    <earlysign.api.ab_tests.BinomialABTest object at 0x...>
    >>> BinomialABTest("duckdb://:memory:", "my_exp")
    <earlysign.api.ab_tests.BinomialABTest object at 0x...>
    """

    def __init__(
        self,
        connector: BaseBackend | str,
        experiment_id: str,
        table_name: str | None = None,
    ) -> None:
        """Initialize BinomialABTest.

        Parameters
        ----------
        connector : BaseBackend | str
            Database connector or connection string
        experiment_id : str
            Unique identifier for this experiment
        table_name : str | None
            Custom table name for ledger (defaults to experiment_id)

        Examples
        --------
        >>> import ibis
        >>> test = BinomialABTest(ibis.connect("duckdb://:memory:"), "exp1")
        >>> test.experiment_id
        'exp1'
        """
        if isinstance(connector, str):
            self.connector = ibis.connect(connector)
        elif isinstance(connector, BaseBackend):
            self.connector = connector
        self.experiment_id = experiment_id

        ## Use experiment_id if table_name is not specified
        self.ledger = Ledger(
            self.connector, table_name if table_name is not None else experiment_id
        ).bind(experiment_id=self.experiment_id)
        self.ledger.ensure()

    def set_design(self, payload: Dict[str, Any]) -> None:
        """Set the group sequential design.

        Parameters
        ----------
        payload : Dict[str, Any]
            Design configuration including:
            - 'planned_max_n': Maximum sample size
            - 'planned_info_times': Planned information times for analyses (optional)
            - Other design parameters (alpha, tails, scale, efficacy, futility)

        Examples
        --------
        >>> import ibis
        >>> test = BinomialABTest(ibis.connect("duckdb://:memory:"), "exp1")
        >>> design = {
        ...     "alpha": 0.05, "tails": 2, "scale": "z",
        ...     "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
        ...     "futility": {"mode": "symmetric"},
        ...     "planned_max_n": 1000,
        ...     "planned_info_times": [0.33, 0.67, 1.0]
        ... }
        >>> test.set_design(design)
        """
        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        design.insert(payload)

    def update(self, payload: Dict[str, Any]) -> None:
        """Update experiment with new observations.

        Records incremental observations and automatically triggers analysis
        (boundary computation and decision) when current information time
        reaches or exceeds a planned look that has not yet been executed.

        Parameters
        ----------
        payload : Dict[str, Any]
            Incremental counts: {"nA": int, "mA": int, "nB": int, "mB": int}

        Returns
        -------
        None
            This method performs ledger writes (observations, statistics,
            information time, and decisions when triggered) and does not
            return a value. Query the ledger for status after calling.

        Notes
        -----
        Analysis is triggered when:
        1. current_info_time >= planned_info_time[i]
        2. Look i has not been executed yet
        3. i is the smallest index satisfying conditions 1 and 2

        This means you can update with arbitrary batch sizes, and the system
        will automatically execute the appropriate analysis when each planned
        information time is reached.

        Examples
        --------
        >>> import ibis
        >>> test = BinomialABTest(ibis.connect("duckdb://:memory:"), "exp1")
        >>> design = {"alpha": 0.05, "tails": 2, "scale": "z",
        ...           "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
        ...           "futility": {"mode": "symmetric"}, "planned_max_n": 1000,
        ...           "planned_info_times": [0.5, 1.0]}
        >>> test.set_design(design)
        >>> # First update doesn't trigger (info_time < 0.5)
        >>> test.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
        >>> # Second update may trigger first look (info_time >= 0.5)
        >>> test.update({"nA": 150, "mA": 15, "nB": 150, "mB": 18})
        """
        ## Record incremental observation (delta)
        obs = BinomialCountsRecord("observation").attach(self.ledger)
        obs.insert(**payload)

        ## Compute cumulative snapshot
        snapshot_op = BinomialCountsSnapshot(self.ledger, obs=obs, out_id="snapshot")
        snapshot_op.run()
        snapshot_record = snapshot_op.outputs.snapshot

        ## Compute statistic (using snapshot)
        stat_op = WaldZStatistic(
            self.ledger, cum_counts=snapshot_record, pooled=True, out_id="statistic"
        )
        stat_op.run()
        stat_record = stat_op.outputs.wald

        ## Read design info
        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        design_latest = design.latest().execute()
        planned_max_n = int(design_latest["planned_max_n"].iloc[0])
        planned_info_times: list[float] = design_latest["planned_info_times"].iloc[0]

        ## Compute information time
        info_op = InformationTime(
            self.ledger,
            out_id="info_time",
            cum_counts=snapshot_record,
            planned_max_n=planned_max_n,
        )
        info_op.run()
        info_record = info_op.outputs.info

        # Get current info time
        current_info_time = float(
            cast(Any, info_record.latest()["info_time"].execute().iloc[0])
        )

        # Use latest decision's timestamp to determine if a new look is due
        decision_record = GroupSequentialDecisionSignalRecord("decision").attach(
            self.ledger
        )
        latest_decision_df = decision_record.latest(explode=True).execute()

        # Get the last info_time before or at the last decision (if any)
        if len(latest_decision_df) > 0:
            last_decision_ts = latest_decision_df["ts"].iloc[0]
            info_before_decision = info_record.latest_before(
                last_decision_ts, include_ts=True, explode=True
            ).execute()
            last_info_time_before_decision = (
                float(info_before_decision["info_time"].iloc[0])
                if len(info_before_decision) > 0
                else 0.0
            )
        else:
            last_info_time_before_decision = 0.0

        # Trigger only if any planned look is newly due since last decision
        if not any(
            last_info_time_before_decision < planned_time <= current_info_time
            for planned_time in planned_info_times
        ):
            return

        ## Compute boundary
        boundary_op = BoundaryFromDesign(
            self.ledger, design=design, info=info_record, out_id="boundary"
        )
        boundary_op.run()
        boundary_record = boundary_op.outputs.boundary

        ## Record the decision
        decision_op = GSDecisionFromWaldZ(
            self.ledger,
            wald=stat_record,
            boundary=boundary_record,
            out_id="decision",
        )
        decision_op.run()

    def status(self) -> State:
        decision_record = GroupSequentialDecisionSignalRecord("decision").attach(
            self.ledger
        )
        latest_decision = decision_record.latest().execute()
        if len(latest_decision) == 0:
            return State(stop_recommended=False)
        if "stop" in latest_decision["signal"].iloc[0]:
            return State(stop_recommended=True)
        else:
            return State(stop_recommended=False)

    def plot_design(self, n_points: int = 50) -> Figure:
        """Plot the group sequential design boundaries.

        Parameters
        ----------
        n_points : int, optional
            Number of information time points to evaluate boundaries at, by default 50

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        # Read design from ledger
        design_record = GroupSequentialDesignRecord("design").attach(self.ledger)
        design = design_record.latest(explode=False).execute()
        if len(design) == 0:
            raise ValueError("No design found in ledger. Call set_design() first.")

        # Delegate to reporting module
        return plot_design_boundaries(
            design_payload=design.iloc[0]["payload"],
            resolve_boundary_func=resolve_boundary_from_design,
            n_points=n_points,
        )
