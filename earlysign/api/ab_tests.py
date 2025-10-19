from dataclasses import dataclass
from typing import Any, Dict

import ibis
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
    GroupSequentialBoundaryRecord,
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
)


@dataclass
class State:
    stop_recommended: bool = False


class BinomialABTest(tpl.TemplateBase):
    """
    Example:
    >>> import ibis
    >>> BinomialABTest(ibis.connect("duckdb://:memory:"), "my_exp")  # doctest: +ELLIPSIS
    <earlysign.api.ab_tests.BinomialABTest object at 0x...>
    >>> BinomialABTest("duckdb://:memory:", "my_exp")  # doctest: +ELLIPSIS
    <earlysign.api.ab_tests.BinomialABTest object at 0x...>
    """

    def __init__(
        self,
        connector: BaseBackend | str,
        experiment_id: str,
        table_name: str | None = None,
    ) -> None:
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
            Design configuration including 'planned_max_n' for maximum sample size,
            along with other design parameters (alpha, tails, scale, efficacy, futility).
        """
        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        design.insert(payload)

    def update(self, payload: Dict[str, Any]) -> None:
        ## Record incremental observation (delta)
        obs = BinomialCountsRecord("observation").attach(self.ledger)
        obs.insert(**payload)

        ## Compute cumulative snapshot
        snapshot_op = BinomialCountsSnapshot(self.ledger, obs=obs, out_id="snapshot")
        snapshot_op.run()
        snapshot_rec: BinomialCountsSnapshotRecord = snapshot_op.outputs.snapshot

        ## Compute statistic (using snapshot)
        stat = WaldZStatistic(
            self.ledger, cum_counts=snapshot_rec, pooled=True, out_id="statistic"
        )
        stat.run()

        stat_record = stat.outputs.wald

        ## Read design info
        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        planned_max_n = design.latest()["planned_max_n"].execute().iloc[0]

        ## Compute information time
        info_op = InformationTime(
            self.ledger,
            out_id="info_time",
            cum_counts=snapshot_rec,
            planned_max_n=planned_max_n,
        )
        info_op.run()
        info = info_op.outputs.info

        ## Compute boundary of this run
        boundary_op = BoundaryFromDesign(
            self.ledger, design=design, info=info, out_id="boundary"
        )
        boundary_op.run()
        boundary: GroupSequentialBoundaryRecord = boundary_op.outputs.boundary

        ## Record the decision
        decision_op = GSDecisionFromWaldZ(
            self.ledger, wald=stat_record, boundary=boundary, out_id="decision"
        )
        decision_op.run()

    def status(self) -> State:
        decision = GroupSequentialDecisionSignalRecord("decision").attach(self.ledger)
        if "stop" in decision.latest()["signal"].execute().iloc[0]:
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
        ddf = design_record.latest().select(design=design_record.t.payload).execute()
        if len(ddf) == 0:
            raise ValueError("No design found in ledger. Call set_design() first.")
        design_payload = ddf.iloc[0]["design"]

        # Delegate to reporting module
        return plot_design_boundaries(
            design_payload=design_payload,
            resolve_boundary_func=resolve_boundary_from_design,
            n_points=n_points,
        )
