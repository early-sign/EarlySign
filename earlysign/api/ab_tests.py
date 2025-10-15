from dataclasses import dataclass
from typing import Any, Dict

import ibis
from ibis import BaseBackend

from earlysign.core.ledger import Ledger
from earlysign.stats.common.group_sequential.records import GroupSequentialDecisionSignalRecord
from earlysign.stats.common.group_sequential.design import (
    BoundaryFromDesign,
    GroupSequentialDesignRecord,
)
from earlysign.stats.common.group_sequential.info_time import (
    InformationTime,
)
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
)
from earlysign.stats.schemes.two_proportions.group_sequential import (
    GSDecisionFromWaldZ,
)
from earlysign.stats.schemes.two_proportions.operators import (
    WaldZStatistic,
)
from earlysign.stats.schemes.two_proportions.records import (
    BinomialCountsRecord,
    WaldZStatisticRecord,
)


@dataclass
class State:
    stop_recommended: bool = False


class BinomialABTest:
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
            Design configuration including 'max_n' for maximum sample size,
            along with other design parameters (alpha, tails, scale, efficacy, futility).
        """
        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        design.insert(payload)

    def update(self, payload: Dict[str, Any]) -> None:
        obs = BinomialCountsRecord("observation").attach(self.ledger)
        obs.insert(**payload)

        stat = WaldZStatistic(self.ledger, counts=obs, pooled=True, out_id="statistic")
        stat.run()

        stat_record: WaldZStatisticRecord = stat.outputs["wald"]  # type: ignore

        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        max_n = design.latest()["max_n"].execute().iloc[0]

        info_op = InformationTime(
            self.ledger, out_id="info_time", counts=obs, max_n=max_n
        )
        info_op.run()
        info = info_op.outputs["info"]

        boundary_op = BoundaryFromDesign(
            self.ledger, design=design, info=info, out_id="boundary"
        )
        boundary_op.run()
        boundary: GroupSequentialBoundaryRecord = boundary_op.outputs["boundary"]  # type: ignore

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
