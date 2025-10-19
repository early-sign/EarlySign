"""
Boundary operator for group sequential testing.

Reads Design and InfoTime records, computes boundaries using essentials functions.
"""

from typing import Dict, Optional

from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.essentials.boundaries import (
    resolve_boundary_from_design,
)
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDesignRecord,
    InformationTimeRecord,
)


class BoundaryFromDesign(LedgerOperator):
    """
    Read latest Design & InfoTime, compute and insert boundary record.

    Parameters
    ----------
    scoped : Ledger
        Scoped ledger instance.
    design : GroupSequentialDesignRecord (attached)
        Design record to read from.
    info : InformationTimeRecord (attached)
        Information time record to read from.
    out_id : str
        ID of boundary record to create.
    look : int, optional
        Look number (1-indexed), required for significance_level style.

    Examples
    --------
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.stats.common.group_sequential.operators.design_op import (
    ...     GroupSequentialDesign
    ... )
    >>> from earlysign.stats.common.group_sequential.operators.boundary_op import (
    ...     BoundaryFromDesign
    ... )
    >>> from earlysign.stats.common.group_sequential.records import (
    ...     GroupSequentialDesignRecord, InformationTimeRecord
    ... )
    >>> import ibis
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>>
    >>> # Write design
    >>> design_op = GroupSequentialDesign(
    ...     ledger, out_id="design1",
    ...     design={"alpha": 0.05, "tails": 2, "scale": "z",
    ...             "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...             "futility": {"mode": "none"}}
    ... )
    >>> design_op.run()
    >>>
    >>> # Write info time
    >>> info_rec = InformationTimeRecord(id="info1").attach(ledger)
    >>> info_rec.insert({"info_time": 0.5})
    >>>
    >>> # Compute boundary
    >>> boundary_op = BoundaryFromDesign(
    ...     ledger,
    ...     design=GroupSequentialDesignRecord(id="design1").attach(ledger),
    ...     info=info_rec,
    ...     out_id="bound1",
    ...     look=2
    ... )
    >>> boundary_op.run()
    """

    design: GroupSequentialDesignRecord
    info: InformationTimeRecord
    out_id: str
    look: Optional[int]

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"boundary": GroupSequentialBoundaryRecord(id=self.out_id)}

    def run(self) -> None:
        design_rec = self.design
        info_rec = self.info
        out = self.outputs["boundary"]
        look = getattr(self, "look", None)

        # Read latest design
        ddf = design_rec.latest().select(design=design_rec.t.payload).execute()
        if len(ddf) == 0:
            return
        design_payload = ddf.iloc[0]["design"]

        # Read latest info time
        idf = (
            info_rec.latest()
            .select(info_time=info_rec.t.payload["info_time"].cast("float64"))
            .execute()
        )
        if len(idf) == 0:
            return
        t = float(idf.iloc[0]["info_time"])

        # Compute boundaries using essentials
        up, lo, scale = resolve_boundary_from_design(
            design_payload=design_payload, info_time=t, look=look
        )

        # Compose payload
        payload = {
            # Flat fields for easy consumption
            "upper": float(up),
            "lower": float(lo),
            # Structured fields for richer consumers
            "efficacy": {"upper": float(up)},
            "futility": {
                "lower": float(lo),
                "binding": bool(
                    design_payload.get("futility", {}).get("binding", False)
                ),
                "mode": str(
                    design_payload.get("futility", {}).get("mode", "none")
                ).lower(),
            },
            "scale": str(scale),
            "alpha": float(design_payload["alpha"]),
            "tails": int(design_payload["tails"]),
            "info_time": float(t),
        }
        if look is not None:
            payload["look"] = int(look)

        out.insert(payload)
