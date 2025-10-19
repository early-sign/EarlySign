"""
Design operator for group sequential testing.

Wraps design validation and storage in the Ledger.
"""

from typing import Any, Dict

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.essentials.design_schema import (
    validate_design_payload,
)
from earlysign.stats.common.group_sequential.records import GroupSequentialDesignRecord


class GroupSequentialDesign(LedgerOperator):
    """
    Validate and insert a Group Sequential Design record.

    Parameters
    ----------
    scoped : Ledger
        Scoped ledger instance.
    out_id : str
        ID of GroupSequentialDesignRecord to create.
    design : dict
        Design specification with keys: alpha, tails, scale, efficacy, futility.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> design = {
    ...     "alpha": 0.05,
    ...     "tails": 2,
    ...     "scale": "z",
    ...     "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...     "futility": {"mode": "none"},
    ... }
    >>> op = GroupSequentialDesign(ledger, out_id="design1", design=design)
    >>> op.run()
    """

    out_id: str

    def __init__(self, scoped: Ledger, *, out_id: str, design: Dict[str, Any]):
        super().__init__(scoped, out_id=out_id, design=design)

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"design": GroupSequentialDesignRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs["design"]
        design = dict(getattr(self, "design"))

        # Validate design using essentials
        validate_design_payload(design)

        # Legacy compatibility: if style at top-level, move to efficacy
        if "style" in design and "efficacy" not in design:
            eff = {
                k: design[k]
                for k in ("style", "family", "alpha_levels", "gamma")
                if k in design
            }
            design["efficacy"] = eff

        # Insert
        out.insert(design)
