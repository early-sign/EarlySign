"""Group sequential testing namespace.

This package centralises design helpers, ledger operators, and plotting tools
for group sequential procedures.

Examples
--------
>>> import earlysign.v0.methods.group_sequential as GST
>>> isinstance(GST.boundary.BoundaryCalculator, type)
True
>>> hasattr(GST.decision, "GSDecision")
True
"""

from earlysign.v0.methods.group_sequential import (  # noqa: F401
    asn,
    boundary,
    decision,
    info_time,
    operating_characteristics,
    simulation,
    spending,
)
from earlysign.v0.methods.group_sequential.design import initial_design  # noqa: F401
