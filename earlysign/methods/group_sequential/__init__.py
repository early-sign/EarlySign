"""Group sequential testing namespace.

This package centralises design helpers, ledger operators, and plotting tools
for group sequential procedures.

Examples
--------
>>> import earlysign.methods.group_sequential as GST
>>> isinstance(GST.boundary.BoundaryCalculator, type)
True
>>> hasattr(GST.decision, "GSDecision")
True
"""

from earlysign.methods.group_sequential import (  # noqa: F401
    asn,
    boundary,
    decision,
    info_time,
    operating_characteristics,
    sample_size_reestimation,
    simulation,
    spending,
)
from earlysign.methods.group_sequential.design import initial_design  # noqa: F401
