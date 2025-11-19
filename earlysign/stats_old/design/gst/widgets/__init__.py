"""
Design mode controllers for Group Sequential Trial UI.

This module contains designer classes for each design mode:
- FixedPowerDesigner: Fixed Power → Compute Required N
- FixedTimingDesigner: Fixed Timing → Compute Boundaries
- NMaxFixedMinMDEDesigner: N-Max Fixed → Find Min MDE
- OptimizeASNDesigner: Fixed Max N → Optimize ASN
- OptimizeDesignDesigner: Flexible Parameters → Optimize Design
"""

from earlysign.stats_old.design.gst.widgets.fixed_power import FixedPowerDesigner
from earlysign.stats_old.design.gst.widgets.fixed_timing import FixedTimingDesigner
from earlysign.stats_old.design.gst.widgets.nmax_fixed_min_mde import (
    NMaxFixedMinMDEDesigner,
)
from earlysign.stats_old.design.gst.widgets.optimize_asn import OptimizeASNDesigner
from earlysign.stats_old.design.gst.widgets.optimize_design import (
    OptimizeDesignDesigner,
)

__all__ = [
    "FixedPowerDesigner",
    "FixedTimingDesigner",
    "NMaxFixedMinMDEDesigner",
    "OptimizeASNDesigner",
    "OptimizeDesignDesigner",
]
