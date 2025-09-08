"""
earlysign.core.names
====================

Typed names shared across the package.

- `Namespace`: an Enum for well-known ledger namespaces.
- `ExperimentId`, `StepKey`, `TimeIndex`: NewType wrappers for clarity.
- Common `Literal` tags for GST-two-proportions (optional; extend per method).

Examples
--------
>>> from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
>>> Namespace.OBS.value
'obs'
>>> eid = ExperimentId("exp#1"); isinstance(eid, str)
True
"""

from __future__ import annotations
from enum import Enum
from typing import Literal, NewType


class Namespace(str, Enum):
    """Well-known ledger namespaces.

    - OBS: raw observations
    - STATS: statistics (derived)
    - CRITERIA: critical values / boundaries / thresholds
    - SIGNALS: emitted signals / decisions / recommendations
    """

    OBS = "obs"
    STATS = "stats"
    CRITERIA = "criteria"
    SIGNALS = "signals"


# Optional: typed aliases for logical identifiers (thin wrappers over str).
ExperimentId = NewType("ExperimentId", str)
StepKey = NewType("StepKey", str)
TimeIndex = NewType("TimeIndex", str)

# Common tags (extend as needed).
WaldZTag = Literal["stat:waldz"]
GstBoundaryTag = Literal["crit:gst"]
GstDecisionTag = Literal["gst:decision"]
GstRecommendTag = Literal["gst:recommend"]
