"""
earlysign.stats.common.tags
============================

Common literal tags used across statistical methods.

This module contains type-safe literal tags for:
- Statistics (e.g., Wald Z-test)
- Criteria (e.g., GST boundaries)
- Decisions and recommendations
"""

from typing import Literal

# Statistics tags
WaldZTag = Literal["stat:waldz"]

# Criteria tags
GstBoundaryTag = Literal["crit:gst"]

# Decision and recommendation tags
GstDecisionTag = Literal["gst:decision"]
GstRecommendTag = Literal["gst:recommend"]
