"""
Entity subpackage for aggregates with consistent identity.

Provides base classes and implementations for entities that maintain
consistent identity across event-sourced trajectories.
"""

from earlysign.framework.entity.base import BaseEntity
from earlysign.framework.entity.core import Entity
from earlysign.framework.entity.sequential import (
    LatestStateProjector,
    PointwiseTrajectoryProjector,
    SequentialEntity,
)
from earlysign.framework.entity.simple import SimpleEntity, SimpleSequentialEntity
from earlysign.framework.entity.snapshot import Snapshot

__all__ = [
    "BaseEntity",
    "Entity",
    "LatestStateProjector",
    "PointwiseTrajectoryProjector",
    "SequentialEntity",
    "SimpleEntity",
    "SimpleSequentialEntity",
    "Snapshot",
]
