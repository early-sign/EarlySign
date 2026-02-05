"""
Entity subpackage for aggregates with consistent identity.

Provides base classes and implementations for entities that maintain
consistent identity across event-sourced trajectories.
"""

from earlysign.v1.framework.entity.base import BaseEntity
from earlysign.v1.framework.entity.core import Entity
from earlysign.v1.framework.entity.sequential import (
    LatestStateProjector,
    PointwiseTrajectoryProjector,
    SequentialEntity,
)
from earlysign.v1.framework.entity.simple import SimpleEntity, SimpleSequentialEntity
from earlysign.v1.framework.entity.snapshot import Snapshot

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
