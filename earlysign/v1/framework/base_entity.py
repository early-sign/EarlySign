from abc import ABC
from typing import TypeVar

from earlysign.v1.framework.projector import Projector

T = TypeVar("T")


class BaseEntity(Projector[T], ABC):
    """
    Base class for identifiable projections (Facets).
    """

    def __init__(self, identity: str):
        self.identity = identity
