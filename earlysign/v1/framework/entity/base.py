from abc import ABC
from typing import TypeVar

from earlysign.v1.framework.projector import Projector

T = TypeVar("T")


class BaseEntity(Projector[T], ABC):
    """Base class for identifiable projections (Facets).

    Attributes:
        identity (str): The unique identity of the entity.
    """

    def __init__(self, identity: str):
        """Initializes the BaseEntity.

        Args:
            identity (str): The unique identity of the entity.
        """
        self.identity = identity
