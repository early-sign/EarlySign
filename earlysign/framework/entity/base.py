"""
Base class for entities in the framework.
"""

from abc import ABC
from typing import TypeVar

from earlysign.framework.projector import Projector

T = TypeVar("T")
"""Generic type placeholder for entity state"""


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
