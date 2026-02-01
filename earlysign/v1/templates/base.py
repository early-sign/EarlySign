from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel

from earlysign.v1.framework.session import Session

TProtocol = TypeVar("TProtocol", bound=BaseModel)


class TemplateBase(ABC, Generic[TProtocol]):
    """
    Base class for all Orchestration Templates.

    Provides common functionality for Ledger initialization and Protocol registration.
    """

    @property
    @abstractmethod
    def _protocol_class(self) -> Type[TProtocol]:
        """Subclasses must define the protocol type."""
        ...

    ledger: Any

    def set_protocol(self, protocol: TProtocol) -> None:
        """
        Persists the trial protocol to the ledger.
        """
        # Validate against schema
        protocol = self._protocol_class.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.Commit(protocol)
