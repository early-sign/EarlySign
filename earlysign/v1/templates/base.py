from typing import Generic, Type, TypeVar

from pydantic import BaseModel

from earlysign.v1.framework.session import Session

TProtocol = TypeVar("TProtocol", bound=BaseModel)


class TemplateBase(Generic[TProtocol]):
    """
    Base class for all Orchestration Templates.

    Provides common functionality for Ledger initialization and Protocol registration.
    """

    _protocol_class: Type[TProtocol]

    def set_protocol(self, protocol: TProtocol) -> None:
        """
        Persists the trial protocol to the ledger.
        """
        if not hasattr(self, "_protocol_class"):
            raise NotImplementedError(
                "Subclasses must define _protocol_class or override set_protocol."
            )

        # Validate against schema
        protocol = self._protocol_class.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.Commit(protocol)
