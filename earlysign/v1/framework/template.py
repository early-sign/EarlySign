from abc import ABC, abstractmethod
from typing import Any, Generic, Self, Type, TypeVar

from pydantic import BaseModel, Field, model_validator

from earlysign.v1.framework.session import Session

TProtocol = TypeVar("TProtocol", bound=BaseModel)


class TemplateBase(ABC, Generic[TProtocol]):
    """Base class for all Orchestration Templates.

    Provides common functionality for Ledger initialization and Protocol registration.
    """

    @property
    @abstractmethod
    def _protocol_class(self) -> Type[TProtocol]:
        """Subclasses must define the protocol type."""
        ...

    ledger: Any

    def set_protocol(self, protocol: TProtocol) -> None:
        """Persists the trial protocol to the ledger.

        Args:
            protocol: The protocol instance to persist.
        """
        # Validate against schema
        protocol = self._protocol_class.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.commit(protocol)


class AutoNameMixin(BaseModel):
    """
    Mixin to automatically populate the 'name' field with the fully qualified class name
    if it is not provided.
    """

    name: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def default_name_pre(cls, data: Any) -> Any:
        # Handle dict input
        if isinstance(data, dict):
            if "name" not in data or not data["name"]:
                data["name"] = f"{cls.__module__}.{cls.__name__}"
        return data

    @model_validator(mode="after")
    def default_name_post(self) -> Self:
        # Handle object init missing explicit name (if default was used)
        if not self.name:
            self.name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return self
