from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Self, Type, TypeVar

from pydantic import BaseModel, Field, model_validator

from earlysign.framework.session import Session

TProtocol = TypeVar("TProtocol", bound=BaseModel)


class RichDisplayMixin:
    """Mixin to provide rich HTML representation in Jupyter/Colab notebooks."""

    def _repr_html_(self) -> str:
        if isinstance(self, BaseModel):
            data = self.model_dump()
        else:
            data = vars(self)

        import html

        def _format_val(v: Any) -> str:
            if isinstance(v, (dict, list)):
                import json

                return f"<pre style='margin: 0;'>{html.escape(json.dumps(v, indent=2))}</pre>"
            return html.escape(str(v))

        rows = "".join(
            f"<tr><th style='text-align: left; vertical-align: top; padding: 8px; border: 1px solid #ddd;'>{html.escape(k)}</th>"
            f"<td style='text-align: left; padding: 8px; border: 1px solid #ddd;'>{_format_val(v)}</td></tr>"
            for k, v in data.items()
        )
        return (
            f"<div style='overflow-x: auto;'>"
            f"<table style='border-collapse: collapse; width: 100%; font-family: sans-serif; border: 1px solid #ddd;'>"
            f"<thead><tr style='background-color: #f8f9fa;'><th colspan='2' style='padding: 12px; text-align: left; border: 1px solid #ddd;'>"
            f"✨ {html.escape(self.__class__.__name__)}</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )


class Controller(ABC, Generic[TProtocol]):
    """Base class for all Orchestration Controllers.

    Provides common functionality for Ledger initialization and Protocol registration.
    A Controller is an interface for executing commands and queries against the Ledger.
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

    def backtest(self, batches: Any) -> Dict[str, Any]:
        """Historical Analysis: Replays data and stops immediately on a stopping decision."""
        raise NotImplementedError("Backtesting is not implemented for this controller.")

    def backtest_from_table(
        self,
        table: Any,
        *,
        arm_col: str = "arm",
        total_col: str = "total",
        success_col: str = "success",
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Historical Analysis from an Ibis table."""
        raise NotImplementedError("Backtesting is not implemented for this template.")

    def backtest_from_df(self, df: Any, **kwargs: Any) -> Dict[str, Any]:
        """Historical Analysis from a Pandas DataFrame.

        Args:
            df: Pandas DataFrame containing historical data.
            **kwargs: Arguments passed to `backtest_from_table`.
        """
        import ibis

        table = ibis.memtable(df)
        return self.backtest_from_table(table, **kwargs)

    def describe_protocol(self) -> str:
        """Reconstructs the protocol from the ledger and describes it."""
        from earlysign.framework.projector import ProtocolProjector
        from earlysign.framework.session import Session

        with Session(self.ledger) as sess:
            traced = sess.read(ProtocolProjector(self._protocol_class))
            return self.describe_protocol_instance(traced.data)

    @classmethod
    def describe_protocol_instance(cls, protocol: BaseModel) -> str:
        """Provides a human-readable description of a protocol instance.

        Subclasses should override this for method-specific summaries.
        The default implementation returns a pretty-printed JSON.
        """
        return protocol.model_dump_json(indent=2)


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
