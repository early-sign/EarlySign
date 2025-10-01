from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Union,
    Mapping,
    Optional,
    Protocol,
    Self,
    overload,
    Dict,
    Tuple,
    TypeAlias,
    cast,
)
from ibis.expr.types import Table as TableExpr
from earlysign.core.ledger import Ledger
from earlysign.util.pydantic_ibis import explode_json_with_pydantic
import pydantic
from pydantic.fields import FieldInfo


PydanticField: TypeAlias = Union[
    type | str,  # e.g., "int" or int
    Tuple[type | str],  # e.g., ("int",) or (int,)
    Tuple[type | str, Any],  # e.g., ("int", 0) or (int, 0)
    Tuple[type | str, FieldInfo],  # e.g., ("int", Field(...)) or (int, Field(...))
]


class _LedgerRW(Protocol):
    """
    The protocol satisfying the required properties to be a Ledger Reader-Writer.
    """

    @property
    def t(self) -> TableExpr: ...

    @property
    def pydantic_model(self) -> type[pydantic.BaseModel]: ...

    def _explode_with_pydantic(self, t: TableExpr) -> TableExpr: ...


class QueryMixin:
    """
    Helpers for ledger-backed records.

    Assumes the concrete class provides:
      - property `t: TableExpr`.
    """

    def _explode_with_pydantic(self: _LedgerRW, t: TableExpr) -> TableExpr:
        return explode_json_with_pydantic(t, self.pydantic_model)

    def latest(self: _LedgerRW, explode: bool = True) -> TableExpr:
        t = self.t
        t = t.order_by(t.ts.desc()).limit(1)
        if not explode:
            return t
        else:
            return self._explode_with_pydantic(t)

    def order_by_ts(self: _LedgerRW, ascending: bool = True) -> TableExpr:
        t = self.t
        keys = (t.ts.asc(), t.uuid.asc()) if ascending else (t.ts.desc(), t.uuid.desc())
        return t.order_by(*keys)

    def since(self: _LedgerRW, ts: Any) -> TableExpr:
        t = self.t
        return t.filter(t.ts >= ts)

    def between(
        self: _LedgerRW, start: Any, end: Any, *, include_end: bool = False
    ) -> TableExpr:
        t = self.t
        expr = t.filter(t.ts >= start)
        return expr.filter(t.ts <= end) if include_end else expr.filter(t.ts < end)


class LedgerRecord:
    """
    Base class for typed views over ledger rows for a given payload_type and record_id.

    - `id` is REQUIRED: every persisted row will be tagged with labels["record_id"] = id
    - `schema` is a dict of type identifiers. It is converted to a Pydantic model by `pydantic.create_model()`.
        At least, the value of the dict can be one of the following:
        - A type identifier (e.g., `int`, `"int"`)
        - A tuple of (type, default value) (e.g., `(int, 0)`)
        - A tuple of (type, `pydantic.fields.FieldInfo`)
    - `payload_type` is set by subclasses as a class attribute
    - `ledger` is attached via .attach(ledger)

    Subclasses should define as class attributes:
    - `payload_type`: str - the payload type string
    - `schema`: Dict[str, PydanticField] - the schema definition

    Usage:
        class MyRecord(LedgerRecord, QueryMixin):
            payload_type = "MySchema"
            schema = {
                "field1": int,                    # simple type
                "field2": (str, "default"),      # type with default
                "field3": (int, Field(...)),     # type with FieldInfo
            }
    """

    # These should be overridden in subclasses as class attributes
    payload_type: str = ""
    schema: Dict[str, PydanticField] = {}

    def __init__(self, id: str):
        self.id = id
        self.ledger: Optional[Ledger] = None

    def attach(self, ledger: Ledger) -> Self:
        self.ledger = ledger
        return self

    @property
    def pydantic_model(self) -> type[pydantic.BaseModel]:
        """Create Pydantic model from schema class attribute."""
        return pydantic.create_model(
            self.payload_type, **cast(Dict[str, Any], self.schema)
        )

    @property
    def t(self) -> TableExpr:
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")
        t = self.ledger.t
        t = t.filter(
            cast(Any, t.payload_type == self.payload_type)
        )  # Cast to satisfy mypy type check
        t = t.filter(t.labels["record_id"].str == str(self.id))
        return t

    # --- overloads -----------------------------------------------------------
    @overload
    def insert(
        self, payload: Mapping[str, Any], *, labels: Optional[Mapping[str, Any]] = ...
    ) -> None: ...
    @overload
    def insert(
        self, *, labels: Optional[Mapping[str, Any]] = ..., **payload: Any
    ) -> None: ...

    # ------------------------------------------------------------------------

    def insert(self, *args: Any, **kwargs: Any) -> None:
        """
        Insert one row for this record.

        Usage:
            rec.insert(payload={"a": 1, "b": 2})
            rec.insert(a=1, b=2)                # kwargs form
            rec.insert(a=1, b=2, labels={"foo": "bar"})
        """
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")

        # labels are extracted from kwargs (both in kwargs and Mapping cases）
        labels = kwargs.pop("labels", None) or {}

        if args:
            # Mapping-type args
            if len(args) != 1 or not isinstance(args[0], Mapping):
                raise TypeError(
                    "insert() expects a single Mapping payload or keyword fields"
                )
            payload: Mapping[str, Any] = args[0]
        else:
            # kwargs-type args
            payload = kwargs

        self.ledger.insert(
            payload_type=self.payload_type,
            payload=payload,
            labels={**labels, "record_id": self.id},
        )
