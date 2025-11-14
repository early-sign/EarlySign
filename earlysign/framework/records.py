import json
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Self,
    Tuple,
    TypeAlias,
    Union,
    cast,
    overload,
)

import pydantic
from ibis.expr.types import Table as TableExpr
from pydantic.fields import FieldInfo

from earlysign.core.ledger import Ledger
from earlysign.core.util.pydantic_ibis import explode_json_with_pydantic

PydanticType: TypeAlias = Any | str  # e.g., "int" or int
PydanticField: TypeAlias = Union[
    PydanticType,  # e.g., "int" or int
    Tuple[PydanticType],  # e.g., ("int",) or (int,)
    Tuple[PydanticType, Any],  # e.g., ("int", 0) or (int, 0)
    Tuple[PydanticType, FieldInfo],  # e.g., ("int", Field(...)) or (int, Field(...))
]


class _LedgerRW(Protocol):
    """
    The protocol satisfying the required properties to be a Ledger Reader-Writer.
    """

    @property
    def t(self) -> TableExpr: ...

    @property
    def schema_pydantic_model(self) -> type[pydantic.BaseModel]: ...


class QueryMixin:
    """
    Helpers for ledger-backed records.

    Assumes the concrete class provides:
      - property `t: TableExpr`.
    """

    def all(self: _LedgerRW, explode: bool = True) -> TableExpr:
        """Return all records, optionally with payload exploded."""
        t = self.t
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def latest(self: _LedgerRW, explode: bool = True) -> TableExpr:
        t = self.t
        t = t.order_by(t.ts.desc()).limit(1)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def latest_before(
        self: _LedgerRW, ts: Any, *, include_ts: bool = True, explode: bool = True
    ) -> TableExpr:
        """Return the latest record before (or at) the given timestamp."""
        t = self.t
        t = t.filter(t.ts <= ts) if include_ts else t.filter(t.ts < ts)
        t = t.order_by(t.ts.desc()).limit(1)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def order_by_ts(
        self: _LedgerRW, ascending: bool = True, explode: bool = True
    ) -> TableExpr:
        t = self.t
        keys = (t.ts.asc(), t.uuid.asc()) if ascending else (t.ts.desc(), t.uuid.desc())
        t = t.order_by(*keys)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def since(self: _LedgerRW, ts: Any, explode: bool = True) -> TableExpr:
        t = self.t
        t = t.filter(t.ts >= ts)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def between(
        self: _LedgerRW,
        start: Any,
        end: Any,
        *,
        include_end: bool = False,
        explode: bool = True,
    ) -> TableExpr:
        t = self.t
        expr = t.filter(t.ts >= start)
        t = expr.filter(t.ts <= end) if include_end else expr.filter(t.ts < end)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)


class LedgerRecord:
    """
    Base class for typed views over ledger rows for a given payload_type and record_name.

    - `name` is REQUIRED: every persisted row will be tagged with labels["record_name"] = name
    - `schema` is a dict of type identifiers. It is converted to a Pydantic model by `pydantic.create_model()`.
        At least, the value of the dict can be one of the following:
        - A type identifier (e.g., `int`, `"int"`)
        - A tuple of (type, default value) (e.g., `(int, 0)`)
        - A tuple of (type, `pydantic.fields.FieldInfo`)
    - `payload_type` is auto-generated from module path and class name.
    - `ledger` is attached via .attach(ledger)

    Subclasses should define as class attributes:
    - `schema`: Dict[str, PydanticField] - the schema definition

    Usage:
        class MyRecord(LedgerRecord, QueryMixin):
            schema = {
                "field1": int,                    # simple type
                "field2": (str, "default"),      # type with default
                "field3": (int, Field(...)),     # type with FieldInfo
            }
    """

    schema: Dict[str, PydanticField] = {}

    def __init__(self, name: str, *, ledger: Optional[Ledger] = None):
        self.name = name
        self.ledger: Optional[Ledger] = None
        if ledger is not None:
            self.attach(ledger)

    def attach(self, ledger: Ledger) -> Self:
        self.ledger = ledger
        return self

    @classmethod
    def payload_type_name(cls) -> str:
        """
        Auto-generated payload type from module path and class name.

        Returns dot-separated module path and class name.
        Example: earlysign.stats.common.anytime_valid.records.EProcessRecord
                 -> stats.common.anytime_valid.records.EProcessRecord
        """
        module = cls.__module__
        cls_name = cls.__name__

        # Remove 'earlysign.' prefix if present
        path = module.removeprefix("earlysign.")

        # Return dot-separated path and class name
        return f"{path}.{cls_name}"

    @property
    def payload_type(self) -> str:
        return self.__class__.payload_type_name()

    @classmethod
    @lru_cache(maxsize=None)
    def _schema_model(cls) -> type[pydantic.BaseModel]:
        """Cached Pydantic model from schema class attribute."""
        return pydantic.create_model(
            cls.payload_type_name(), **cast(Dict[str, Any], cls.schema)
        )

    @property
    def schema_pydantic_model(self) -> type[pydantic.BaseModel]:
        return self.__class__._schema_model()

    @property
    def t(self) -> TableExpr:
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")
        t = self.ledger.t
        t = t.filter(
            cast(Any, t.payload_type == self.payload_type)
        )  # Cast to satisfy mypy type check
        t = t.filter(t.labels["record_name"].str == str(self.name))
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

    def insert(self, *args: Any, **kwargs: Any) -> None:
        """
        Insert one row for this record.

        Usage:
            rec.insert(payload={"a": 1, "b": 2})
            rec.insert(a=1, b=2)                # kwargs form
            rec.insert(a=1, b=2, labels={"foo": "bar"})
        The input is casted to the schema defined as the property of the specific LedgerRecord subclass.
        """
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")

        # labels are extracted from kwargs (both in kwargs and Mapping cases)
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
            # Support callers that pass the payload as a single keyword: insert(payload={...})
            # If provided, merge the inner mapping with any other explicit kwargs so callers
            # can do insert(payload={...}, extra_field=...)
            if "payload" in kwargs and isinstance(kwargs["payload"], Mapping):
                inner = dict(kwargs.pop("payload"))
                # remaining kwargs (except labels which were popped) override inner
                inner.update(kwargs)
                payload = inner
            else:
                payload = kwargs

        payload_with_defaults = self._validate_payload(payload)
        self.ledger.insert(
            payload_type=self.payload_type,
            payload=payload_with_defaults,
            labels={**labels, "record_name": self.name},
        )

    def _validate_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate payload against schema and return a plain dict."""
        model = self.schema_pydantic_model
        parsed = model.model_validate(dict(payload))
        dumped: Dict[str, Any] = parsed.model_dump()
        return dumped

    @overload
    def latest_payload(
        self,
        *,
        default: Optional[Mapping[str, Any]] = ...,
        include_ts: Literal[False] = ...,
    ) -> Dict[str, Any]: ...

    @overload
    def latest_payload(
        self,
        *,
        default: Optional[Mapping[str, Any]] = ...,
        include_ts: Literal[True],
    ) -> tuple[Dict[str, Any], Any]: ...

    def latest_payload(
        self,
        *,
        default: Optional[Mapping[str, Any]] = None,
        include_ts: bool = False,
    ) -> Dict[str, Any] | tuple[Dict[str, Any], Any]:
        """
        Fetch the latest payload as a Python dict.

        Parameters
        ----------
        default : Mapping, optional
            Returned when no row exists. Raises LookupError otherwise.
        """
        if self.ledger is None:
            raise RuntimeError("Record is not attached. Call .attach(ledger).")
        tbl = self.t
        tbl = tbl.order_by(tbl.ts.desc(), tbl.uuid.desc()).limit(1)
        df = tbl.select(tbl.ts, tbl.payload).execute()
        if df.empty:
            if default is not None:
                return dict(default)
            raise LookupError(f"No rows found for record_name={self.name}")
        row = df.iloc[0]
        raw_payload = row["payload"]
        ts_value = row["ts"]
        if raw_payload is None:
            data = {}
        else:
            if isinstance(raw_payload, str):
                raw_payload = json.loads(raw_payload)
            data = dict(cast(Mapping[str, Any], raw_payload))
        if include_ts:
            return data, ts_value
        return data
