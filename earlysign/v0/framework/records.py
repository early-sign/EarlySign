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
        t = t.order_by(t.timestamp.desc()).limit(1)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def latest_before(
        self: _LedgerRW,
        timestamp: Any,
        *,
        include_ts: bool = True,
        explode: bool = True,
    ) -> TableExpr:
        """Return the latest record before (or at) the given timestamp."""
        t = self.t
        t = (
            t.filter(t.timestamp <= timestamp)
            if include_ts
            else t.filter(t.timestamp < timestamp)
        )
        t = t.order_by(t.timestamp.desc()).limit(1)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def order_by_ts(
        self: _LedgerRW, ascending: bool = True, explode: bool = True
    ) -> TableExpr:
        t = self.t
        keys = (
            (t.timestamp.asc(), t.uuid.asc())
            if ascending
            else (t.timestamp.desc(), t.uuid.desc())
        )
        t = t.order_by(*keys)
        if not explode:
            return t
        else:
            return explode_json_with_pydantic(t, self.schema_pydantic_model)

    def since(self: _LedgerRW, timestamp: Any, explode: bool = True) -> TableExpr:
        t = self.t
        t = t.filter(t.timestamp >= timestamp)
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
        expr = t.filter(t.timestamp >= start)
        t = (
            expr.filter(t.timestamp <= end)
            if include_end
            else expr.filter(t.timestamp < end)
        )
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
        # Return only class name for consistency with Ledger v2 inference
        return cls.__name__

    @property
    def payload_type(self) -> str:
        return self.schema_pydantic_model.__name__

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
            cast(Any, t.type == self.payload_type)
        )  # Cast to satisfy mypy type check
        t = t.filter(t.attributes["record_name"].str == str(self.name))
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
        # Note: In Ledger v2, we pass the data dict directly, and type is derived.
        # However, for LedgerRecord, we want to maintain the specific payload_type.
        # The Ledger.insert method derives type from data.__class__.__name__ or uses "Record".
        # We'll pass it as 'data' and the Ledger will use "Record" as type,
        # but we rely on the payload_type filter in .t
        self.ledger.insert(
            data=payload_with_defaults,
            attributes={**labels, "record_name": self.name},
        )

    def _validate_payload(self, payload: Mapping[str, Any]) -> pydantic.BaseModel:
        """Validate payload against schema and return a Pydantic model instance."""
        model = self.schema_pydantic_model
        return model.model_validate(dict(payload))

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
        tbl = tbl.order_by(tbl.timestamp.desc(), tbl.uuid.desc()).limit(1)
        df = tbl.select(tbl.timestamp, tbl.payload).execute()
        if df.empty:
            if default is not None:
                return dict(default)
            raise LookupError(f"No rows found for record_name={self.name}")
        row = df.iloc[0]
        raw_payload = row["payload"]
        ts_value = row["timestamp"]
        if raw_payload is None:
            data = {}
        else:
            if isinstance(raw_payload, str):
                raw_payload = json.loads(raw_payload)
            data = dict(cast(Mapping[str, Any], raw_payload))
        if include_ts:
            return data, ts_value
        return data


class SnapshotLedgerRecord(LedgerRecord, QueryMixin):
    """Ledger record that represents snapshots of another record's state."""

    snapshot_of: type[LedgerRecord] | None = None

    def __init__(
        self,
        name: str,
        *,
        ledger: Optional[Ledger] = None,
        snapshot_of: type[LedgerRecord] | None = None,
    ):
        super().__init__(name, ledger=ledger)
        if snapshot_of is not None:
            self.snapshot_of = snapshot_of

    def source_class(self) -> type[LedgerRecord]:
        """Return the LedgerRecord class this snapshot represents."""

        if self.snapshot_of is None:
            raise RuntimeError(
                "Snapshot record does not define a source class. Set 'snapshot_of'."
            )
        return self.snapshot_of

    def latest_snapshot_ts(self) -> Any | None:
        """Return the timestamp of the latest snapshot row, if any."""

        tbl = self.t.order_by(self.t.timestamp.desc(), self.t.uuid.desc()).limit(1)
        df = tbl.select(self.t.timestamp).execute()
        if df.empty:
            return None
        return df.iloc[0]["timestamp"]

    def diff_since_last_snapshot(
        self,
        source: LedgerRecord,
        *,
        include_equal_ts: bool = False,
    ) -> TableExpr:
        """Return source rows added since the last snapshot.

        Parameters
        ----------
        source : LedgerRecord
            The record whose rows are being snapshotted.
        include_equal_ts : bool, default False
            When True, include rows whose timestamps equal the snapshot timestamp.
        """

        if source.ledger is None:
            raise RuntimeError("Source record must be attached to a ledger.")

        if self.snapshot_of is not None and not isinstance(source, self.snapshot_of):
            raise TypeError("Source record does not match snapshot_of class.")

        t = source.t
        last_ts = self.latest_snapshot_ts()
        if last_ts is None:
            return t
        if include_equal_ts:
            return t.filter(t.timestamp >= last_ts)
        return t.filter(t.timestamp > last_ts)
