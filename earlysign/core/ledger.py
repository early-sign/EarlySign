"""
High level Ledger facade over a backend table (Ibis).

Design
------
- Append-only event ledger with JSON payload and labels.
- Labels can be "bound" (like a scope) so that queries and inserts
  automatically apply those filters / label merges.
- `bind(**labels)` returns a new Ledger with additional labels bound.
- `unbind(*selectors)` removes bound labels.

Table contract
--------------
Base table has columns:
  - uuid: string (auto-generated)
  - type: string
  - payload: json
  - attributes: json
  - timestamp: timestamp (UTC) (auto-generated)
  - metadata: json (contains trace)

Doctests
--------
>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger import Ledger
>>> import re

# Setup
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events"); ledger.ensure()

# Bind two attributes, then drop one (exact key)
>>> experiment_ledger = ledger.bind(experiment_id="exp1", env="prod")
>>> experiment_ledger.attributes == {"experiment_id": "exp1", "env": "prod"}
True
>>> reduced_ledger = experiment_ledger.unbind("env")
>>> reduced_ledger.attributes == {"experiment_id": "exp1"}
True

# Insert with only remaining bound attribute applied
>>> class MyEvent:
...     def __init__(self, a): self.a = a
>>> _ = reduced_ledger.insert(data=MyEvent(a=1))

# Some backends differ in JSON key equality semantics; materialize and check in Python.
>>> df = ledger.t.execute()
>>> any(rec["attributes"].get("experiment_id") == "exp1" for rec in df.to_dict("records"))
True

# Regex unbind: drop all keys starting with 'site_'
>>> geo_ledger = experiment_ledger.bind(site_eu=True, site_us=True)
>>> geo_ledger.attributes == {"experiment_id": "exp1", "env": "prod", "site_eu": True, "site_us": True}
True
>>> trimmed_ledger = geo_ledger.unbind(r"^site_.*")
>>> "site_eu" in trimmed_ledger.attributes or "site_us" in trimmed_ledger.attributes
False
>>> set(trimmed_ledger.attributes.keys()) == {"experiment_id", "env"}
True

# Regex unbind with compiled pattern
>>> p = re.compile(r"^exp.*")
>>> no_exp_ledger = trimmed_ledger.unbind(p)
>>> "experiment_id" in no_exp_ledger.attributes
False

# Drop all bound attributes
>>> unbound_ledger = experiment_ledger.unbind()
>>> unbound_ledger.attributes
{'experiment_id': 'exp1', 'env': 'prod'}
"""

import json
import re
import uuid as uuidlib
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Pattern,
    Self,
    TypeVar,
    Union,
    cast,
)

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.expr.types import Table

from earlysign import __version__
from earlysign.core.util.sanitize_for_json import sanitize_for_json

T = TypeVar("T")


def bq_parse_json(col: Any) -> Any:
    """BigQuery specific: Use PARSE_JSON to convert string to JSON type."""
    return ibis.literal("PARSE_JSON(").concat(col).concat(ibis.literal(")"))


class Ledger:
    """
    Append-only Event Ledger.

    The Ledger provides a unified interface for recording and querying events.
    It manages the mapping between high-level domain records and the physical
    append-only table.

    Attributes:
        connector: The Ibis backend connector.
        table_name: The physical table name.
        attributes: Default attributes applied to all operations in the current scope.
    """

    def __init__(
        self,
        connector: Optional[ibis.BaseBackend] = None,
        table_name: str = "events",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.connector = connector
        self.table_name = table_name
        self.attributes = attributes if attributes is not None else {}

    # --------- lifecycle ----------
    def set_connector(self, connector: ibis.BaseBackend) -> Self:
        return cast(Self, Ledger(connector, self.table_name, dict(self.attributes)))

    def use_default_table(self, name: str = "events") -> Self:
        return cast(Self, Ledger(self.connector, name, dict(self.attributes)))

    @property
    def _schema(self) -> sch.Schema:
        """Standard schema for the ledger table."""
        return sch.schema(
            dict(
                uuid=dt.string,
                type=dt.string,
                payload=dt.json,
                attributes=dt.json,
                timestamp=dt.timestamp(timezone="UTC"),
                metadata=dt.json,
            )
        )

    def ensure(self) -> None:
        """Ensure the ledger table (with standard schema) exists."""
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        if self.table_name in self.connector.list_tables():
            return
        self.connector.create_table(self.table_name, schema=self._schema)

    # --------- binding / scoping ----------
    def bind(self, **attributes: Any) -> Self:
        """Return a new Ledger whose scope includes the given attributes."""
        merged = dict(self.attributes)
        merged.update(attributes)
        return cast(Self, Ledger(self.connector, self.table_name, merged))

    def unbind(self, *patterns: Union[str, Pattern[str]]) -> Self:
        """
        Return a new Ledger with bound attributes removed if their keys match ANY pattern.
        Keys for which ANY compiled pattern .search(key) succeeds will be removed.

        Each argument in `patterns` may be:
          - str: compiled via re.compile(...)
          - re.Pattern[str]: used as-is

        Examples:
        >>> from earlysign.core.ledger import Ledger
        >>> import ibis, duckdb, re
        >>> con = ibis.duckdb.connect(":memory:")
        >>> base = Ledger(con, "events"); base.ensure()
        >>> experiment_ledger = base.bind(experiment_id="exp1", env="prod")
        >>> class MyEvent:
        ...     def __init__(self, a): self.a = a
        >>> _ = experiment_ledger.insert(data=MyEvent(a=1))
        >>> df = base.t.execute()
        >>> any(rec["attributes"].get("experiment_id") == "exp1" for rec in df.to_dict("records"))
        True

        # Regex unbind: drop all keys starting with 'site_'
        >>> geo_ledger = experiment_ledger.bind(site_eu=True, site_us=True)
        >>> geo_ledger.attributes == {"experiment_id": "exp1", "env": "prod", "site_eu": True, "site_us": True}
        True
        >>> trimmed_ledger = geo_ledger.unbind(r"^site_.*")
        >>> "site_eu" in trimmed_ledger.attributes or "site_us" in trimmed_ledger.attributes
        False
        >>> set(trimmed_ledger.attributes.keys()) == {"experiment_id", "env"}
        True

        # Regex unbind with compiled pattern
        >>> p = re.compile(r"^exp.*")
        >>> no_exp_ledger = trimmed_ledger.unbind(p)
        >>> "experiment_id" in no_exp_ledger.attributes
        False
        """
        if not patterns:
            return self

        compiled: list[Pattern[str]] = []
        for p in patterns:
            if isinstance(p, str):
                compiled.append(re.compile(p))
            else:
                compiled.append(p)

        def _keep_key(k: str) -> bool:
            return not any(rx.search(k) for rx in compiled)

        remaining = {k: v for k, v in self.attributes.items() if _keep_key(k)}
        return cast(Self, Ledger(self.connector, self.table_name, remaining))

    @property
    def t(self) -> Table:
        """Return a scoped TableExpr filtered by bound attributes."""
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        t = self.connector.table(self.table_name)
        for k, v in self.attributes.items():
            t = t.filter(t.attributes[k].str == str(v))
        return t

    # --------- write ----------
    def insert(
        self,
        data: Any,
        attributes: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Insert one row (append-only). Auto-fills uuid and timestamp.
        The current scope attributes (self.attributes) are ALWAYS merged into `attributes`.

        Note: This method intentionally does not return anything.
        In event sourcing all derived state should be obtained
        by projecting from the ledger via Read, which provides Traced[T] with
        proper lineage.
        """
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")

        # Derive type from data
        payload_type = data.__class__.__name__

        combined_attributes: Dict[str, Any] = dict(self.attributes)
        if attributes:
            combined_attributes.update(attributes)

        combined_metadata: Dict[str, Any] = {"pkg_version": f"earlysign=={__version__}"}
        if metadata:
            combined_metadata.update(metadata)

        if hasattr(data, "model_dump"):
            payload_obj = data.model_dump(mode="json", exclude_none=True)
        elif hasattr(data, "dict"):
            payload_obj = data.dict()
        else:
            payload_obj = sanitize_for_json(data)

        payload = json.dumps(payload_obj)

        ts = datetime.now(timezone.utc)
        row = {
            "uuid": uuidlib.uuid4().hex,
            "type": payload_type,
            "payload": payload,
            "attributes": json.dumps(sanitize_for_json(combined_attributes)),
            "timestamp": ts,
            "metadata": json.dumps(sanitize_for_json(combined_metadata)),
        }

        if self.connector.name == "bigquery":
            self._insert_bigquery(row)
        else:
            self._insert_ibis(row)

    def _insert_bigquery(self, row: Dict[str, Any]) -> None:
        """BigQuery SDK insert for robustness with JSON types."""
        # Client reference is dynamic on the connector
        client = getattr(self.connector, "client", None)
        dataset_id = getattr(self.connector, "dataset_id", None)
        project_id = getattr(self.connector, "project_id", None)
        
        if not client:
             raise RuntimeError("BigQuery connector missing client.")

        table_id = f"{project_id}.{dataset_id}.{self.table_name}"

        # Convert row to strict format for JSON API
        api_row = row.copy()
        # Timestamp must be ISO string
        if isinstance(api_row["timestamp"], datetime):
            api_row["timestamp"] = api_row["timestamp"].isoformat()
        
        # Note: row["payload"], row["attributes"], row["metadata"] are already JSON strings
        # matching what the BQ SDK expects for JSON columns.
        
        errors = client.insert_rows_json(table_id, [api_row])
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors}")

    def _insert_ibis(self, row: Dict[str, Any]) -> None:
        """Standard Ibis insert via memtable."""
        # Define schema for the local memtable (all strings for local stability)
        insert_schema = sch.schema(
            dict(
                uuid=dt.string,
                type=dt.string,
                payload=dt.string,
                attributes=dt.string,
                timestamp=dt.timestamp(timezone="UTC"),
                metadata=dt.string,
            )
        )
        mem_table = ibis.memtable([row], schema=insert_schema)
        
        to_insert = mem_table.select(
            *[
                mem_table[name].cast(self._schema[name])
                for name in self._schema.names
            ]
        )
        self.connector.insert(self.table_name, to_insert)

    # --------- scientific horizon support ----------
    @property
    def latest_ts(self) -> Any:
        """Returns the latest timestamp from the ledger."""
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        return self.t.timestamp.max().execute()

    # --------- utility ----------
    def show(self, all: bool = False) -> Any:
        if all:
            return self.t.execute()
        else:
            t = self.t.order_by("timestamp")
            return (
                t.mutate(
                    type=t.type.cast("string").split(".")[-1],
                    identity=t.attributes["entity_identity"]
                    .cast("string")
                    .re_replace('^"|"$', ""),
                    trace=t.metadata["trace"],
                )
                .drop("uuid", "timestamp", "metadata")
                .select("type", "identity", "trace", "payload", "attributes")
                .execute()
            )
