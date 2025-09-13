"""
earlysign.core.ledger
=====================

Core ibis-framework based ledger implementation with simplified design.

This module provides an ibis-based ledger system focused on essential functionality:
- Backend-agnostic via ibis-framework
- JSON payload support with type-based wrap/unwrap
- Automatic earlysign_version tracking
- Simple get() interface with raw/unwrapped options

Examples:
---------
>>> import ibis
>>> from earlysign.core.ledger import Ledger, create_test_connection
>>> from earlysign.core.names import Namespace
>>>
>>> # Create test connection
>>> conn = create_test_connection("duckdb")
>>> ledger = Ledger(conn)
>>>
>>> # Write event
>>> ledger.write_event(
...     time_index="t1", namespace=Namespace.OBS, kind="observation",
...     experiment_id="exp1", step_key="s1", payload_type="TwoProportion",
...     payload={"n_treatment": 100, "n_control": 95}
... )
>>>
>>> # Query data (raw JSON)
>>> query = ledger.table.filter(ledger.table.payload_type == "TwoProportion")
>>> results = query.execute()
>>> len(results)
1
>>>
>>> # Query data (unwrapped)
>>> rows = ledger.unwrap_results(results)
>>> rows[0]["payload"]["n_treatment"]
100
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Type
import uuid as uuid_module
import json

# Direct ibis import - required dependency
import ibis
from ibis import BaseBackend
from ibis.expr.types import Table

from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.__version__ import __version__

# Type aliases
NamespaceLike = Union[Namespace, str]


def get_ledger_schema() -> ibis.Schema:
    """Get the standardized ledger schema using ibis.Schema."""
    return ibis.schema(
        [
            ("uuid", "string"),
            ("ledger_name", "string"),
            ("time_index", "string"),
            ("ts", "timestamp"),
            ("namespace", "string"),
            ("kind", "string"),
            ("entity", "string"),
            ("snapshot_id", "string"),
            ("tag", "string"),
            ("payload_type", "string"),
            ("payload", "json"),  # JSON data - supports ibis JSON operations
            ("earlysign_version", "string"),  # Auto-populated version
        ]
    )


class PayloadType(ABC):
    """Abstract base class for payload type handlers."""

    @abstractmethod
    def wrap(self, data: Any) -> str:
        """Convert data to JSON string for storage."""
        pass

    @abstractmethod
    def unwrap(self, json_str: str) -> Any:
        """Convert JSON string back to data."""
        pass


class JSONPayloadType(PayloadType):
    """Default JSON payload type handler."""

    def wrap(self, data: Any) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, separators=(",", ":"))

    def unwrap(self, json_str: str) -> Any:
        """Convert JSON string back to data."""
        return json.loads(json_str)


class PayloadTypeRegistry:
    """Registry for payload type handlers."""

    _handlers: Dict[str, PayloadType] = {}
    _default_handler = JSONPayloadType()

    @classmethod
    def register(cls, payload_type: str, handler: PayloadType) -> None:
        """Register a payload type handler."""
        cls._handlers[payload_type] = handler

    @classmethod
    def get_handler(cls, payload_type: str) -> PayloadType:
        """Get handler for payload type, fallback to default JSON handler."""
        return cls._handlers.get(payload_type, cls._default_handler)

    @classmethod
    def wrap(cls, payload_type: str, data: Any) -> str:
        """Wrap data using appropriate handler."""
        handler = cls.get_handler(payload_type)
        return handler.wrap(data)

    @classmethod
    def unwrap(cls, payload_type: str, json_str: str) -> Any:
        """Unwrap data using appropriate handler."""
        handler = cls.get_handler(payload_type)
        return handler.unwrap(json_str)


class Ledger:
    """
    Clean, focused ledger class using ibis-framework.

    Responsibilities:
    - Database connection management
    - Schema guarantee and table lifecycle
    - Automatic ledger_name and earlysign_version injection
    - Payload wrapping/unwrapping via PayloadTypeRegistry

    Query construction, aggregation, and complex transformations
    are delegated to callers using ibis table expressions.
    """

    def __init__(
        self,
        connection: BaseBackend,
        ledger_name: str = "default",
        table_name: str = "ledger",
    ):
        """Initialize ledger with connection and names.

        Parameters
        ----------
        connection : BaseBackend
            Ibis backend connection
        ledger_name : str
            Name of this ledger instance (for multi-ledger support)
        table_name : str
            Name of the table in the backend
        """
        self.connection = connection
        self.ledger_name = ledger_name
        self.table_name = table_name
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Create table with standardized schema if it doesn't exist."""
        try:
            self.connection.table(self.table_name)
        except Exception:
            # Use DDL to create table with proper JSON type
            ddl = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                uuid STRING,
                ledger_name STRING,
                time_index STRING,
                ts TIMESTAMP,
                namespace STRING,
                kind STRING,
                entity STRING,
                snapshot_id STRING,
                tag STRING,
                payload_type STRING,
                payload JSON,
                earlysign_version STRING
            )
            """
            self.connection.raw_sql(ddl)

    @property
    def table(self) -> Table:
        """
        Get ibis table filtered by ledger name.

        This is the main interface for querying - callers use this
        to build ibis expressions for filtering, aggregation, etc.

        Returns
        -------
        Table
            Ibis table expression filtered to this ledger's name

        Examples
        --------
        >>> conn = create_test_connection("duckdb")
        >>> ledger = Ledger(conn, "test_ledger")
        >>> # Direct ibis querying
        >>> filtered = ledger.table.filter(ledger.table.namespace == "OBS")
        >>> count = filtered.count().execute()

        >>> # Complex aggregations
        >>> import ibis
        >>> aggregated = (ledger.table
        ...     .filter(ledger.table.payload_type == "TwoProportion")
        ...     .group_by("entity")
        ...     .aggregate(total_events=ibis._.count())
        ...     .execute())
        """
        table = self.connection.table(self.table_name)
        return table.filter(table.ledger_name == self.ledger_name)

    @property
    def raw_table(self) -> Table:
        """
        Get raw ibis table without ledger filtering.

        Useful for meta-analysis across multiple ledgers.

        Returns
        -------
        Table
            Unfiltered ibis table expression
        """
        return self.connection.table(self.table_name)

    def write_event(
        self,
        *,
        time_index: Union[TimeIndex, str],
        namespace: NamespaceLike,
        kind: str,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        payload_type: str,
        payload: Any,
        tag: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """Write a typed event to the ledger.

        Parameters
        ----------
        time_index : TimeIndex or str
            Time index for the event
        namespace : NamespaceLike
            Event namespace
        kind : str
            Event kind/type
        experiment_id : ExperimentId or str
            Experiment identifier
        step_key : StepKey or str
            Step key within experiment
        payload_type : str
            Type of payload for wrap/unwrap handling
        payload : Any
            Payload data to be wrapped
        tag : str, optional
            Optional tag for filtering
        ts : datetime, optional
            Timestamp, defaults to now
        """
        if ts is None:
            ts = datetime.now(timezone.utc)
        elif ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Convert experiment_id + step_key to entity format
        entity = f"{experiment_id}#{step_key}"

        # Wrap payload using registered handler
        payload_json = PayloadTypeRegistry.wrap(payload_type, payload)

        # For JSON columns, we need to store the dict object, not JSON string
        import json
        if isinstance(payload_json, str):
            try:
                payload_obj = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                payload_obj = payload_json
        else:
            payload_obj = payload_json

        # Create record
        record = {
            "uuid": str(uuid_module.uuid4()),
            "ledger_name": self.ledger_name,
            "time_index": str(time_index),
            "ts": ts,
            "namespace": str(namespace),
            "kind": kind,
            "entity": entity,
            "snapshot_id": str(step_key),
            "tag": tag or "",
            "payload_type": payload_type,
            "payload": payload_obj,
            "earlysign_version": __version__,
        }

        # Insert using ibis - this approach works for most backends
        # Create a temporary table with the single record and union it
        import pandas as pd

        temp_df = pd.DataFrame([record])

        try:
            # Try to get existing data and append new record
            existing_data = self.raw_table.to_pandas()
            new_data = pd.concat([existing_data, temp_df], ignore_index=True)

            # Overwrite table with combined data
            self.connection.create_table(
                self.table_name, ibis.memtable(new_data), overwrite=True
            )
        except Exception:
            # If table doesn't exist or is empty, create with new record
            self.connection.create_table(
                self.table_name, ibis.memtable(temp_df), overwrite=True
            )

    def unwrap_payload(self, payload_type: str, payload_json: str) -> Any:
        """
        Unwrap payload using PayloadTypeRegistry.

        This is a convenience method for callers who need to decode
        payloads from query results.

        Parameters
        ----------
        payload_type : str
            Type of payload for unwrap handling
        payload_json : str
            JSON payload string to unwrap

        Returns
        -------
        Any
            Unwrapped payload data
        """
        return PayloadTypeRegistry.unwrap(payload_type, payload_json)

    def unwrap_results(self, df: Any) -> List[Dict[str, Any]]:
        """
        Convenience method to unwrap payloads in query results.

        Takes a pandas DataFrame from ibis query execution and
        unwraps the payload column using payload_type.

        Parameters
        ----------
        df : pandas.DataFrame
            Query results with payload and payload_type columns

        Returns
        -------
        List[Dict[str, Any]]
            Records with unwrapped payloads
        """
        records: List[Dict[str, Any]] = df.to_dict("records")

        for record in records:
            if "payload" in record and "payload_type" in record:
                record["payload"] = self.unwrap_payload(
                    record["payload_type"], record["payload"]
                )

        return records


def create_test_connection(backend: str = "duckdb") -> BaseBackend:
    """Create a test connection for testing purposes.

    Parameters
    ----------
    backend : str
        Backend type ("duckdb" or "polars")

    Returns
    -------
    BaseBackend
        Ibis backend connection

    Examples
    --------
    >>> conn = create_test_connection("duckdb")
    >>> ledger = Ledger(conn, "test")
    >>> ledger.write_event(
    ...     time_index="t1", namespace=Namespace.OBS, kind="test",
    ...     experiment_id="exp1", step_key="s1", payload_type="TestData",
    ...     payload={"value": 42}
    ... )
    >>> results = ledger.table.execute()
    >>> records = ledger.unwrap_results(results)
    >>> len(records)
    1
    >>> records[0]["payload"]["value"]
    42
    """
    if backend == "duckdb":
        return ibis.duckdb.connect(":memory:")
    elif backend == "polars":
        return ibis.polars.connect()
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'duckdb' or 'polars'.")
