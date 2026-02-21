"""
Core Ledger Operations
----------------------

This module tests low-level ledger operations and JSON access.

Setup
^^^^^

.. code-block:: python

    >>> import ibis, duckdb  # noqa: F401

JSON strategy
^^^^^^^^^^^^^

payload/labels access behaves like Ibis JSON column:

.. code-block:: python

    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()  # should be a no-op since table exists

    >>> # Empty select should still compile and execute:
    >>> df = ledger.t
    >>> from earlysign.core.util.json_ops import extract_json_scalar
    >>> q = df.select(
    ...     df.uuid,
    ...     x = extract_json_scalar(df.payload, "x", "string"),          # JSON scalar (string) extraction
    ...     kind = extract_json_scalar(df.attributes, "kind", "string"),     # JSON scalar from attributes
    ... )
    >>> out = q.execute()
    >>> list(out.columns)
    ['uuid', 'x', 'kind']

    >>> # Insert a couple of JSON events and read back via JSON accessors:
    >>> rows = [
    ...     dict(type="", payload={"x":"A"}, attributes={"kind":"k1"}),
    ...     dict(type="", payload={"x":"B"}, attributes={"kind":"k2"}),
    ... ]
    >>> _ = ledger.insert(data=rows[0]["payload"], attributes=rows[0]["attributes"])
    >>> _ = ledger.insert(data=rows[1]["payload"], attributes=rows[1]["attributes"])
    >>> from earlysign.core.util.json_ops import extract_json_scalar
    >>> got = df.select(extract_json_scalar(df.payload, "x", "string").name("x")).execute().to_dict("records")
    >>> sorted(v["x"] for v in got)
    ['A', 'B']

Delegation check
^^^^^^^^^^^^^^^^
Delegation check: these methods must return Ibis TableExpr unmodified:

.. code-block:: python

    >>> from ibis.expr.types import Table as TableExpr
    >>> isinstance(df.select("uuid"), TableExpr)
    True
    >>> isinstance(df.filter(df.type == "json"), TableExpr)
    True
    >>> isinstance(df.limit(1), TableExpr)
    True
    >>> df  # the underlying ibis table
    DatabaseTable: events
      uuid       string
      type       string
      payload    json
      attributes json
      timestamp  timestamp('UTC', 6)
      ledger_id  string
      metadata   json

Ledger Setup
^^^^^^^^^^^^
Doctests for Ledger:

.. code-block:: python

    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger

    # Setup with JSON-based ledger:
    >>> con = ibis.duckdb.connect(":memory:")
    >>> L = Ledger(con, "events")
    >>> # ensure must succeed when base table already exists
    >>> L.ensure()

Batch Insert
^^^^^^^^^^^^
Insert multiple json events:

.. code-block:: python

    >>> _ = [L.insert(data=row["payload"], attributes=row["attributes"]) for row in [
    ...   dict(payload={"nA": 100, "mA": 38, "nB": 120, "mB": 51}, attributes={"kind":"observation","batch":1}),
    ...   dict(payload={"nA":  80, "mA": 22, "nB":  90, "mB": 30}, attributes={"kind":"observation","batch":2}),
    ... ]]

JSON Accessors
^^^^^^^^^^^^^^
Read using JSON accessors:

.. code-block:: python

    >>> from earlysign.core.util.json_ops import extract_json_scalar
    >>> obs = (
    ...   L.t
    ...     .select(
    ...       "uuid",
    ...       nA=extract_json_scalar(L.t.payload, "nA", "int64"),
    ...       mA=extract_json_scalar(L.t.payload, "mA", "int64"),
    ...       nB=extract_json_scalar(L.t.payload, "nB", "int64"),
    ...       mB=extract_json_scalar(L.t.payload, "mB", "int64"),
    ...     )
    ...     .execute()
    ... )
    >>> set({"uuid","nA","mA","nB","mB"}) <= set(obs.columns)
    True

Verification of JSON access
^^^^^^^^^^^^^^^^^^^^^^^^^^
Insert more data and verify JSON access works:

.. code-block:: python

    >>> L.insert(
    ...   data={"nA":150,"mA":60,"nB":140,"mB":48},
    ...   attributes={"kind":"observation","batch":3}
    ... )
    UUID(...)
    >>> from earlysign.core.util.json_ops import extract_json_scalar
    >>> q2 = (
    ...     L.t.filter(L.t.type == "dict")
    ...     .order_by(L.t.timestamp.desc())
    ...     .limit(1)
    ...     .select(n_treat=extract_json_scalar(L.t.payload, "nA", "int64"))
    ... )
    >>> rows = q2.execute().to_dict("records")
    >>> rows[0]["n_treat"]
    150

Basic Operations
^^^^^^^^^^^^^^^^
Simple verification that basic operations work:

.. code-block:: python

    >>> len_before = len(L.t.execute())
    >>> L.insert(data={"test": True}, attributes={"kind": "test"})
    UUID(...)
    >>> len_after = len(L.t.execute())
    >>> len_after > len_before
    True

JSON Sanitization (BigQuery Safety)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Ledger must sanitize values like Infinity and NaN to None (JSON null)
to ensure compatibility with BigQuery.

.. code-block:: python

    >>> import numpy as np
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> L = Ledger(con, "sanitization_test")
    >>> L.ensure()

    >>> # 1. Test Infinity and NaN in payload
    >>> data = {
    ...     "inf": float("inf"),
    ...     "neg_inf": float("-inf"),
    ...     "nan": float("nan"),
    ...     "normal": 1.23
    ... }
    >>> _ = L.insert(data=data)

    >>> # 2. Read back and verify sanitization
    >>> row = L.t.order_by(ibis.desc("timestamp")).limit(1).execute().to_dict("records")[0]
    >>> payload = row["payload"]
    >>> if isinstance(payload, str):
    ...     import json
    ...     payload = json.loads(payload)
    >>> payload["inf"] is None
    True
    >>> payload["neg_inf"] is None
    True
    >>> payload["nan"] is None
    True
    >>> payload["normal"]
    1.23

    >>> # 3. Test NumPy Infinity and NaN
    >>> data_np = {
    ...     "inf": np.float64(np.inf),
    ...     "nan": np.float32(np.nan)
    ... }
    >>> _ = L.insert(data=data_np)
    >>> row_np = L.t.order_by(ibis.desc("timestamp")).limit(1).execute().to_dict("records")[0]
    >>> payload_np = row_np["payload"]
    >>> if isinstance(payload_np, str):
    ...     import json
    ...     payload_np = json.loads(payload_np)
    >>> payload_np["inf"] is None
    True
    >>> payload_np["nan"] is None
    True
"""
