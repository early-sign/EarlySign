"""
Ledger: The Source of Truth
---------------------------

At the heart of `EarlySign` is the **Ledger**. Instead of storing the *current state*
of an experiment in a traditional database table, we store the **history of events**.
This approach, known as **Event Sourcing**, allows us to reconstruct the state of
an analysis at any point in time with full auditability and lineage.

Why Event Sourcing for Statistics?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. **Reproducibility**: You can exactly replicate a decision by replaying the events.
2. **Auditability**: Every statistical fact (Z-score, p-value) is linked to a specific set of raw observations.
3. **Flexibility**: You can change your analysis logic and "back-fill" results from the same raw data without losing history.

1. Setup
^^^^^^^^

We start by creating an in-memory database (DuckDB) and initializing the `Ledger`.
A Ledger is a table with a specific schema designed for performance: `timestamp`,
`ledger_id` (primary identifier), `uuid`, `type`, `payload` (JSON), `attributes` (JSON),
and `metadata` (JSON).

.. code-block:: python

    >>> import ibis
    >>> from pydantic import BaseModel
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.connect("duckdb://:memory:")
    >>> # Initialize a Ledger named 'tutorial_events' with a specific ledger_id for performance/clustering
    >>> ledger = Ledger(con, "tutorial_events", ledger_id="101_demo")
    >>> # Ensure the table exists with the correct schema
    >>> ledger.ensure()

2. Writing to the Ledger
^^^^^^^^^^^^^^^^^^^^^^^^

Every "write" is an immutable event. We use types (classes) to distinguish between
different kinds of events.

.. code-block:: python

    >>> class Design(BaseModel):
    ...     test_name: str
    ...     alpha: float
    ...     metric: str
    >>> class Observation(BaseModel):
    ...     look_index: int
    ...     arms: list[dict]
    >>> # 1. Record the Design of our experiment
    >>> _ = ledger.insert(
    ...     data=Design(
    ...         test_name="Binomial A/B",
    ...         alpha=0.05,
    ...         metric="conversion_rate",
    ...     )
    ... )
    >>> # 2. Record a batch of Observations
    >>> _ = ledger.insert(
    ...     data=Observation(
    ...         look_index=1,
    ...         arms=[
    ...             {"name": "control", "n": 500, "success": 55},
    ...             {"name": "treatment", "n": 500, "success": 72},
    ...         ],
    ...     )
    ... )

3. Reading and Deriving State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The power of the Ledger is that we can use `Ibis` (a portable dataframe-like API)
to query events. The `ledger.t` property gives us an Ibis expression pointing
to the table.

.. code-block:: python

    >>> # View the raw records (as a Pandas DataFrame)
    >>> df = ledger.t.execute()
    >>> len(df)
    2
    >>> set(df['type'].unique()) == {'Design', 'Observation'}
    True

4. Deriving Statistics on the Fly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of reading a pre-computed "conversion_rate" column, we calculate it
from the raw events. Because we use **Ibis**, this same code works identically
on DuckDB, BigQuery, or Snowflake.

.. code-block:: python

    >>> def calculate_conversion(t):
    ...     import json
    ...     obs = t.filter(t.type == "Observation")
    ...     pdf = obs.execute()
    ...     for _, row in pdf.iterrows():
    ...         p = row["payload"]
    ...         if isinstance(p, str): p = json.loads(p)
    ...         print(f"Look {p['look_index']}:")
    ...         for arm in p["arms"]:
    ...             rate = arm["success"] / arm["n"]
    ...             print(f"  - {arm['name']}: {rate:.1%} ({arm['success']}/{arm['n']})")
    >>> calculate_conversion(ledger.t)
    Look 1:
      - control: 11.0% (55/500)
      - treatment: 14.4% (72/500)

Advanced: Deep JSON Access
^^^^^^^^^^^^^^^^^^^^^^^^^^

`EarlySign` provides utilities to extract data from JSON columns directly
within Ibis expressions, which is more efficient on production backends
like BigQuery or Snowflake.

.. code-block:: python

    >>> from earlysign.core.util.json_ops import extract_json_scalar
    >>> df = ledger.t
    >>> q = df.filter(df.type == "Design").select(
    ...     name = extract_json_scalar(df.payload, "test_name", "string"),
    ...     alpha = extract_json_scalar(df.payload, "alpha", "float64"),
    ... )
    >>> q.execute().to_dict("records")[0]
    {'name': 'Binomial A/B', 'alpha': 0.05}

Advanced: Low-level API
^^^^^^^^^^^^^^^^^^^^^^^

You can also bypass Pydantic and insert raw dictionaries. By default,
the `type` for such records is "dict".

.. code-block:: python

    >>> _ = ledger.insert(
    ...     data={"nA": 100, "mA": 38, "nB": 120, "mB": 51},
    ...     attributes={"kind": "observation", "batch": 1}
    ... )
    >>> # Verify record saved with type "dict"
    >>> q = ledger.t.filter(ledger.t.type == "dict").order_by(ledger.t.timestamp.desc()).limit(1)
    >>> res = q.select(n_treat=extract_json_scalar(ledger.t.payload, "nA", "int64")).execute()
    >>> int(res.iloc[0]["n_treat"])
    100
    >>> # attributes are also searchable via JSON accessors
    >>> q_attr = ledger.t.filter(extract_json_scalar(ledger.t.attributes, "kind", "string") == "observation")
    >>> len(q_attr.execute()) >= 1
    True

Verification & Integrity
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> # Delegation check: Ledger should return Ibis TableExpr unmodified
    >>> from ibis.expr.types import Table as TableExpr
    >>> isinstance(ledger.t.select("uuid"), TableExpr)
    True
    >>> # Integrity check: sequential inserts
    >>> len_before = len(ledger.t.execute())
    >>> _ = ledger.insert(data={"test": True}, attributes={"kind": "test"})
    >>> len_after = len(ledger.t.execute())
    >>> len_after == len_before + 1
    True

Production Safety: JSON Sanitization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Ledger must sanitize values like Infinity and NaN to None (JSON null)
to ensure compatibility with production warehouses like BigQuery.

.. code-block:: python

    >>> import numpy as np
    >>> data = {
    ...     "inf": float("inf"),
    ...     "nan": float("nan"),
    ...     "normal": 1.23
    ... }
    >>> _ = ledger.insert(data=data)
    >>> row = ledger.t.order_by(ibis.desc("timestamp")).limit(1).execute().to_dict("records")[0]
    >>> payload = row["payload"]
    >>> if isinstance(payload, str):
    ...     import json
    ...     payload = json.loads(payload)
    >>> payload["inf"] is None and payload["nan"] is None
    True
    >>> payload["normal"]
    1.23

Summary
^^^^^^^

- **Source of Truth**: The Ledger is the definitive record of what happened.
- **ledger_id**: We use it for high-performance grouping and clustering in production warehouses.
- **Attributes**: We continue to use them for secondary, dynamic JSON-based labeling.
- **Derivation**: State is reconstructed by querying history, not by updating rows.
"""
