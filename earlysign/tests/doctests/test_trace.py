"""
Trace System Doctests
---------------------

Design Philosophy
^^^^^^^^^^^^^^^^^
In event sourcing, ALL state and lineage flows through Read operations (Projectors).
Write operations (Session.Commit, Ingest) are "fire and forget" - they record facts to the
ledger but do not return identifiers. Trace information comes from projecting
the ledger via Read, which provides Traced[T] with proper lineage.

These tests verify:
1. Basic Traced container behavior
2. Trace flows through Read operations (Projectors)
3. Implicit trace from session.read is used when trace is not explicitly passed

--- Setup ---

.. code-block:: python

    >>> import json
    >>> import ibis
    >>> from pydantic import BaseModel
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.framework.projector import Projector, ProjectionResult
    >>> from earlysign.framework.session import Session
    >>> from earlysign.framework.trace import Traced, TraceId, extract_traces

--- Test Models ---

.. code-block:: python

    >>> class Fact(BaseModel):
    ...     val: int

    >>> class Result(BaseModel):
    ...     total: int

.. code-block:: python

    >>> # Basic Traced wraps data with trace list
    >>> t = Traced(data=42, trace=[TraceId("abc123")])
    >>> t.data
    42
    >>> t.trace
    ['abc123']
    >>>
    >>> # Traced can have empty trace (root data)
    >>> t = Traced(data="root", trace=[])
    >>> t.data
    'root'
    >>> t.trace
    []

--- Test: extract_traces ---

.. code-block:: python

    >>> # extract_traces extracts TraceIds from Traced containers
    >>> t1 = Traced(10, [TraceId("h1")])
    >>> t2 = Traced(20, [TraceId("h2"), TraceId("h3")])
    >>> result = extract_traces(t1, t2)
    >>> sorted(result)
    ['h1', 'h2', 'h3']
    >>>
    >>> # extract_traces keeps duplicates (same trace used multiple times)
    >>> t1 = Traced(10, [TraceId("h1")])
    >>> extract_traces(t1, t1)
    ['h1', 'h1']
    >>>
    >>> # extract_traces returns None when no Traced containers present
    >>> extract_traces(1, 2, "string", {"key": "value"}) is None
    True
    >>>
    >>> # extract_traces returns [] when Traced has empty trace
    >>> t = Traced(10, [])
    >>> extract_traces(t)
    []
    >>>
    >>> # extract_traces handles nested structures
    >>> t1 = Traced(1, [TraceId("a")])
    >>> t2 = Traced(2, [TraceId("b")])
    >>> result = extract_traces([t1, {"nested": t2}])
    >>> sorted(result)
    ['a', 'b']

.. code-block:: python

    >>> # Create a simple projector that extracts uuids as trace
    >>> class FactProjector(Projector[Fact]):
    ...     def project(self, table: ibis.Expr) -> ProjectionResult[Fact]:
    ...         facts = table.filter(table.type == "Fact").execute()
    ...         total = facts["payload"].apply(lambda x: x["val"]).sum() if len(facts) > 0 else 0
    ...         ids = facts["uuid"].tolist() if len(facts) > 0 else []
    ...         return ProjectionResult(data=Fact(val=total), trace=[TraceId(u) for u in ids])

    >>> # Setup fresh ledger and ingest data
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()

    >>> with Session(ledger) as sess:
    ...     sess.commit(Fact(val=10), trace=[])
    ...     sess.commit(Fact(val=20), trace=[])
    UUID(...)
    UUID(...)

    >>> # Trace comes from Read, not from Commit return values
    >>> with Session(ledger) as sess:
    ...     traced_facts = sess.read(FactProjector())
    ...     len(traced_facts.trace) == 2  # Two facts, two ids
    True
    >>> traced_facts.data.val  # Sum of 10 + 20
    30

--- Test: Session accumulates trace from Read ---

    >>> # Session trace is populated by Read operations
    >>> with Session(ledger) as sess:
    ...     traced = sess.read(FactProjector())
    ...     len(sess.trace) == 2
    True

--- Test: Implicit trace used when not specified in Commit ---

    >>> # When Session.commit is called without explicit trace, session.trace is used
    >>> with Session(ledger) as sess:
    ...     _ = sess.read(FactProjector())  # Populates session.trace
    ...     sess.commit(Result(total=30))  # Uses implicit trace
    UUID(...)

    >>> # Verify the Result was committed with trace from the Read
    >>> # Use Ibis to inspect the JSON metadata column
    >>> t = ledger.t
    >>> # Extract trace field from metadata JSON directly using Ibis expression
    >>> q = t.filter(t.type == "Result").order_by(t.timestamp.desc()).limit(1)
    >>> res = q.select(trace=q.metadata["trace"]).execute()
    >>> len(res.iloc[0]["trace"]) == 2
    True

--- Test: Explicit trace overrides implicit ---

    >>> with Session(ledger) as sess:
    ...     _ = sess.read(FactProjector())  # Populates session.trace with 2 items
    ...     sess.commit(Result(total=99), trace=[])  # Explicit empty trace
    UUID(...)

    >>> t = ledger.t
    >>> q = t.filter(t.type == "Result").order_by(t.timestamp.desc()).limit(1)
    >>> res = q.select(trace=q.metadata["trace"]).execute()
    >>> trace_val = res.iloc[0]["trace"]
    >>> # Trace should be empty list
    >>> len(trace_val or []) == 0
    True

--- Test: Multiple Reads accumulate trace ---

    >>> with Session(ledger) as sess:
    ...     _ = sess.read(FactProjector())
    ...     _ = sess.read(FactProjector())
    ...     len(sess.trace) == 4  # 2 facts x 2 reads = 4 traces
    True
"""
