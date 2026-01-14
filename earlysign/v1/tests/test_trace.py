"""
Doctests for the trace system (TraceId, Traced, explicit/implicit trace propagation).

Design Philosophy:
==================
In event sourcing, ALL state and lineage flows through Read operations (Projectors).
Write operations (Commit, Ingest) are "fire and forget" - they record facts to the
ledger but do not return identifiers. Trace information comes from projecting
the ledger via Read, which provides Traced[T] with proper lineage.

These tests verify:
1. Basic Traced container behavior
2. Trace flows through Read operations (Projectors)
3. Implicit trace from session.Read is used when trace is not explicitly passed

--- Setup ---
>>> import json
>>> import ibis
>>> from pydantic import BaseModel
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.v1.framework.projector import Projector, ProjectionResult
>>> from earlysign.v1.framework.session import Session
>>> from earlysign.v1.framework.trace import Traced, TraceId, extract_traces
>>> from earlysign.v1.framework.write_models import WriteModel

--- Test Models ---
>>> class Fact(BaseModel):
...     val: int

>>> class Result(BaseModel):
...     total: int

--- Test: Traced Container ---

# Basic Traced wraps data with trace list
>>> t = Traced(data=42, trace=[TraceId("abc123")])
>>> t.data
42
>>> t.trace
['abc123']

# Traced can have empty trace (root data)
>>> t = Traced(data="root", trace=[])
>>> t.data
'root'
>>> t.trace
[]

--- Test: extract_traces ---

# extract_traces extracts TraceIds from Traced containers
>>> t1 = Traced(10, [TraceId("h1")])
>>> t2 = Traced(20, [TraceId("h2"), TraceId("h3")])
>>> result = extract_traces(t1, t2)
>>> sorted(result)
['h1', 'h2', 'h3']

# extract_traces keeps duplicates (same trace used multiple times)
>>> t1 = Traced(10, [TraceId("h1")])
>>> extract_traces(t1, t1)
['h1', 'h1']

# extract_traces returns None when no Traced containers present
>>> extract_traces(1, 2, "string", {"key": "value"}) is None
True

# extract_traces returns [] when Traced has empty trace
>>> t = Traced(10, [])
>>> extract_traces(t)
[]

# extract_traces handles nested structures
>>> t1 = Traced(1, [TraceId("a")])
>>> t2 = Traced(2, [TraceId("b")])
>>> result = extract_traces([t1, {"nested": t2}])
>>> sorted(result)
['a', 'b']

--- Test: Trace Flows Through Read (Projectors) ---

# Create a simple projector that extracts uuids as trace
>>> class FactProjector(Projector[Fact]):
...     def project(self, table: ibis.Expr) -> ProjectionResult[Fact]:
...         facts = table.filter(table.payload_type == "Fact").execute()
...         total = facts["payload"].apply(lambda x: x["val"]).sum() if len(facts) > 0 else 0
...         uuids = facts["uuid"].tolist() if len(facts) > 0 else []
...         return ProjectionResult(data=Fact(val=total), trace=[TraceId(u) for u in uuids])

# Setup fresh ledger and ingest data
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events")
>>> ledger.ensure()

>>> with Session(ledger) as sess:
...     WriteModel.Commit(sess, Fact(val=10), trace=[])
...     WriteModel.Commit(sess, Fact(val=20), trace=[])

# Trace comes from Read, not from Commit return values
>>> with Session(ledger) as sess:
...     traced_facts = sess.Read(FactProjector())
...     len(traced_facts.trace) == 2  # Two facts, two uuids
True

>>> traced_facts.data.val  # Sum of 10 + 20
30

--- Test: Session accumulates trace from Read ---

# Session trace is populated by Read operations
>>> with Session(ledger) as sess:
...     traced = sess.Read(FactProjector())
...     len(sess.trace) == 2
True

--- Test: Implicit trace used when not specified in Commit ---

# When Commit is called without explicit trace, session.trace is used
>>> with Session(ledger) as sess:
...     _ = sess.Read(FactProjector())  # Populates session.trace
...     WriteModel.Commit(sess, Result(total=30))  # Uses implicit trace

# Verify the Result was committed with trace from the Read
>>> df = ledger.t.execute()
>>> result_row = df[df["payload_type"] == "Result"].iloc[0]
>>> stored_trace = json.loads(result_row["trace"])
>>> len(stored_trace) == 2  # Should have trace from the two Facts
True

--- Test: Explicit trace overrides implicit ---

>>> with Session(ledger) as sess:
...     _ = sess.Read(FactProjector())  # Populates session.trace with 2 items
...     WriteModel.Commit(sess, Result(total=99), trace=[])  # Explicit empty trace

>>> df2 = ledger.t.execute()
>>> result_rows = df2[df2["payload_type"] == "Result"]
>>> last_result = result_rows.iloc[-1]
>>> last_trace = json.loads(last_result["trace"]) if last_result["trace"] else []
>>> len(last_trace) == 0  # Empty because we explicitly passed empty trace
True

--- Test: Multiple Reads accumulate trace ---

>>> with Session(ledger) as sess:
...     _ = sess.Read(FactProjector())
...     _ = sess.Read(FactProjector())
...     len(sess.trace) == 4  # 2 facts x 2 reads = 4 traces
True
"""
