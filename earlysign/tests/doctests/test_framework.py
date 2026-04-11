"""
Framework: Patterns of Scientific Logic
---------------------------------------

The `EarlySign` **Framework layer** introduces an architecture that provides
core abstractions to manage the lifecycle and lineage of an analysis:

1. **Projectors**: Pure functional views that translate history into scientific objects.
2. **Sessions & Writers**: Orchestrators that handle reading, writing, and lineage tracking.
3. **Entities**: Identifiable aggregates that support efficient snapshotting.
4. **Trace & Lineage**: The statistical "provenance" that links results to raw data.

1. Setup
^^^^^^^^

.. code-block:: python

    >>> import ibis, json
    >>> from typing import Optional
    >>> from pydantic import BaseModel
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.framework.session import Session

    >>> # Initialize an in-memory ledger for demonstration
    >>> con = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(con, "framework_demo")
    >>> ledger.ensure()

    >>> # Set pandas options for consistent display
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', None)
    >>> pd.set_option('display.width', 1000)

2. Core Ledger Read/Write
^^^^^^^^^^^^^^^^^^^^^^^^^

The Ledger is a low-level, append-only store. You can record any Pydantic model
to capture domain events or manual decisions.

.. code-block:: python

    >>> class MyDecision(BaseModel):
    ...     action: str
    ...     reason: str
    >>> _ = ledger.insert(
    ...     data=MyDecision(
    ...         action="CONTINUE",
    ...         reason="Non-binding futility boundary crossed, but clinical relevance remains."
    ...     )
    ... )
    >>> # ledger.show() provides a quick summary view of the ledger
    >>> ledger.show()  # doctest: +ELLIPSIS
             type identity trace                                            payload attributes
    0  MyDecision     None  None  {'action': 'CONTINUE', 'reason': 'Non-binding ...         {}

3. Scoped Views: Ledger Binding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ledgers can be "bound" to attributes, creating scoped views. Reads and writes
automatically apply the bound attributes.

.. code-block:: python

    >>> # Bind to a specific experiment_id
    >>> exp_ledger = ledger.bind(experiment_id="EXP001")
    >>> _ = exp_ledger.insert(data=MyDecision(action="STOP", reason="Scoped test"))
    >>> # Records inserted with one binding don't appear in another
    >>> other_ledger = ledger.bind(experiment_id="OTHER")
    >>> len(other_ledger.t.filter(other_ledger.t.type == "MyDecision").execute())
    0

4. Pure Functional Views: Projectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **`Projector`** takes a stream of events and "projects" them into a meaningful
object. This follows the **State-as-a-Fold** pattern: your state is simply the result
of "folding" (reducing) history.

Projectors are **purely functional**. Instead of returning just a value, they
return a **`ProjectionResult`**, which automatically bundles:
- **`data`**: The projected state (e.g., a count or a model).
- **`trace`**: A list of `TraceId`s that contributed to this specific result.

.. code-block:: python

    >>> from earlysign.framework.projector import Projector, ProjectionResult
    >>> from earlysign.framework.trace import TraceId

    >>> class CounterState(BaseModel):
    ...     total: int = 0

    >>> class IncrementProjector(Projector[CounterState]):
    ...     def project(self, table: ibis.Expr) -> ProjectionResult[CounterState]:
    ...         matches = table.filter(table.type == "Increment")
    ...         pdf = matches.execute()
    ...         if pdf.empty:
    ...             return ProjectionResult(data=CounterState(total=0), trace=[])
    ...         total = pdf["payload"].apply(
    ...             lambda x: json.loads(x)["value"] if isinstance(x, str) else x["value"]
    ...         ).sum()
    ...         trace = [TraceId(str(u)) for u in pdf["uuid"]]
    ...         return ProjectionResult(data=CounterState(total=total), trace=trace)

    >>> # Projector for MyDecision (finds latest)
    >>> class DecisionProjector(Projector[str]):
    ...     def project(self, table: ibis.Expr) -> ProjectionResult[str]:
    ...         match = table.filter(table.type == "MyDecision").order_by(ibis.desc("timestamp")).limit(1).execute()
    ...         if match.empty: return ProjectionResult(data=None, trace=[])
    ...         p = match.iloc[0]["payload"]
    ...         if isinstance(p, str): p = json.loads(p)
    ...         return ProjectionResult(data=p["action"], trace=[TraceId(str(match.iloc[0]["uuid"]))])

5. Orchestration: Sessions & Writers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **`Session`** tracks lineage across multiple reads and writes, creating an
**Implicit Web of Proof**.

1. **`sess.read(projector)`**: Executes a projector and *merges* its trace into
   the session's memory.
2. **`sess.commit(fact)`**: Writes a new record to the ledger, automatically
   tagging it with all traces accumulated during the session.

.. code-block:: python

    >>> class Increment(BaseModel):
    ...     value: int
    >>> class Summary(BaseModel):
    ...     text: str

    >>> # Add tutorial data using scoped ledger
    >>> sess_ledger = ledger.bind(experiment_id="102_demo")
    >>> _ = sess_ledger.insert(Increment(value=10))
    >>> _ = sess_ledger.insert(Increment(value=5))

    >>> with Session(sess_ledger) as sess:
    ...     # Reading merges trace into session
    ...     res = sess.read(IncrementProjector())
    ...     print(f"Current Total: {res.data.total}")
    ...     # Committing uses session trace for lineage
    ...     _ = sess.commit(Summary(text="Initial count completed"))
    Current Total: 15

6. Identifiable Aggregates: Entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While Projectors process history, **`Entity`** provides identifiable aggregates
with **snapshots** for efficiency. An Entity automatically handles:
1. Loading the **Latest Snapshot**.
2. Finding the **Delta** (new events since the snapshot).
3. **Folding** the delta into the state.
4. **Saving** a new snapshot back to the ledger.

.. code-block:: python

    >>> from earlysign.framework.entity.core import Entity
    >>> from earlysign.framework.entity.snapshot import Snapshot

    >>> class CounterEntity(Entity[CounterState]):
    ...     data_type = CounterState
    ...     @property
    ...     def initial_value(self) -> CounterState: return CounterState(total=0)
    ...     def compute(self, snapshot, delta_expr, full_table):
    ...         current_total = snapshot.data.total if snapshot else 0
    ...         pdf = delta_expr.filter(delta_expr.type == "Increment").execute()
    ...         for _, row in pdf.iterrows():
    ...             p = row["payload"]
    ...             val = json.loads(p)["value"] if isinstance(p, str) else p["value"]
    ...             current_total += val
    ...         trace = [TraceId(str(u)) for u in pdf["uuid"]]
    ...         if snapshot and snapshot.uuid: trace.insert(0, TraceId(str(snapshot.uuid)))
    ...         return ProjectionResult(data=CounterState(total=current_total), trace=trace)

    >>> with Session(sess_ledger) as sess:
    ...     counter = CounterEntity(identity="shared_counter")
    ...     result = sess.read(counter)
    ...     counter.save(sess, result)  # Save snapshot
    ...     print(f"Entity Total: {result.data.total}")
    Entity Total: 15

7. Scientific Lineage (Trace)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the heart of `EarlySign` is the **Trace**. We don't just care about the
current value; we care about **causality**.

- **`TraceId`**: A pointer to a specific event in the Ledger.
- **`Traced[T]`**: A wrapper that bundles a result of type `T` with its `TraceId` lineage.

Think of it like being able to point to any "pixel on the screen" (a result)
and seeing the direct line through every calculation back to the raw observations.

.. code-block:: python

    >>> with Session(sess_ledger) as sess:
    ...     _ = sess.read(IncrementProjector())
    ...     # Lineage is automatically accumulated during Read
    ...     len(sess.trace) > 0
    True

8. Session Scoping: Horizon & Local Visibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sessions define a "Scientific Horizon"—a point-in-time snapshot. Analysis is
protected from concurrent external writes while allowing regional consistency
of local commits.

.. code-block:: python

    >>> # Horizon test
    >>> # 1. Record exists BEFORE session starts
    >>> _ = sess_ledger.insert(MyDecision(action="PRE_SESSION", reason="baseline"))
    >>> with Session(sess_ledger) as sess:
    ...     # 2. Record inserted outside AFTER session started
    ...     _ = ledger.insert(MyDecision(action="EXTERNAL", reason="Mid-session"))
    ...     # 3. Read should only see PRE_SESSION
    ...     res = sess.read(DecisionProjector())
    >>> res.data
    'PRE_SESSION'

    >>> # Local Visibility test
    >>> with Session(sess_ledger) as sess:
    ...     # 1. Commit something within the session
    ...     _ = sess.commit(MyDecision(action="LOCAL_A", reason="Within sess"))
    ...     # 2. Directly insert something in the ledger (simulating external write)
    ...     _ = ledger.insert(MyDecision(action="EXTERNAL_3", reason="Outside sess"))
    ...     # 3. Read should see LOCAL_A but NOT EXTERNAL_3
    ...     res = sess.read(DecisionProjector())
    >>> res.data
    'LOCAL_A'

Advanced Topics
^^^^^^^^^^^^^^^

The Framework also supports complex domain-specific aggregates and time-series
trajectories.

.. code-block:: python

    >>> from earlysign.parts.trackers.binomial import Scoreboard
    >>> from earlysign.schema.ES3.trackers.binomial import BinomialArmData
    >>> with Session(sess_ledger) as sess:
    ...     _ = sess.commit(BinomialArmData(total=10, success=2, arm="A"))
    >>> with Session(sess_ledger) as sess:
    ...     state = sess.read(Scoreboard(identity="metrics"))
    >>> state.data.arms["A"].metrics.total
    10

    >>> from typing import List, Tuple
    >>> from earlysign.framework.entity.sequential import SequentialEntity
    >>> class SequentialCounter(SequentialEntity[int, CounterState]):
    ...     data_type = List[Tuple[int, CounterState]]
    ...     index_field = "step"
    ...     @property
    ...     def initial_value(self): return []
    ...     def get_index_expr(self, table):
    ...         from earlysign.core.util.json_ops import extract_json_scalar
    ...         return extract_json_scalar(table.attributes, self.index_field, "int64")
    ...     def compute_step(self, index, prev_state, delta_expr):
    ...         prev_val = prev_state.total if prev_state else 0
    ...         inc = delta_expr.filter(delta_expr.type == "Increment").execute()["payload"].apply(
    ...             lambda p: p["value"] if "value" in p else 0
    ...         ).sum()
    ...         return CounterState(total=prev_val + inc)
    >>> # Prepare data for staggered steps
    >>> sess_ledger.insert(Increment(value=10), attributes={"step": 1})
    UUID(...)
    >>> sess_ledger.insert(Increment(value=5), attributes={"step": 2})
    UUID(...)
    >>> with Session(sess_ledger) as sess:
    ...     counter = SequentialCounter(identity="demo_seq")
    ...     trajectory = sess.read(counter).data
    >>> len(trajectory)
    2
    >>> trajectory[1][1].total
    15

Summary
^^^^^^^

- **Projectors**: Pure functional interpretation of history.
- **Sessions**: Managers for causality and the "Scientific Horizon".
- **Entities**: Efficient, identifiable, and snapshot-capable aggregates.
- **Trace**: The immutable web of evidence connecting every result to its origin.
"""
