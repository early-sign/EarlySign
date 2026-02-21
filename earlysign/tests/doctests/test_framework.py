"""
Introduction to the EarlySign Framework
---------------------------------------

This module provides a comprehensive introduction to the EarlySign framework through functional doctests.

Setup
^^^^^

.. code-block:: python

    >>> import ibis
    >>> from earlysign.core.ledger import Ledger

    # Allow more columns in displaying pandas dataframes:
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', None)
    >>> pd.set_option('display.width', 1000)

    # Initialize an in-memory ledger for demonstration purposes:
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()

Core features
^^^^^^^^^^^^^

Ledger Read/Write
~~~~~~~~~~~~~~~~~
The Ledger is a low-level, append-only event store. You can record any Pydantic model
to capture domain events or manual decisions without using the full framework.

.. code-block:: python

    >>> from pydantic import BaseModel
    >>> class MyDecision(BaseModel):
    ...     action: str
    ...     reason: str

    >>> # Record a manual decision to continue despite crossing a non-binding boundary
    >>> ledger.insert(
    ...     data=MyDecision(
    ...         action="CONTINUE",
    ...         reason="Non-binding futility boundary crossed, but clinical relevance remains."
    ...     )
    ... )
    UUID(...)

ibis-framework
~~~~~~~~~~~~~~
Since the Ledger is backed by Ibis, you can perform powerful queries using
standard Ibis expressions.

.. code-block:: python

    >>> table = ledger.t
    >>> table.filter(table.type == "MyDecision").payload["action"].execute().tolist()
    ['CONTINUE']

    >>> # ledger.show() provides a quick summary view of the ledger
    >>> ledger.show()
             type identity trace                                            payload attributes
    0  MyDecision     None  None  {'action': 'CONTINUE', 'reason': 'Non-binding ...         {}

Ledger Binding
~~~~~~~~~~~~~~
Ledgers can be "bound" to attributes, creating scoped views. Reads and writes
automatically apply the bound attributes. Use `unbind()` to remove attributes.

.. code-block:: python

    >>> exp_ledger = ledger.bind(experiment_id="EXP001")
    >>> # Every record has a unique id and timestamp (timestamp)
    >>> df = ledger.t.execute()
    >>> 'uuid' in df.columns and 'timestamp' in df.columns
    True
    >>> len(df.iloc[0]['uuid']) == 32  # hex uuid
    True

    >>> # Records inserted with one binding don't appear in a differently-bound ledger
    >>> other_ledger = ledger.bind(experiment_id="OTHER")
    >>> from pydantic import BaseModel
    >>> class MyDecision(BaseModel):
    ...     action: str
    ...     reason: str
    >>> _ = other_ledger.insert(data=MyDecision(action="OTHER", reason="scoped"))
    >>> len(other_ledger.t.filter(other_ledger.t.type == "OTHER").execute())
    0

Framework features
^^^^^^^^^^^^^^^^^^

Write Model
~~~~~~~~~~~
The Framework provides `Session.commit` to record events with scientific lineage.
`sess.commit` records a model, while `sess.call_and_commit` records the result of a function.

.. code-block:: python

    >>> from earlysign.framework.session import Session
    >>> with Session(ledger) as sess:
    ...    sess.commit(MyDecision(action="STOP", reason="Safety concern"))
    UUID(...)

Projector
~~~~~~~~~
Projectors are "State-as-a-Fold" operators. They reconstruct high-level facts
from the event stream.

.. code-block:: python

    >>> import ibis
    >>> from earlysign.framework.projector import ProjectionResult
    >>> from earlysign.framework.trace import TraceId
    >>> class DecisionProjector:
    ...     def project(self, table):
    ...         match = table.filter(table.type == "MyDecision").order_by(ibis.desc("timestamp")).limit(1).execute()
    ...         if match.empty: return ProjectionResult(data=None, trace=[])
    ...         return ProjectionResult(data=match.iloc[0]["payload"]["action"], trace=[TraceId(str(match.iloc[0]["uuid"]))])

    >>> with Session(ledger) as sess:
    ...     latest_action = sess.read(DecisionProjector())
    >>> latest_action.data
    'STOP'

Session (Horizon)
~~~~~~~~~~~~~~~~~
A Session defines a "Scientific Horizon"—a point-in-time snapshot of the ledger.
Analysis within a session is protected from concurrent writes.

.. code-block:: python

    >>> with Session(ledger) as sess:
    ...     # 1. Write something outside (directly to ledger) AFTER session started
    ...     _ = ledger.insert(data=MyDecision(action="EXTERNAL_B", reason="outside"))
    ...     # 2. Projection within session only sees records up to the horizon (e.g., 'STOP')
    ...     res = sess.read(DecisionProjector())
    >>> res.data
    'STOP'

Session (Local Visibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~
A Session allows reading its own committed data even while the Scientific Horizon
is fixed for external data. This allows multi-step updates within a single session.

.. code-block:: python

    >>> with Session(ledger) as sess:
    ...     # 1. Commit something within the session
    ...     _ = sess.commit(MyDecision(action="LOCAL_A", reason="within session"))
    ...     # 2. Directly insert something in the ledger (simulating external write)
    ...     _ = ledger.insert(data=MyDecision(action="EXTERNAL_C", reason="external concurrent"))
    ...     # 3. Read should see LOCAL_A but NOT EXTERNAL_C
    ...     res = sess.read(DecisionProjector())
    ...
    >>> res.data
    'LOCAL_A'

Trace
~~~~~
Scientific Lineage (Trace) is automatically accumulated as you Read data in a Session.

.. code-block:: python

    >>> with Session(ledger) as sess:
    ...     # Reading records their causal IDs in the session trace
    ...     res = sess.read(DecisionProjector())
    ...     len(sess.trace) > 0
    True

Entity
~~~~~~
Entities are special aggregates with identity. `Entity` supports differential folding.

.. code-block:: python

    >>> from earlysign.methods.binomial import Scoreboard
    >>> from earlysign.schema.ES3.Binomial import BinomialArmData
    >>> fact = Scoreboard(identity="metrics")
    >>> # Pre-populate data in a separate session so it's visible in the next horizon
    >>> with Session(ledger) as sess:
    ...     sess.commit(BinomialArmData(total=10, success=2, arm="A"))
    UUID(...)

    >>> with Session(ledger) as sess:
    ...     state = sess.read(fact)
    >>> state.data.arms["A"].metrics.total
    10

Sequential Entity
~~~~~~~~~~~~~~~~~
Sequential Entities evolve over time (e.g., Test Statistics, Spending Boundaries).
They represent a trajectory of states $(S_0, S_1, \\dots, S_n)$ indexed by a sequential coordinate.

.. code-block:: python

    >>> from typing import List, Tuple
    >>> from pydantic import BaseModel
    >>> from earlysign.framework.entity.sequential import SequentialEntity
    >>>
    >>> class CounterState(BaseModel):
    ...     count: int
    >>>
    >>> class Increment(BaseModel):
    ...     step: int
    ...     val: int
    >>>
    >>> class SequentialCounter(SequentialEntity[int, CounterState]):
    ...     data_type = List[Tuple[int, CounterState]]
    ...     index_field = "step"
    ...
    ...     @property
    ...     def initial_value(self) -> List[Tuple[int, CounterState]]:
    ...         return []
    ...
    ...     def get_index_expr(self, table: ibis.Expr) -> ibis.Expr:
    ...         from earlysign.core.util.json_ops import extract_json_scalar
    ...         return extract_json_scalar(table.attributes, self.index_field, "int64")
    ...
    ...     def compute_step(self, index, prev_state, delta_expr) -> CounterState:
    ...         prev_val = prev_state.count if prev_state else 0
    ...         increments = delta_expr.filter(delta_expr.type == "Increment").execute()
    ...         total_inc = increments["payload"].apply(lambda p: p["val"] if isinstance(p, dict) else 0).sum()
    ...         return CounterState(count=prev_val + total_inc)
    >>>
    >>> # 1. Prepare data for two steps
    >>> ledger.insert(Increment(step=1, val=10), attributes={"step": 1})
    UUID(...)
    >>> ledger.insert(Increment(step=2, val=5), attributes={"step": 2})
    UUID(...)
    >>>
    >>> # 2. Project the trajectory
    >>> with Session(ledger) as sess:
    ...     counter = SequentialCounter(identity="demo_counter")
    ...     trajectory = sess.read(counter).data
    >>> len(trajectory)
    2
    >>> trajectory[0]
    (1, CounterState(count=10))
    >>> trajectory[1]
    (2, CounterState(count=15))
"""
