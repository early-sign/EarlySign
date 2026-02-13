"""
This module provides a comprehensive introduction to the EarlySign framework through functional doctests.

--- Setup ---
>>> import ibis
>>> from pydantic import BaseModel
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.v1.framework.session import Session
>>> from earlysign.v1.framework.projector import ProtocolProjector
>>> from earlysign.v1.framework.trace import Traced, TraceId
>>> from earlysign.v1.methods.binomial import Scoreboard
>>> from earlysign.schema.ES3.Binomial import ArmData
>>> from earlysign.v1.templates.GST_Spending_JennisonTurnbull2000 import JennisonTurnbull2000Template as BinomialABTemplate, JennisonTurnbull2000TaskSpec as BinomialABTaskSpec
>>> from earlysign.v1.methods.group_sequential.execution.binomial import BinomialGSTEngine
>>> import earlysign.schema.ES3.Base as ES3_BASE
>>> import earlysign.schema.ES3.GST as GST
>>> from earlysign.schema.ES3.GST.Log import DecisionStatus

Allow more columns in displaying pandas dataframes
>>> import pandas as pd
>>> pd.set_option('display.max_columns', None)
>>> pd.set_option('display.width', 1000)

# Initialize an in-memory ledger for demonstration purposes for all subsequent examples
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events"); ledger.ensure()

# Core features
## Ledger Read/Write
The Ledger is a low-level, append-only event store. You can record any Pydantic model
to capture domain events or manual decisions without using the full framework.

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

## ibis-framework
Since the Ledger is backed by Ibis, you can perform powerful queries using
standard Ibis expressions.

>>> table = ledger.t
>>> table.filter(table.type == "MyDecision").payload["action"].execute().tolist()
['CONTINUE']

>>> # ledger.show() provides a quick summary view of the ledger
>>> ledger.show()
         type identity trace                                            payload attributes
0  MyDecision     None  None  {'action': 'CONTINUE', 'reason': 'Non-binding ...         {}

## Ledger Binding
Ledgers can be "bound" to attributes, creating scoped views. Reads and writes
automatically apply the bound attributes. Use `unbind()` to remove attributes.

>>> exp_ledger = ledger.bind(experiment_id="EXP001")
>>> # Every record has a unique id and timestamp (timestamp)
>>> df = ledger.t.execute()
>>> 'uuid' in df.columns and 'timestamp' in df.columns
True
>>> len(df.iloc[0]['uuid']) == 32  # hex uuid
True

>>> # Records inserted with one binding don't appear in a differently-bound ledger
>>> other_ledger = ledger.bind(experiment_id="OTHER")
>>> other_ledger.insert(data=MyDecision(action="OTHER", reason="scoped")) # Insert something to be sure
UUID(...)
>>> len(other_ledger.t.filter(other_ledger.t.type == "OTHER").execute()) # Assuming you might want to check
0

# Framework features
## Write Model
The Framework provides `Session.commit` to record events with scientific lineage.
`sess.commit` records a model, while `sess.call_and_commit` records the result of a function.

>>> with Session(ledger) as sess:
...    sess.commit(MyDecision(action="STOP", reason="Safety concern"))
UUID(...)

## Projector
Projectors are "State-as-a-Fold" operators. They reconstruct high-level facts
from the event stream.

>>> from earlysign.v1.framework.projector import ProjectionResult
>>> from earlysign.v1.framework.projector import ProjectionResult
>>> class DecisionProjector:
...     def project(self, table):
...         # Reify the table to get a concrete result with trace
...         match = table.filter(table.type == "MyDecision").order_by(ibis.desc("timestamp")).limit(1).execute()
...         if match.empty: return ProjectionResult(data=None, trace=[])
...         return ProjectionResult(data=match.iloc[0]["payload"]["action"], trace=[TraceId(str(match.iloc[0]["uuid"]))])

>>> with Session(ledger) as sess:
...     latest_action = sess.read(DecisionProjector())
>>> latest_action.data
'STOP'

## Session (Horizon)
A Session defines a "Scientific Horizon"—a point-in-time snapshot of the ledger.
Analysis within a session is protected from concurrent writes.

>>> with Session(ledger) as sess:
...     # 1. Write something outside (directly to ledger) AFTER session started
...     _ = ledger.insert(data=MyDecision(action="EXTERNAL_B", reason="outside"))
...     # 2. Projection within session only sees records up to the horizon (e.g., 'STOP')
...     res = sess.read(DecisionProjector())
>>> res.data
'STOP'

## Session (Local Visibility)
A Session allows reading its own committed data even while the Scientific Horizon
is fixed for external data. This allows multi-step updates within a single session.

>>> with Session(ledger) as sess:
...     # 1. Commit something within the session
...     sess.commit(MyDecision(action="LOCAL_A", reason="within session"))
...     # 2. Directly insert something in the ledger (simulating external write)
...     ledger.insert(data=MyDecision(action="EXTERNAL_C", reason="external concurrent"))
...     # 3. Read should see LOCAL_A but NOT EXTERNAL_C
...     res = sess.read(DecisionProjector())
UUID(...)
>>> res.data
'LOCAL_A'

## Trace
Scientific Lineage (Trace) is automatically accumulated as you Read data in a Session.

>>> with Session(ledger) as sess:
...     # Reading records their causal IDs in the session trace
...     res = sess.read(DecisionProjector())
...     len(sess.trace) > 0
True

## Entity
Entities are special aggregates with identity. `Entity` supports differential folding.

>>> fact = Scoreboard(identity="metrics")
>>> # Pre-populate data in a separate session so it's visible in the next horizon
>>> with Session(ledger) as sess:
...     sess.commit(ArmData(n=10, success=2, arm="A"))
UUID(...)

>>> with Session(ledger) as sess:
...     state = sess.read(fact)
>>> state.data.arms["A"].metrics.n
10

## Sequential Entity
Sequential Entities evolve over time (e.g., Test Statistics, Spending Boundaries).
They are projected similarly but represent a path of decisions.

## Predefined Entities
The library provides off-the-shelf entities for common trial components.

>>> # Experiment State (via summary facts), Boundaries, and Test Statistics
>>> protocol = GST.Protocol(
...     name="Example Trial",
...     task=GST.TaskSpec(
...         kind="group_sequential",
...         arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
...         response_type=GST.ResponseType.BINARY,
...         efficacy=GST.EfficacyRequirement(alpha=0.05),
...         hypotheses=GST.HypothesisSpec(
...             h_null_description="p_t <= p_c", h_alt_description="p_t > p_c",
...             test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
...             target_effect=GST.BinaryEffectSize(proportions={"control": 0.2, "treatment": 0.3})
...         )
...     ),
...     method=GST.MethodSpec(
...         kind="group_sequential",
...         stopping_policy=GST.StoppingPolicySpec(
...             statistic=GST.TwoArmBinomialZ(variance_estimation=GST.VarianceEstimation.POOLED),
...             strategy=GST.AlphaSpendingStrategy(
...                 spending_fn=GST.SpendingFunction(family="obrien_fleming"),
...                 budget=0.05,
...                 sided=GST.Sided.ONE,
...                 statistical_model=GST.CanonicalGaussianModel(),
...             ),
...             timer=GST.SampleSizeTimer(unit=GST.Unit.INDIVIDUALS, max_sample_size=100),
...             schedule=GST.FixedSchedule(analyses=[0.5, 1.0])
...         ),
...     )
... )
>>> engine = BinomialGSTEngine(protocol=protocol)
>>> # Projection of boundary at 50% info time
>>> engine.get_boundary_at_look(0, 0.5)
2.3261743106419144

## ES3 Schema
### Protocol
ES3 provides a standardized schema for trial protocols.
>>> protocol.name
'Example Trial'

# Preset Templates
Templates provide a high-level API for running standard trial designs.

## GST Design, Backtest, Run, Report
>>> template = BinomialABTemplate(ledger)
>>> task = BinomialABTaskSpec(
...     arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
...     efficacy=GST.EfficacyRequirement(alpha=0.05),
...     futility=GST.FutilityRequirement(power=0.8),
...     hypotheses=GST.HypothesisSpec(
...         h_null_description="p_t <= p_c", h_alt_description="p_t > p_c",
...         test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
...         target_effect=GST.BinaryEffectSize(proportions={"control": 0.2, "treatment": 0.3})
...     )
... )
>>> # Design
>>> protocol = template.design(
...     task=task, looks=2, alpha=0.05, power=0.8, spending_function="obrien_fleming",
...     designer_params={"model": "canonical_joint", "model_params": {"rng_seed": 42}}
... )
>>> template.set_protocol(protocol)
>>> # Update & Report
>>> batch = [ArmData(n=100, success=25, arm="control"), ArmData(n=100, success=35, arm="treatment")]
>>> template.update(batch)
>>> report = template.report_progress()
>>> report['status']
'continue'
"""
