# Implementation Plan: Pattern G (EarlySign v1) - The Scientific Standard

This plan outlines the definitive implementation of "Pattern G" within the `earlysign/v1/` directory. For the foundational design philosophy, see the [Concept Design](file:///Users/teshima/2025/EarlySign/docs/explanation/concept_design.md).

> [!IMPORTANT]
> The implementation is strictly confined to `earlysign/v1/` to avoid collisions with the existing "v0" codebase. We leverage `earlysign/core` as the foundational layer.

## Trace: Scientific Provenance

**Trace** is the mechanism for tracking causality and scientific lineage in the framework.

### Design Principles

1. **Trace = List of Parent UUIDs**: Each record's trace is a list of Ledger `uuid` values representing the parent records that contributed to its creation.
2. **No Hashing**: Traces use raw UUIDs directly. No cryptographic hashing is involved.
3. **Immutability Guarantees Acyclicity**: The Ledger is append-only. Records can only reference previously existing records, so circular references are impossible.
4. **First-Class Column**: The `trace` column is a top-level field in the Ledger schema (stored as JSON-serialized string for backend compatibility).

### Why Write Operations Return Nothing

In Event Sourcing, all "state" is derived from the Ledger via projections.
Therefore, any variable holding computed results must be generated via Read from the Ledger.

```python
# ❌ Wrong design: managing dependencies via Commit return value
uuid = Writer.Commit(sess, my_record)
next_result = some_function(uuid)  # passing uuid around

# ✅ Correct design: all state comes from Read
Writer.Commit(sess, my_record)  # just record, no return
traced_data = sess.Read(MyProjector())  # get dependencies via Read
# traced_data.trace contains the list of parent uuids
```

**Write** = record events (fire and forget)
**Read** = reconstruct state (always returns Traced[T] with dependencies)

When you call Read within a Session, the projection's trace accumulates in `session.trace`.
When you call Commit without an explicit trace, this accumulated trace is used automatically.

### Key Components

| Component | Role |
|-----------|------|
| `TraceId` | Typed alias for `str` (NewType) representing a Ledger uuid |
| `Traced[T]` | Container wrapping `data: T` with `trace: List[TraceId]` |
| `trace` column | Ledger column storing JSON array of parent uuids |

### Trace Flow

```
[Root Record]              trace: []
       ↓
[Derived Record A]         trace: [uuid_of_root]
       ↓
[Derived Record B]         trace: [uuid_of_A, uuid_of_root]
```

---

## Proposed Changes

### 1. Framework Foundations (`earlysign/v1/framework/`)

#### trace.py (Causality Primitives)
- **`TraceId = NewType("TraceId", str)`**: Type safety for Ledger uuids.
- **`Traced[T]`**: Dataclass wrapping `data: T` and `trace: List[TraceId]`.
- **`extract_traces(*args)`**: Recursively extracts TraceIds from Traced containers.

#### projector.py (Read Side: State-as-a-Fold)
- **`ProjectionResult[T]`**: Container for hydrated `data` and its evidentiary `trace`.
- **`Projector[T]` Protocol**: Defines `project(data: ibis.Expr) -> ProjectionResult[T]`.
- **Trace Extraction**: Projectors extract `uuid` from Ledger rows for trace building.

#### session.py (Context Isolation)
- **`Session`**: Context manager capturing ledger state at initiation.
- **`_session_trace`**: Accumulates TraceIds from all Read operations.
- **`Read(projector)`**: Executes projection and accumulates trace.

#### writer.py (Write Side: Primitives)
- **`Commit(session, record, trace=None)`**: Records a Pydantic model with its parent trace. Returns the new record's uuid as TraceId.
- **`CallAndCommit(session, result_type, func, *args)`**: Executes function and commits result with lineage from input traces.

---

### 2. Domain Layer (`earlysign/v1/methods/`)

#### actions.py (Ubiquitous Language Writers)
- **`Decision(session, decision, trace=None)`**: Records operational conclusion.
- **`UpdateProtocol(session, protocol, trace=None)`**: Records design adaptation.
- **`Ingest(session, batch)`**: Records raw evidence with empty trace (root).

---

### 3. Ledger Schema (`earlysign/core/ledger.py`)

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | string | Auto-generated unique identifier |
| `ts` | timestamp | UTC timestamp |
| `trace` | string | JSON array of parent uuids |
| `payload_type` | string | Record type name |
| `payload` | json | Record content |
| `labels` | json | Metadata/scope labels |
| `identity` | string | State/stream identity |

The `insert()` method accepts a `trace` parameter and returns the generated `uuid`.

---

### 4. Quality Standards

- **`doctest` Integration**: Every public method has a doctest.
- **Scientific Docstrings**: Explain rationale for each operation.
