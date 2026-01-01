# Implementation Plan: Pattern G (EarlySign v1) - The Scientific Standard

This plan outlines the definitive implementation of "Pattern G" within the `earlysign/v1/` directory. Pattern G untangles complexity by strictly separating scientific concerns, focusing on technical robustness and a natural domain language for sequential inference.

## User Review Required

> [!IMPORTANT]
> The implementation will be strictly confined to `earlysign/v1/` to avoid collisions with the existing "v0" codebase. We will leverage `earlysign/core` as the foundational layer.

## Proposed Changes

### 1. Framework Foundations (`earlysign/v1/framework/`)

The framework provides the low-level ES/CQRS primitives required for scientific rigor.

#### [NEW] trace.py (Causality & Hashing)
- **`TraceHash = NewType("TraceHash", str)`**: Type safety for identifiers.
- **`Traced[T]`**: Dataclass wrapping `data: T` and `trace: List[TraceHash]`.
- **`stable_hash(*args)`**: Deterministic SHA-256 for lineage and idempotency.

#### [NEW] projector.py (Read Side: State-as-a-Fold)
- **`ProjectionResult[T]`**: Dataclass container for hydrated `data` and its evidentiary `trace`.
- **`Projector[T]` Protocol**: Defines the `project(self, data: ibis.Expr) -> ProjectionResult[T]` interface.
- **Ibis Integration**: The `project` method receives an Ibis table (filtered by the scientific horizon) to perform high-performance server-side aggregation.
- **Incremental Projection**: Implementers are responsible for their own state management, potentially leveraging internal snapshots.

#### [NEW] snapshot.py (Optimization)
- **Mechanism**: Implements intermediate state persistence for Event Sourcing.
- **Usage**: Projectors utilize these snapshots to resume aggregation from the last valid checkpoint, essential for verifying $O(N)$ recovery performance.

#### [NEW] session.py (Context Isolation)
- **`Session`**: A context manager (`with Session(ledger) as ep:`) that captures the ledger's latest state at initiation.
- **Horizon Isolation**: Ensures all `Read` operations within the session are mutually consistent.
- **`_session_trace`**: Manages the "Implicit Trace Accumulation" for all reads.

#### [NEW] write_models.py (Write Side: Primitives)
The fundamental operations for asserting facts into the Ledger.
- **`Commit(session, record: BaseModel, trace=None)`**:
    - **Pydantic Integration**: Accepts a Pydantic `BaseModel`. Uses `record.model_dump()` for payload and its class name for schema tracking.
- **`CommitCallResult(session, result_type: Type[BaseModel], func, *args, **kwargs)`**:
    - **Result Schema**: The `result_type` MUST be a Pydantic `BaseModel` subclass.
    - **Scientific Provenance**: Skips `func` execution if a matching `trace_hash` exists in the Ledger.

---

### 2. Domain Layer & Migration Strategy (`earlysign/v1/methods/`)

Bridging the framework to the ubiquitous language of sequential statistical work.

#### [NEW] actions.py (Ubiquitous Language Writers)
Domain-specific wrappers built on Pydantic models.
- **`Decision(session, decision: BaseModel, trace=None)`**: semantic operational conclusion.
- **`UpdateProtocol(session, protocol: BaseModel, trace=None)`**: structural design adaptation.
- **`Ingest(session, batch: ObservationBatch)`**: raw evidence persistence. `ObservationBatch` is a typed Pydantic record.

#### Protocol Definitions (`protocols.py`)
All protocols are implemented as **Pydantic `BaseModel`** subclasses to ensure schema integrity.
- **`group_sequential/protocols.py`**: `GSTProtocol` (spending, alpha, K, etc.).
- **`anytime_valid/protocols.py`**: `EProcessProtocol` (null/alternative hypotheses, alpha).

#### Ported Logic (v1-First)
High-quality statistical logic ported to become the authoritative v1 version.
- `group_sequential/`: `spending.py`, `canonical_dist.py`, `boundary.py`.
- `anytime_valid/`: `e_process.py`.
- `stats/binary.py`: Ibis-based aggregations (Projectors with Snapshot support).

---

### 3. Workflow Templates (`earlysign/v1/templates/`)

High-level pre-composed patterns for end-users.

- **`binomial_ab.py`**: `BinomialABTemplate` with `set_protocol()` and `update()`.
- **`binomial_monitoring.py`**: `BinomialMonitoringTemplate` for e-process monitoring.

---

### 4. Quality Standards & Verification Plan

#### Full-Stack Verification (`verify_v1_asos.py`)
The verification script will exercise the **entire Pattern G stack**:
1.  **Ingest**: Load ASOS data incrementally (`Ingest`).
2.  **Setup**: Define and persist the trial rules (`UpdateProtocol` with `GSTProtocol`).
3.  **Read**: Execute `Read` using `BinomialSummaryProjector`, demonstrating both automatic trace accumulation and **Snapshot-based recovery** (by re-running the script).
4.  **Compute**: Use `CommitCallResult` to solve boundaries and test statistics, verifying skip-logic on the second run.
5.  **Act**: Issue a `Decision` based on the results and verify it's only appended once.

#### Standard Requirements
- **`doctest` Integration**: Every public method MUST have a `doctest` showcasing standard usage.
- **Scientific Docstrings**: Explain the rationale for each operation.
