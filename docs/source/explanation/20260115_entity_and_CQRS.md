# Optimistic Concurrency Control (OCC) in Event Store

EarlySign's development framework supports defining a special type of aggregate, namely Entities.
In the Event Sourcing pattern, there are no first-class entities, but all constructs are derived from events.
However, for efficiency of computation, we support the concept of entities as special types of aggregates having consistent identity.
Not all aggregates appearing in sequential procedures are entities.

Under Command Query Responsibility Segregation (CQRS), the computation of an entity's state may occur without locks since they stem from projections of events.
Nevertheless, we can ensure the consistency of entities by using the Optimistic Concurrency Control (OCC) pattern. The key is to use the version number of the entity as a marker of its state.

## 1. Purpose
This document defines the implementation guidelines for ensuring the identity and consistency of Entities (Aggregates) within a CQRS/Event Sourcing architecture. It physically prevents "data forking" when multiple processes attempt to update the same Entity simultaneously, ensuring that a single source of truth (Canonical State) is always maintained.

## 2. Core Concept: Expected Version (Sequence Number)
Each Entity (Event Stream) is assigned a continuous **Version Number (Sequence Number)** starting from `0`.

Updates must follow these "Verification Rules":
1. **Read**: The client fetches the current events along with the latest version number (e.g., `V5`).
2. **Write Request**: The client sends the new event along with an `expected_version = 5`, proving the calculation was based on `V5`.
3. **Server-side Check**: 
   - If the current version on the server is indeed `5`, the write is permitted, and the version is incremented to `6`.
   - If someone else has already updated it to `6` or higher, the server **rejects the write (throws an exception)** due to a version mismatch.

## 3. Database Design (SQL Example)
When using an RDB, a composite unique constraint on `stream_id` and `version` is used to physically enforce architectural consistency.

```sql
CREATE TABLE events (
    id UUID PRIMARY KEY,
    stream_id UUID NOT NULL,       -- Identifier for the Entity
    version INT NOT NULL,          -- Continuous sequence number (0, 1, 2, ...)
    event_type TEXT NOT NULL,      -- Name/Type of the event
    payload JSONB NOT NULL,        -- Event data payload
    created_at TIMESTAMP NOT NULL,
    
    -- Physically prevents duplicate version numbers within the same Entity
    UNIQUE (stream_id, version)
);
```

## 4. Client-side (Command Handler) Implementation Pattern
The writing side should implement a "Retry (Reload)" strategy to handle conflicts.

```python
def handle_command(command):
    while True:  # Loop to retry on conflict
        # 1. Load the current stream and identify the version
        events = event_store.load_stream(command.entity_id)
        current_version = events[-1].version if events else -1
        
        try:
            # 2. Execute business logic based on the current state
            # (e.g., statistical calculations involving external dependencies)
            new_event_payload = domain_logic.calculate(events, command)
            
            # 3. Write with the expected version
            # current_version must match the DB state at the moment of execution
            event_store.append_event(
                stream_id=command.entity_id,
                expected_version=current_version,
                event=Event(
                    version=current_version + 1, 
                    payload=new_event_payload
                )
            )
            break  # Success: exit the loop
            
        except ConcurrencyError:
            # Conflict: another process updated first. Reload and try again.
            continue 
```

## 5. Server-side (Event Store) Writing Constraints
The Event Store `append` operation performs the version check within an atomic transaction.

```python
def append_event(self, stream_id, expected_version, event):
    with db.transaction():
        # Identify the current latest version in the DB
        # (via SELECT MAX... or a dedicated stream management table)
        actual_version = db.query(
            "SELECT MAX(version) FROM events WHERE stream_id = :id", 
            id=stream_id
        ).scalar()
        
        if actual_version is None:
            actual_version = -1
        
        # If expected != actual, a fork has occurred
        if actual_version != expected_version:
            raise ConcurrencyError("Conflict: The entity has been updated by another process.")
            
        # Save only if the check passes
        db.insert(event)
```

## 6. Key Benefits
- **Absolute Protection of Identity**: Regardless of parallel requests, events are always "linearized" at the point of entry into the Event Store.
- **Reliability of Canonical Models**: The "Shared Read Model (Canonical Shadow)" generated from these linearized events will always represent a single, consistent truth.
- **Computational Validity**: Ensures that updates to data with strong inter-dependencies (e.g., accumulated statistical values or intermediate states) are never inconsistent.

---
*Note: Since this implementation uses "Optimistic Concurrency Control," it is recommended to use partitioning (routing events for specific IDs to specific servers) in high-load environments to minimize conflict frequency.*

## 7. Sequential Entities
A **Sequential Entity** is a specialized aggregate for sequential procedures (e.g., sequential testing) where the state is indexed by a temporal or ordinal coordinate (the **Index**). Unlike a standard entity, the Sequential Entity must ensure the monotonicity of its information trajectory.

### 7.1 Intrinsic Indexing
The `index` (e.g., Look $N$ or Sample $N$) is a first-class property of the Sequential Entity's snapshot. The transition $S_n \rightarrow S_{n+1}$ is governed by a **Sequential-OCC** pattern where the database ensures that the next index is strictly successor to the current one, preventing gaps or out-of-order computations.

### 7.2 History Retention Strategies
Sequential Entities support two canonical modes for maintaining the trajectory (history) of values. The choice depends on the trade-off between update frequency and read requirements.

#### Strategy A: Projective Mode (Lean Snapshot)
- **Concept**: Stores the value at the current index and the **Sufficient Statistics** required for the next computation.
- **Snapshot Composition**: $(v_n, \theta_n)$ where $\theta$ represents the cumulative information required to compute $S_{n+1}$ without re-reading the whole history.
- **Trajectory Reconstruction**: To view the full path, the system collects all previous index records.
- **Best Use Case**: High-frequency real-time updates where storage overhead is a concern.

#### Strategy B: Cumulative Mode (Fat Snapshot)
- **Concept**: Stores the entire history of values up to the current index, alongside the Sufficient Statistics.
- **Snapshot Composition**: $(\{v_i\}_{i=1}^n, \theta_n)$.
- **Trajectory Reconstruction**: The latest snapshot provides the full trajectory instantly without extra lookups.
- **Best Use Case**: Low-to-moderate frequency updates (e.g., Group Sequential Design) where instant visualization of the trajectory is required.

### 7.3 Summary of Projections
Sequential Entities provide two canonical projections:
1. **Latest State Projection**: The head of the snapshot, providing the current value and statistical status.
2. **Trajectory Projection**: The history of the process, retrieved via collection (Projective Mode) or direct reading (Cumulative Mode).

```json
# Example of a Sequential Snapshot structure
{
    "entity_id": "uuid",
    "index": 12,
    "strategy": "collective", 
    "data": {
        "current_value": 2.45,
        "sufficient_stats": { "sum_x": 120.5, "n": 100 },
        "history": [1.1, 1.5, ..., 2.45] # Empty if PROJECTIVE
    }
}
```