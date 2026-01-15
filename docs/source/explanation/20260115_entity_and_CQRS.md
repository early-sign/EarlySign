# Optimistic Concurrency Control (OCC) in Event Store

EarlySign's development framework supports defining special types of aggregates, namely Entities.
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
