# EarlySign: Event-Sourced Sequential Testing Framework (Explanation)

This document explains the core architectural concepts and design principles behind EarlySign, a framework for sequential statistical testing that employs an **event-sourcing pattern** to ensure reproducible, auditable, and composable experimental analysis.

## 1. Event-Sourcing Architecture

EarlySign is built around the **event-sourcing pattern**, a powerful architectural approach where:

- **All changes are captured as events**: Rather than storing current state, we record a complete history of domain events
- **State is derived from events**: Current state is reconstructed by replaying the event sequence
- **Immutable event log**: Events are append-only and never modified, ensuring complete auditability
- **Temporal reasoning**: We can examine the system state at any point in time or replay decisions

### 1.1 The Ledger as Event Store

The **ledger** serves as our event store—a single, authoritative record of all events that have occurred in sequential experiments:

- **Single source of truth**: All facts are appended to the ledger (append-only)
- **Complete event history**: Every observation, statistic computation, decision boundary update, and stopping signal is recorded as an event
- **Temporal ordering**: Events maintain strict ordering through `time_index` and timestamps
- **Event identity**: Each event has a unique UUID for precise identification across distributed scenarios

### 1.2 Event Structure and Metadata

Each event in the ledger follows a consistent structure with rich metadata:

**Core Event Fields:**
- `uuid`: Unique identifier for the event
- `time_index`: Logical time ordering (e.g., "t001", "t002")
- `ts`: Physical timestamp (ISO 8601 format)
- `namespace`: Domain category (obs, stats, criteria, signals, etc.)
- `kind`: Event type within namespace (registered, updated, emitted)
- `entity`: Experiment identifier this event belongs to
- `snapshot_id`: State snapshot identifier
- `tag`: Semantic labeling for event querying and fusion

**Event Payload:**
- `payload_type`: Type identifier for the event data
- `payload`: JSON-serialized event data with domain-specific structure

### 1.3 Decoupled Event Processing

The framework employs **decoupled event processing** where:

- **Events are protocol-agnostic**: Raw events stored as JSON can be decoded into typed domain objects as needed
- **Flexible interpretation**: The same event can be interpreted differently by different components
- **Version tolerance**: Event structures can evolve while maintaining backward compatibility
- **Dynamic querying**: Events can be filtered and aggregated using tags and metadata without affecting the core event log
## 2. Domain Event Types and Lifecycle

Sequential experiments in EarlySign follow a well-defined event lifecycle, with events categorized into distinct domains:

### 2.1 Design Events
- **Purpose**: Register experimental design parameters and methodology
- **Namespace**: `design`
- **When**: At experiment initialization
- **Examples**: Significance levels, power requirements, spending functions, sample size calculations
- **Immutability**: Design events establish the experimental protocol and should not change during execution

### 2.2 Observation Events
- **Purpose**: Record raw experimental data as it becomes available
- **Namespace**: `obs`
- **When**: Each time new data is collected (batch or streaming)
- **Examples**: Treatment/control group counts, individual measurements, survival times
- **Aggregation**: Multiple observation events can be reduced to sufficient statistics

### 2.3 Statistic Events
- **Purpose**: Record computed test statistics based on accumulated observations
- **Namespace**: `stats`
- **When**: After new observations are processed
- **Examples**: Z-statistics, log-likelihood ratios, e-values, Bayes factors
- **Dependencies**: Derived from current observation state and design parameters

### 2.4 Criteria Events
- **Purpose**: Update decision boundaries and stopping rules
- **Namespace**: `criteria`
- **When**: Computed alongside or after statistics
- **Examples**: Group sequential boundaries, e-value thresholds, posterior probability criteria
- **Adaptivity**: Can incorporate information accrual and spending function updates

### 2.5 Signal Events
- **Purpose**: Emit actionable decisions (stop/continue/modify)
- **Namespace**: `signals`
- **When**: When criteria are met or external triggers occur
- **Examples**: Stopping for efficacy, futility, safety; sample size re-estimation signals
- **Auditability**: Captures the complete decision rationale and supporting evidence

### 2.6 Lifecycle Events
- **Purpose**: Track runtime and orchestration metadata
- **Namespace**: `lifecycle`
- **When**: At experiment start/stop and major state transitions
- **Examples**: Runtime initialization, component execution timing, error handling
- **Operational**: Supports debugging, performance monitoring, and compliance

## 3. Component Architecture and Event Flow

EarlySign employs a **component-based architecture** where atomic, composable components process events in a coordinated fashion:

### 3.1 Atomic Components

Each component type follows the **single responsibility principle** and operates through well-defined event interfaces:

**Statistic Components**
- Read: Design + observation events
- Write: Statistic events
- Responsibility: Transform raw data into inferential statistics
- Examples: `WaldZStatistic`, `LogLikelihoodRatio`, `BetaBinomialEValue`

**Criteria Components**
- Read: Design + statistic events
- Write: Criteria events
- Responsibility: Compute decision boundaries and stopping rules
- Examples: `LanDeMetsBoundary`, `EValueThreshold`, `PosteriorProbability`

**Signaler Components**
- Read: Statistic + criteria events
- Write: Signal events
- Responsibility: Apply stopping logic and emit decisions
- Examples: `PeekSignaler`, `FutilitySignaler`, `SafetySignaler`

### 3.2 Event-Driven Coordination

Components coordinate through **event-driven messaging**:

1. **Loose coupling**: Components only know about event schemas, not other components
2. **Temporal consistency**: Event ordering ensures components see consistent state snapshots
3. **Replayability**: Component execution can be replayed from any point in the event log
4. **Composability**: Different combinations of components can be orchestrated without code changes

### 3.3 Runtime Orchestration

**Runtime components** manage the overall event flow and component coordination:

- **Sequential execution**: Ensures proper ordering of statistic → criteria → signal computation
- **Parallel execution**: Allows independent components to process events concurrently when safe
- **Error handling**: Captures failures as events for debugging and recovery
- **Resource management**: Coordinates backend resources and manages computational state

## 4. Event Store Implementation

### 4.1 Backend Agnosticism

EarlySign's event store is **backend-agnostic**, supporting various storage implementations:

- **In-memory**: For prototyping, testing, and lightweight scenarios
- **File-based**: Using Parquet, CSV, or JSON formats for persistence
- **Database**: SQL or NoSQL databases for enterprise deployment
- **Distributed**: Event streaming platforms like Kafka for high-throughput scenarios

### 4.2 Storage Schema and Querying

**Event Record Schema:**

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `uuid` | String | Unique event identifier | `550e8400-e29b-41d4-a716-446655440000` |
| `time_index` | String | Logical ordering | `t001`, `t002`, `t003` |
| `ts` | Timestamp | Physical time | `2025-09-07T10:00:00Z` |
| `namespace` | String | Event domain | `obs`, `stats`, `criteria`, `signals` |
| `kind` | String | Event type | `registered`, `updated`, `emitted` |
| `entity` | String | Experiment ID | `exp#42`, `study_001` |
| `snapshot_id` | String | State snapshot | `design-v1`, `snap-001` |
| `tag` | String | Semantic label | `stat:waldz`, `crit:gst`, `obs:batch` |
| `payload_type` | String | Data schema | `WaldZ`, `GSTBoundary`, `TwoPropObs` |
| `payload` | JSON | Event data | `{"z": 2.10, "se": 0.45, "nA": 10, "nB": 10}` |

**Sample Event Log:**

| uuid | time_index | ts | namespace | kind | entity | snapshot_id | tag | payload_type | payload |
|------|------------|----|-----------|----|--------|-------------|-----|-------------|---------|
| 550e8400-e29b-41d4-a716-446655440000  | t001       | 2025-09-07T10:00:00Z      | design    | registered  | exp#42 | design-v1   | design        | GSTTwoPropDesign      | {"design_id":"exp#42-design","version":1,"alpha":0.025,"spending":"obrien","sides":"two"}   |
| 550e8400-e29b-41d4-a716-446655440001  | t002       | 2025-09-07T10:05:00Z      | obs       | observation | exp#42 | snap-001    | obs           | TwoSampleBinomialObs  | {"nA":10,"nB":10,"mA":8,"mB":1}                                                              |
| 550e8400-e29b-41d4-a716-446655440002  | t003       | 2025-09-07T10:06:00Z      | stats     | updated     | exp#42 | snap-001    | stat:waldz    | WaldZ                 | {"z":2.10,"se":0.45,"nA":10,"nB":10,"mA":8,"mB":1}                                          |
| 550e8400-e29b-41d4-a716-446655440003  | t004       | 2025-09-07T10:06:01Z      | criteria  | updated     | exp#42 | crit-001    | crit:gst      | GSTBoundary           | {"upper":1.96,"lower":-1.96,"info_time":0.35,"design_ref":{"design_id":"exp#42-design","version":1}} |
| 550e8400-e29b-41d4-a716-446655440004  | t005       | 2025-09-07T10:06:02Z      | signals   | emitted     | exp#42 | sig-001     | gst:decision  | Signal                | {"topic":"gst:decision","body":{"action":"stop","z":2.10}}                                  |
| 550e8400-e29b-41d4-a716-446655440005  | t006       | 2025-09-07T10:07:00Z      | lifecycle | lifecycle   | exp#42 | run-001     | runtime       | LifecycleStart        | {"runtime":"GroupSequentialRuntime","status":"start"}                                       |
| 550e8400-e29b-41d4-a716-446655440006  | t007       | 2025-09-07T10:08:00Z      | lifecycle | lifecycle   | exp#42 | run-001     | runtime       | LifecycleStop         | {"runtime":"GroupSequentialRuntime","status":"ok"}                                          |

---

## 3. Spec Tables (Protocol Design)

### 3.1 Ledger

| Aspect        | Spec                                                                 |
|---------------|----------------------------------------------------------------------|
| Purpose       | Store all facts (Observation / Statistic / Criteria / Signal / Design / Notes / Lifecycle) as append-only records |
| Write API     | `append(...)`, `emit_signal(...)`, `lifecycle_start(...)`, `lifecycle_stop(...)` |
| Read API      | `reader().iter(...)`, `reader().latest(...)`, `reader().count(...)` (with typed decoding via `as_=`) |
| Record Schema | Metadata header + `payload_type` + `payload(JSON)`                   |
| Tag           | `tag: Optional[str]` used for fusion, query, or separation of streams |
| Guarantees    | Append-only, total order, reproducibility, random UUID per record    |
| Notes         | Payloads should be minimal but sufficient for recomputation          |

---

### 3.2 Atomic Components

Atomic classes directly **read/write** the ledger. They are first-class citizens invoked by runtimes.

| Layer       | Protocol | Method | Inputs | Reads (Ledger) | Writes (Ledger) | Example Tag | Example Payload |
|-------------|----------|--------|--------|----------------|-----------------|-------------|-----------------|
| **Design**  | Register design | `register(ledger)` | design params (α, spending, sides, …) | latest design (idempotent check) | `("design","registered")` | `design:{id,ver}` | `{"alpha":0.025,"spending":"obrien"}` |
| **Observation** | Register data | `step(ledger, data)` | domain-specific obs (arm,y / counts) | – | `("observation","registered")` | `obs` | `{"nA":10,"nB":10,"mA":8,"mB":1}` |
| **Statistic** | Update statistic | `step(ledger, data)` | – | latest design, obs | `("statistic","updated")` | `stat:waldz` | `{"z":2.1,"se":0.45}` |
| **Criteria** | Update boundary/rule | `step(ledger, data)` | – | design, info_time, stat | `("criteria","updated")` | `crit:gst` | `{"upper":1.96,"lower":-1.96}` |
| **Signal**  | Emit signal | `step(ledger, data)` | – | stat, criteria | `("signal","emitted")` | `gst:decision` | `{"action":"stop","reason":{...}}` |

---

### 3.3 Runtimes

| Aspect    | Spec                                                                 |
|-----------|----------------------------------------------------------------------|
| Role      | Orchestrator: executes Atomic steps in sequence/parallel as a Plan   |
| Interface | `run(ledger, data)`                                                  |
| Behavior  | Receives input data → runs `Observation → Statistic → Criteria → Signal` steps |
| Examples  | GroupSequentialRuntime, AnytimeValidRuntime, composite runtimes      |
| Notes     | Runtimes themselves don’t write to ledger (except lifecycle start/stop) |

---

## 4. Example Flow (Two-Sample Binomial, GST)

1. **Design registered** → `("design","registered")`
2. **Observation appended** → `("observation","registered")`
3. **Statistic updated** (Wald Z) → `("statistic","updated")`
4. **Criteria updated** (boundary) → `("criteria","updated")`
5. **Signal emitted** (stop decision) → `("signal","emitted")`
6. **Lifecycle events** (runtime start/stop) → `("lifecycle","lifecycle")`

All facts are preserved in the ledger for **auditability, reproducibility, and reporting**.
