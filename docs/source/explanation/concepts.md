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
- Examples: `BinomialWaldZ`, `LogLikelihoodRatio`, `BetaBinomialEValue`

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

### 3.4 Custom Templates for Domain-Specific Workflows

When you need a **custom combination** of parameters, effect size definitions, statistics, or stopping rules that isn't covered by the built-in templates, you can create your own **custom Template** by inheriting from the base class.

**Why Create Custom Templates?**

- **Domain-specific requirements**: Your field may have specialized effect size definitions (e.g., clinical meaningful difference, business impact metrics, survival hazards)
- **Custom statistics**: You might need non-standard test statistics (e.g., rank-based tests, variance-weighted combinations, Bayesian posteriors)
- **Specialized stopping rules**: Your experimental context may require unique stopping logic (e.g., regulatory constraints, multi-arm rules, futility boundaries with specific thresholds)
- **Team standardization**: Encode your organization's experimental protocols into reusable, shareable templates

**Benefits of Custom Templates:**

1. **Portability**: Templates are self-contained files that can be shared across teams and projects
2. **Reproducibility**: Complete experimental protocol is captured in code, ensuring consistent execution
3. **Backend agnostic**: Same template works with DuckDB, Polars, or any other ibis-supported backend
4. **Version control**: Templates can be versioned, reviewed, and stored in Git repositories
5. **Auditability**: Template definitions are part of the event log, ensuring complete experimental traceability

**Template Structure:**

A custom template typically inherits from `ExperimentTemplate` and implements:

```python
from earlysign.templates.base import ExperimentTemplate

class MyCustomTemplate(ExperimentTemplate):
    """Custom template for domain-specific sequential testing."""

    def setup(self, ledger, design_params):
        """
        Initialize experiment design and register to ledger.

        - Define custom effect size parameterization
        - Configure spending functions
        - Set up domain-specific boundaries
        """
        pass

    def step(self, ledger, observation_data):
        """
        Process one observation batch.

        - Compute custom statistics
        - Update decision criteria
        - Emit signals based on custom stopping rules
        """
        pass

    def analyze(self, ledger):
        """
        Generate analysis report from ledger events.

        - Extract relevant events
        - Compute summary statistics
        - Generate visualizations
        """
        pass
```

**Example Use Case: Clinical Trial with Custom Endpoints**

```python
class ClinicalTrialWithQALY(ExperimentTemplate):
    """
    Sequential testing for quality-adjusted life years (QALY).

    - Effect size: Mean QALY difference (clinical meaningful difference = 0.5)
    - Statistic: Variance-stabilized z-score with time-to-event adjustment
    - Stopping rule: Group sequential with binding futility for ethical early stop
    """

    def setup(self, ledger, alpha=0.025, beta=0.10, cmd=0.5):
        # Register custom design with QALY-specific parameters
        design = {
            "effect_measure": "qaly_difference",
            "clinically_meaningful_difference": cmd,
            "alpha": alpha,
            "beta": beta,
            "spending_function": "obrien_fleming",
            "binding_futility": True
        }
        ledger.write_event(namespace="design", kind="registered",
                          payload_type="QALYDesign", payload=design)

    def step(self, ledger, qaly_data):
        # Compute variance-stabilized statistic
        # Update group sequential boundaries
        # Check stopping rules with ethical considerations
        pass
```

**Sharing and Reusability:**

Once created, your custom template becomes a **portable experimental protocol**:

```python
# Team member A creates template
template = ClinicalTrialWithQALY()

# Team member B uses same template on different backend
import ibis
conn_duckdb = ibis.connect("duckdb://data.db")
conn_polars = ibis.polars.connect()

# Same template, different backends
ledger_duck = Ledger(conn_duckdb, "clinical_trial_001")
ledger_polars = Ledger(conn_polars, "clinical_trial_002")

template.setup(ledger_duck, alpha=0.025, beta=0.10, cmd=0.5)
template.setup(ledger_polars, alpha=0.025, beta=0.10, cmd=0.5)
```

**Best Practices:**

1. **Document thoroughly**: Include docstrings explaining the statistical rationale and domain assumptions
2. **Validate inputs**: Check parameter constraints and raise informative errors
3. **Use typed payloads**: Define clear payload schemas for your custom event types
4. **Test across backends**: Verify your template works with multiple ibis backends
5. **Version your templates**: Use semantic versioning for template definitions stored in the ledger

## 4. Event Store Implementation

### 4.1 Backend Agnosticism

EarlySign's event store is **backend-agnostic**, supporting various storage implementations:

- **In-memory**: For prototyping, testing, and lightweight scenarios
- **File-based**: Using Parquet, CSV, or JSON formats for persistence
- **Database**: SQL or NoSQL databases for enterprise deployment
- **Distributed**: Event streaming platforms like Kafka for high-throughput scenarios

### 4.2 Storage Schema and Querying

**Event Record Schema:**

The Ledger stores events in a flattened schema, with domain-specific metadata packed into JSON columns.

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `uuid` | String | Unique event identifier (auto-generated) | `550e8400-e29b-41d4-a716-446655440000` |
| `type` | String | Data schema identifier (Pydantic model name) | `GSTTwoPropDesign`, `LookResult` |
| `payload` | JSON | Event data serialized as JSON | `{"z": 2.10, "status": "STOP_EFFICACY"}` |
| `attributes` | JSON | Logical indexing tags (Entity ID, Namespace, Scope) | `{"entity_identity": "exp#42", "namespace": "stats"}` |
| `timestamp` | Timestamp | Physical time (UTC) | `2025-09-07T10:00:00Z` |
| `metadata` | JSON | System metadata (Trace lineage, Package version) | `{"trace": ["parent-uuid-1"], "pkg_version": "0.1.0"}` |

**Sample Event Log:**

| uuid | type | payload | attributes | timestamp |
|------|------|---------|------------|-----------|
| ...00 | `GSTTwoPropDesign` | `{"alpha":0.025,"method":"..."}` | `{"entity_identity":"exp#42", "namespace":"design"}` | 2025-09-07T10:00:00Z |
| ...01 | `BinomialObs` | `{"nA":10,"nB":10,"yA":8,"yB":1}` | `{"entity_identity":"exp#42", "namespace":"obs"}` | 2025-09-07T10:05:00Z |
| ...02 | `LookResult` | `{"z_stat":2.1,"status":"STOP"}` | `{"entity_identity":"exp#42", "namespace":"stats"}` | 2025-09-07T10:06:00Z |

---

## 3. Spec Tables (Protocol Design)

### 3.1 Ledger

| Aspect        | Spec                                                                 |
|---------------|----------------------------------------------------------------------|
| Purpose       | Store all facts as append-only records with trace lineage.           |
| Write API     | `ledger.insert(data, attributes=..., metadata=...)`                  |
| Read API      | `ledger.t` (Ibis Table Expression) for filtering and projection.     |
| Record Schema | `uuid`, `type`, `payload`, `attributes`, `timestamp`, `metadata`     |
| Scoping       | `ledger.bind(**attrs)` / `ledger.unbind(*keys)` manage attribute context. |
| Lineage       | `metadata["trace"]` stores list of parent UUIDs used to derive event.|

---

### 3.2 Framework Primitives

The core framework provides three primitives for interacting with the Ledger, implementing the Event Sourcing pattern:

| Primitive | Role | Description |
|-----------|------|-------------|
| **Projector** | Read | Defines how to reconstruct a specific view or state from the raw event log. Used via `Session.Read(projector)`. |
| **Logic** | Process | Pure domain logic (Functions/Entities) that transforms inputs (Traced Data) into results. |
| **Writer** | Write | Records new events to the Ledger. Used via `Session.Commit(record)`. "Fire and forget". |

**Session**: The `Session` manages the "Scientific Horizon" (snapshot of time) and implicit trace accumulation, ensuring that every committed event is causally linked to the events read during the transaction.

### 3.3 Atomic Components (Conceptual)

While the implementation uses Projectors and Writers, conceptually the system operates via these logical roles:

| Role | Protocol | Inputs (Read) | Output (Write) | Example Payload |
|------|----------|---------------|----------------|-----------------|
| **Design** | `register` | User Config | Design Spec | `GSTTwoPropDesign` |
| **Observation** | `step` | User Data | Observation Event | `BinomialObs` |
| **Statistic** | `compute` | Design + Obs | Statistic Event | `LookResult` (z-stat) |
| **Criteria** | `check` | Design + Stat | Decision Status | `LookResult` (status) |

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
