---
applyTo: "earlysign/**"
---

# EarlySign Framework Implementation Guidelines

This document provides implementation-specific guidelines and patterns for developing with the EarlySign event-sourced sequential testing framework.

## Event-Sourcing Pattern Implementation

### Ledger as Event Store
- **Immutable events**: Never modify events after they are written to the ledger
- **Event versioning**: Use `payload_type` versioning for schema evolution (e.g., `WaldZ_v1`, `WaldZ_v2`)
- **Idempotent operations**: Components should handle replay scenarios gracefully
- **Event ordering**: Respect `time_index` ordering for temporal consistency

### Component Design Patterns

#### Statistic Components
```python
@dataclass(kw_only=True)
class MyStatistic(Statistic):
    """Template for statistic components."""

    tag_stats: str = "stat:mystat"

    def step(self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        # 1. Read required events from ledger
        obs_events = ledger.iter_ns(namespace=Namespace.OBS, experiment_id=experiment_id)
        design_event = ledger.latest(namespace=Namespace.DESIGN, experiment_id=experiment_id)

        # 2. Compute statistic from events
        statistic_value = self._compute_statistic(obs_events, design_event)

        # 3. Write statistic event
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="updated",
            experiment_id=experiment_id,
            step_key=step_key,
            payload_type="MyStatistic",
            payload={"value": statistic_value},
            tag=self.tag_stats,
        )
```

#### Criteria Components
```python
@dataclass(kw_only=True)
class MyCriteria(Criteria):
    """Template for criteria components."""

    tag_crit: str = "crit:mycrit"
    threshold: float = 1.96

    def step(self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        # Read latest statistic
        stat_event = ledger.latest(namespace=Namespace.STATS, experiment_id=experiment_id)

        # Compute boundary/threshold
        boundary = self._compute_boundary(stat_event, self.threshold)

        # Write criteria event
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.CRITERIA,
            kind="updated",
            experiment_id=experiment_id,
            step_key=step_key,
            payload_type="MyCriteria",
            payload={"boundary": boundary, "threshold": self.threshold},
            tag=self.tag_crit,
        )
```

#### Signaler Components
```python
@dataclass(kw_only=True)
class MySignaler(Signaler):
    """Template for signaler components."""

    decision_topic: str = "decision"

    def step(self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        # Read latest statistic and criteria
        stat_event = ledger.latest(namespace=Namespace.STATS, experiment_id=experiment_id)
        crit_event = ledger.latest(namespace=Namespace.CRITERIA, experiment_id=experiment_id)

        if not stat_event or not crit_event:
            return

        # Apply decision logic
        decision = self._make_decision(stat_event.payload, crit_event.payload)

        if decision["should_signal"]:
            ledger.emit(
                time_index=time_index,
                experiment_id=experiment_id,
                step_key=step_key,
                topic=self.decision_topic,
                body=decision,
                tag=f"{self.decision_topic}:decision",
            )
```

### Payload Design Guidelines

#### Type Safety and Validation
- Use `TypedDict` for payload schemas to ensure type safety
- Validate payload structure in component constructors when possible
- Include version information in payload types for schema evolution

#### Payload Registry Pattern
```python
from earlysign.core.ledger import PayloadRegistry

# Register decoders for typed payload parsing
PayloadRegistry.register("MyStatistic", lambda d: MyStatisticPayload(**d))
PayloadRegistry.register("MyCriteria", lambda d: MyCriteriaPayload(**d))
```

#### Minimal Payload Principle
- Include only essential data needed for downstream components
- Avoid redundant information that can be derived from other events
- Balance completeness with storage efficiency

### State Reconstruction Patterns

#### Event Replay
```python
def reconstruct_state_at_time(ledger: LedgerReader, experiment_id: str, target_time: str) -> Dict[str, Any]:
    """Reconstruct experiment state by replaying events up to target_time."""
    state = {}

    for event in ledger.iter_rows(entity=experiment_id):
        if event.time_index > target_time:
            break

        # Apply event to state based on namespace and kind
        if event.namespace == "obs" and event.kind == "registered":
            state = apply_observation_event(state, event)
        elif event.namespace == "stats" and event.kind == "updated":
            state = apply_statistic_event(state, event)
        # ... handle other event types

    return state
```

#### Snapshot Optimization
- Create periodic snapshots for performance optimization
- Use `snapshot_id` to link events to state snapshots
- Implement incremental state updates for efficient replay

### Error Handling and Recovery

#### Event-Driven Error Handling
```python
def handle_component_error(ledger: Ledger, error: Exception, context: Dict[str, Any]) -> None:
    """Record errors as events for debugging and recovery."""
    ledger.write_event(
        time_index=context["time_index"],
        namespace=Namespace.ERRORS,
        kind="error_occurred",
        experiment_id=context["experiment_id"],
        step_key=context["step_key"],
        payload_type="ComponentError",
        payload={
            "component": context["component_name"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        },
        tag="error:component",
    )
```

#### Graceful Degradation
- Components should continue processing even when optional dependencies fail
- Use partial results when complete computation is not possible
- Emit warning events for non-critical failures

### Testing Patterns

#### Component Unit Testing
```python
def test_my_statistic():
    """Test component in isolation using fake ledger."""
    from earlysign.backends.polars.ledger import PolarsLedger

    ledger = PolarsLedger()

    # Setup test events
    ledger.write_event(...)  # observation events
    ledger.write_event(...)  # design events

    # Execute component
    component = MyStatistic()
    component.step(ledger, "test_exp", "test_step", "t001")

    # Verify expected statistic event was written
    stat_event = ledger.latest(namespace=Namespace.STATS, tag="stat:mystat")
    assert stat_event is not None
    assert stat_event.payload["value"] == expected_value
```

#### Integration Testing with Event Sequences
```python
def test_component_integration():
    """Test component interactions through event flows."""
    ledger = PolarsLedger()

    # Setup component chain
    statistic = MyStatistic()
    criteria = MyCriteria()
    signaler = MySignaler()

    # Execute event flow
    statistic.step(ledger, "test_exp", "step1", "t001")
    criteria.step(ledger, "test_exp", "step1", "t002")
    signaler.step(ledger, "test_exp", "step1", "t003")

    # Verify complete event chain
    events = list(ledger.reader().iter_rows(entity="test_exp"))
    assert len(events) >= 3  # stat + criteria + signal events
```

### Performance Optimization

#### Event Filtering and Indexing
- Use namespace and tag filtering to minimize event processing
- Implement efficient queries for latest events by experiment
- Consider backend-specific indexing strategies for large event volumes

#### Lazy Computation
- Defer expensive computations until results are actually needed
- Use caching for computationally expensive statistics
- Implement incremental updates when possible

#### Batching Strategies
- Process multiple observations in single events when appropriate
- Batch write multiple events for performance-critical scenarios
- Balance real-time requirements with computational efficiency

## Method-Specific Guidelines

### Safe Testing (e-values)
- Design events should include betting strategies and prior specifications
- Statistic events should capture e-values, confidence sequences, and betting outcomes
- Criteria events should specify anytime-valid thresholds and stopping rules
- Ensure e-processes maintain anytime validity properties through event replay

### Bayesian Methods
- Design events should capture prior distributions and decision criteria
- Statistic events should include posterior summaries, Bayes factors, and predictive probabilities
- Support streaming posterior updates through incremental Bayesian computation
- Maintain posterior state consistency across event replay scenarios

### Group Sequential Testing
- Design events should specify spending functions and information fractions
- Implement Lan-DeMets spending through criteria components
- Support adaptive information monitoring through observation events
- Ensure boundary calculations remain consistent across method implementations

## Code Organization

### Module Structure
```
earlysign/
├── methods/
│   ├── safe_testing/        # Safe testing implementations
│   ├── group_sequential/    # Group sequential methods
│   ├── bayesian/           # Bayesian sequential methods
│   └── {method_name}/      # New method implementations
│
├── schemes/
│   ├── two_proportions/    # Two-group binary outcome schemes
│   ├── continuous/         # Continuous outcome schemes
│   └── {outcome_type}/     # Domain-specific data schemes
│
├── backends/               # Storage backend implementations
├── core/                  # Framework core (ledger, components, traits)
├── api/                   # High-level API and compatibility layers
└── reporting/             # Analysis and visualization components
```

### Import Conventions
- Always use absolute imports: `from earlysign.core.ledger import Ledger`
- Import typing utilities: `from typing import Union, Optional, Dict, Any`
- Import framework protocols: `from earlysign.core.components import Statistic, Criteria, Signaler`

### Documentation Requirements
- Include comprehensive docstrings with mathematical formulation where appropriate
- Provide doctest examples that demonstrate component usage
- Document payload schemas using TypedDict
- Explain event flow and dependencies in component docstrings
