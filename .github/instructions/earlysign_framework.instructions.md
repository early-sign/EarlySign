---
applyTo: "earlysign/**"
---

# 🛠️ EarlySign Framework Implementation Guide

## 🔄 Event-Sourcing Implementation

### Core Principles
- **Immutable events**: Never modify events after ledger writes
- **Event versioning**: Use `payload_type` for schema evolution (`StatType_v1`, `StatType_v2`)
- **Idempotent operations**: Components handle replay gracefully
- **Temporal consistency**: Respect `time_index` ordering

### Ledger as Event Store
```python
# ✅ Correct: Write immutable events
ledger.write_event(
    time_index=time_index,
    namespace=Namespace.STATS,
    kind="updated",
    experiment_id=experiment_id,
    payload_type="MyStatistic_v1",
    payload={"value": computed_value}
)

# ❌ Incorrect: Never modify existing events
# ledger.update_event(...)  # This doesn't exist
```

## 🏗️ Component Design Patterns

### Statistic Components
```python
@dataclass(kw_only=True)
class MyStatistic(Statistic):
    """
    Compute [statistical measure] from observations.

    Mathematical formulation:
        τ = f(X₁, X₂, ..., Xₙ)

    Events consumed:
        - Namespace.OBS: observation data
        - Namespace.DESIGN: experiment configuration

    Events produced:
        - Namespace.STATS: computed statistic value
    """

    tag_stats: str = "stat:mystat"

    def step(self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        # 1. Read required events
        obs_events = ledger.iter_ns(namespace=Namespace.OBS, experiment_id=experiment_id)
        design_event = ledger.latest(namespace=Namespace.DESIGN, experiment_id=experiment_id)

        # 2. Compute statistic
        statistic_value = self._compute_statistic(obs_events, design_event)

        # 3. Write result event
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="updated",
            experiment_id=experiment_id,
            step_key=step_key,
            payload_type="MyStatistic_v1",
            payload={"value": statistic_value, "n_obs": len(list(obs_events))},
            tag=self.tag_stats,
        )
```

### Criteria Components
```python
@dataclass(kw_only=True)
class MyCriteria(Criteria):
    """
    Compute decision boundaries for [statistical test].

    Mathematical formulation:
        boundary = g(α, β, information_fraction)
    """

    tag_crit: str = "crit:mycrit"
    alpha: float = 0.05

    def step(self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        # Read latest statistic and design
        stat_event = ledger.latest(namespace=Namespace.STATS, experiment_id=experiment_id)
        design_event = ledger.latest(namespace=Namespace.DESIGN, experiment_id=experiment_id)

        if not stat_event or not design_event:
            return

        # Compute boundary
        boundary = self._compute_boundary(stat_event.payload, design_event.payload)

        # Write criteria event
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.CRITERIA,
            kind="updated",
            experiment_id=experiment_id,
            step_key=step_key,
            payload_type="MyCriteria_v1",
            payload={"boundary": boundary, "alpha": self.alpha},
            tag=self.tag_crit,
        )
```

### Signaler Components
```python
@dataclass(kw_only=True)
class MySignaler(Signaler):
    """
    Generate decision signals based on statistic vs. criteria comparison.

    Decision logic:
        signal = "stop" if |statistic| > boundary else "continue"
    """

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

## 📦 Payload Design

### Type Safety
```python
from typing import TypedDict

class MyStatisticPayload(TypedDict):
    """Payload schema for MyStatistic events."""
    value: float
    n_obs: int
    confidence_interval: tuple[float, float]

class MyCriteriaPayload(TypedDict):
    """Payload schema for MyCriteria events."""
    boundary: float
    alpha: float
    information_fraction: float

# Register for typed parsing
from earlysign.core.ledger import PayloadRegistry
PayloadRegistry.register("MyStatistic_v1", lambda d: MyStatisticPayload(**d))
PayloadRegistry.register("MyCriteria_v1", lambda d: MyCriteriaPayload(**d))
```

### Design Guidelines
- **Minimal principle**: Include only essential data for downstream components
- **Version compatibility**: Design payloads to support schema evolution
- **Type validation**: Use TypedDict for compile-time and runtime checking
- **Documentation**: Document payload schemas in component docstrings

## 🧪 Testing Patterns

### Component Unit Tests
```python
def test_my_statistic():
    """Test component behavior in isolation."""
    from earlysign.backends.polars.ledger import PolarsLedger

    ledger = PolarsLedger()

    # Setup test events
    ledger.write_event(
        time_index="t001",
        namespace=Namespace.OBS,
        kind="registered",
        experiment_id="test_exp",
        payload_type="Observation_v1",
        payload={"treatment": "A", "outcome": 1}
    )

    # Execute component
    component = MyStatistic()
    component.step(ledger, "test_exp", "test_step", "t002")

    # Verify result
    stat_event = ledger.latest(namespace=Namespace.STATS, tag="stat:mystat")
    assert stat_event is not None
    assert stat_event.payload["value"] == expected_value
```

### Integration Tests
```python
def test_component_workflow():
    """Test end-to-end component interactions."""
    ledger = PolarsLedger()

    # Setup components
    statistic = MyStatistic()
    criteria = MyCriteria(alpha=0.05)
    signaler = MySignaler()

    # Add observations
    for i, outcome in enumerate([1, 0, 1, 1, 0]):
        ledger.write_event(
            time_index=f"t{i:03d}",
            namespace=Namespace.OBS,
            kind="registered",
            experiment_id="test_exp",
            payload_type="Observation_v1",
            payload={"treatment": "A" if i % 2 == 0 else "B", "outcome": outcome}
        )

    # Execute workflow
    time_index = "t100"
    statistic.step(ledger, "test_exp", "step1", time_index)
    criteria.step(ledger, "test_exp", "step1", time_index)
    signaler.step(ledger, "test_exp", "step1", time_index)

    # Verify complete event chain
    events = list(ledger.reader().iter_rows(entity="test_exp"))
    stat_events = [e for e in events if e.namespace == "stats"]
    crit_events = [e for e in events if e.namespace == "criteria"]

    assert len(stat_events) >= 1
    assert len(crit_events) >= 1
```

## 📁 Module Organization

```
earlysign/
├── methods/                    # Statistical method implementations
│   ├── safe_testing/          # E-values, always-valid inference
│   ├── group_sequential/      # Lan-DeMets, O'Brien-Fleming
│   ├── bayesian/             # Bayesian sequential testing
│   └── {method_name}/        # New method families
│
├── schemes/                   # Domain-specific data handling
│   ├── two_proportions/      # Binary outcomes, A/B tests
│   ├── continuous/           # Continuous outcomes
│   └── {outcome_type}/       # New outcome types
│
├── backends/                 # Storage implementations
│   ├── polars/              # Polars DataFrame backend
│   └── {backend_name}/      # Additional backends
│
├── core/                    # Framework essentials
│   ├── components.py        # Component protocols
│   ├── ledger.py           # Event store interface
│   ├── modules.py          # Module composition
│   └── traits.py           # Shared component traits
│
├── api/                     # High-level interfaces
│   ├── experiment.py       # Experiment management
│   ├── ab_test.py          # A/B testing API
│   └── compatibility/      # External tool compatibility
│
└── reporting/              # Analysis and visualization
    ├── generic.py          # Generic reporting components
    └── {domain_specific}.py # Domain-specific reports
```

## 🚀 Performance Guidelines

### Event Processing
- **Efficient queries**: Use namespace and tag filtering to minimize event processing
- **Latest event optimization**: Cache frequently accessed latest events
- **Batch processing**: Process multiple observations when possible

### Memory Management
- **Lazy evaluation**: Defer expensive computations until needed
- **Streaming processing**: Handle large event volumes without loading everything into memory
- **Resource cleanup**: Properly close ledger connections and free resources

### Backend Considerations
- **Backend-specific optimizations**: Leverage each backend's strengths
- **Indexing strategies**: Use appropriate indexing for query patterns
- **Connection pooling**: Reuse connections for better performance

