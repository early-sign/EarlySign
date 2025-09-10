---
applyTo: "earlysign/**"
---

# 🛠️ EarlySign Framework Implementation Guide

## 🔄 Event-Sourcing Implementation

### Core Principles
- **Immutable events**: Never modify events after ledger writes
- **Event versioning**: Use `payload_type` for schema evolution (`WaldZ_v1`, `WaldZ_v2`)
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
    step_key=step_key,
    payload_type="WaldZ",
    payload={"z": z_value, "se": standard_error, "nA": n_treatment, "nB": n_control}
)

# ❌ Incorrect: Never modify existing events
# ledger.update_event(...)  # This doesn't exist
```

## 🏗️ Component Design Patterns

### Statistic Components
```python
from dataclasses import dataclass
from typing import Optional, Union
from earlysign.core.components import Statistic
from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.core.ledger import Ledger

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

    tag_stats: Optional[str] = "stat:mystat"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        # 1. Read required events
        obs_events = ledger.iter_ns(namespace=Namespace.OBS, experiment_id=str(experiment_id))
        design_event = ledger.latest(namespace=Namespace.DESIGN, experiment_id=str(experiment_id))

        # 2. Compute statistic
        statistic_value = self._compute_statistic(obs_events, design_event)

        # 3. Write result event
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="updated",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="MyStatistic",
            payload={"value": statistic_value, "n_obs": len(list(obs_events))},
            tag=self.tag_stats,
        )
```

### Criteria Components
```python
from dataclasses import dataclass
from typing import Optional, Union
from earlysign.core.components import Criteria

@dataclass(kw_only=True)
class MyCriteria(Criteria):
    """
    Compute decision boundaries for [statistical test].

    Mathematical formulation:
        boundary = g(α, β, information_fraction)
    """

    tag_crit: Optional[str] = "crit:mycrit"
    alpha: float = 0.05

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        # Read latest statistic and design
        stat_event = ledger.latest(namespace=Namespace.STATS, experiment_id=str(experiment_id))
        design_event = ledger.latest(namespace=Namespace.DESIGN, experiment_id=str(experiment_id))

        if not stat_event or not design_event:
            return

        # Compute boundary
        boundary = self._compute_boundary(stat_event.payload, design_event.payload)

        # Write criteria event
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.CRITERIA,
            kind="updated",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="MyCriteria",
            payload={"boundary": boundary, "alpha": self.alpha},
            tag=self.tag_crit,
        )
```

### Signaler Components
```python
from dataclasses import dataclass
from typing import Union
from earlysign.core.components import Signaler

@dataclass(kw_only=True)
class MySignaler(Signaler):
    """
    Generate decision signals based on statistic vs. criteria comparison.

    Decision logic:
        signal = "stop" if |statistic| > boundary else "continue"
    """

    decision_topic: str = "decision"

    def step(
        self,
        ledger: Ledger,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        # Read latest statistic and criteria
        stat_event = ledger.latest(namespace=Namespace.STATS, experiment_id=str(experiment_id))
        crit_event = ledger.latest(namespace=Namespace.CRITERIA, experiment_id=str(experiment_id))

        if not stat_event or not crit_event:
            return

        # Apply decision logic
        decision = self._make_decision(stat_event.payload, crit_event.payload)

        if decision["should_signal"]:
            ledger.emit(
                time_index=time_index,
                experiment_id=str(experiment_id),
                step_key=str(step_key),
                topic=self.decision_topic,
                body=decision,
                tag=f"{self.decision_topic}:decision",
            )
```

## 📦 Payload Design

### Type Safety with TypedDict
```python
from typing import TypedDict, Tuple

class WaldZPayload(TypedDict):
    """Payload schema for Wald Z-test statistic events."""
    z: float
    se: float
    nA: int
    nB: int
    mA: int
    mB: int
    pA_hat: float
    pB_hat: float

class GstBoundaryPayload(TypedDict):
    """Payload schema for Group Sequential Testing boundary events."""
    boundary: float
    alpha_total: float
    t: float  # information fraction
    cumulative_alpha: float

# Register for typed parsing (if needed)
from earlysign.core.ledger import PayloadRegistry
PayloadRegistry.register("WaldZ", lambda d: WaldZPayload(**d))
PayloadRegistry.register("GstBoundary", lambda d: GstBoundaryPayload(**d))
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
        step_key="step1",
        payload_type="TwoPropObsBatch",
        payload={"nA": 10, "nB": 10, "mA": 2, "mB": 5}
    )

    # Execute component
    component = MyStatistic()
    component.step(ledger, "test_exp", "step1", "t002")

    # Verify result
    stat_event = ledger.latest(namespace=Namespace.STATS, experiment_id="test_exp", tag="stat:mystat")
    assert stat_event is not None
    assert stat_event.payload["value"] == expected_value
```

### Integration Tests
```python
def test_component_workflow():
    """Test end-to-end component interactions."""
    ledger = PolarsLedger()

    # Setup components
    statistic = WaldZStatistic()
    criteria = LanDeMetsBoundary(alpha_total=0.05, t=0.5, style="obf")
    signaler = PeekSignaler()

    # Add observations
    for i, (treatment, outcome) in enumerate([("A", 1), ("B", 0), ("A", 1), ("B", 1), ("A", 0)]):
        ledger.write_event(
            time_index=f"t{i:03d}",
            namespace=Namespace.OBS,
            kind="registered",
            experiment_id="test_exp",
            step_key="step1",
            payload_type="TwoPropObsBatch",
            payload={"nA": 3, "nB": 2, "mA": 2, "mB": 1}
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
├── stats/                     # Statistical method implementations
│   ├── methods/              # Core statistical methods
│   │   ├── safe_testing/     # E-values, always-valid inference
│   │   ├── group_sequential/ # Lan-DeMets, O'Brien-Fleming
│   │   ├── common/          # Shared statistical utilities
│   │   └── {method_name}/   # New method families
│   │
│   └── schemes/             # Domain-specific data handling
│       ├── two_proportions/ # Binary outcomes, A/B tests
│       ├── continuous/      # Continuous outcomes (future)
│       └── {outcome_type}/  # New outcome types
│
├── backends/                # Storage implementations
│   ├── polars/             # Polars DataFrame backend
│   │   ├── ledger.py       # Core ledger implementation
│   │   └── io.py           # Persistence utilities
│   └── {backend_name}/     # Additional backends
│
├── core/                   # Framework essentials
│   ├── components.py       # Component base classes
│   ├── ledger.py          # Event store interfaces
│   ├── traits.py          # LedgerOps mixin
│   └── names.py           # Shared types and namespaces
│
├── api/                    # High-level interfaces
│   ├── ab_test.py         # A/B testing API
│   └── compatibility/     # External tool compatibility
│
├── runtime/               # Orchestration and modules
│   ├── experiment_template.py # Base experiment module
│   └── runners.py         # Execution runners
│
└── reporting/            # Analysis and visualization
    ├── generic.py        # Generic reporting components
    └── two_proportions.py # Domain-specific reports
```

## 🚀 Performance Guidelines

### Event Processing
- **Efficient queries**: Use namespace and tag filtering to minimize event processing
```python
# ✅ Efficient: Use specific namespace and filters
obs_events = ledger.iter_ns(namespace=Namespace.OBS, experiment_id=experiment_id)
latest_stat = ledger.latest(namespace=Namespace.STATS, experiment_id=experiment_id, tag="stat:waldz")

# ❌ Inefficient: Process all events
all_events = list(ledger.reader().iter_rows())
```

- **Latest event optimization**: Cache frequently accessed latest events
- **Batch processing**: Process multiple observations when possible

### Memory Management
- **Lazy evaluation**: Defer expensive computations until needed
- **Streaming processing**: Handle large event volumes without loading everything into memory
- **Resource cleanup**: Properly close ledger connections and free resources

### Backend Considerations
- **Backend-specific optimizations**: Leverage each backend's strengths (Polars vectorization)
- **Indexing strategies**: Use appropriate indexing for query patterns
- **Connection pooling**: Reuse connections for better performance

## 🔧 Runtime and Execution Patterns

### Experiment Modules
```python
from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult
from earlysign.stats.schemes.two_proportions.statistics import (
    WaldZStatistic, LanDeMetsBoundary, PeekSignaler
)

class TwoPropGSTModule(ExperimentTemplate):
    """Two-proportions Group Sequential Testing module."""

    def __init__(self, experiment_id: str, alpha_total: float = 0.05, looks: int = 4):
        super().__init__(experiment_id)
        self.alpha_total = alpha_total
        self.looks = looks

    def configure_components(self) -> Dict[str, Any]:
        return {
            "statistic": WaldZStatistic(),
            "criteria": LanDeMetsBoundary(
                alpha_total=self.alpha_total,
                t=1.0 / self.looks,  # Equal spacing
                style="obf"
            ),
            "signaler": PeekSignaler(),
        }

    def register_design(self, ledger: Ledger) -> None:
        ledger.write_event(
            time_index="t000",
            namespace=Namespace.DESIGN,
            kind="registered",
            experiment_id=str(self.experiment_id),
            step_key="design",
            payload_type="TwoPropGSTDesign",
            payload={"alpha_total": self.alpha_total, "looks": self.looks}
        )
```

### Sequential Runners
```python
from earlysign.runtime.runners import SequentialRunner

# Setup
experiment = TwoPropGSTModule("checkout_v1", alpha_total=0.05, looks=4)
runner = SequentialRunner(experiment, PolarsLedger())

# Run analysis
batch_data = {"nA": 100, "nB": 100, "mA": 12, "mB": 8}
result = runner.analyze(batch_data, look_number=1)

if result.should_stop:
    print(f"Early stopping recommended: {result.statistic_value} vs {result.threshold_value}")
```

## 🎯 API Design Patterns

### Business-Friendly Interfaces
```python
from earlysign.api.ab_test import interim_analysis, guardrail_monitoring

# High-level A/B testing API
experiment = interim_analysis(
    experiment_id="checkout_test",
    alpha=0.05,
    looks=4,
    spending="conservative"  # Maps to O'Brien-Fleming
)

# Guardrail monitoring
guardrail = guardrail_monitoring(
    experiment_id="payment_safety",
    sensitivity="balanced"
)
```

### Compatibility Layer
```python
# Support for common A/B testing tools
from earlysign.api.compatibility import from_optimizely_config, to_statsig_format

config = from_optimizely_config(optimizely_experiment_json)
experiment = interim_analysis(**config)
```

This implementation guide reflects the actual EarlySign framework structure and provides concrete examples based on the existing codebase.

