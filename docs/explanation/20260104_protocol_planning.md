# Protocol Architecture 2026: Modular Composition & Alternative Patterns

**Date**: 2026-01-04
**Objective**: Evaluation of Protocol structures to cover the full scope of Jennison & Turnbull (2000) and modern adaptivity.

## Abstract

This document examines the data structures required to represent and persist **Analysis Protocols** (method design information). The objective is to verify that the proposed schema is rigorously designed to:

1.  **Cover the J&T Scope**: Support all methods described in Jennison & Turnbull (2000).
2.  **Ensure High Extensibility**: Accommodate future adaptive methods and user-customized logic with flexibility.
3.  **Formal Definition**: Data structures are described using **TypeSpec** format for clarity and precision.

---

## 1. Context: J&T 2000 Design Compatibility Analysis

The target library must accurately represent the following standard designs.

| J&T 2000 Design Family | Configuration Strategy | Compatibility |
| :--- | :--- | :--- |
| **Pocock (1977)** | `EfficacySpec(style="fixed", spending="pocock")` | ✅ Supported |
| **O'Brien & Fleming** | `EfficacySpec(style="fixed", spending="obrien_fleming")` | ✅ Supported |
| **Wang & Tsiatis** Power | `EfficacySpec(style="power", delta=Delta)` | ✅ Supported |
| **Error Spending** | `EfficacySpec(style="spending", rho=3.0)` | ✅ Supported |
| **Haybittle-Peto** | `EfficacySpec(style="constrained", constraint=3.0)` | ✅ Supported |
| **One-sided Binding** | `FutilitySpec(binding=true)` | ✅ Supported |
| **One-sided Non-Binding** | `FutilitySpec(binding=false)` | ✅ Supported |
| **Two-sided Asymmetric** | Defined by independent Upper/Lower specs | ✅ Supported |
| **Triangular Test** | Special boundary calculation class | ✅ Supported |
| **Adaptive (Ch 14)** | Updates Protocol instance between cohorts | ✅ Supported |

---

## 1.2 Advanced & Future Scope Requirements

The schema must not only handle standard J&T but also accommodate the following complex and future patterns.

### A. Data & Metric Complexity
-   **Outcome Type**: Binary, Continuous, Time-to-Event (Survival).
-   **Metric Definition**: Testing Z-Score, Log-Hazard Ratio, or Kaplan-Meier difference?
-   **Distributional Model**:
    -   *Canonical*: Asymptotic Gaussian Process ($Z \sim N(\theta\sqrt{I}, 1)$).
    -   *Approximation*: Significance Level Approach (working on P-value scale).
    -   *Exact*: Exact Binomial enumeration.

### B. Multi-Dimensional Designs
-   **Hybrid Endpoints**: Using Survival data for **Futility** while using Binary Response for **Efficacy** (Asymmetric testing on different series).
-   **Multiple Endpoints**: Testing Primary + Secondary endpoints. Requires **Alpha Splitting** (e.g., Bonferroni, Gatekeeping) within the protocol.
-   **Multi-Arm (MAMS)**: Dropping arms, selecting "Best" (requiring arm-specific boundaries).

### C. Adaptive Mechanisms
-   **Internal Pilot**: Re-estimating standard deviation ($\sigma$) from early data to adjust sample size (Blind or Unblind SSR).
-   **Conditional Power**: Using $CP(\theta)$ instead of fixed boundaries for decision making.
-   **Adaptive Randomization (RAR)**: Updating allocation probabilities based on intermediate results.
-   **Bayesian**: Pure posterior probability decision rules (Metric is $P(\theta > 0 | D)$).

### D. Inference & Estimation
-   **Repeated Confidence Intervals (RCI)**: Inverting the group sequential test to obtain valid confidence intervals at every look.
-   **Bias-Corrected Estimation**: Adjusting point estimates for the bias introduced by the stopping rule.

The chosen architecture (Idea A vs C) must demonstrate how it would—eventually—support these scenarios.

---

## 2. Design Idea A: Modular Composition (Concern-Based)

This design separates uniqueness into three independent axes: **Logic** (Why), **Time** (When), and **Space** (How).

### Schema (TypeSpec)

```typespec
// The "Why": Logic
union DesignSpec {
  Superiority: { mode: "superiority" },
  NonInferiority: { mode: "non_inferiority"; margin: float64 },
  Equivalence: { mode: "equivalence"; margin: float64 }
}

// The "When": Time
union ScheduleSpec {
  Fixed: { mode: "fixed"; K: integer },
  EventDriven: { mode: "event_driven"; events: integer[] },
  Adaptive: { mode: "adaptive"; min_interval: duration }
}

// The "How": Space
model BoundarySpec {
  upper: GeneratorSpec;
  lower?: GeneratorSpec;
  binding: boolean;
}

// The Composition
model GSTProtocol {
  design: DesignSpec;
  schedule: ScheduleSpec;
  boundary: BoundarySpec;
  
  alpha: float64;
  power: float64;
}
```

### Philosophy
- **Orthogonal Concerns**: Unaffected axes remain stable when one changes.
- **J&T Alignment**: Direct mapping to standard "Design + Schedule + Boundary" mental model.

---

## 3. Design Idea C: Template-Centric Schema (Decentralized)

**Definition**: [ES3 v1 (EarlySign Standard Schema)](./es3_v1.tsp)

**Core Concept**: There is no single "Universal GST Protocol" class. Instead, each **Template** defines and owns the specific Protocol schema it requires.

### Philosophy
-   **Template Authority**: The `BinomialABTemplate` defines `BinomialABProtocol`. A `MAMS_Template` defines `MAMSProtocol`.
-   **Standardization via Composition**: While Protocols are specific, they are composed of standard, reusable **Parts** (e.g., `StandardBoundarySpec`, `ScheduleSpec`) produced by generic `Designers`.
-   **Evolution via Isolation**: Adding a complex MAMS design does not require changing the schema used by simple A/B tests.

### Architecture: The "Peeling" Consumption Model

The system is layered so that complexity is "peeled off" as we go deeper.

1.  **Producer (`SequentialDesignPlanner`)**:
    -   Generates the "Standard Parts" (e.g., calculates OBF boundaries and returns a `StandardBoundarySpec`).
    -   It does not need to know about the full Template context, only the math.

2.  **Orchestrator (`Template`)**:
    -   Holds the full **Template-Specific Protocol** (e.g., embedding the `StandardBoundarySpec` along with other template settings like `n_max`).
    -   In `update()`, it selects the appropriate **Engine** based on its protocol data.
    -   *Example*: `BinomialABTemplate` sees `n_max` changed, calls Planner to get new `BoundarySpec`, updates internal `BinomialABProtocol`.

3.  **Consumer (`Engine`)**:
    -   The `Engine` (e.g., `BinomialEngine`) accepts the Protocol (or a subset of it).
    -   It executes the core loop: `compute(protocol, data)`. It uses Projectors to get data and writes results to the Ledger.

    *Update 2026-01-06 (Refinement: Consumer-Based Granularity)*:
    -   Engines should be granular and defined by **Protocol Consumers** (or Responsibility).
    -   Instead of a monolithic `BinomialGSTEngine` handling everything, we decompose into:
        -   `GSTStoppingRuleEngine`: Consumes `StoppingRule` (Schedule + Boundary). Handles "Should we stop?". Reusable across endpoints.
        -   `BinomialStatisticsEngine`: Consumes Data + Hypothesis. Handles "Calculate Z".
    -   The `BinomialGSTEngine` acts as an **Orchestrator** (Wiring), injecting the appropriate sub-engines.
    -   **Dependency Injection**: Ideally, `GSTStoppingRuleEngine` receives dependencies (like `BinaryModel`) injected by the Orchestrator, ensuring it remains pure and testable.


### Schema (TypeSpec)

```typespec
// --- Reusable Parts (Produced by Planner) ---
model StandardBoundarySpec {
  upper_values: float64[];
  lower_values: float64[];
}

model ScheduleSpec { ... }

// --- Template-Specific Protocols ---

// 1. Simple Binomial A/B (Template 1)
model BinomialABProtocol {
  // Embeds standard parts
  boundary: StandardBoundarySpec;
  schedule: ScheduleSpec;
  
  // Template specific
  arm_names: string[];
}

// 2. Complex Asynchronous Monitor (Template 2)
model AsyncMonitorProtocol {
  // Does NOT use StandardBoundarySpec
  // Defines its own custom structure
  efficacy_rule: EfficacyRule;
  futility_rule: FutilityRule;
}
```

### Evaluation against Stress Tests
*   **J&T Coverage**: `BinomialABTemplate` uses a protocol embedding standard `Design/Schedule/Boundary` specs. Perfect fit.
*   **Scenario X (Async)**: A new `AsyncMonitoringTemplate` is created with a custom schema. It doesn't fight with the `BinomialABProtocol`.
*   **Scenario Z (Logic Switch)**: The Template's `update()` method detects the condition and swaps the `Engine` or modifies the Protocol state explicitly.

---

## 4. Deep Application Study (Critical Stress Tests)

### Scenario X: Asynchronous Asymmetric Monitoring
*   **Requirement**: "Check Efficacy every patient, but Futility only at N=200, 400."
*   **Evaluation**:
    *   **Idea A**: **Fails** (or requires complex recursion). `ScheduleSpec` implies a global clock.
    *   **Idea C**: **Passes**. `AsyncMonitoringTemplate` defines a protocol with `efficacy_schedule` and `futility_schedule` properties.

### Scenario Y: Hybrid Metrics (Scale Mixing)
*   **Requirement**: "Efficacy on Bayesian Posterior > 0.98, Futility on Z-score < -1.5."
*   **Evaluation**:
    *   **Idea A**: **Weak**. `BoundarySpec` assumes a single statistic scale.
    *   **Idea C**: **Passes**. `HybridTemplate` defines a protocol where `efficacy_rule` uses posterior and `futility_rule` uses Z-score.

### Scenario Z: State-Dependent Logic Switch
*   **Requirement**: "If Z > 2.5 at Interim 1, switch spending function from OBF to Pocock."
*   **Evaluation**:
    *   **Idea A**: **Impossible**.
    *   **Idea C**: **Passes**. `Template.update()` explicitly modifies the Protocol's state or switches the Engine strategy.

---

## 5. Comparative Conclusion

| Feature | Idea A (Universal Union) | Idea C (Template-Centric) |
| :--- | :--- | :--- |
| **Complexity** | High (Giant Union of all possibilities) | Low (Decentralized Simplicity) |
| **Flexibility** | Limited by monolithic schema | Infinite (New Template = New Schema) |
| **Coordination** | Hard (One size fits all) | Natural (Template Orchestration) |

**Recommendation**: **Idea C** aligns perfectly with the "Peeling" architecture.
-   **Planner** produces math (Parts).
-   **Template** owns the Protocol (Structure).
-   **Engine** consumes the Protocol (Execution).


---

## 6. Component Reusability & Naming Convention

To balance "Ease of Use" and "Portability", we avoid monolithic inheritance. Instead, we provide a **Standard Library of Leaf Components** that Custom Engines can compose.

### A. The "Standard Parts" (Reusable Leaf Nodes)

These are the "Ready-made" components that any Custom Protocol or Engine can reference. They are pure, stateless, and focused on a single math/data problem.

| Category | Component Name | Role (The "What") | Example Usage |
| :--- | :--- | :--- | :--- |
| **Data Types** | `DesignSpec` | Defines Hypothesis Logic (Sup/NI). | Used by Solvers to shift margins. |
|  | `ScheduleSpec` | Defines Timing ($N$ or Events). | Used by Solvers to generate $I$. |
|  | `BoundarySpec` | Defines Critical Values (Space). | The input to any decision logic. |
| **Logic (Math)** | `BoundarySolver` | $Design + Schedule \to Boundary$. | `solver.solve(design, schedule)` |
|  | `ZScoreProjector` | $Data \to (Z, I)$. | `proj.project(df, n_max)` |
|  | `LogRankProjector` | $SurvivalData \to (Z, I)$. | `proj.project(df, events)` |

### B. The "Custom Engine" (The Orchestrator)

**Analogy: `Engine` $\approx$ `nn.Module` (PyTorch)**

Just as an `nn.Module` holds sub-modules (layers) and defines a `forward()` pass, an **Engine** holds sub-components (Projectors, Solvers) and defines a `compute()` pipeline.

*   **State**: The Engine holds the "Weights" (or in our case, the specific `Protocol` configuration).
*   **Forward**: The `compute(data)` method runs the data through the sub-components.

*Example: `BinomialGSTEngine` wrapping a Standard GST Flow*

```python
class BinomialGSTEngine:  # Like a Custom Module
    def __init__(self, protocol: BinomialABProtocol):
        self.protocol = protocol
        
        # Sub-modules (The "Layers")
        # specific logic can optionally be swapped here
        self.solver = StandardBoundarySolver() 
        self.projector = ZScoreProjector()

    def compute(self, data: DataFrame):  # The "forward()" pass
        # 1. Unpack the embedded Standard GST Protocol
        gst_params = self.protocol.engine_protocol 
        
        # 2. Run the Layers
        stat = self.projector.project(data, gst_params.schedule)
        boundary = self.solver.solve(gst_params.task, gst_params.schedule)
        
        # 3. Custom Logic (e.g., checking arm names defined in outer protocol)
        return self._check_stopping(stat, boundary)
```

### C. Naming & Design Conventions (The "Embedded" Pattern)

We use **Composition** to reuse standard structures without enforcing inheritance.

1.  **`StandardGSTProtocol` (The Reusable Block)**:
    -   Bundles the J&T essentials: `TaskSpec`, `ScheduleSpec`, `BoundarySpec`.
    -   Can be used standalone or embedded.

2.  **`*Protocol` (The Template Manifest)**:
    -   Embeds `StandardGSTProtocol` as a field (e.g., `engine_protocol`).
    -   Adds template-specific context (e.g., `arms`).

    ```typespec
    // Reusable Standard Block
    model StandardGSTProtocol {
      task: TaskSpec;           // Renamed to TaskSpec (The "What")
      schedule: ScheduleSpec;   // The "When"
      boundary: BoundarySpec;   // The "How"
      adaptation?: AdaptationSpec; // SSR, Drop-the-loser, etc.
    }

    // Template-Specific Manifest
    model BinomialABProtocol {
      // 1. Context Information
      template: "BinomialAB";
      arms: string[];
      
      // 2. The Core Mathematical Design (Embedded)
      engine_protocol: StandardGSTProtocol; 
    }
    ```

### D. User Value ("The Sweet Spot")

-   **Modular**: Users can swap the `gst` block entirely (e.g. switch from OBF to Pocock) while keeping `arms` constants.
-   **Familiar**: The `Engine` / `Protocol` split mirrors `Model` / `Config` or `Layer` / `Hyperparams` patterns found in ML frameworks.

### E. Architectural Justification: Why Task/Schedule/Boundary?

The user asked: *"Should we split components by the Consumer (e.g., ProjectorConfig, SolverConfig) instead of Axes?"*

**Analysis: Cross-Cutting Dependencies**

| Spec (Axis) | Used By Projector? | Used By Solver? | Role |
| :--- | :--- | :--- | :--- |
| **`ScheduleSpec`** | ✅ Need $N$ to calc $I$. | ✅ Need $I$ to calc $b_k$. | **Common Time Basis** |
| **`TaskSpec`** | ✅ Need `margin` for Z. | ✅ Need `side` for $H_0$. | **Common Logic** |
| **`BoundarySpec`** | ❌ | ✅ Need shape/alpha. | Solver-Specific |

**Conclusion**:
If we split by Consumer (`ProjectorConfig`, `SolverConfig`), we would forced to **duplicate** `Schedule` and `Task` in both configs (or create a complex sync requirements).
By keeping **Axes (`Task`, `Schedule`)**, we treat them as **Shared Context**. The `Engine` acts as the **Binder**, routing the shared context to each consumer.
-   `Task` = The Question (Shared).
-   `Schedule` = The Timeline (Shared).
-   `Boundary` = The Critical Values (Solver).

