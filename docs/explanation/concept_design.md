# Concept Design: Sequential Statistical Computing via Event Sourcing

This document defines the core conceptual framework for EarlySign, bridging the requirements of sequential statistical inference with architectural patterns.

---

# Part I: Foundations of Sequential Statistical Computing

## I-1. Bridging Statistical Requirements and Event Sourcing

Sequential testing in real-world environments, such as clinical trials, demands a high degree of historical fidelity. In practice, ad-hoc judgments by **Data Monitoring Committees (DMC)** or unplanned analyses are inevitable, requiring the system to preserve the exact sequence of events—the "full course"—rather than just the latest summary state. Recording every observation, setup parameter, and decision captures the exhaustive context of the experiment, which corresponds mathematically to the filtration $\{\mathcal{F}_t\}_{t \ge 0}$ that generates our statistical insights. Crucially, the necessity of this exhaustive record stems from the fact that we cannot predict in advance what facts or events may become relevant, or how they may be revisited to make reasonable statistical inference—potentially using methodologies that were not devised at the study's outset.

To address these needs, EarlySign utilizes **Event Sourcing (ES)** as its foundational architectural pattern (Fowler 2005). In this approach, we do not simply store the latest state of a trial; instead, we persist the complete, immutable sequence of events that led to that state. This ensures that the current scientific context can always be reconstructed by replaying the series—mathematically representing a left-fold over a stream of events. EarlySign implements this through four core conceptual components:

*   **The Ledger**: The physical **Event Store**. It is the manifestation of the filtration process, storing an immutable, append-only sequence of events.
*   **Projections**: Interpretation of the Ledger is achieved through **Projections**. Formalized in the context of Event Sourcing and CQRS (Command Query Responsibility Segregation) to describe the derivation of read-models from an immutable event log (Young, Versioning in an Event Sourced System (v10 ed.)), Projections implement the **State-as-a-Fold** principle. This involves a deterministic transformation of historical facts into a usable representation via a left-fold operation: $\text{State}_t = \text{fold}(\text{Events}_{1 \dots t})$.
*   **The Write Model**: The command-handling interface responsible for asserting new facts (e.g., analyses, decisions) and appending them to the Ledger.
*   **Operations**: Logical units that accept Projections as input, perform computations, and use the Write Model to persist realizations.

By preserving the full trajectory through this **Event Sourcing** foundation, the library ensures there is **"room for flexible response"** to the evolving needs of the study. This historical fidelity provides **procedural flexibility**: should unexpected timing of decisions arise, plans change, or the need to switch methodologies occur, the system can reliably account for everything that has happened up to that point.
## I-2. The Episode: Structure and Event Categories

The core domain object in EarlySign is the **Episode**, which represents the complete lifecycle of a sequential analysis session. Unlike a traditional domain object characterized by mutable fields, an Episode is an **ever-growing set of immutable events**. 

However, it is important to recognize that while the Episode's core identity is this event stream, specific subsets of these events—such as cumulative statistics—**can also manifest as** a "state" in the conventional sense. The **State of Episode** is therefore a flexible, case-by-case **Projection** derived from the event stream, where statistics or parameters hydrate into a form suitable for calculation.

To maintain statistical integrity while ensuring procedural flexibility, an Episode's events are categorized into four distinct types:

*   **Protocol**: Events that define or update the rules of the study (e.g., `ProtocolUpdated`, alpha-spending functions).
*   **Observations**: External evidentiary events (the source of the filtration $\mathcal{F}_t$).
*   **Analyses**: Materialized statistical derivations. These capture "what the statistician saw" as a historical fact, ensuring the rationale for decisions remains reconstructible even if methodologies change.
*   **Decisions**: Operational conclusions or actions taken based on an analysis (e.g., "Stop for efficacy").

---

## I-3. The Essential Utility of Event Sourcing and CQRS in Sequential Design

In EarlySign, **Event Sourcing** and **CQRS** are not merely implementation details; they are fundamental to scientific rigor across four dimensions:

### 1. Physical Realization of Filtrations ($\mathcal{F}_t$)
In sequential statistics, every decision must be **measurable** with respect to the information set $\{\mathcal{F}_t\}$ available at that time. By utilizing an append-only Ledger, EarlySign provides a physical implementation of this mathematical filtration. The immutable nature of the log ensures that the order of evidence is preserved and tamper-proof, satisfying the basic requirement for provable sequential inference.

### 2. Separation of Fact and Interpretation (Causal Decoupling)
Clinical trials distinguish between raw data (facts) and the statistical models used to interpret them. By separating the **Write Model** (appending facts) from the **Read Model** (Projections), EarlySign allows a trial to be "re-interpreted" without altering its historical evidence. This decoupling is essential for responding to unplanned events or updating methodologies while maintaining a consistent audit trail.

### 3. Structural Scientific Provenance
Traditional imperative designs often obscure the link between a result and the data that produced it. The Trace mechanism in EarlySign, enabled by the clear boundaries of CQRS, allows the system to automatically track the "causal chain" between Projections and Commits. Every decision is structurally bound to the specific evidence state it represents, providing a verifiable rationale that is inherent to the data structure itself.

### 4. Reproducibility of Scientific Context
For audits and verification, it is often necessary to reproduce a specific decision-making moment. Because EarlySign stores state as a "Fold" over events, any historical point can be replayed with bit-level accuracy. Even stochastic realizations (such as MCMC samples) are "frozen" once committed to the Ledger as facts, allowing the surrounding logic to remain deterministic during subsequent replays.

---

# Part II: Practical Design Requirements and Optimization

To transition from a mathematical concept to a usable library, EarlySign addresses implementation complexities through specific design strategies.

## II-1. Practical Implementation Requirements

*   **Strict Statelessness (Batch-Job Compatibility)**: Computing environments are often volatile (e.g., short-lived batch jobs). To ensure continuity, all necessary context for resumption is stored in the Ledger, and the library provides a "Hydrate $\to$ Process $\to$ Commit" cycle that prevents the loss of critical state between executions.
*   **Complexity Management ($O(N)$ vs $O(N^2)$)**: Replaying the entire history for every new data point is computationally expensive. To counter this, Projections are designed to update incrementally, and the library supports **Persistent Snapshots**. These snapshots are stored in the Ledger as special events (or state tables), ensuring they survive process restarts. New processes can "cold-start" by loading the latest snapshot and replaying only the subsequent differential events ($\Delta N$).
*   **Handling Stochasticity**: Procedures like Bayesian MCMC introduce randomness that can break reproducibility. EarlySign addresses this by adopting **Stochastic Materialization** as a compromise: once a stochastic value is computed, it is written to the Ledger as an immutable fact, "freezing" the realization to ensure the audit trail remains deterministic for all subsequent steps.
*   **Portability and Low Lock-in**: To ensure that the library remains a valuable tool rather than a restrictive silo, core statistical logic must be decoupled from the Event Sourcing framework. Practical implementations should allow the same calculation code to be reused in other contexts (e.g., standard simulation studies or non-ES pipelines) without significant refactoring or dependency on the library's internal state management.
*   **Idempotent Execution by Design**: Statistical workflows must be idempotent. Unless new information (e.g., data ingestion, protocol update) is added to the Ledger, re-running an analysis command should produce no new events. This design prevents duplicate analysis records and ensures that the audit trail accurately reflects the progression of evidence rather than the frequency of job execution.

## II-2. Implementation Details

*   **Execution Strategy**: While Ibis is used for scalable data retrieval and initial aggregation (Projections), core statistical computations are executed within the **Python layer**. This choice stems from the discovery that pushing complex statistical expression trees directly into backend SQL engines via Ibis can lead to excessive translation overhead and deep, unperformant query plans.
*   **The Horizon Boundary**: EarlySign introduces the concept of a **Horizon** (implemented via a `with` context). This serves as the explicit boundary for **Snapshot Isolation**. Upon entering a horizon, the library locks the current state of the Ledger, ensuring that all subsequent `Read` operations are mutually consistent and immune to external updates during the session. Once the horizon is exited, all transient state is discarded. This design enforces **Strict Statelessness**, ensuring that every new interaction must begin by objectively re-hydrating the current reality from the Ledger.
*   **Natural Integration**: The library is designed to "play along" with the Ibis ecosystem, using it as a high-performance "Read Side" while maintaining the flexibility of Python for the "Command Side" (analytical logic).

## II-3. Taxonomy of Sequential Adaptivity

To validate the "Episode" and "Ledger" abstractions, EarlySign categorizes sequential statistical methods by their **Level of Adaptivity**. This taxonomy moves from rigid, pre-planned monitoring to dynamic, evolutionary trial structures, each placing unique demands on the Event Sourcing architecture.

### Level 1: Rigid Monitoring (Scheduled GST)
Standard **Group Sequential Tests (GST)** with fixed interim schedules and boundaries (e.g., Pocock, O'Brien-Fleming). These designs require the Ledger to track a fixed number of "Looks" and ensure that the correct boundary is applied at the correct sample size.

### Level 2: Context-Adaptive Monitoring (Flexible Space)
Designs where the timing of analysis is driven by the data rather than a fixed schedule.
*   **Spending Functions (Lan-DeMets)**: $\alpha$ and $\beta$ spending based on **Information Time**, requiring the $ \mathcal{F}_t $ to reconstruct history to compute the residual error budget.
*   **Modern Continuous Monitoring**: Implementation of **Safe Tests** and **e-processes**, allowing for valid inference under optional stopping without a predefined schedule.
*   **Sequential Time-to-Event Analysis**: Log-rank tests where "Information" is defined by event counts rather than sample size.

### Level 3: Structural Adaptation (Parameter-Driven)
Designs that modify the trial's design parameters based on interim evidence.
*   **Sample Size Re-estimation (SSR)**: Dynamic adjustment of the maximum information $I_{max}$ or treatment effect weights (Cui-Hung-Wang).
*   **Adaptive Enrichment**: Narrowing the inclusion criteria to a specific biomarker subgroup if the treatment effect is heterogeneous.
*   **Internal Pilot Designs**: Estimating nuisance parameters (e.g., variance) from early data to refine the final sample size without breaking the treatment blind.

### Level 4: Evolutionary Designs (Population & Arm Adaptation)
Designs where the very components of the experiment (arms, populations) change over time.
*   **Multi-Arm Multi-Stage (MAMS)**: Dropping non-performing arms or selecting the "Best" arm for confirmatory testing (Seamless Phase II/III).
*   **Platform and Basket Trials**: Open-ended trials where new arms enter or exit the Episode. This requires the Ledger to handle a multi-generational narrative of facts.
*   **Response-Adaptive Randomization (RAR)**: Dynamically updating allocation probabilities based on therapeutic success (e.g., Thompson Sampling).

### Level 5: Direct Inference & Decision (Policy-Driven)
*   **Bayesian Dose-Finding (CRM/BOIN)**: Real-time decision-making for dose escalation in Phase I, where each new observation immediately influences the next "Decision" event in the Ledger.

---

## II-4. Architectural Stress Tests: The "Attacks" on Design

The taxonomy above serves as a set of **architectural stress tests**—conceptual "attacks" that the API must defend against to ensure it truly "untangles" complexity.

*   **The Idempotency Challenge**: In adaptive designs, how does the system ensure that re-running an analysis on the same evidence state produces no new events, even if the **Protocol** has slightly shifted its parameters in the meantime?
*   **The Governance and Provenance Challenge**: Every **Decision** must be auditably linked to the specific **Analysis** that justified it. Furthermore, any subsequent **Protocol Update** (e.g., SSR) must record which specific statistical "Insight" triggered the change.
*   **The Latency and Pipeline Challenge**: Sequential monitoring must distinguish between **Observed Evidence** (finalized outcomes) and **Pipeline Information** (enrolled subjects whose outcomes are pending). The API must allow solvers to reason about the "Total Trial State" to prevent premature stoppage in the face of significant outstanding data.
*   **The Stochastic Reproducibility Challenge**: Bayesian MCMC or randomization steps must have their realizations "frozen" via **Stochastic Materialization**. The API must ensure that these realizes are treated as immutable facts in the Ledger to guarantee deterministic replay.
*   **The Multi-Generation Challenge (O(N))**: For Platform Trials spanning years, state reconstruction must remain $O(N)$ with respect to the total event stream, requiring efficient snapshotting and incremental logic that spans multiple protocol versions.

---

# Part III: API Design and Integrity Patterns

The candidates below represent a progression from **Explicit Manual Control** to **Transparent Logical Autopilot**. They share the same underlying "Episodes" and "Ledger" but differ in their degree of structural integrity, operational efficiency, and the level of "Architectural Consciousness" required from the developer.

**Summary of Evolution (Selection Guide):**
- **Pattern A/B**: Best for simple, one-off scripts where explicit control is preferred.
- **Pattern C**: Useful for scripts where literal string identifiers are preferred over formal types.
- **Pattern F**: Recommended standard for production-grade trials. It provides strict type safety, automatic validation, and clear schema definitions via Pydantic.
- **Pattern G**: The most refined standard, adding **Traced[T]** containers and polymorphic trace merging for precise scientific context management.
- **Pattern D/E**: Future experimental patterns for complex orchestration.

## III-1. Pattern A: Raw Procedural API (Manual Integrity)

This is the baseline API. It provides direct, procedural access to Ledger operations. It is intuitive but places the burden of integrity (e.g. idempotency, provenance) entirely on the developer.

```python
with ledger.bind(episode_id="TRIAL_01").Horizon() as ep:
    # 1. Read: Projections hydrate state
    summary = ep.Read(BinomialSummary())
    protocol = ep.Read(Latest(StudyProtocol))

    # 2. Logic: Direct calculation
    z_stat = compute_z(summary)
    look = compute_boundaries(protocol)
    
    # 3. Commit: Explicitly record the analysis
    # NOTE: Re-running this script will blindly append duplicate events.
    ep.Analysis("LookResult", z=z_stat, bounds=look)
    
    # 4. Action: No formal link to the Analysis result
    if z_stat > look.efficacy:
        ep.Decision("STOP_EFFICACY")
```

## III-2. Pattern B: Managed Procedural API (Structural Integrity)

Pattern B is **Pattern A with an "Anchor"**. It uses the same procedural flow but introduces `ep.Commit` to solve idempotency, governance, and stochastic challenges structurally. The name `Commit` emphasizes that we are finalizing a statistical insight into the immutable Ledger within the current **Horizon**.

```python
episode = ledger.bind(episode_id="TRIAL_01")
with episode.session() as ep:
    # 1. Read: Hydrate a summary using a Projector
    # The session internally records the specific data context of this read.
    summary = ep.Read(BinomialSummaryProjector(arm="A"))
    
    # 2. Derive: Scientific Computation
    z_val, lookup = perform_analysis(summary)
    
    # 3. Managed Commitment:
    # Trace Hash is automatically built from the session's read history.
    res = ep.Commit(AnalysisResult(z=z_val, bounds=lookup))
    
    # 4. Governance Action:
    # The decision's Trace Hash includes the analysis result 'res'.
    if res.z > res.bounds.efficacy:
        ep.Decision(Decision.StopEfficacy())
```

## III-3. Pattern C: Functional/Lazy Procedural (High Efficiency)

Pattern C is the standard evolution, designed for high-performance and distributed environments. It extends Pattern B by replacing direct calculation with **Deferred Execution (Lambdas)**. This allows the library to verify the **Trace Hash** *before* running any heavy statistical logic, enabling a "Skip-then-Compute" model and multi-step checkpointing.

### User Workspace: Checkpointing and Resumption

```python
    episode = ledger.bind(episode_id="TRIAL_01")
    with episode.session() as ep:
        ctx = ep.Read(InferenceProjector())

        # Step 1: Preprocessing (Lazy)
        clean = ep.CommitCallResult(DataRecord, slow_preprocess, ctx)

        # Step 2: Analysis (Lazy)
        res = ep.CommitCallResult(BayesianResult, perform_mcmc, clean)

        if res.p_value < 0.05:
            ep.Decision(Decision.Success())
```

### Note on "Read" Skipping in Pattern C:
The `ep.Read()` calls are executed to prepare arguments for `ep.CommitCallResult()`. While the calculation body is skipped if the fingerprint matches, the data retrieval logic remains visible and active. This is suitable for workflows where data retrieval overhead is negligible compared to statistical computation.

## III-4. Pattern F: Typed Record API (Recommended Standard)

Pattern F is the standard implementation pattern. It focuses on the use of **Pydantic Models** to define the schema of every committed event.

Crucially, Pattern F introduces **Implicit Trace Accumulation**:
- Every `ep.Read(projector)` adds the specific data references (the "Trace") identified by the Projector to the session's context.
- Every subsequent `Commit` or `Decision` automatically includes the cumulative session trace in its fingerprint.
- This ensures **Referential Transparency**: the developer writes simple imperative code, but the library enforces strict scientific causality.

### User Workspace: Projectors and Sessions

```python
# 1. Define the Result Schema
class ZTestResult(BaseModel):
    z_stat: float
    p_value: float

# 2. Define the Projector
class BinomialProjector(Projector[BinomialSummary]):
    arm: str
    def project(self, data: ibis.Expr) -> BinomialSummary:
        # High-performance fold logic on the Ibis table...
        stats = data.filter(data.arm == self.arm).aggregate(n=data.count(), k=data.y.sum())
        return BinomialSummary.model_validate(stats.to_pandas().to_dict('records')[0])

episode = ledger.bind(episode_id="TRIAL_01")
with episode.session() as ep:
    # 1. Traced Read: Returns a container (data + specific trace)
    summary = ep.Read(BinomialProjector(arm="A"))

    # 2. Automated Commitment: 
    # extract trace from 'summary' automatically.
    res = ep.CommitCallResult(ZTestResult, perform_analysis, summary)

    # 3. Governance:
    # Uses the implicit session trace (including 'res') by default.
    if res.p_value < 0.05:
        ep.Decision(Decision.Success())
```

### Why Pattern F is the Recommended Pattern

1.  **Explicit Context Separation**: `episode.session()` clearly distinguishes the long-term Episode from the transient work Session.
2.  **Modular Projections**: `Projector` classes centralize complex "Read" logic (folds), keeping the analysis script focused on scientific rules.
3.  **Schema Persistence**: Committing with Typed Records stores the schema identity (model name/version) in the Ledger, allowing future tools to parse event payloads without guessing.

## III-5. Pattern G: Traced Procedural API (The Refined Standard)

Pattern G is the **Refined Standard** for EarlySign. It builds upon Pattern F by introducing the `Traced[T]` container and advanced trace orchestration. While Pattern F handles schema, Pattern G ensures that scientific causality is handled with both automation and surgical precision.

### Key Features of Pattern G:

1.  **Implicit Trace Accumulation**: Every `ep.Read(projector)` adds the specific data references (the "Trace") identified by the Projector to the session's context.
2.  **`Traced[T]` Container**: `Read` returns a wrapper that holds both the hydrated `data` and its scientific `trace`. This allows the trace to be passed explicitly as an argument.
3.  **Polymorphic Trace Merging**: Methods like `Commit` accept a `List[Union[Traced, str]]`, automatically flattening containers to extract their lineage.
4.  **Implicit Access**: The `ep.trace` property provides direct access to the session's cumulative "implicit trace."

### User Workspace: Advanced Orchestration

```python
episode = ledger.bind(episode_id="TRIAL_01")
with episode.session() as ep:
    # 1. Targeted Read: returns Traced[BinomialSummary]
    summary_a = ep.Read(BinomialProjector(arm="A"))
    summary_b = ep.Read(BinomialProjector(arm="B"))

    # 2. Polymorphic Merging:
    # The Trace for this result is automatically merged from (summary_a, summary_b)
    res = ep.CommitCallResult(ComparativeResult, calculate_diff, summary_a, summary_b)

    # 3. Explicit Narrowing:
    # We choose to justify this decision ONLY based on the comparative result 'res', 
    # ignoring other unrelated session activity.
    if res.p_value < 0.05:
        ep.Decision(Decision.Success(), trace=[res])
```

### Control Modes of Pattern G:

1.  **Automatic (Convenience)**: By default, `Commit` and `Decision` use the cumulative session trace. 
2.  **Narrowed (Explicit & Polymorphic)**: Pass a `List[Traced | str]` to `Commit(trace=...)`. The library flattens any `Traced` containers to extract their scientific lineage.
3.  **Explicit Isolation**: Passing `trace=[]` (empty list) binds the record *only* to the current Snapshot, isolating it from session-level interactions.
4.  **Implicit Access**: The `ep.trace` property provides direct access to the current cumulative "implicit trace" of the session.

### How it Works: The Trace Hash Ingredients

The **Trace Hash** is the soul of the system's integrity. It represents a **Scientific Lineage**, ensuring that every fact in the Ledger is provably tied to its evidence. Technically, it is a stable hash computed from four discrete ingredients:

1.  **Induction Snapshot**: The Ledger's state ID (LSN or cumulative hash) representing the observation cutoff.
2.  **Parent Trace Hashes**: The `trace_hash` values extracted from every record/container passed as an argument to the current call. This creates the dependency link.
3.  **Action Identity**: The `name` or `type` of the operation (e.g., "ComparativeResult").
4.  **Parameter State**: Literal values of other `*args` and `**kwargs`.

### Why this is User-Friendly:

*   **Linear code, Non-linear reality**: You write a simple, top-to-bottom script. The library "remembers" what has already been logically satisfied.
*   **Stochastic Safety**: In Bayesian trials (MCMC), the first result is the *only* result. Future runs retrieve the exact same posterior-samples from the Ledger, ensuring your audit trail is deterministic.
*   **Pipeline Awareness**: It solves the "Waitlist" problem automatically. If 20 new patients enroll, the fingerprint changes, allowing a new "Look" that accounts for the increased information weight.

## III-6. Library-Side Logic: Implementing Pattern G

Pattern G leverages Traced[T] containers and a polymorphic Trace Resolver to guarantee scientific causality.

```python
# Generic container for data + scientific trace
class Traced(Generic[T]):
    data: T
    trace: List[str]

class Session:
    def __init__(self, snapshot_id):
        self.snapshot_id = snapshot_id
        self._session_trace = [] # Cumulative trace of all session activity

    @property
    def trace(self) -> List[str]:
        """Returns the cumulative 'implicit trace' of the session."""
        return self._session_trace

    def _resolve_trace(self, trace: Optional[List[Union[Traced, str]]]) -> List[str]:
        """Polymorphic helper to flatten Traced objects and strings into a list of hashes."""
        if trace is None:
            return self.trace # Default to implicit session trace
            
        flat_trace = []
        for item in trace:
            if isinstance(item, Traced):
                flat_trace.extend(item.trace)
            else:
                flat_trace.append(item)
        return flat_trace

    def Read(self, projector: Projector[T]) -> Traced[T]:
        # a. Execute the projection
        result, projection_trace = projector.project(self.ledger.get_data(self.snapshot_id))
        
        # b. Update session context
        self._session_trace.append(projection_trace)
        
        # c. Return wrapped result
        return Traced(data=result, trace=projection_trace)

    def Commit(self, record: TypedRecord, trace: Optional[List[Union[Traced, str]]] = None):
        # Resolve causality: explicit list (flattened) or session default
        target_trace = self._resolve_trace(trace)
        
        trace_hash = hash(self.snapshot_id, target_trace, record.__class__.__name__, record.model_dump())
        
        # Idempotency check...
        # Store to Ledger...
        self._session_trace.append(trace_hash)
        return record.with_trace(trace_hash)

    def CommitCallResult(self, result_type: Type[TypedRecord], func, *args, **kwargs):
        # a. Automatic Trace Extraction from arguments
        explicit_traces = [v.trace for v in list(args) + list(kwargs.values()) if isinstance(v, Traced)]
        
        # b. Resolve context: 
        # Merged traces from Traced args, or fall back to cumulative session trace
        target_trace = explicit_traces if explicit_traces else self.trace
        
        trace_hash = hash(self.snapshot_id, target_trace, result_type.__name__, args, kwargs)
        
        # c. Execute logic (passing raw data to the user function)
        raw_args = [v.data if isinstance(v, Traced) else v for v in args]
        raw_kwargs = {k: v.data if isinstance(v, Traced) else v for k, v in kwargs.items()}
        
        # ... (Idempotency and Ledger Append)
        self._session_trace.append(trace_hash)
        return event


## III-7. Architectural Proofs: Defending Against Complex Attacks (Pattern G)

To demonstrate that the Typed Procedural API is rigorous enough for professional adaptivity, we examine its behavior in complex scenarios.

### Proof 1: Sample Size Re-estimation (Level 3 - Parameter Adaptation)
**Challenge**: Ensure that a protocol update is strictly schema-validated and correctly linked to its analysis basis.

```python
episode = ledger.bind(episode_id="TRIAL_01")
with episode.session() as ep:
    ctx = ep.Read(SSRContextProjector())
    
    # Pass 'ctx' directly. The fingerprint caches the result.
    insight = ep.CommitCallResult(SSRResult, compute_cp_and_n, ctx)
    
    # Protocol updates are also adaptive actions.
    if insight.recommended_n > 500:
        ep.UpdateProtocol(MaxNUpdate(new_n=insight.recommended_n))
```

### Proof 2: Platform Trials (Level 4 - Arm Adaptation)
**Challenge**: Managing multiple arms entering/exiting, ensuring Arm A's results don't collide with Arm B.

```python
episode = ledger.bind(episode_id="TRIAL_PLATFORM")
with episode.session() as ep:
    platform = ep.Read(PlatformProjector())
    
    for arm_id in platform.active_arms:
        # 1. Targeted Read for a specific arm
        summary = ep.Read(BinomialProjector(arm=arm_id))
        
        # 2. The analysis is strictly bound to 'summary'
        res = ep.CommitCallResult(AnalysisResult, perform_analysis, summary)
        
        if res.z > res.bounds.futility:
             ep.Decision(Decision.ArmFutility(arm=arm_id))
```

### Proof 3: Latency & Pipeline (The Waitlist Challenge)
**Challenge**: Preventing "Double Stopping" when multiple analyses are run while outcome data is still in the pipeline.

By including the **Pipeline Hash** (enrolled subjects with pending outcomes) in the `CommitCallResult` trace hash, Pattern G ensures that a heavy analysis is only "New" if the pipeline has changed. If the number of pending patients remains the same and no outcomes have arrived, `CommitCallResult` returns the existing `res` immediately, and the library's internal `ep.Decision` check prevents a redundant action.

### Proof 4: Stochastic Reproducibility (Bayesian MCMC)
**Challenge**: Guaranteeing that stochastic computations (e.g., MCMC sampling) are "frozen" once computed, ensuring deterministic replay.

The architecture ensures that once a stochastic result is committed to the Ledger, all subsequent sessions within the same scientific context retrieve that exact realization rather than re-executing the randomness. 

```python
episode = ledger.bind(episode_id="TRIAL_01")
with episode.session() as ep:
    ctx = ep.Read(InferenceProjector())
    
    # 1. perform_mcmc executes on the first run.
    # 2. Subsequent runs detect that 'ctx' (scientific trace) hasn't changed.
    # 3. The library skips the solver and returns the committed record.
    res = ep.CommitCallResult(BayesianResult, perform_mcmc, ctx, seed=42)
    
    if res.posterior_mean > threshold:
        ep.Decision(Decision.EfficacySignal())
```

### Proof 5: Multi-Generation Challenge (Platform Trials)
**Challenge**: Maintaining $O(N)$ performance for state reconstruction in long-running platform trials with evolving protocols.

Pattern G, combined with **Persistent Snapshots** (II-1), addresses this by ensuring that `ep.Read()` operations can leverage the latest snapshot, and `ep.CommitCallResult()` trace hashes only need to consider the differential events since that snapshot. The `Induction Snapshot` in the trace hash ensures that even if the underlying protocol changes, the system correctly identifies when a re-computation is necessary, while the snapshotting mechanism keeps the replay cost manageable.

## III-8. The Repeatable Process Pattern (Idempotency of Action)

In production, the library is typically used within a repeatable `process(data)` loop. For this to work without "double-shooting" (e.g., stopping a trial multiple times), the library ensures that not only are **Analyses** idempotent, but **Actions** (Decisions, Protocol Updates) are too.

### The Ingest -> Process Cycle

```python
def process_batch(episode_id, new_data):
    episode = ledger.bind(episode_id=episode_id)
    with episode.session() as ep:
        # 1. Ingest new evidence using a Model
        ep.Ingest(ObservationBatch(data=new_data))
        
        # 2. Re-run the same inference logic
        run_dmc_logic(ep)
```

### Library-Side Guarantee: Basis Idempotency

The library protects the Ledger by tracking the relationship between a `basis` and an `Action`.

```python
def Decision(self, name, **kwargs):
    # Current scientific trace for this action
    trace_hash = hash(self.snapshot_id, self.trace, name, kwargs)

    # Check for existing action linked to this scientific lineage
    if self.ledger.query(DecisionRecord, trace_hash=trace_hash):
        return # Skip redundant action
        
    self.ledger.AppendDecision(name, trace_hash, kwargs)
```

**Conclusion on Loop Compatibility**:
Because `ep.Commit` is idempotent based on the *Evidence*, and `ep.Decision` is idempotent based on the *Basis*, the entire complex branching script becomes a **declarative description of intent** that can be executed repeatedly. The developer does not need to check "Is the trial already stopped?" or "Have I already updated the protocol?". They simply state the rules, and the library ensures the Ledger remains a lean, non-redundant audit trail.

---

---

## IV. Future Considerations (Experimental Patterns)

The follow patterns are considered "Power Features" for highly complex trial orchestration and are not intended for the initial implementation phase.

### IV-1. Pattern E: Transparent Policy Pattern (Logical Autopilot)
Uses a **Replay-Aware Policy** mechanism and a `@ep.policy` decorator to track execution paths across multiple sessions. While powerful for modular, multi-disciplinary DMC logic, it introduces complexity that may be non-intuitive for standard procedural workflows.

### IV-2. Pattern D: Block-based Imperative (Define-by-Run)

Pattern D provides a **Define-by-Run** experience, where the execution path *is* the definition. It avoids the functional abstraction (lambdas) of Pattern C in favor of an imperative block structure. This allows users to write complex, branching logic directly in the main scope while the library manages "Step skipping" based on the evidence fingerprint.

```python
episode = ledger.bind(episode_id="TRIAL_01")
with episode.session() as ep:
    ctx = ep.Read(InferenceProjector())

    with ep.Step("HeavyAnalysis") as s:
        if s.active: 
             result = perform_mcmc(ctx)
             s.Commit(AnalysisResult(z=result.z, p=result.p))
        
        res = s.result

    if res.p < 0.05:
        ep.Decision(Decision.StopEfficacy(), basis=res)
```

---

---

---

## VI. Appendix: Baseline Structural Integrity Tests (Pattern B)

To ensure the underlying Ledger logic is sound, we maintain these baseline stress tests for the Managed Procedural API (Pattern B). These demonstrate that even without the deferred execution of Pattern C, the Fingerprint mechanism provides fundamental defense against adaptive trial "attacks".

### Stress Test B-1: Explicit SSR Linkage
Ensures that a protocol update ($N_{max}$) is strictly measurable with respect to the specific analysis event that triggered it.

```python
# [Pattern B Logic]
insight = ep.Commit("SSR_Analysis", cp=0.45, new_n=500)
ep.UpdateProtocol(MaxNUpdate(new_n=500))
```

### Stress Test B-2: Subgroup/Arm Isolation
Demonstrates that multi-arm analysis events are physically separated by their event names and evidence fingerprints, preventing cross-arm decision leakage.

---

## VII. Appendix: Lazy Procedural Stress Tests (Pattern C Baseline)

These tests validate the deferred execution logic of Pattern C, ensuring that "heavy" computations are only invoked when the scientific context (Snapshot + Reference-Set) has meaningfully changed.

### Stress Test C-1: Deferred SSR Calculation
```python
# [Pattern C Logic]
# compute_cp_and_n is ONLY called on the first encounter of this context.
insight = ep.CommitCallResult("SSR_Analysis", compute_cp_and_n, history)
```

---

## References

*   Fowler, M. (2005, December 12). *Event Sourcing*. Martinfowler.Com. https://martinfowler.com/eaaDev/EventSourcing.html
*   Young, G. (2017). *Versioning in an Event Sourced System* (v10 ed.). Leanpub. https://leanpub.com/read/esversioning
