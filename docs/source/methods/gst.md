# Group Sequential Tests
**Terminate futile experiments early with planned interim analyses.**

This is a classic and powerful method where you pre-define several points for "interim analysis" during the experiment (e.g., every week, every 1000 users). At each point, you evaluate for statistical significance or futility (i.e., concluding that continuing the test is unlikely to yield a significant result).

* **Use Cases:**
    * When analysis timing is fixed, such as for weekly business reviews.
    * For structured experiments like clinical trials that require strict error control.
* **Key Features:**
    * **Spending Functions:** Supports various functions like O'Brien-Fleming (conservative) and Pocock (aggressive) to control the probability of early termination.
    * **Stopping for Efficacy or Futility:** Allows for early stopping based on conclusions of either effectiveness ("there is an effect") or futility ("unlikely to find a significant difference even if continued").

---

## Group Sequential Test — Functional Design System
(with explicit PowerCurve and consistent bracket notation)

### Core notation
- `H₀, H₁(δ)`: null / alternative models
- `α(t), β(t)`: error spending functions
- `{tᵢ}`: information schedule (0 < t₁ < … < t_K ≤ 1)
- `{cᵢ}`: boundary vector (upper/lower)
- `Powerᵢ(δ)`: power at look i for effect δ
- `PowerCurve(δ)`: overall power as a function of δ
- `ASN, TotalSN`: average / maximal sample size
- `BindingMode`: {Binding | NonBinding}
- `Objective`: e.g., minimize E_{H₁}[N_stop]

### 1) Core functions (signatures)

#### Fixed-sample reference
- `CriticalValue`: (α, H₀) → c
- `SampleSize`: (β, c, H₁) → N
- `PowerFunction`: (N, c, H₁) → (1−β)
- `EffectSizeDesign`: (α, β, δ) → (c, N)
- `PowerCurve_Fixed`: (N, α, H₀, H₁) → [δ ↦ Power(δ)]

#### Group Sequential Test (GST)
- `Design`: (α(t), β(t), H₀, BindingMode, {tᵢ}) → {cᵢ}
- `Performance`: ({tᵢ}, {cᵢ}, H₁) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
- `PowerCurve_GST`: ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
- `Sensitivity`: ({tᵢ}, {cᵢ}, H₀) → ({MDEᵢ})
- `SpendingOptimization`: (H₀, H₁, Objective) → (α*(t), β*(t))
- `ScheduleOptimization`: (Objective, α(t), β(t), H₀, H₁) → {tᵢ*}
- `InverseDesign`: ({TargetPowerᵢ* | MDEᵢ*}, H₁) → ({cᵢ}, {tᵢ})
- `FitSpending`: ({cᵢ}, {tᵢ}) → (α̂(t), β̂(t))  [fit spending to boundaries]

#### Explicit schedule choice (user- or data-driven)
- `ChooseT.UserProvided`: ({t_given}) → {tᵢ}
- `ChooseT.CalendarDriven`: (calendar, accrual, lag) → {tᵢ}

### 2) Binding futility boundary
- **Binding**: futility contributes to α → efficacy re-optimized (less conservative)
- **NonBinding**: futility excluded from α → efficacy fixed (more conservative)

### 3) Pipelines (Inputs → Outputs; {tᵢ} explicit)

#### Forward (spending-based):
```
(α(t), β(t), H₀, BindingMode, ChooseT_params)
  |> ChooseT.*
     : (ChooseT_params) → {tᵢ}
  |> Design
     : (α(t), β(t), H₀, BindingMode, {tᵢ}) → {cᵢ}
  |> Performance
     : ({tᵢ}, {cᵢ}, H₁, δ₀) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ}, {cᵢ}, H₀, target_power) → ({MDEᵢ})
```

#### Inverse (power/MDE-based):
```
({TargetPowerᵢ* | MDEᵢ*}, H₀, H₁, δ₀)
  |> InverseDesign
     : ({TargetPowerᵢ* | MDEᵢ*}, H₀, H₁, δ₀) → ({cᵢ}, {tᵢ})
  |> FitSpending
     : ({cᵢ}, {tᵢ}, H₀) → (α̂(t), β̂(t))
  |> Performance
     : ({tᵢ}, {cᵢ}, H₁, δ₀) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ}, {cᵢ}, H₀, target_power) → ({MDEᵢ})
```

#### Optimization (spending and/or schedule):
```
(H₀, H₁, δ₀, target_power, max_N, Objective)
  |> SpendingOptimization
     : (H₀, H₁, δ₀, target_power, max_N, Objective) → (α*(t), β*(t))
  |> ScheduleOptimization
     : (α*(t), β*(t), H₀, H₁, δ₀, target_power, max_N, Objective) → {tᵢ*}
  |> Design
     : (α*(t), β*(t), H₀, BindingMode, {tᵢ*}) → {cᵢ*}
  |> Performance
     : ({tᵢ*}, {cᵢ*}, H₁, δ₀) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ*}, {cᵢ*}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ*}, {cᵢ*}, H₀, target_power) → ({MDEᵢ})
```

### 4) Use-case Scenarios (I/O annotated with PowerCurve)

#### Digital A/B test (NonBinding; frequent peeks)
```
(α(t), β(t), H₀, n_analyses)
  |> ChooseT.EquallySpaced
     : (n_analyses) → {tᵢ}
  |> Design
     : (α(t), β(t), H₀, BindingMode=NonBinding, {tᵢ}) → {cᵢ}
  |> Performance
     : ({tᵢ}, {cᵢ}, H₁, δ₀) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ}, {cᵢ}, H₀, target_power) → ({MDEᵢ})
```
```
Given: (α, β, H₀, H₁(δ), peek_schedule)
(α(t):=OBF, β(t), H₀)
  |> ChooseT.UserProvided(peek_schedule)
     : ({t_peek}) → {tᵢ}
  |> Design
     : (α(t), β(t), H₀, BindingMode.NON_BINDING, {tᵢ}) → {cᵢ}
  |> Performance
     : ({tᵢ}, {cᵢ}, H₁(δ)) → ({Powerᵢ(δ)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ}, {cᵢ}, H₀) → ({MDEᵢ})
```
**Note**: Use O'Brien-Fleming spending for conservative Type I error control. Frequent peeks are allowed since spending function guarantees α control regardless of peek frequency.

#### Clinical Phase III (Binding; ethical early stop)
```
(H₀, H₁, δ₀, target_power, max_N, Objective:=min E_{H₁}[N_stop])
  |> SpendingOptimization
     : (H₀, H₁, δ₀, target_power, max_N, Objective) → (α*(t), β*(t))
  |> ScheduleOptimization
     : (α*(t), β*(t), H₀, H₁, δ₀, target_power, max_N, Objective) → {tᵢ*}
  |> Design
     : (α*(t), β*(t), H₀, BindingMode=Binding, {tᵢ*}) → {cᵢ*}
  |> Performance
     : ({tᵢ*}, {cᵢ*}, H₁, δ₀) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ*}, {cᵢ*}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ*}, {cᵢ*}, H₀, target_power) → ({MDEᵢ})
```
```
Given: (α, β, H₀, H₁(δ₀), Objective:=min E_{H₁}[N_stop], N_max)
(H₀, H₁(δ₀), Objective)
  |> SpendingOptimization
     : (α, β, {tᵢ}, δ₀, target_power, N_max) → γ*(α), γ*(β)
  |> α*(t):=HSD(γ*(α)), β*(t):=HSD(γ*(β))
  |> ScheduleOptimization
     : (α*(t), β*(t), k, δ₀, target_power, N_max) → {tᵢ*}
  |> Design
     : (α*(t), β*(t), H₀, BindingMode.BINDING, {tᵢ*}) → {cᵢ*}
  |> Performance
     : ({tᵢ*}, {cᵢ*}, H₁(δ₀)) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : ({tᵢ*}, {cᵢ*}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity
     : ({tᵢ*}, {cᵢ*}, H₀) → ({MDEᵢ})
```

#### Promising-zone adaptive re-estimation
```
(α(t), β(t), H₀, {t_given})
  |> ChooseT.UserProvided
     : ({t_given}) → {tᵢ₀}
  |> Design
     : (α(t), β(t), H₀, BindingMode, {tᵢ₀}) → {cᵢ₀}
Interim (at t_current):
  ConditionalUpdate
     : (observed_Z, t_current, {tᵢ₀}, {cᵢ₀}, α, assumed_δ) → ({cᵢ′}, {tᵢ′}, N′, decision)
  Performance
     : ({tᵢ′}, {cᵢ′}, H₁, δ₀) → ({Powerᵢ′(δ₀)}, ASN′)
  PowerCurve_GST
     : ({tᵢ′}, {cᵢ′}, H₁) → [δ ↦ Power′(δ)]
  Sensitivity
     : ({tᵢ′}, {cᵢ′}, H₀, target_power) → ({MDEᵢ′})
```
```
Given: (α, β, H₀, {tᵢ₀}, CP_threshold)
Setup:
(α(t), β(t), H₀)
  |> ChooseT.UserProvided
     : ({t_given}) → {tᵢ₀}
  |> Design
     : (α(t), β(t), H₀, BindingMode, {tᵢ₀}) → {cᵢ₀}

Interim (at look j):
(Z_observed, t_j, {tᵢ₀}, {cᵢ₀}, δ_assumed)
  |> ConditionalUpdate.conditional_power
     : (Z_j, t_j, t_final, c_final, δ) → CP
  |> ConditionalUpdate.promising_zone_decision
     : (Z_j, t_j, c_j^eff, c_j^fut, t_final, c_final, δ, CP_threshold) → decision
  |> If decision="continue":
       ConditionalUpdate.update_remaining_boundaries
         : (t_j, {t_{j+1:K}}, α, γ, α_spent) → ({cᵢ′}, {tᵢ′})
  |> Performance
     : ({tᵢ′}, {cᵢ′}, H₁(δ)) → ({Powerᵢ′(δ)}, ASN′)
  |> PowerCurve_GST
     : ({tᵢ′}, {cᵢ′}, H₁) → [δ ↦ Power′(δ)]
  |> Sensitivity
     : ({tᵢ′}, {cᵢ′}, H₀) → ({MDEᵢ′})
```

### 5) Optimization: Objectives vs Constraints

**Important distinction:**

- **Optimization objectives** (what we minimize/maximize):
  - `Objective := min E_{H₁}[N_stop]` (minimize expected sample size under H₁)
  - `Objective := max Power(δ₀)` (maximize power at target effect size)

- **Constraints** (what we must satisfy):
  - `max_N`: Maximum allowable sample size (budget/resource constraint)
  - `target_power`: Minimum required power (e.g., ≥ 0.90)
  - `α`: Overall Type I error rate (must be ≤ α)
  - `β`: Overall Type II error rate (must be ≤ β)

**Common pattern:** N-constrained schedule optimization
```
Given: (α, β, H₀, H₁, δ₀, target_power, max_N)
Objective: Minimize E_{H₁}[N_stop]
Constraints: Power(δ₀) ≥ target_power, N_max ≤ max_N

This is NOT "fixed N with MDE optimization" — MDE is a result metric, not the optimization target.
The actual optimization is: find {tᵢ*} that minimizes ASN while respecting N_max constraint.
```

**Note on MDE:**
- MDE (Minimum Detectable Effect) is computed AFTER design optimization as a sensitivity metric
- It answers: "Given this design, what's the smallest effect we can detect with target_power?"
- It is NOT an optimization parameter — we don't optimize γ or {tᵢ} to achieve specific MDE
- Use `Sensitivity` function to compute MDE from finalized design: `({tᵢ}, {cᵢ}, H₀, target_power) → ({MDEᵢ})`

### 6) Power curve family

- `PowerCurve_Fixed`: (N, α, H₀, H₁) → [δ ↦ Power(δ)]
- `PowerCurve_GST`: ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
- `PowerSurface`: ({tᵢ}, {cᵢ}, H₁) → [ (t, δ) ↦ Power(t, δ) ]
  - allows visualization of per-look accumulation of power
- `MDEProfile`: ({tᵢ}, {cᵢ}, H₀) → [ t ↦ MDE(t) ]

### 7) Quick chooser (scenario → pipeline core)

- **Digital A/B test**: ChooseT.UserProvided(peek_schedule) → Design(OBF, NonBinding) → Performance → PowerCurve_GST → Sensitivity
- **Clinical (Binding)**: SpendingOptimization → ScheduleOptimization → Design(Binding) → Performance → PowerCurve_GST → Sensitivity
- **N-constrained schedule optimization**: SpendingOptimization(max_N) → ScheduleOptimization(max_N, min ASN) → Design → Performance → Sensitivity
- **Early detection pattern**: InverseDesign → FitSpending → Performance → PowerCurve_GST → Sensitivity
- **Promising-zone adaptive**: Design({tᵢ₀}) → ConditionalUpdate(CP, decision) → Performance → PowerCurve_GST → Sensitivity

---

## Function Implementation Reference

| Functional Design | EarlySign Implementation | Status |
|-------------------|-------------------------|--------|
| **Spending Functions** | | |
| `obf_spending(t, α)` | `spending.obf_spending` | ✅ |
| `pocock_spending(t, α)` | `spending.pocock_spending` | ✅ |
| `hsd_spending(t, α, γ)` | `spending.hsd_spending` | ✅ |
| `beta_obf_spending(t, β)` | `spending.beta_obf_spending` | ✅ |
| `beta_pocock_spending(t, β)` | `spending.beta_pocock_spending` | ✅ |
| `beta_hsd_spending(t, β, γ)` | `spending.beta_hsd_spending` | ✅ |
| **Design & Boundaries** | | |
| `Design` | `BoundaryCalculator.compute_boundary` | ✅ |
| `Design` (batch) | `BoundaryCalculator.compute_boundaries` | ✅ |
| Efficacy boundary | `boundaries.efficacy_boundary_from_spending` | ✅ |
| Futility boundary | `boundaries.futility_boundary_from_spending` | ✅ |
| **Information Time** | | |
| `ChooseT.UserProvided` | `information.choose_t_user_provided` | ✅ |
| `ChooseT.EquallySpaced` | `information.choose_t_equally_spaced` | ✅ |
| `info_time_from_sample_size` | `information.info_time_from_sample_size` | ✅ |
| `info_time_from_ratio` | `information.info_time_from_ratio` | ✅ |
| `info_time_from_variance` | `information.info_time_from_variance` | ✅ |
| `info_time_from_sd` | `information.info_time_from_sd` | ✅ |
| `info_time_from_fisher` | `information.info_time_from_fisher` | ✅ |
| `ChooseT.CalendarDriven` | `calendar.choose_t_calendar_driven` | ✅ |
| **Performance & Analysis** | | |
| `Performance` | `performance.performance` | ✅ |
| `PowerCurve_GST` | `performance.power_curve` | ✅ |
| `Sensitivity` | `performance.sensitivity` | ✅ |
| `FitSpending` | `fitting.fit_spending` | ✅ |
| `InverseDesign` | `inverse_design.inverse_design_from_power` | ✅ |
| `InverseDesign` (MDE) | `inverse_design.inverse_design_from_mde` | ✅ |
| `InverseDesign` (sample size) | `inverse_design.optimize_sample_size_for_power` | ✅ |
| **Scale Conversions** | | |
| `cumulative_to_nominal_z` | `boundary.nominal_z_from_spent_alpha` | ✅ |
| `level_to_nominal_z` | `boundary.nominal_z_from_level` | ✅ |
| `z_to_brownian` | `boundary.z_to_brownian` | ✅ |
| `brownian_to_z` | `boundary.brownian_to_z` | ✅ |
| `convert_scale` | `boundary.convert_statistic_scale` | ✅ |
| **Type Definitions** | | |
| `BindingMode` | string literal (`"binding"`, `"non_binding"`) | ✅ |
| `SpendingFamily` | string literal (`"obrien_fleming"`, `"pocock"`, `"hsd"`) | ✅ |
| `BoundaryScale` | string literal (`"z"`, `"bm"`) | ✅ |
| `FutilityMode` | string literal (`"none"`, `"symmetric"`, `"fixed_threshold"`, `"beta_spending"`, `"custom"`) | ✅ |
| `DesignPayload` | `applications.design.group_sequential.initial_design.schema.DesignPayloadModel` | ✅ |
| **Ledger Operators** | | |
| — | `records.GroupSequentialDesignRecord` (insert) | ✅ |
| — | `methods.group_sequential.boundary.BoundaryFromDesign` | ✅ |
| — | `methods.group_sequential.information_time.InformationTime` | ✅ |
| — | `methods.group_sequential.information_time.InformationTimeFromRatio` | ✅ |
| — | `methods.group_sequential.decision.GSDecision` | ✅ |
| — | `methods.group_sequential.decision.GSDecisionFromWaldZ` | ✅ |
| **Advanced & Visualization** | | |
| `InverseDesign` (power) | `inverse_design.inverse_design_from_power` | ✅ |
| `InverseDesign` (MDE) | `inverse_design.inverse_design_from_mde` | ✅ |
| `PowerSurface` | `visualization.power_surface` | ✅ |
| `PowerSurface` (grid) | `visualization.power_surface_grid` | ✅ |
| `MDEProfile` | `visualization.mde_profile` | ✅ |
| `PowerContours` | `visualization.power_contours` | ✅ |
| `SpendingOptimization` | `spending_optimization.optimize_spending_for_asn` | ✅ |
|  | `spending_optimization.optimize_spending_for_power` | ✅ |
| `ScheduleOptimization` | `optimize_timing.minimize_asn.minimize_asn_schedule` | ✅ |
|  | `optimize_timing.minimize_asn.summarize_schedule` | ✅ |
| `ConditionalUpdate` | `group_sequential.sample_size_reestimation.operators.ConditionalPowerCalculation` | ✅ |
|  | `conditional_update.update_remaining_boundaries` | ✅ |
|  | `conditional_update.promising_zone_decision` | ✅ |

**Legend**: ✅ Implemented

**Module locations**: Core `essentials` live in `earlysign.methods.group_sequential.<module>` with adaptive-routine implementations in `earlysign.methods.adaptive_group_sequential.conditional_update`. Execution-layer adaptive operators/records live in `earlysign.methods.group_sequential.sample_size_reestimation`.
