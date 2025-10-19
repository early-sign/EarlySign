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
(α(t), β(t), H₀, BindingMode)
  |> ChooseT.*                        : (…) → {tᵢ}
  |> Design                           : (α(t), β(t), H₀, BindingMode, {tᵢ}) → {cᵢ}
  |> Performance                      : ({tᵢ}, {cᵢ}, H₁) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST                   : ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity                      : ({tᵢ}, {cᵢ}, H₀) → ({MDEᵢ})
```

#### Inverse (power/MDE-based):
```
({TargetPowerᵢ* | MDEᵢ*}, H₁)
  |> InverseDesign                    : (…) → ({cᵢ}, {tᵢ})
  |> FitSpending                      : ({cᵢ}, {tᵢ}) → (α̂(t), β̂(t))
  |> Performance                      : ({tᵢ}, {cᵢ}, H₁) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST                   : ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity                      : ({tᵢ}, {cᵢ}, H₀) → ({MDEᵢ})
```

#### Optimization (spending and/or schedule):
```
(H₀, H₁, Objective)
  |> SpendingOptimization             : (…) → (α*(t), β*(t))
  |> ScheduleOptimization             : (Objective, α*(t), β*(t), H₀, H₁) → {tᵢ*}
  |> Design                           : (α*(t), β*(t), H₀, BindingMode, {tᵢ*}) → {cᵢ*}
  |> Performance                      : ({tᵢ*}, {cᵢ*}, H₁) → ({Powerᵢ(δ₀)}, ASN)
  |> PowerCurve_GST                   : ({tᵢ*}, {cᵢ*}, H₁) → [δ ↦ Power(δ)]
  |> Sensitivity                      : ({tᵢ*}, {cᵢ*}, H₀) → ({MDEᵢ})
```

### 4) Use-case Scenarios (I/O annotated with PowerCurve)

#### Digital A/B test (NonBinding; frequent peeks)
```
(α(t), β(t), H₀)
  |> ChooseT.FrequencyGuard
     : (…) → {tᵢ}
  |> Design
     : (…) → {cᵢ}
  |> Performance
     : (…) → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST
     : (…) → [δ ↦ Power(δ)]
  |> Sensitivity
     : (…) → ({MDEᵢ})
```

#### Clinical Phase III (Binding; ethical early stop)
```
(H₀, H₁, Objective:=min E_{H₁}[N_stop])
  |> SpendingOptimization → (α*(t), β*(t))
  |> ScheduleOptimization → {tᵢ*}
  |> Design → {cᵢ*}
  |> Performance → ({Powerᵢ(δ₀)}, ASN, TotalSN)
  |> PowerCurve_GST → [δ ↦ Power(δ)]
  |> Sensitivity → ({MDEᵢ})
```

#### Promising-zone adaptive re-estimation
```
(α(t), β(t), H₀)
  |> ChooseT.UserProvided → {tᵢ₀}
  |> Design → {cᵢ₀}
Interim:
  ConditionalUpdate → ({cᵢ′}, {tᵢ′}, N′)
  Performance → ({Powerᵢ′(δ₀)}, ASN′)
  PowerCurve_GST → [δ ↦ Power′(δ)]
  Sensitivity → ({MDEᵢ′})
```

### 5) Power curve family

- `PowerCurve_Fixed`: (N, α, H₀, H₁) → [δ ↦ Power(δ)]
- `PowerCurve_GST`: ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]
- `PowerSurface`: ({tᵢ}, {cᵢ}, H₁) → [ (t, δ) ↦ Power(t, δ) ]
  - allows visualization of per-look accumulation of power
- `MDEProfile`: ({tᵢ}, {cᵢ}, H₀) → [ t ↦ MDE(t) ]

### 6) Quick chooser (scenario → pipeline core)

- **Clinical (Binding)**: SpendingOptimization → ScheduleOptimization → Design → Performance → PowerCurve_GST → Sensitivity
- **A/B test (NonBinding)**: ChooseT.FrequencyGuard → Design → Performance → PowerCurve_GST → Sensitivity
- **Early detection pattern**: InverseDesign → ImpliedSpending → Performance → PowerCurve_GST → Sensitivity
- **Promising-zone adaptive**: Design({tᵢ₀}) → ConditionalUpdate → Performance → PowerCurve_GST → Sensitivity
