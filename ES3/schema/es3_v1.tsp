import "@typespec/openapi3";

@service(#{ title: "EarlySign Standard Schema (ES3)" })
namespace ES3;

/**
 * EarlySign Standard Schema (ES3) - Manifest
 * Version: v1.0.0
 * 
 * NAMING CONVENTIONS:
 * 1. Properties (model fields): snake_case
 *    - EarlySign is primarily Python-based; snake_case is the natural choice
 *    - Example: `h_null`, `test_logic`, `response_type`
 * 
 * 2. Models: PascalCase
 *    - Follows TypeSpec/TypeScript conventions as well as Python class naming conventions
 *    - Example: `Protocol`, `TaskSpec`, `BoundarySpec`
 * 
 * 3. Enums:
 *    - Enum name: PascalCase (e.g., `DecisionStatus`, `SpendingFunctionType`)
 *    - Enum keys: PascalCase (e.g., `ObrienFleming`, `StopEfficacy`)
 *    - Enum values: snake_case (e.g., `"obrien_fleming"`, `"stop_efficacy"`)
 *    - Rationale: Keys are TypeScript-style for IDE support; values are lowercase for modern API design
 * 
 * 4. Namespaces: PascalCase
 *    - Example: `ES3`, `GST`, `Base`, `AVI`
 * 
 * 5. Inline literal union values: snake_case
 *    - When enum extraction is unnecessary
 *    - Example: `"z_score" | "p_value" | "score_statistic"`
 * 
 * 6. Constants (default values): match their context
 *    - String literals: snake_case (e.g., `"group_sequential"`)
 *    - Version strings: semantic versioning format (e.g., `"v1.0.0"`)
 *
 * DESIGN PHILOSOPHY:
 * 1. Polymorphic Schema Design:
 *    The schema is polymorphic: some properties exist conditionally based on the values of other
 *    properties. This design allows for flexibility but requires careful handling at the implementation level.
 *    
 *    Facade Granularity:
 *    When implementing a specific Protocol's engine or facade (e.g., an EarlySign Template),
 *    polymorphism should be largely resolved at the implementation level. In other words,
 *    the facade code should be defined granularly enough so that most schema choices are fixed.
 *    If a facade leaves too much polymorphism unresolved, it should likely be split into more
 *    specific facades.
 * 
 * 2. Producer/Consumer Correspondence:
 *    Sub-schemas should ideally correspond to their respective producer or consumer components. This alignment
 *    ensures that the handling of schema instances remains abstract and decoupled.
 * 
 * 3. Semantic Versioning:
 *    We use semantic versioning for the ES3 schema. This ensures that changes to the schema are clearly documented
 *    and that the schema can be easily updated in the future.
 */

/** Schema version */
const ES3_VERSION = "v1.0.0";

namespace Base {
  /**
   * Metadata for EarlySign events and records.
   */
  model Metadata {
    /** Schema version */
    ES3_version?: string = ES3_VERSION;
  }

  /**
   * The Root Protocol Container.
   * Defines the high-level structure of any EarlySign protocol.
   */
  model Protocol {
    /** Schema version */
    ES3_version?: string = ES3_VERSION;
    name: string;
    task: TaskSpec;
    method: MethodSpec;
  }

  /**
   * Base class for Problem Definition.
   * Concrete protocols should extend this (e.g., GST.TaskSpec).
   */
  model TaskSpec {
    kind: string;
  }

  /**
   * Base class for Operational Method.
   * Concrete protocols should extend this (e.g., GST.MethodSpec).
   */
  model MethodSpec {
    kind: string;
  }

  /**
   * Base log/event model for metadata inclusion.
   */
  model Log {
    /** Schema version */
    ES3_version?: string = ES3_VERSION;
  }
}

namespace GST {
  /**
   * GST-Specific Protocol Container.
   * Enforces that task and method belong to the GST domain.
   */
  model Protocol extends Base.Protocol {
    task: TaskSpec;
    method: MethodSpec;
  }

  /**
   * GST-Specific Task Definition (Univariate).
   */
  model TaskSpec extends Base.TaskSpec {
    kind: "group_sequential" = "group_sequential"; // Discriminator value for Base
    
    arms: string[]; 
    response_type: "binary" | "continuous" | "time_to_event";
    hypotheses: HypothesisSpec; 

    // Design Requirements (Designer Input)
    // Presence implies the intent to stop for this reason.
    efficacy?: EfficacyRequirement;
    futility?: FutilityRequirement;
  }

  /**
   * GST-Specific Method Definition (Univariate).
   */
  model MethodSpec extends Base.MethodSpec {
    kind: "group_sequential" = "group_sequential"; 
    
    // Stopping Policy (discriminated union)
    stopping_policy: StoppingPolicySpec;
    schedule: ScheduleSpec;
    
    adaptation?: AdaptationSpec;
  }

  // --- Sub-Components ---

  /**
   * Efficacy requirement for group sequential design.
   * Specifies the Type I error rate (alpha) to control.
   */
  model EfficacyRequirement {
    alpha: float64; // Type I Error target
  }

  /**
   * Futility requirement for group sequential design.
   * Specifies the target power (1 - beta) and binding behavior.
   */
  model FutilityRequirement {
    power: float64; // Target Power (1 - Beta)
    binding?: boolean = false; // Non-binding default (regulatory best practice)
  }

  /**
   * Discriminated Union for Stopping Policy types.
   */
  union StoppingPolicySpec {
    AlphaSpendingPolicy,
    BetaSpendingPolicy,
    AlphaBetaSpendingPolicy,
    WhiteheadPolicy,
    OBrienFlemingPolicy,
  }

  /**
   * Abstract Base for Stopping Policy.
   */
  model StoppingPolicyBase {
    kind: string;
  }

  /**
   * Alpha (Efficacy) Spending Policy.
   * Supports one-sided or two-sided tests.
   */
  model AlphaSpendingPolicy extends StoppingPolicyBase {
    kind: "alpha_spending" = "alpha_spending";
    spending_fn: SpendingFunctionSpec;
    alpha: float64; // Fallback to Task.efficacy.alpha if omitted
    sided: "one" | "two" = "two";  // default two-sided
  }

  /**
   * Beta (Futility) Spending Policy.
   * Always one-sided (lower boundary).
   */
  model BetaSpendingPolicy extends StoppingPolicyBase {
    kind: "beta_spending" = "beta_spending";
    spending_fn: SpendingFunctionSpec;
    beta: float64; // Fallback to Task.futility.beta (1-power) if omitted
  }

  /**
   * Combined Alpha-Beta Spending Policy.
   * Efficacy (upper) + Futility (lower) boundaries.
   */
  model AlphaBetaSpendingPolicy extends StoppingPolicyBase {
    kind: "alpha_beta_spending" = "alpha_beta_spending";
    alpha_spending_fn: SpendingFunctionSpec;
    beta_spending_fn: SpendingFunctionSpec;
    alpha: float64; // Fallback to Task.efficacy.alpha if omitted
    beta: float64;  // Fallback to Task.futility.beta (1-power) if omitted
    alpha_binding?: boolean = true;
    beta_binding?: boolean = false;
  }

  /**
   * Classic O'Brien-Fleming Policy.
   * Shortcut for O'Brien-Fleming spending.
   */
  model OBrienFlemingPolicy extends StoppingPolicyBase {
    kind: "obrien_fleming" = "obrien_fleming";
    alpha: float64; // Fallback to Task.efficacy.alpha if omitted
    sided: "one" | "two" = "two";
  }

  /**
   * Whitehead's Triangular Test Policy.
   * Typically defines both efficacy and futility boundaries.
   */
  model WhiteheadPolicy extends StoppingPolicyBase {
    kind: "whitehead" = "whitehead";
    alpha: float64; // Fallback to Task.efficacy.alpha if omitted
    power: float64; // Fallback to Task.futility.power if omitted
  }

  /**
   * Logical Margins and Test Objectives.
   */
  model HypothesisParameters {
    kind: string;
  }

  model EqualityHypothesis extends HypothesisParameters { kind: "equality"; }
  model SuperiorityHypothesis extends HypothesisParameters { kind: "superiority"; superiority_margin: float64 = 0.0; }
  model NonInferiorityHypothesis extends HypothesisParameters { kind: "non_inferiority"; non_inferiority_margin: float64; }
  model EquivalenceHypothesis extends HypothesisParameters { kind: "equivalence"; lower_margin: float64; upper_margin: float64; }

  model HypothesisSpec {
    h_null: string;
    h_alt: string;
    test_logic: HypothesisParameters;
    target_effect: EffectSizeSpec;
  }

  /**
   * Numerical assumptions for the Alternative Hypothesis (H1).
   * Note: This defines the "ball position" (simulation target), separate from the "goal posts" (Hypothesis Margins).
   */
  model EffectSizeSpec {
    type: string;
  }

  model BinaryEffectSize extends EffectSizeSpec {
    type: "binary";
    proportions: Record<float64>; // Arm name -> Expected rate
  }

  model ContinuousEffectSize extends EffectSizeSpec {
    type: "continuous";
    means: Record<float64>; // Arm name -> Expected mean
    standard_deviation: float64;
  }

  model SurvivalEffectSize extends EffectSizeSpec {
    type: "time_to_event";
    hazard_ratios: Record<float64>; // Arm name -> Hazard Ratio (relative to control)
    median_survival_times?: Record<float64>; // Optional: for sample size calc
    event_rate?: float64; // Overall event rate if needed
  }

  /**
   * The "Core": Reference Model (Statistical Engine Assumptions).
   */
  model ReferenceModelSpec {
    kind: string;
  }

  model BinaryModel extends ReferenceModelSpec {
    kind: "binary" = "binary";
    test_statistic: "Z" | "chi_square" | "exact";
    link_function: "identity" | "log" | "logit" = "identity";
    use_canonical_joint_distribution: boolean;
  }

  model ContinuousModel extends ReferenceModelSpec {
    kind: "continuous" = "continuous";
    test_statistic: "Z" | "t";
    variance_assumption: "known" | "unknown" | "heteroscedastic";
    use_canonical_joint_distribution: boolean;
  }

  model SurvivalModel extends ReferenceModelSpec {
    kind: "time_to_event" = "time_to_event";
    test_statistic: "log_rank" | "cox_ph";
    use_canonical_joint_distribution: boolean;
  }

  model ScheduleSpec {
    unit: "sample_size" | "information_fraction" | "calendar_time" | "events";
    // Usually defined by N looks and spacing, or specific points.
    n_looks?: int32;
    interim_points?: float64[]; // Optional explicit points
  }

  model StoppingRule {
    schedule: ScheduleSpec; // The timing of the tests
    boundary: BoundarySpec; // The decision rule
  }

  // --- Boundary Definitions ---

  /**
   * Abstract Base for a Boundary Definition.
   */
  model BoundarySpec {
    kind: string;
    reference_model: ReferenceModelSpec; 
    boundary_scale: "z_score" | "p_value" | "score_statistic" = "z_score";
    binding: boolean = true;
  }

  /**
   * Available spending function types for group sequential design.
   */
  enum SpendingFunctionType {
    ObrienFleming: "obrien_fleming",
    Pocock: "pocock",
    KimDemets: "kim_demets",
    LanDemets: "lan_demets",
    PowerFamily: "power_family",
    HwangShihDecani: "hwang_shih_decani",
  }

  model SpendingFunctionSpec {
    family: SpendingFunctionType | string;
    params?: Record<float64>; // e.g. { "rho": 2.0, "gamma": -4.0 }
  }

  model SpendingBoundary extends BoundarySpec {
    kind: "spending";
    spending_function: SpendingFunctionSpec;
  }

  model FixedBoundary extends BoundarySpec {
    kind: "fixed";
    value: float64; // The fixed constant (e.g., 3.0)
  }

  model CustomBoundary extends BoundarySpec {
    kind: "custom";
    values: float64[]; // Explicit boundary values per look
  }

  namespace Log {
    /**
     * Status of an ongoing or completed Group Sequential Test.
     */
    enum DecisionStatus {
      Continue: "continue",
      StopEfficacy: "stop_efficacy",
      StopFutility: "stop_futility",
      StopPlanEndReached: "stop_plan_end_reached",
      Stop: "stop",
    }

    /**
     * Defines the reason / cause for an operational event (e.g., Analysis Execution).
     */
    model Trigger {
      kind: string;
    }

    /**
     * Triggered by planned protocol milestones or schedules.
     */
    model ScheduleTrigger extends Trigger {
      kind: "schedule";
      index: int32;
      value: float64;
    }

    /**
     * Triggered manually by a human operator.
     */
    model ManualTrigger extends Trigger {
      kind: "manual";
      reason: string;
    }

    /**
     * Structural Record of a GST Analysis Execution.
     * This captures the Context (Trigger), Metadata, and factual Result.
     */
    model Analysis extends Base.Log {
      trigger?: Trigger;
      
      // Results
      look: int32;
      statistic: float64;
      info_frac: float64;
      decision: DecisionStatus | string;
      
      efficacy_boundary?: float64;
      futility_boundary?: float64;
    }

    /**
     * Formalized Decision Event.
     */
    model Decision extends Base.Log {
      status: DecisionStatus | string;
      message: string;
    }

    /**
     * Result of a statistical test/evaluation for the study at a look.
     */
    model LookResult {
      look?: int32;
      sample_n: int32;
      info_frac: float64;
      z_stat: float64;

      efficacy_boundary?: float64;
      is_efficacy_crossed: boolean;

      futility_boundary?: float64;
      is_futility_crossed: boolean;

      status: DecisionStatus | string;
    }
  }

  model AdaptationSpec {
    type: string;
  }

  model SampleSizeReestimationSpec extends AdaptationSpec {
    type: "sample_size_reestimation";
    method: "conditional_power" | "predictive_power";
    target_power: float64;
    n_range: [uint32, uint32]; 
  }
}

/** Anytime-Valid Inference (AVI) Models */
namespace AVI {
  model TaskSpec extends Base.TaskSpec {
    kind: "anytime_valid" = "anytime_valid";
    null_p: float64;
    alt_p?: float64;
    alpha: float64;
  }

  model MethodSpec extends Base.MethodSpec {
    kind: "e_process" = "e_process";
    growth_rate?: float64;
  }

  model Protocol extends Base.Protocol {
    task: TaskSpec;
    method: MethodSpec;
  }
}

namespace GST.Multivariate {
  /**
   * Multivariate Task (Multiple Endpoints).
   * Composed of multiple Univariate GST Tasks.
   */
  model TaskSpec extends Base.TaskSpec {
    kind: "group_sequential_multivariate" = "group_sequential_multivariate"; 
    
    sub_tasks: GST.TaskSpec[]; // Reusing the Standard GST TaskSpec
    
    // Global parameters
    arms: string[]; // Study-wide arms (must be consistent with sub_tasks)
    
    /**
     * Global Hypothesis (e.g., "At least one effective" or "Global Null").
     * Typically represents the Union-Intersection or Intersection-Union test.
     */
    hypotheses: GST.HypothesisSpec; 
    
    alpha?: float64; // Global FWER
    beta?: float64;  // Global Power
  }

  model MethodSpec extends Base.MethodSpec {
    kind: "group_sequential_multivariate" = "group_sequential_multivariate";
    
    multiplicity_adjustment: "none" | "bonferroni" | "holm" | "hochberg" | "gatekeeping";
    correlation_matrix?: float64[][];
  }

  model Protocol extends Base.Protocol {
    task: TaskSpec;
    method: MethodSpec;
  }
}

namespace Binomial {
  /**
   * Raw evidence: data for a single arm.
   */
  model ArmData extends Base.Log {
    n: int32;
    success: int32;
    arm: string;
  }

  /**
   * Standard statistics for a single arm (Bernoulli/Binomial).
   */
  model ArmMetrics {
    n: int32;
    successes: int32;
    p_hat: float64;
  }

  /**
   * Latest status and metrics for a single arm.
   */
  model ArmStatus {
    metrics: ArmMetrics;
    is_active: boolean = true;
  }

  /**
   * Collective state of all arms (Scoreboard).
   */
  model Scoreboard {
    arms: Record<ArmStatus>;
  }
}
