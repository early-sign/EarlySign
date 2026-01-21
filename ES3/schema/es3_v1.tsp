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
   * @example
   * {
   *   "kind": "group_sequential",
   *   "arms": ["treatment", "control"],
   *   "response_type": "binary",
   *   "hypotheses": { ... },
   *   "efficacy": { "alpha": 0.025 }
   * }
   */
  model TaskSpec extends Base.TaskSpec {
    kind: "group_sequential" = "group_sequential"; // Discriminator value for Base
    
    arms: string[]; 
    response_type: "binary" | "continuous" | "time_to_event";
    hypotheses: HypothesisSpec; 

    /**
     * Design Requirements (Designer Input)
     * Presence implies the intent to stop for this reason.
     */
    efficacy?: EfficacyRequirement;
    futility?: FutilityRequirement;
  }

  /**
   * GST-Specific Method Definition (Univariate).
   */
  model MethodSpec extends Base.MethodSpec {
    kind: "group_sequential" = "group_sequential"; 

    /** Stopping Policy (discriminated union) */
    stopping_policy: StoppingPolicySpec;
    
    adaptation?: AdaptationSpec;
  }

  // --- Sub-Components ---

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

  model SpendingFunction {
    family: SpendingFunctionType | string;
    /** e.g. { "rho": 2.0, "gamma": -4.0 } */
    params?: Record<float64>;
  }

  /**
   * Efficacy requirement for group sequential design.
   * Specifies the Type I error rate (alpha) to control.
   */
  model EfficacyRequirement {
    alpha: float64;
  }

  /**
   * Futility requirement for group sequential design.
   * Specifies the target power (1 - beta) and binding behavior.
   */
  model FutilityRequirement {
    /** Target Power (1 - Beta) */
    power: float64;
    /** Non-binding default (regulatory best practice) */
    binding?: boolean = false;
  }

  /**
   * Discriminated Union for Decision Strategies (The "Rule").
   */
  union DecisionStrategy {
    AlphaSpendingStrategy,
    BetaSpendingStrategy,
    AlphaBetaSpendingStrategy,
    WhiteheadStrategy,
    OBrienFlemingStrategy,
    PocockStrategy,
    WangTsiatisStrategy,
  }

  /**
   * Abstract Base for Decision Strategy.
   * Holds the Inferential Model (Assumption).
   */
  model DecisionStrategyBase {
    kind: string;
    statistical_model: StatisticalModel;
  }

  /**
   * Abstract Base for Spending-Function Strategies.
   */
  model SpendingFunctionStrategyBase extends DecisionStrategyBase {}

  /**
   * Abstract Base for Shape-Based Boundary Strategies.
   */
  model BoundaryFunctionStrategyBase extends DecisionStrategyBase {}

  /**
   * Alpha (Efficacy) Spending Strategy.
   */
  model AlphaSpendingStrategy extends SpendingFunctionStrategyBase {
    kind: "alpha_spending" = "alpha_spending";
    spending_fn: SpendingFunction;
    budget: float64;
    sided: "one" | "two" = "two";
  }

  /**
   * Beta (Futility) Spending Strategy.
   */
  model BetaSpendingStrategy extends SpendingFunctionStrategyBase {
    kind: "beta_spending" = "beta_spending";
    spending_fn: SpendingFunction;
    budget: float64;
  }

  /**
   * Combined Alpha-Beta Spending Strategy.
   */
  model AlphaBetaSpendingStrategy extends SpendingFunctionStrategyBase {
    kind: "alpha_beta_spending" = "alpha_beta_spending";
    alpha_spending_fn: SpendingFunction;
    beta_spending_fn: SpendingFunction;
    alpha_budget: float64;
    beta_budget: float64;
    alpha_binding?: boolean = true;
    beta_binding?: boolean = false;
  }

  /**
   * Classic O'Brien-Fleming Strategy.
   */
  model OBrienFlemingStrategy extends BoundaryFunctionStrategyBase {
    kind: "obrien_fleming" = "obrien_fleming";
    alpha: float64;
    sided: "one" | "two" = "two";
  }

  /**
   * Whitehead's Triangular Test Strategy.
   */
  model WhiteheadStrategy extends BoundaryFunctionStrategyBase {
    kind: "whitehead" = "whitehead";
    alpha: float64;
    beta: float64;
  }

  /**
   * Pocock Strategy.
   */
  model PocockStrategy extends BoundaryFunctionStrategyBase {
    kind: "pocock" = "pocock";
    alpha: float64;
    sided: "one" | "two" = "two";
  }

  /**
   * Wang-Tsiatis Power Family Strategy.
   */
  model WangTsiatisStrategy extends BoundaryFunctionStrategyBase {
    kind: "wang_tsiatis" = "wang_tsiatis";
    alpha: float64;
    delta: float64;
    sided: "one" | "two" = "two";
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
    /** Arm name -> Expected rate */
    proportions: Record<float64>;
  }

  model ContinuousEffectSize extends EffectSizeSpec {
    type: "continuous";
    /** Arm name -> Expected mean */
    means: Record<float64>;
    standard_deviation: float64;
  }

  model SurvivalEffectSize extends EffectSizeSpec {
    type: "time_to_event";
    /** Arm name -> Hazard Ratio (relative to control) */
    hazard_ratios: Record<float64>;
    /** Optional: for sample size calc */
    median_survival_times?: Record<float64>;
    /** Overall event rate if needed */
    event_rate?: float64;
  }

  /**
   * The "Model": Statistical Assumptions for Inference.
   */
  /**
   * The "Model": Statistical Assumptions for Inference.
   */
  model StatisticalModel {
    kind: string;
  }

  model CanonicalGaussianModel extends StatisticalModel {
    kind: "canonical_gaussian" = "canonical_gaussian";
    /** Link function for the effect scale (Risk Difference, Risk Ratio, Odds Ratio). */
    link_function?: "identity" | "log" | "logit";
  }

  model ExactBinomialModel extends StatisticalModel {
    kind: "exact_binomial" = "exact_binomial";
  }

  model TProcessModel extends StatisticalModel {
    kind: "t_process" = "t_process";
    /** Degrees of Freedom method (e.g., "satterthwaite") */
    degrees_of_freedom_method: string;
  }

  /**
   * The "Statistic": Operational Definition of the Test Statistic.
   */
  model TestStatisticSpec {
    kind: string;
  }

  /**
   * Two-Arm Wald Z-Statistic.
   * Computes Z = (p_t - p_c) / SE.
   */
  model TwoArmBinomialWaldZ extends TestStatisticSpec {
    kind: "two_arm_binomial_wald_z" = "two_arm_binomial_wald_z";
    /** Variance estimation: pooled (common p) or unpooled. */
    variance_estimation: "pooled" | "unpooled" = "pooled";
  }

  /**
   * Two-Arm Score Z-Statistic.
   */
  model TwoArmBinomialScoreZ extends TestStatisticSpec {
    kind: "two_arm_binomial_score_z" = "two_arm_binomial_score_z";
  }

  /**
   * One-Arm Wald Z-Statistic.
   */
  model OneArmBinomialWaldZ extends TestStatisticSpec {
    kind: "one_arm_binomial_wald_z" = "one_arm_binomial_wald_z";
  }

  model BinomialExactStatistic extends TestStatisticSpec {
    kind: "binomial_exact" = "binomial_exact";
  }

  model ContinuousWaldZ extends TestStatisticSpec {
    kind: "continuous_wald_z" = "continuous_wald_z";
    variance_assumption: "known" | "estimated";
  }

  /**
   * Composite Policy Container.
   */
  model StoppingPolicySpec {
    statistic: TestStatisticSpec;
    strategy: DecisionStrategy;
    schedule: ScheduleSpec;
  }

  model ScheduleSpec {
    unit: "sample_size" | "information_fraction" | "calendar_time" | "events";
    /** Usually defined by N looks and spacing, or specific points. */
    n_looks?: int32;
    /** Optional explicit points */
    interim_points?: float64[];
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
      
      /** Results */
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
    
    /** Reusing the Standard GST TaskSpec */
    sub_tasks: GST.TaskSpec[]; 
    
    /** Global parameters */
    /** Study-wide arms (must be consistent with sub_tasks) */
    arms: string[];
    
    /**
     * Global Hypothesis (e.g., "At least one effective" or "Global Null").
     * Typically represents the Union-Intersection or Intersection-Union test.
     */
    hypotheses: GST.HypothesisSpec; 
    
    /** Global FWER */
    alpha?: float64;
    /** Global Power */
    beta?: float64;
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
