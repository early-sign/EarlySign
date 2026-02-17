
namespace ES3.GST;

/**
 * GST-Specific Protocol Container.
 * Enforces that task and method belong to the GST domain.
 */
model Protocol extends Base.Protocol {
  /** The GST task specification. */
  task: TaskSpec;
  /** The GST method specification. */
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
  
  arms: Base.ArmStructure; 
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
  /** The kind of method (discriminator). */
  kind: "group_sequential" = "group_sequential"; 

  /** Stopping Policy (discriminated union). */
  stopping_policy: StoppingPolicySpec;
  
  /** Technical specification for interim adaptation (SSR, etc.). */
  adaptation?: AdaptationSpec;

  /** Snapshot of internal state during adaptation, used for Weighted Z-score correction. */
  adaptation_snapshot?: AdaptationSnapshot;
}

// --- Sub-Components ---

/**
 * Available spending function types for group sequential design.
 */
enum SpendingFunctionType {
  ObrienFleming: "obrien_fleming",
  Pocock: "pocock",
  PowerFamily: "power_family",
  HwangShihDeCani: "hwang_shih_decani",
}

model SpendingFunction {
  family: SpendingFunctionType | string;
  /** e.g. { "rho": 2.0, "gamma": -4.0 } */
  params?: Record<float64>;
}

/**
 * Efficacy requirement for group sequential design.
 * Specifies the Type I error rate (alpha) to control and binding behavior.
 */
model EfficacyRequirement {
  alpha: float64;
  /**
   * Binding (defaults to true: usually true for efficacy).
   */
  binding?: boolean = true;
}

/**
 * Futility requirement for group sequential design.
 * Specifies the target power (1 - beta) and binding behavior.
 */
model FutilityRequirement {
  /** Target Power (1 - Beta) */
  power: float64;
  /** Non-binding (defaults to false: regulatory best practice) */
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
  /**
   * Is the Alpha (Efficacy) boundary binding?
   * If True, crossing this boundary considers the trial stopped, providing slack for Beta calculations (fewer cases pass to the later stages for futility stopping, which reduces the false rejection cases, hence the boundary is less required to care the false rejection rate).
   */
  alpha_binding?: boolean = true;
  /**
   * Is the Beta (Futility) boundary binding?
   * If True, crossing this boundary considers the trial stopped, providing slack for Alpha calculations (fewer cases pass to the later stages for efficacy stopping, which reduces the false rejection cases, hence the boundary is less required to care the false rejection rate).
   */
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
model HypothesisParametersBase {
  kind: string;
}

union HypothesisParameters {
  EqualityHypothesis,
  SuperiorityHypothesis,
  NonInferiorityHypothesis,
  EquivalenceHypothesis,
}

model EqualityHypothesis extends HypothesisParametersBase { kind: "equality"; }
model SuperiorityHypothesis extends HypothesisParametersBase { kind: "superiority"; superiority_margin: float64 = 0.0; }
model NonInferiorityHypothesis extends HypothesisParametersBase { kind: "non_inferiority"; non_inferiority_margin: float64; }
model EquivalenceHypothesis extends HypothesisParametersBase { kind: "equivalence"; lower_margin: float64; upper_margin: float64; }

model HypothesisSpec {
  h_null_description: string;
  h_alt_description: string;
  test_logic: HypothesisParameters;
  target_effect: EffectSizeUnion;
}

union EffectSizeUnion {
  BinaryEffectSize,
  ContinuousEffectSize,
  SurvivalEffectSize,
}

/**
 * Numerical assumptions for the Alternative Hypothesis (H1).
 * Note: This defines the "ball position" (simulation target), separate from the "goal posts" (Hypothesis Margins).
 */
model EffectSizeSpecBase {
  type: string;
}

model BinaryEffectSize extends EffectSizeSpecBase {
  type: "binary";
  /** Arm name -> Expected rate */
  proportions: Record<float64>;
}

model ContinuousEffectSize extends EffectSizeSpecBase {
  type: "continuous";
  /** Arm name -> Expected mean */
  means: Record<float64>;
  standard_deviation: float64;
}

model SurvivalEffectSize extends EffectSizeSpecBase {
  type: "time_to_event";
  /** Arm name -> Hazard Ratio (relative to control) */
  hazard_ratios: Record<float64>;
  /** Optional: for sample size calc */
  median_survival_times?: Record<float64>;
  /** Overall event rate if needed */
  event_rate?: float64;
}


/**
 * Effect Measure for Binomial comparisons.
 */
model EffectMeasureBase {
  kind: string;
  value: float64;
}

model AbsoluteDifference extends EffectMeasureBase {
    kind: "absolute_difference" = "absolute_difference";
}

model OddsRatio extends EffectMeasureBase {
    kind: "odds_ratio" = "odds_ratio";
}

model RelativeRisk extends EffectMeasureBase {
    kind: "relative_risk" = "relative_risk";
}

union EffectMeasure {
  AbsoluteDifference,
  OddsRatio,
  RelativeRisk,
}

/**
 * The "Model": Statistical Assumptions for Inference.
 */
model StatisticalModel {
  kind: string;
}

model CanonicalGaussianModel extends StatisticalModel {
  kind: "canonical_gaussian" = "canonical_gaussian";
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
 * Two-Arm Binomial Z-Statistic.
 * Covers both Wald Z (unpooled) and Score Z (pooled).
 */
model TwoArmBinomialZ extends TestStatisticSpec {
  kind: "two_arm_binomial_z" = "two_arm_binomial_z";
  information_unit: "fisher_information" = "fisher_information";

  /** 
   * Variance estimation method.
   * - "unpooled": Wald Z (Observed).
   * - "pooled": Score Z (Null/Common).
   */
  variance_estimation: "pooled" | "unpooled" = "pooled";
}

/**
 * One-Arm Binomial Z-Statistic.
 */
model OneArmBinomialZ extends TestStatisticSpec {
  kind: "one_arm_binomial_z" = "one_arm_binomial_z";
  information_unit: "fisher_information" = "fisher_information";

  /**
   * Variance source.
   * - "sample": Uses observed estimate (Wald-like).
   * - "null_hypothesis": Uses null parameter (Score-like).
   */
  variance_source: "sample" | "null_hypothesis" = "null_hypothesis"; 
}

model BinomialExactStatistic extends TestStatisticSpec {
  kind: "binomial_exact" = "binomial_exact";
}

/**
 * Two-Arm Continuous Z-Statistic.
 */
model TwoArmContinuousZ extends TestStatisticSpec {
  kind: "two_arm_continuous_z" = "two_arm_continuous_z";
  information_unit: "fisher_information" = "fisher_information";
  variance: TwoArmContinuousVariance;
}

/**
 * One-Arm Continuous Z-Statistic.
 */
model OneArmContinuousZ extends TestStatisticSpec {
  kind: "one_arm_continuous_z" = "one_arm_continuous_z";
  information_unit: "fisher_information" = "fisher_information";
  variance: OneArmContinuousVariance; 
}

/**
 * Discriminated Union for Timer (The "Clock").
 */
union InformationTimer {
  SampleSizeTimer,
  FisherInformationTimer,
  EventCountTimer,
}

model InformationTimerBase {
  kind: string;
}

model SampleSizeTimer extends InformationTimerBase {
  kind: "sample_size" = "sample_size";
  /**
   * "individuals": Count individuals (e.g. n1 + n2).
   * "effective_size": Effective sample size (e.g. 4*n1*n2/(n1+n2)).
   */
  unit: "individuals" | "effective_size" = "individuals"; 
  max_sample_size: int32; 
}

model FisherInformationTimer extends InformationTimerBase {
  kind: "fisher_information" = "fisher_information";
  max_information: float64;
}

model EventCountTimer extends InformationTimerBase {
  kind: "event_count" = "event_count";
  max_events: int32;
}

/**
 * Discriminated Union for Schedule (The "Checkpoints").
 */
union ScheduleSpec {
  FixedSchedule,
  EquidistantSchedule,
}

model ScheduleBase {
  kind: string;
}

model FixedSchedule extends ScheduleBase {
  kind: "fixed" = "fixed";
  analyses: float64[]; // Explicit list of information fractions (e.g. [0.5, 1.0])
}

model EquidistantSchedule extends ScheduleBase {
  kind: "equidistant" = "equidistant";
  n_looks: int32; // Number of equally spaced looks
}

/**
 * Shared Known Variance model.
 */
model KnownVariance {
  kind: "known" = "known";
  value: float64; // sigma^2
}

/**
 * Two-Arm Variance options.
 */
model TwoArmEstimatedVariance {
  kind: "estimated" = "estimated";
  /**
   * - "pooled": Equal variance assumption.
   * - "unpooled": Unequal variance (Welch).
   */
  method: "pooled" | "unpooled" = "pooled";
}

union TwoArmContinuousVariance {
  KnownVariance,
  TwoArmEstimatedVariance,
}

/**
 * One-Arm Variance options.
 */
model OneArmEstimatedVariance {
  kind: "estimated" = "estimated"; 
  // No options (always sample variance)
}

union OneArmContinuousVariance {
  KnownVariance,
  OneArmEstimatedVariance,
}

/**
 * Composite Policy Container.
 */
model StoppingPolicySpec {
  statistic: TestStatisticSpec;
  strategy: DecisionStrategy;
  timer: InformationTimer;
  schedule: ScheduleSpec;
  
  /**
   * Strategy for triggering interim analyses.
   * Defaults to "due_look" (trigger when a milestone is reached).
   */
  trigger_strategy?: TriggerStrategySpec = #{ kind: "due_look", tolerance: 0.0 };
}

/**
 * Discriminated Union for Trigger Strategies.
 */
union TriggerStrategySpec {
  DueLookTrigger,
}

model TriggerStrategySpecBase {
  kind: string;
}

/**
 * Triggers a look when a planned look is reached (due) and hasn't been analyzed.
 */
model DueLookTrigger extends TriggerStrategySpecBase {
  kind: "due_look";
  /** Information time tolerance (e.g. 0.01 for 1% buffer). */
  tolerance?: float64 = 0.0;
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
    /** Look index (1-indexed). */
    look?: int32;
    /** The trigger for this look. */
    trigger?: Trigger;
    /** Current cumulative sample size. */
    sample_n: int32;
    /** Current fraction of information accrued. */
    info_frac: float64;
    /** Current calculated Z-statistic. */
    z_stat: float64;

    /** The calculated efficacy boundary at this look. */
    efficacy_boundary?: float64;
    /** Whether the efficacy boundary was crossed. */
    is_efficacy_crossed: boolean;

    /** The calculated futility boundary at this look. */
    futility_boundary?: float64;
    /** Whether the futility boundary was crossed. */
    is_futility_crossed: boolean;

    /** Cumulative Alpha spent up to this look. */
    alpha_spent?: float64;
    /** Cumulative Beta spent up to this look. */
    beta_spent?: float64;

    /** The decision status resulting from this look. */
    status: DecisionStatus | string;
  }

  enum PromisingZoneStatus {
    Futility: "futility",
    Efficacy: "efficacy",
    Promising: "promising",
    Continue: "continue",
  }

  /**
   * Log entry for an adaptation event (e.g., SSR).
   */
  model AdaptationLog extends Base.Log {
    /** Look index at which adaptation was calculated. */
    look: int32;
    /** Calculated conditional power based on current data. */
    conditional_power: float64;
    /** Status of the promising zone logic. */
    promising_zone_status: PromisingZoneStatus | string;
    /** Human-readable recommendation or message. */
    promising_zone_recommendation: string;
    /** Original planned max sample size. */
    original_sample_size: int32;
    /** Newly recommended max sample size (if SSR triggered). */
    recommended_sample_size?: int32;
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

namespace Multivariate {
  /**
   * Multivariate Task (Multiple Endpoints).
   * Composed of multiple Univariate GST Tasks.
   */
  model TaskSpec extends Base.TaskSpec {
    kind: "group_sequential_multivariate" = "group_sequential_multivariate"; 
    
    /** Reusing the Standard GST TaskSpec */
    sub_tasks: ES3.GST.TaskSpec[]; 
    
    /** Global parameters */
    /** Study-wide arms (must be consistent with sub_tasks) */
    arms: string[];
    
    /**
     * Global Hypothesis (e.g., "At least one effective" or "Global Null").
     * Typically represents the Union-Intersection or Intersection-Union test.
     */
    hypotheses: ES3.GST.HypothesisSpec; 
    
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

/**
 * State captured during an adaptation event to preserve Type I error control.
 */
model AdaptationSnapshot {
  /** Observed Z-statistic at the time of adaptation. */
  z_t: float64;
  /** Information fraction at the time of adaptation (t). */
  info_frac: float64;
  /** The maximum sample size before re-planning. */
  original_max_sample_size: int32;
  /** Whether to use the Cui-Hung-Wang weighted statistic (defaults to true). */
  use_weighted_statistic?: boolean = true;
}
