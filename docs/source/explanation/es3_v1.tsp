import "@typespec/openapi";

/**
 * EarlySign Standard Schema (ES3) - Manifest
 * Version: v1.0.0
 * 
 * BASIC POLICY & DESIGN PHILOSOPHY:
 * 1. Naming Convention: All property names are defined in snake_case for Pydantic (Python) compatibility.
 * 2. Polymorphism Resolution Policy:
 *    We use TypeSpec "discriminator" decorators for polymorphism, which is unavoidable in deep hierarchies.
 *    However, when implementing a specific Protocol's engine or facade (e.g., an EarlySign Template),
 *    this polymorphism should be largely resolved. In other words, the facade code should be defined granularly enough
 *    so that most schema choices are fixed. If a facade leaves too much polymorphism unresolved, it should likely be
 *    split into more specific facades.
 * 3. Producer/Consumer Correspondence:
 *    Sub-schemas should ideally correspond to their respective producer or consumer components. This alignment
 *    ensures that the handling of schema instances remains abstract and decoupled.
 * 4. Semantic Versioning:
 *    We use semantic versioning for the ES3 schema. This ensures that changes to the schema are clearly documented
 *    and that the schema can be easily updated in the future.
 */

namespace ES3.Base {
  /**
   * The Root Protocol Container.
   * Defines the high-level structure of any EarlySign protocol.
   */
  model Protocol {
    ES3_version: "v1.0.0" = "v1.0.0";
    name: string;
    task: TaskSpec;
    method: MethodSpec;
  }

  /**
   * Base class for Problem Definition.
   * Concrete protocols should extend this (e.g., ES3.GST.TaskSpec).
   */
  @discriminator("kind")
  model TaskSpec {}

  /**
   * Base class for Operational Method.
   * Concrete protocols should extend this (e.g., ES3.GST.MethodSpec).
   */
  @discriminator("kind")
  model MethodSpec {}
}

namespace ES3.GST {
  /**
   * GST-Specific Protocol Container.
   * Enforces that task and method belong to the GST domain.
   */
  model Protocol extends ES3.Base.Protocol {
    task: TaskSpec;
    method: MethodSpec;
  }

  /**
   * GST-Specific Task Definition (Univariate).
   */
  model TaskSpec extends ES3.Base.TaskSpec {
    kind: "group_sequential"; // Discriminator value for Base
    
    arms: string[]; 
    response_type: "binary" | "continuous" | "time_to_event";
    hypotheses: HypothesisSpec; 

    // Design Requirements (Designer Input)
    // Presence implies the intent to stop for this reason.
    efficacy?: {
      alpha: float64; // Type I Error target
    };
    futility?: {
      power: float64; // Target Power (1 - Beta)
      binding?: boolean; // Default: false (Non-binding is standard for regulatory)
    };
  }  

  /**
   * GST-Specific Method Definition (Univariate).
   */
  model MethodSpec extends ES3.Base.MethodSpec {
    kind: "group_sequential"; 
    
    // Method Implementation Components
    efficacy?: StoppingRule;
    futility?: StoppingRule;
    
    adaptation?: AdaptationSpec;
  }

  /**
   * Logical Margins and Test Objectives.
   */
  @discriminator("kind")
  model HypothesisParameters {}

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
  @discriminator("type")
  model EffectSizeSpec {}

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
  @discriminator("kind")
  model ReferenceModelSpec {}

  model BinaryModel extends ReferenceModelSpec {
    kind: "binary";
    test_statistic: "Z" | "chi_square" | "exact";
    link_function: "identity" | "log" | "logit" = "identity";
    use_canonical_joint_distribution: boolean;
  }

  model ContinuousModel extends ReferenceModelSpec {
    kind: "continuous";
    test_statistic: "Z" | "t";
    variance_assumption: "known" | "unknown" | "heteroscedastic";
    use_canonical_joint_distribution: boolean;
  }

  model SurvivalModel extends ReferenceModelSpec {
    kind: "time_to_event";
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
  @discriminator("kind")
  model BoundarySpec {
    reference_model: ReferenceModelSpec; 
    boundary_scale: "z_score" | "p_value" | "score_statistic" = "z_score";
    binding: boolean = true;
  }

  model SpendingFunctionSpec {
    type: "obrien_fleming" | "pocock" | "kim_demets" | "lan_demets" | "power_family" | "hwang_shih_decani" | "custom";
    params?: Record<float64>; // e.g. { "rho": 2.0, "alpha": 0.025 }
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
      CONTINUE,
      STOP_EFFICACY,
      STOP_FUTILITY,
      STOP_PLAN_END_REACHED,
      STOP,
    }

    /**
     * Defines the reason / cause for an operational event (e.g., Analysis Execution).
     */
    @discriminator("kind")
    model Trigger {}

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
    model Analysis {
      trigger?: Trigger;
      
      // Results
      look: int32;
      statistic: float64;
      info_frac: float64;
      decision: DecisionStatus;
      
      efficacy_boundary?: float64;
      futility_boundary?: float64;
    }

    /**
     * Formalized Decision Event.
     */
    model Decision {
      status: DecisionStatus;
      message: string;
    }
  }

  @discriminator("type")
  model AdaptationSpec {}

  model SampleSizeReestimationSpec extends AdaptationSpec {
    type: "sample_size_reestimation";
    method: "conditional_power" | "predictive_power";
    target_power: float64;
    n_range: [uint32, uint32]; 
  }
}

namespace ES3.GST.Multivariate {
  /**
   * Multivariate Task (Multiple Endpoints).
   * Composed of multiple Univariate GST Tasks.
   */
  model TaskSpec extends ES3.Base.TaskSpec {
    kind: "group_sequential"; 
    
    sub_tasks: ES3.GST.TaskSpec[]; // Reusing the Standard GST TaskSpec
    
    // Global parameters
    arms: string[]; // Study-wide arms (must be consistent with sub_tasks)
    
    /**
     * Global Hypothesis (e.g., "At least one effective" or "Global Null").
     * Typically represents the Union-Intersection or Intersection-Union test.
     */
    hypotheses: ES3.GST.HypothesisSpec; 
    
    alpha?: float64; // Global FWER
    beta?: float64;  // Global Power
  }

  model MethodSpec extends ES3.Base.MethodSpec {
    kind: "group_sequential";
    
    multiplicity_adjustment: "none" | "bonferroni" | "holm" | "hochberg" | "gatekeeping";
    correlation_matrix?: float64[][];
  }

  model Protocol extends ES3.Base.Protocol {
    task: TaskSpec;
    method: MethodSpec;
  }
}

namespace ES3.Examples {
  /**
   * Example Extension: Wang-Tsiatis Power Family Boundary.
   * Demonstrates how to extend the standard boundary definitions.
   */
  model WangTsiatisBoundary extends ES3.GST.BoundarySpec {
    kind: "wang_tsiatis";
    delta: float64; 
  }

  /**
   * TEMPLATE: Binary A/B Superiority.
   * Extends the GST Protocol.
   */
  model BinomialABSuperiorityProtocol extends ES3.GST.Protocol {
    name: string;
    task: ES3.GST.TaskSpec & {
      response_type: "binary";
      // Design Requirements
      efficacy: { alpha: 0.025 };
      futility: { power: 0.8, binding: false }; 
      hypotheses: ES3.GST.HypothesisSpec & {
        test_logic: ES3.GST.SuperiorityHypothesis;
        target_effect: ES3.GST.BinaryEffectSize;
      };
    };
    method: ES3.GST.MethodSpec & {
      // Efficacy Stopping Rule
      efficacy: ES3.GST.StoppingRule & {
        boundary: ES3.GST.SpendingBoundary & {
          reference_model: ES3.GST.BinaryModel & { use_canonical_joint_distribution: true; };
          spending_function: { type: "obrien_fleming" };
          binding: true;
        };
      };
      // Futility Stopping Rule
      futility: ES3.GST.StoppingRule & {
        boundary: ES3.GST.SpendingBoundary & {
            reference_model: ES3.GST.BinaryModel & { use_canonical_joint_distribution: true; };
            spending_function: { type: "pocock" };
            binding: false;
        };    
      };
    };
  }
}
