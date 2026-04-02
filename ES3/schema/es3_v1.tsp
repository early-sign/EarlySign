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
 *    When implementing a specific Protocol's engine or facade (e.g., an EarlySign Controller),
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
    /** Schema version (e.g., "v1.0.0") */
    ES3_version?: string = ES3_VERSION;
  }

  /**
   * The Root Protocol Container.
   * Defines the high-level structure of any EarlySign protocol.
   */
  model Protocol {
    /** The name of the protocol. */
    name: string;
    /** The problem definition. */
    task: TaskSpec;
    /** The operational method. */
    method: MethodSpec;
  }

  /**
   * Base class for Problem Definition.
   * Concrete protocols should extend this (e.g., GST.TaskSpec).
   */
  model TaskSpec {
    /** The kind of task (discriminator). */
    kind: string;
  }

  /**
   * Base class for Operational Method.
   * Concrete protocols should extend this (e.g., GST.MethodSpec).
   */
  model MethodSpec {
    /** The kind of method (discriminator). */
    kind: string;
  }

  /**
   * Base log/event model for metadata inclusion.
   */
  model Log {
  }

  /**
   * Structural definition of study arms and their roles.
   */
  union ArmStructure {
    SingleArm,
    TwoArmComparison,
    MultiArmComparison,
  }

  model ArmStructureBase {
    kind: string;
  }

  /**
   * A single-arm trial design.
   */
  model SingleArm extends ArmStructureBase {
    kind: "single";
    arm_name: string;
  }

  /**
   * A classic two-arm comparison (e.g., Treatment vs. Control).
   */
  model TwoArmComparison extends ArmStructureBase {
    kind: "two_arm";
    control_arm_name: string;
    treatment_arm_name: string;
    /** Mapping of treatment arm name to its ratio relative to control. Key must match treatment_arm_name. */
    allocation_ratios?: Record<float64>;
  }

  /**
   * A multi-arm comparison against a common control.
   */
  model MultiArmComparison extends ArmStructureBase {
    kind: "multi_arm";
    control_arm_name: string;
    treatment_arm_names: string[];
    /** Mapping of treatment arm names to their ratio relative to control (n_treatment / n_control). If None, equal allocation is assumed. */
    allocation_ratios?: Record<float64>;
  }
}

namespace Binomial {
  /**
   * Raw evidence: data for a single arm (Binomial).
   */
  model BinomialArmData extends Base.Log {
    /** Total number of trials. */
    total: int32;
    /** Total number of successes. */
    success: int32;
    /** The name/id of the arm. */
    arm: string;
  }

  /**
   * Standard statistics for a single arm (Bernoulli/Binomial).
   */
  model ArmMetrics {
    /** Total number of trials. */
    total: int32;
    /** Total number of successes. */
    successes: int32;
    /** Success probability estimate (p-hat). */
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
namespace Continuous {
  /**
   * Raw evidence: data for a single arm (continuous observation).
   */
  model Observation extends Base.Log {
    /** The observed value. */
    value: float64;
    /** Arm identifier. */
    arm: string;
  }

  /**
   * Raw evidence: aggregated data for a single arm (continuous).
   */
  model ContinuousArmData extends Base.Log {
    /** Sample size. */
    total: int32;
    /** Sum of observations. */
    sum_x: float64;
    /** Sum of squared observations. */
    sum_x2: float64;
    /** Arm identifier. */
    arm: string;
  }

  /**
   * Standard statistics for a single arm (Continuous).
   */
  model ArmMetrics {
    total: int32;
    mean: float64;
    variance: float64;
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
    /** Map of arm IDs to status. */
    arms: Record<ArmStatus>;
  }
}
