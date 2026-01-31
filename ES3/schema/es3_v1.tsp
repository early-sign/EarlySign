import "@typespec/openapi3";
import "./es3_v1_imports/YEAST.tsp";
import "./es3_v1_imports/AVI.tsp";
import "./es3_v1_imports/GST.tsp";
import "./es3_v1_imports/SequentialQuantile.tsp";

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
namespace Continuous {
  /**
   * Raw evidence: data for a single arm (continuous observation).
   */
  model Observation extends Base.Log {
    value: float64;
    arm: string;
  }

  /**
   * Raw evidence: aggregated data for a single arm (continuous).
   */
  model ArmData extends Base.Log {
    n: int32;
    sum_x: float64;
    sum_x2: float64;
    arm: string;
  }

  /**
   * Standard statistics for a single arm (Continuous).
   */
  model ArmMetrics {
    n: int32;
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
    arms: Record<ArmStatus>;
  }
}
