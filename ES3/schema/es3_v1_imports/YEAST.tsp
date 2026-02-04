namespace ES3.YEAST;

model TaskSpec extends Base.TaskSpec {
  kind: "yeast" = "yeast";
  arms: string[];
  response_type: "binary" | "continuous" = "binary";
}

model MethodSpec extends Base.MethodSpec {
  kind: "yeast" = "yeast";
  significance_level: float64;
  expected_num_observations: int32;
  /**
   * The estimated variance of a single increment (difference).
   * If not specified, conservatively defaults to the max variance for a difference of two Binomials (0.5).
   */
  estimated_variance: float64 = 0.5;
}

model Protocol extends Base.Protocol {
  task: TaskSpec;
  method: MethodSpec;
}

namespace Log {
  /**
   * Status of an ongoing or completed YEAST Test.
   */
  enum DecisionStatus {
    Continue: "continue",
    StopEfficacy: "stop_efficacy",
    StopPlanEndReached: "stop_plan_end_reached",
    Stop: "stop",
  }

  model Boundary extends Base.Log {
    value?: float64;
  }

  /**
   * Result of a statistical test/evaluation for the study at a look (YEAST).
   */
  model LookResult extends Base.Log {
    /** Total number of samples (cumulative). */
    sample_n: int32;
    /** Fraction of information accrued. */
    info_frac: float64;
    
    /** The calculated trajectory value (Your Evidence). */
    trajectory: float64;

    /** The efficacy boundary at this look. */
    efficacy_boundary?: float64;
    /** Whether the trajectory crossed the efficacy boundary. */
    is_efficacy_crossed: boolean;

    /** Current status of the YEAST test. */
    status: DecisionStatus | string;
  }
}