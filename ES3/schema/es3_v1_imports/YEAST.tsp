namespace ES3.YEAST;

model TaskSpec extends Base.TaskSpec {
  kind: "yeast" = "yeast";
  arms: string[];
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
    value: float64;
  }

  /**
   * Result of a statistical test/evaluation for the study at a look.
   */
  model LookResult extends Base.Log {
    sample_n: int32;
    info_frac: float64;
    
    /** The calculated trajectory value */
    trajectory: float64;

    efficacy_boundary?: float64;
    is_efficacy_crossed: boolean;

    status: DecisionStatus | string;
  }
}