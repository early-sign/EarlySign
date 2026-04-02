import "@earlysign/es3-standard";

namespace ES3.AVI;

model BaseAVIMethodSpec extends Base.MethodSpec {
  /**
   * Optional burn-in sample size before checking stopping rules.
   * Recommended when using plug-in variance estimators to mitigate instability
   * at very small sample sizes, or for asymptotic approximations that require a
   * minimum sample size to become valid.
   */
  burn_in?: int32 = 100;
}

model GAVIMethodSpec extends BaseAVIMethodSpec {
  kind: "GAVI" = "GAVI";
  /** Significance level. */
  alpha: float64;
  /**
   * Estimated variance (sigma squared).
   * If null, the variance is estimated from the data (Maharaj et al., 2023).
   */
  variance?: float64;
  /** One-sided or two-sided test. */
  sides: "one" | "two";
  /** Maximum sample size, used for rho calculation. */
  max_n: int32;
}

model MSPRTMethodSpec extends BaseAVIMethodSpec {
  kind: "mSPRT" = "mSPRT";
  /** Significance level. */
  alpha: float64;
  /**
   * Estimated variance (sigma squared).
   * If null, the variance is estimated from the data (Maharaj et al., 2023).
   */
  variance?: float64;
  /** One-sided or two-sided test. */
  sides: "one" | "two";
  /** Minimum Detectable Effect (absolute difference), used for phi calculation. */
  mde: float64;
}

model SequentialQuantileMethodSpec extends BaseAVIMethodSpec {
  kind: "SequentialQuantile" = "SequentialQuantile";
  /** Target quantile (e.g., 0.5 for median). */
  quantile: float64;
  /** Significance level. */
  alpha: float64;
  /** Optional maximum sample size for the horizon. */
  max_n?: int32;
}

union MethodSpec {
  GAVIMethodSpec,
  MSPRTMethodSpec,
  SequentialQuantileMethodSpec,
}

model TaskSpec extends Base.TaskSpec {
  kind: "AVI" = "AVI";
  arms: Base.ArmStructure;
  response_type: "binary" | "continuous";
}

model Protocol extends Base.Protocol {
  task: TaskSpec;
  method: MethodSpec;
}

namespace Log {
  enum DecisionStatus {
    Continue: "continue",
    StopDetected: "stop_detected",
    StopPlanEndReached: "stop_plan_end_reached",
  }

  model LookResult extends Base.Log {
    /** The total sample size (cumulative) at this look. */
    sample_n: int32;
    /** The calculated trajectory statistic. */
    trajectory: float64;
    /** The current boundary value. */
    boundary: float64;
    /** Whether the trajectory crossed the boundary. */
    is_crossed: boolean;
    /** The decision status after this look. */
    status: DecisionStatus | string;
  }

  /**
   * Result of a sequential quantile test look.
   */
  model SequentialQuantileLookResult extends Base.Log {
    /** Current cumulative sample size. */
    sample_n: int32;
    /** Point estimate of the target quantile. */
    estimated_quantile: float64;
    /** Lower confidence bound. */
    interval_lower: float64;
    /** Upper confidence bound. */
    interval_upper: float64;
    /** Current status of the test. */
    status: DecisionStatus | string;
  }
}

model AVIMetadata {
  /** Metadata version */
  version: string;
}
