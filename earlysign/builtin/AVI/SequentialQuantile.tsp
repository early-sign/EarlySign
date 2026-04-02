import "@earlysign/es3-standard";

namespace ES3.SequentialQuantile;

@doc("Metrics for a single arm in a sequential quantile test.")
model ArmMetrics extends ES3.Base.Log {
  @doc("Sample size n at the time of calculation.")
  total: int32;

  @doc("The value of the lower confidence bound (X at rank L_t).")
  ci_lower: float64;

  @doc("The value of the upper confidence bound (X at rank U_t).")
  ci_upper: float64;

  @doc("The point estimate of the target quantile.")
  quantile_estimate: float64;
}

@doc("Projection of all arm metrics in a sequential quantile test.")
model Scoreboard {
  arms: Record<ArmStatus>;
}

@doc("Status wrapper for arm metrics.")
model ArmStatus {
  metrics: ArmMetrics;
  is_active: boolean;
}
