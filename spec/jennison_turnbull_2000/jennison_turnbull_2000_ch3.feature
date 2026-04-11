Feature: Jennison & Turnbull (2000) Chapter 3 (Two-Sided Tests: General Applications)
  As a statistical designer, I want to verify the library's outputs match the results from the classic textbook
  Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with Applications to Clinical Trials. Chapman and Hall/CRC.

  Background:
    Given simulation precision with 2000 samples

  Scenario: Computing O'Brien-Fleming design for a paired comparison (Subsection 3.2.2)
    # Note: The textbook presents this as a two-sided normal mean comparison for paired data.
    # Under the canonical joint model used for protocol design, neither the response distribution
    # (normal) nor the study structure (paired) affect boundary calculation. These details provide
    # context for the example but do not influence the design phase results.
    Given a two-sided "paired" design with alpha 0.05
    And a target power 0.9 at effect 1
    And variance 6
    And 5 looks with "obrien_fleming" spending
    When I compute the design
    Then the fixed sample information (I_f) should be 10.51 with 50.0 precision
    And the maximum information (I_max) should be 10.78 with 50.0 precision
    And the boundary values should be "4.562, 3.226, 2.634, 2.281, 2.040" with 50.0 precision
    And the sample size increment per group per look should be 12.94 with 50.0 precision
    And the total sample size (n_max) should be 64.7 with 50.0 precision
    And the rounded number of pairs per look should be 13

  Scenario: Computing Wang-Tsiatis design for a crossover trial (Subsection 3.2.2)
    # Note: The textbook presents this as a two-sided normal mean comparison for a 2-period crossover trial.
    # Under the canonical joint model used for protocol design, neither the response distribution
    # (normal) nor the study structure (crossover) affect boundary calculation. These details provide
    # context for the example but do not influence the design phase results.
    # Under the canonical joint model used for protocol design, neither the response distribution
    # (normal) nor the study structure (crossover) affect boundary calculation. These details provide
    # context for the example but do not influence the design phase results.
    Given a two-sided "crossover" design with alpha 0.05
    And a target power 0.8 at effect 0.6
    And variance 9
    And 4 looks with "wang_tsiatis" spending
    And a Wang-Tsiatis delta 0.25
    When I compute the design
    Then the fixed sample information (I_f) should be 21.81 with 50.0 precision
    And the maximum information (I_max) should be 23.23 with 15.0 precision
    And the boundary values should be "2.988, 2.513, 2.270, 2.113" with 50.0 precision
    And the sample size increment per group per look should be 26.1 with 15.0 precision
    And the total sample size (n_max) should be 104.4 with 50.0 precision
    And the rounded number of subjects per sequence per look should be 27

  Scenario: Operating characteristics with varying group sizes (Table 3.1)
    # Note: The textbook presents this as a two-sided normal mean comparison.
    # This scenario evaluates robustness when the actual sample sizes deviate from the planned schedule.
    Given a two-sided "normal-mean" design with alpha 0.05
    And a target power 0.9 at effect 1
    And variance 4
    And a planning sample size sequence per group "<n_plan>"
    And spending "<spending>"
    When the actual sample size sequence per group is "<n_actual>"
    Then the actual type-I error should be <alpha_actual> with 50.0 precision
    And the actual power should be <power_actual> with 50.0 precision

    Examples:
      | spending       | n_plan             | n_actual            | alpha_actual | power_actual |
      | pocock         | 21, 42, 63, 84, 105 | 21, 42, 63, 84, 105  | 0.050        | 0.910        |
      | pocock         | 21, 42, 63, 84, 105 | 18, 36, 54, 72, 90   | 0.050        | 0.860        |
      | pocock         | 21, 42, 63, 84, 105 | 23, 46, 69, 92, 115  | 0.050        | 0.934        |
      | pocock         | 21, 42, 63, 84, 105 | 30, 50, 55, 86, 105  | 0.046        | 0.909        |
      | pocock         | 21, 42, 63, 84, 105 | 12, 31, 57, 81, 105  | 0.054        | 0.909        |
      | pocock         | 21, 42, 63, 84, 105 | 13, 42, 56, 78, 99   | 0.051        | 0.892        |
      | pocock         | 21, 42, 63, 84, 105 | 26, 40, 63, 96, 110  | 0.049        | 0.923        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 18, 36, 54, 72, 90   | 0.050        | 0.912        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 16, 32, 48, 64, 80   | 0.050        | 0.877        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 20, 40, 60, 80, 100  | 0.050        | 0.937        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 26, 39, 50, 76, 90   | 0.049        | 0.911        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 10, 27, 55, 66, 90   | 0.051        | 0.912        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 11, 38, 59, 65, 83   | 0.049        | 0.888        |
      | obrien_fleming | 18, 36, 54, 72, 90  | 27, 40, 57, 73, 96   | 0.051        | 0.928        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 18, 36, 54, 72, 90   | 0.050        | 0.901        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 16, 32, 48, 64, 80   | 0.050        | 0.864        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 20, 40, 60, 80, 100  | 0.050        | 0.929        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 26, 39, 50, 76, 90   | 0.049        | 0.901        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 10, 27, 55, 66, 90   | 0.052        | 0.901        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 11, 38, 59, 65, 83   | 0.048        | 0.875        |
      | wang_tsiatis   | 18, 36, 54, 72, 90  | 27, 40, 57, 73, 96   | 0.050        | 0.919        |

  Scenario: Operating characteristics with information mismatch (Table 3.2)
    Given we are planning a two-sided "normal-mean" design
    And alpha is 0.05
    And target power is 0.9
    And a planning information sequence for <K> looks with equal increments
    And spending "<spending>"
    When the actual information sequence is I_k = <pi> * (k/K)^<r> * I_max
    Then the actual type-I error should be <alpha_actual> with 50.0 precision
    And the actual power should be <power_actual> with 50.0 precision

    Examples:
      | K  | spending       | r    | pi  | alpha_actual | power_actual |
      | 2  | pocock         | 0.80 | 0.9 | 0.048        | 0.868        |
      | 2  | pocock         | 0.80 | 1.0 | 0.048        | 0.901        |
      | 2  | pocock         | 0.80 | 1.1 | 0.048        | 0.926        |
      | 2  | pocock         | 1.00 | 0.9 | 0.050        | 0.867        |
      | 2  | pocock         | 1.00 | 1.0 | 0.050        | 0.900        |
      | 2  | pocock         | 1.00 | 1.1 | 0.050        | 0.926        |
      | 2  | pocock         | 1.25 | 0.9 | 0.052        | 0.865        |
      | 2  | pocock         | 1.25 | 1.0 | 0.052        | 0.899        |
      | 2  | pocock         | 1.25 | 1.1 | 0.052        | 0.925        |
      | 2  | obrien_fleming | 0.80 | 0.9 | 0.049        | 0.867        |
      | 2  | obrien_fleming | 0.80 | 1.0 | 0.049        | 0.900        |
      | 2  | obrien_fleming | 0.80 | 1.1 | 0.049        | 0.925        |
      | 2  | obrien_fleming | 1.00 | 0.9 | 0.050        | 0.867        |
      | 2  | obrien_fleming | 1.00 | 1.0 | 0.050        | 0.900        |
      | 2  | obrien_fleming | 1.00 | 1.1 | 0.050        | 0.925        |
      | 2  | obrien_fleming | 1.25 | 0.9 | 0.050        | 0.868        |
      | 2  | obrien_fleming | 1.25 | 1.0 | 0.050        | 0.900        |
      | 2  | obrien_fleming | 1.25 | 1.1 | 0.050        | 0.925        |
      | 5  | pocock         | 0.80 | 0.9 | 0.047        | 0.867        |
      | 5  | pocock         | 0.80 | 1.0 | 0.047        | 0.901        |
      | 5  | pocock         | 0.80 | 1.1 | 0.047        | 0.927        |
      | 5  | pocock         | 1.00 | 0.9 | 0.050        | 0.866        |
      | 5  | pocock         | 1.00 | 1.0 | 0.050        | 0.900        |
      | 5  | pocock         | 1.00 | 1.1 | 0.050        | 0.927        |
      | 5  | pocock         | 1.25 | 0.9 | 0.053        | 0.864        |
      | 5  | pocock         | 1.25 | 1.0 | 0.053        | 0.899        |
      | 5  | pocock         | 1.25 | 1.1 | 0.053        | 0.926        |
      | 5  | obrien_fleming | 0.80 | 0.9 | 0.048        | 0.866        |
      | 5  | obrien_fleming | 0.80 | 1.0 | 0.048        | 0.899        |
      | 5  | obrien_fleming | 0.80 | 1.1 | 0.048        | 0.925        |
      | 5  | obrien_fleming | 1.00 | 0.9 | 0.050        | 0.867        |
      | 5  | obrien_fleming | 1.00 | 1.0 | 0.050        | 0.900        |
      | 5  | obrien_fleming | 1.00 | 1.1 | 0.050        | 0.925        |
      | 5  | obrien_fleming | 1.25 | 0.9 | 0.052        | 0.868        |
      | 5  | obrien_fleming | 1.25 | 1.0 | 0.052        | 0.900        |
      | 5  | obrien_fleming | 1.25 | 1.1 | 0.052        | 0.925        |
      | 10 | pocock         | 0.80 | 0.9 | 0.046        | 0.866        |
      | 10 | pocock         | 0.80 | 1.0 | 0.046        | 0.901        |
      | 10 | pocock         | 0.80 | 1.1 | 0.046        | 0.928        |
      | 10 | pocock         | 1.00 | 0.9 | 0.050        | 0.865        |
      | 10 | pocock         | 1.00 | 1.0 | 0.050        | 0.900        |
      | 10 | pocock         | 1.00 | 1.1 | 0.050        | 0.927        |
      | 10 | pocock         | 1.25 | 0.9 | 0.055        | 0.863        |
      | 10 | pocock         | 1.25 | 1.0 | 0.055        | 0.899        |
      | 10 | pocock         | 1.25 | 1.1 | 0.055        | 0.926        |
      | 10 | obrien_fleming | 0.80 | 0.9 | 0.048        | 0.866        |
      | 10 | obrien_fleming | 0.80 | 1.0 | 0.048        | 0.899        |
      | 10 | obrien_fleming | 0.80 | 1.1 | 0.048        | 0.924        |
      | 10 | obrien_fleming | 1.00 | 0.9 | 0.050        | 0.867        |
      | 10 | obrien_fleming | 1.00 | 1.0 | 0.050        | 0.900        |
      | 10 | obrien_fleming | 1.00 | 1.1 | 0.050        | 0.925        |
      | 10 | obrien_fleming | 1.25 | 0.9 | 0.053        | 0.869        |
      | 10 | obrien_fleming | 1.25 | 1.0 | 0.053        | 0.901        |
      | 10 | obrien_fleming | 1.25 | 1.1 | 0.053        | 0.926        |

  Scenario: Computing O'Brien-Fleming design for a normal mean (Subsection 3.4.2)
    Given a two-sided "normal-mean" design with alpha 0.05
    And a target power 0.8 at effect size 0.5
    And a known variance (sigma squared) 1.2
    And a maximum of 6 looks with "obrien_fleming" spending
    When I compute the normal mean sequential design
    Then the fixed sample information (I_f) should be 31.40 with 50.0 precision
    And the maximum information (I_max) should be 32.40 with 50.0 precision
    And the information levels (I_k) should be "5.40, 10.80, 16.20, 21.60, 27.00, 32.40" with 50.0 precision
    And the critical values (c_k) should be "5.029, 3.556, 2.903, 2.515, 2.249, 2.053" with 50.0 precision
    And the total sample size (n_max) should be 155.4 with 200.0 precision

  Scenario: Computing Pocock design for a single-arm binomial test (Subsection 3.6.1)
    Given a two-sided "single-arm" binomial design with alpha 0.05
    And a target power 0.9 at effect size 0.2
    And a null hypothesis proportion (p_0) 0.6
    And a maximum of 4 looks with "pocock" spending
    When I compute the single-arm binomial sequential design
    Then the fixed sample information (I_f) should be 262.7 with 50.0 precision
    And the maximum information (I_max) should be 310.8 with 50.0 precision
    And the total sample size (n_max) should be 76 with 50.0 precision
    And the sample size increment per group per look should be 19 with 15.0 precision
    And the critical values (c_k) should be 2.361 with 5.0 precision


  Scenario: Computing O'Brien-Fleming design for binomial outcomes (Subsection 3.6.2)
    Given a two-sided binomial A/B test design with alpha 0.05
    And a target power 0.8 at effect size 0.2
    And a baseline proportion (p_control) 0.5
    And a maximum of 8 looks with "obrien_fleming" spending
    When I compute the binomial sequential design
    Then the fixed sample information (I_f) should be 196.2 with 50.0 precision
    And the maximum information (I_max) should be 203.5 with 50.0 precision
    And the total sample size per group (n_g) should be 104 with 50.0 precision
    And the sample size increment per group per look should be 13 with 50.0 precision
    And the O'Brien-Fleming boundary constant (C_OBF) should be 2.072 with 5.0 precision
    And the standardized boundaries (z_k) should be "5.861, 4.144, 3.384, 2.930, 2.621, 2.393, 2.215, 2.072" with 50.0 precision

  Scenario: Computing O'Brien-Fleming design for survival data (Subsection 3.7)
    Given a two-sided "log-rank" design with alpha 0.05
    And a target power 0.8 at hazard ratio 1.5
    And a maximum of 5 looks with "obrien_fleming" spending
    When I compute the log-rank sequential design
    Then the fixed sample information (I_f) should be 47.85 with 50.0 precision
    And the maximum information (I_max) should be 49.19 with 50.0 precision
    And the total number of events (d_max) should be 197 with 50.0 precision
    And the boundary values should be "4.562, 3.226, 2.634, 2.281, 2.04" with 50.0 precision

  Scenario: Properties of group sequential t-tests (Table 3.3)
    # Calculations are performed assuming the total number of subjects (n_k) across
    # two treatment groups at look k, even if it results in fractional subjects per group.
    # Degrees of freedom at look k: nu_k = (k/K)(nu_K + 2) - 2.
    Given a two-sided t-test design with alpha 0.05
    And a maximum of <K> looks with "<spending>" spending
    And a final degrees of freedom (nu_K) <nu_K>
    And simulation precision with 5000 replicates
    When I evaluate the group sequential t-test performance
    Then the actual type-I error should be <alpha_actual> with 50.0 precision
    And the actual power should be <power_actual> with 50.0 precision

    Examples:
      | spending       | K | nu_K | alpha_actual | power_actual |
      | pocock         | 3 | 7    | 0.060        | 0.741        |
      | pocock         | 3 | 13   | 0.054        | 0.774        |
      | pocock         | 5 | 13   | 0.058        | 0.760        |
      | pocock         | 5 | 23   | 0.054        | 0.780        |
      | pocock         | 8 | 22   | 0.058        | 0.774        |
      | pocock         | 8 | 38   | 0.056        | 0.786        |
      | obrien_fleming | 3 | 7    | 0.054        | 0.793        |
      | obrien_fleming | 3 | 13   | 0.052        | 0.798        |
      | obrien_fleming | 5 | 13   | 0.055        | 0.795        |
      | obrien_fleming | 5 | 23   | 0.051        | 0.797        |
      | obrien_fleming | 8 | 22   | 0.055        | 0.799        |
      | obrien_fleming | 8 | 38   | 0.052        | 0.803        |
      | wang_tsiatis   | 3 | 7    | 0.057        | 0.782        |
      | wang_tsiatis   | 3 | 13   | 0.054        | 0.794        |
      | wang_tsiatis   | 5 | 13   | 0.056        | 0.791        |
      | wang_tsiatis   | 5 | 23   | 0.051        | 0.794        |
      | wang_tsiatis   | 8 | 22   | 0.056        | 0.793        |
      | wang_tsiatis   | 8 | 38   | 0.052        | 0.796        |

  Scenario: Two-sample t-test with O'Brien-Fleming boundaries (Subsection 3.8.2, Two-Treatment Comparison)
    Given the problem setup "Two-sample comparison: X_Ai ~ N(mu_A, sigma^2), X_Bi ~ N(mu_B, sigma^2), sigma^2 unknown"
    And we test "H_0: mu_A = mu_B vs H_1: mu_A != mu_B"
    And the stat definition is "T_k = (sum(X_Ai) - sum(X_Bi)) / sqrt(2*m*k * s_k^2)"
    And the degrees of freedom are 2*m*k - 2
    And the design targets alpha 0.01
    And the maximum number of looks is 4 with "obrien_fleming" spending
    And each group contains m = 8 observations per treatment
    When I compute the t-statistic sequential design with the significance-level approach based on the canonical Gaussian process model
    Then the t-statistic thresholds should be "t(14, 1 - Phi(5.218 * 1^(-0.5))), t(30, 1 - Phi(5.218 * 2^(-0.5))), t(46, 1 - Phi(5.218 * 3^(-0.5))), t(62, 1 - Phi(5.218 * 4^(-0.5)))" with 0.001 precision

  Scenario: Covariate-adjusted t-test (Subsection 3.8.2, Two-Treatment Comparison Adjusted for Covariates)
    Given the problem setup "Linear model: Y_i = beta_0 + beta_1 * Z1_i + ... + beta_5 * Z5_i + epsilon_i, sigma^2 unknown"
    And we test "H_0: beta_1 = 0 vs H_1: beta_1 != 0"
    And the stat definition is "T_k = beta_hat_1_k / sqrt(Var_hat(beta_hat_1_k))"
    And we assume p = 6 parameters (5 covariates and 1 treatment effect)
    # The degrees of freedom are nu_k = n_k - p
    And the degrees of freedom are n_k - 6
    And the design targets alpha 0.05
    And the maximum number of looks is 6 with "obrien_fleming" spending
    And we take a total of n_max = 156 observations as a convenient sample size, giving 26 per group
    When I compute the t-statistic sequential design with the significance-level approach based on the canonical Gaussian process model
    Then the t-statistic thresholds should be "t(20, 1 - Phi(6.131 * 1^(-0.5))), t(46, 1 - Phi(6.131 * 2^(-0.5))), t(72, 1 - Phi(6.131 * 3^(-0.5))), t(98, 1 - Phi(6.131 * 4^(-0.5))), t(124, 1 - Phi(6.131 * 5^(-0.5))), t(150, 1 - Phi(6.131 * 6^(-0.5)))" with 10.0 precision
