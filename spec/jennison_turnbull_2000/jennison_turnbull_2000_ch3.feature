# Reference:
# Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with Applications to Clinical Trials. Chapman and Hall/CRC.

Feature: Sequential Test Design Computation (Jennison & Turnbull 2000)
  As a statistical designer
  I want to verify the group sequential design for various outcomes
  So that I can match the results from classic textbooks

  Background:
    Given simulation precision with 60000 samples

  Scenario: Computing two-sided O'Brien-Fleming design for a paired comparison (Subsection 3.2.2)
    Given a two-sided paired comparison design with alpha 0.05
    And a target power 0.9 at effect size 1.0
    And a known variance (sigma squared) 6.0
    And a maximum of 5 looks with "obrien_fleming" spending
    When I compute the normal mean sequential design
    Then the information levels (I_k) should be "2.16, 4.31, 6.47, 8.63, 10.78" with 0.15 precision
    And the critical values (c_k) should be "4.562, 3.226, 2.634, 2.281, 2.040" with 0.05 precision
    And the total number of pairs (n_max) should be 64.7 with 1.0 precision
    And the required pairs per group should be 12.9 with 0.2 precision
    And the rounded pairs per group should be 13

  Scenario: Computing two-sided Wang-Tsiatis design for a 2-period crossover trial (Subsection 3.2.2)
    Given a two-sided crossover trial design with alpha 0.05
    And a target power 0.8 at effect size 0.6
    And a known variance (sigma squared) 9.0
    And a maximum of 4 looks with "wang_tsiatis" spending
    When I compute the normal mean sequential design
    Then the information levels (I_k) should be "5.81, 11.61, 17.42, 23.23" with 0.3 precision
    And the critical values (c_k) should be "2.988, 2.513, 2.271, 2.113" with 0.1 precision
    And the total subjects per sequence (n_max) should be 104.5 with 2.0 precision
    And the required subjects per sequence per group should be 26.1 with 0.4 precision
    And the rounded subjects per sequence per group should be 27

  Scenario Outline: Operating characteristics with varying group sizes (Table 3.1)
    Given a two-sided normal mean design planned for alpha 0.05
    And a planning sample size sequence per group "<n_plan>"
    And a spending function or shape "<spending>"
    When the actual sample size sequence per group is "<n_actual>"
    Then the actual alpha should be <alpha_actual> with 0.015 precision
    And the actual power should be <power_actual> with 0.02 precision for effect 1.0 and variance 4.0

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

  Scenario Outline: Operating characteristics with information mismatch (Table 3.2)
    Given a two-sided normal mean design planned for alpha 0.05 and power 0.9
    And a planning information sequence for <K> looks with equal increments
    And a spending function "<spending>"
    When the actual information sequence is I_k = <pi> * (k/K)^<r> * I_max
    Then the actual alpha should be <alpha_actual> with 0.015 precision
    And the actual power should be <power_actual> with 0.02 precision

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
    Given a two-sided normal mean test design with alpha 0.05
    And a target power 0.8 at effect size 0.5
    And a known variance (sigma squared) 1.2
    And a maximum of 6 looks with "obrien_fleming" spending
    When I compute the normal mean sequential design
    Then the maximum information (I_max) should be 32.22 with 0.1 precision
    And the total sample size (n_max) should be 155.5 with 1.5 precision
    And the boundary values should be "5.029, 3.556, 2.903, 2.514, 2.249, 2.053" with 0.08 precision

  Scenario: Computing Pocock design for a single-arm binomial test (Subsection 3.6.1)
    Given a two-sided single-arm binomial test design with alpha 0.05
    And a target power 0.9 at effect size 0.2
    And a null hypothesis proportion (p_0) 0.6
    And a maximum of 4 looks with "pocock" spending
    When I compute the single-arm binomial sequential design
    Then the maximum information (I_max) should be 311.4 with 0.1 precision
    And the total sample size (n_max) should be 76 with 1.0 precision
    And the Pocock boundary value (C) should be 2.361 with 0.05 precision
    And the critical difference in proportions should be 0.265 with 0.05 precision / sqrt(k)

  Scenario: Operating characteristics of the single-arm binomial test (Subsection 3.6.1)
    Given a two-sided single-arm binomial test design with 4 looks and total sample size 76
    And a null hypothesis proportion (p_0) 0.6
    When I evaluate the operating characteristics
    Then the actual alpha should be 0.050 with 0.01 precision
    And the nominal power at p = 0.8 (using null variance) should be 0.906 with 0.005 precision
    And the true power at p = 0.8 (using alternative variance) should be 0.982 with 0.005 precision

  Scenario: Computing O'Brien-Fleming design for binomial outcomes (Subsection 3.6.2)
    Given a two-sided A/B test design with alpha 0.05
    And a target power 0.8 at effect size 0.2
    And a baseline proportion (p_control) 0.5
    And a maximum of 8 looks with "obrien_fleming" spending
    When I compute the binomial sequential design
    Then the fixed sample information (I_f) should be 196.2 with 0.1 precision
    And the maximum information (I_max) should be 203.2 with 0.1 precision
    And the total sample size per group (n_g) should be 104 with 1.5 precision
    And the sample size increment per group per look should be 13 with 1.0 precision
    And the O'Brien-Fleming boundary constant (C_OBF) should be 2.072 with 0.05 precision
    And the standardized boundary at look k should be 2.072 with 0.08 precision * sqrt(8/k)
    And the critical difference in proportions should be 2.30 with 0.05 precision * sqrt(p_bar * (1-p_bar)) / k

  Scenario: Operating characteristics of the A/B binomial test (Subsection 3.6.2)
    Given a two-sided A/B test design with 8 looks and sample size per group 104
    And a baseline proportion (p_control) 0.5
    When I evaluate the A/B operating characteristics
    Then the actual alpha should be 0.050 with 0.01 precision
    And the nominal power at delta = 0.2 (using sigma^2 = 0.25) should be 0.801 with 0.02 precision
    And the true power at p_A = 0.4, p_B = 0.6 (using sigma^2 = 0.24) should be 0.816 with 0.02 precision

  Scenario: Computing O'Brien-Fleming design for survival data (Subsection 3.7)
    Given a two-sided log-rank test design with alpha 0.05
    And a target power 0.8 at hazard ratio 1.5
    And a maximum of 5 looks with "obrien_fleming" spending
    When I compute the log-rank sequential design
    Then the fixed sample information (I_f) should be 47.74 with 0.1 precision
    And the maximum information (I_max) should be 49.70 with 0.1 precision
    And the total number of events (d_max) should be 197 with 2.5 precision
    And the boundary values should be "4.562, 3.226, 2.634, 2.281, 2.040" with 0.08 precision

  Scenario Outline: Properties of group sequential t-tests (Table 3.3)
    # Reference: Section 3.8.1, Table 3.3
    # Note from Table 3.3 footnote:
    # Calculations are performed assuming the total number of subjects (n_k) across
    # two treatment groups at look k, even if it results in fractional subjects per group.
    # Degrees of freedom at look k: nu_k = (k/K)(nu_K + 2) - 2.
    Given a two-sided t-test design with alpha 0.05
    And a maximum of <K> looks with "<spending>" spending
    And a final degrees of freedom (nu_K) <nu_K>
    When I evaluate the group sequential t-test performance
    Then the actual alpha should be <alpha_actual> with 0.02 precision
    And the actual power should be <power_actual> with 0.02 precision
    And the power approximation (3.20) should be 0.8 with 0.01 precision
    And the power approximation (3.21) should be <approx_321> with 0.05 precision

    Examples:
      | spending       | K | nu_K | alpha_actual | power_actual | approx_321 |
      | pocock         | 3 | 7    | 0.060        | 0.741        | 0.905      |
      | pocock         | 3 | 13   | 0.054        | 0.774        | 0.858      |
      | pocock         | 5 | 13   | 0.058        | 0.760        | 0.858      |
      | pocock         | 5 | 23   | 0.054        | 0.780        | 0.833      |
      | obrien_fleming | 3 | 7    | 0.054        | 0.793        | 0.905      |
      | obrien_fleming | 3 | 13   | 0.052        | 0.798        | 0.858      |
      | obrien_fleming | 5 | 13   | 0.055        | 0.795        | 0.858      |
      | obrien_fleming | 5 | 23   | 0.051        | 0.797        | 0.833      |
      | wang_tsiatis   | 3 | 7    | 0.057        | 0.782        | 0.905      |
      | wang_tsiatis   | 3 | 13   | 0.054        | 0.794        | 0.858      |

  Scenario: Two-sample t-test with O'Brien-Fleming boundaries (Subsection 3.8.2, Example 1)
    Given a two-sided t-test design with alpha 0.01
    And a maximum of 4 looks with "obrien_fleming" spending
    And a final degrees of freedom (nu_K) 62
    And a parameter count (p) 2
    When I evaluate the group sequential t-test performance for effect 1.0 (sigma units)
    Then the actual alpha should be 0.0101 with 0.01 precision
    And the actual power should be 0.9021 with 0.03 precision

  Scenario: Covariate-adjusted t-test (Subsection 3.8.2, Example 2)
    Given a two-sided t-test design with alpha 0.05
    And a maximum of 6 looks with "obrien_fleming" spending
    And a final degrees of freedom (nu_K) 150
    And a parameter count (p) 6
    When I evaluate the group sequential t-test performance for effect 0.5 (beta_1) and sigma_sq 1.2
    Then the power approximation (3.20) should be 0.796 with 0.01 precision
