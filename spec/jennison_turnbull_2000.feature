# Reference:
# Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with Applications to Clinical Trials. Chapman and Hall/CRC.

Feature: Sequential Test Design Computation (Jennison & Turnbull 2000)
  As a statistical designer
  I want to verify the group sequential design for various outcomes
  So that I can match the results from classic textbooks

  Scenario: Computing O'Brien-Fleming design for a paired comparison (Subsection 3.2.2)
    Given a two-sided paired comparison design with alpha 0.05
    And a target power 0.9 at effect size 1.0
    And a known variance (sigma squared) 6.0
    And a maximum of 5 looks with "obrien_fleming" spending
    When I compute the normal mean sequential design
    Then the maximum information (I_max) should be around 10.78
    And the total sample size (n_max) should be 65

  Scenario: Computing Wang-Tsiatis design for a 2-period crossover trial (Subsection 3.2.2)
    Given a two-sided crossover trial design with alpha 0.05
    And a target power 0.8 at effect size 0.6
    And a known variance (sigma squared) 9.0
    And a maximum of 4 looks with "wang_tsiatis" spending
    When I compute the normal mean sequential design
    Then the maximum information (I_max) should be around 23.23
    And the total sample size (n_max) should be 108

  Scenario: Computing O'Brien-Fleming design for a normal mean (Subsection 3.4.2)
    Given a two-sided normal mean test design with alpha 0.05
    And a target power 0.8 at effect size 0.5
    And a known variance (sigma squared) 1.2
    And a maximum of 6 looks with "obrien_fleming" spending
    When I compute the normal mean sequential design
    Then the maximum information (I_max) should be around 32.40
    And the total sample size (n_max) should be around 155.5
    And the boundary values should be around "5.029, 3.556, 2.903, 2.514, 2.249, 2.053"

  Scenario Outline: Operating characteristics with varying group sizes (Table 3.1)
    Given a two-sided normal mean design planned for alpha 0.05
    And a planning sample size sequence per group "<n_plan>"
    And a spending function or shape "<spending>"
    When the actual sample size sequence per group is "<n_actual>"
    Then the actual alpha should be around <alpha_actual>
    And the actual power should be around <power_actual> for effect 1.0 and variance 4.0

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
    Then the actual alpha should be around <alpha_actual>
    And the actual power should be around <power_actual>

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

  Scenario: Computing Pocock design for a single-arm binomial test (Subsection 3.6.1)
    Given a two-sided single-arm binomial test design with alpha 0.05
    And a target power 0.9 at effect size 0.2
    And a null hypothesis proportion (p_0) 0.6
    And a maximum of 4 looks with "pocock" spending
    When I compute the single-arm binomial sequential design
    Then the maximum information (I_max) should be around 310.8
    And the total sample size (n_max) should be 76

  Scenario: Computing O'Brien-Fleming design for binomial outcomes (Subsection 3.6.2)
    Given a two-sided A/B test design with alpha 0.05
    And a target power 0.8 at effect size 0.2
    And a baseline proportion (p_control) 0.5
    And a maximum of 8 looks with "obrien_fleming" spending
    When I compute the binomial sequential design
    Then the maximum information (I_max) should be around 203.5
    And the sample size per group (n_g) should be 104
