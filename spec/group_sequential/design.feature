Feature: Sequential Test Design Computation
  As a statistical designer
  I want to compute the boundaries of a one-sided sequential test for an A/B test
  So that I can control the Type I error rate across multiple interim looks

  Scenario: Computing O'Brien-Fleming boundaries for a 2-look design
    Given a one-sided A/B test design with alpha 0.05
    And an interim schedule at information fractions [0.5, 1.0]
    And an "obrien_fleming" alpha-spending function
    When I compute the sequential boundaries
    Then look 1 should have an upper Z-boundary around 2.326
    And look 2 should have an upper Z-boundary around 1.645
