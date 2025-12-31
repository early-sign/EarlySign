Feature: Jennison & Turnbull (2000) Chapter 7 (Flexible Monitoring: The Error Spending Approach)
  As a statistical designer, I want to verify the library's outputs match the results from the classic textbook
  Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with Applications to Clinical Trials. Chapman and Hall/CRC.

  Background:
    Given simulation precision with 60000 samples

  Scenario: Verification of maximum information test constant R_LD (Table 7.1)
    Given a two-sided maximum information test with alpha 0.05
    And a target power <power> at some effect size
    And a maximum of <K> looks with rho-family spending <rho>
    When I compute the inflation factor R_LD
    Then the result should be <R_LD> with 0.02 precision
    # Note: under the canonical Gaussian process model, R_LD is independent of the effect size.

    Examples: 1-beta=0.8
      | K  | rho | power | R_LD  |
      | 1  | 1   | 0.8   | 1.000 |
      | 2  | 1   | 0.8   | 1.082 |
      | 3  | 1   | 0.8   | 1.117 |
      | 4  | 1   | 0.8   | 1.137 |
      | 5  | 1   | 0.8   | 1.150 |
      | 6  | 1   | 0.8   | 1.159 |
      | 7  | 1   | 0.8   | 1.165 |
      | 8  | 1   | 0.8   | 1.170 |
      | 9  | 1   | 0.8   | 1.174 |
      | 10 | 1   | 0.8   | 1.178 |
      | 11 | 1   | 0.8   | 1.180 |
      | 12 | 1   | 0.8   | 1.183 |
      | 15 | 1   | 0.8   | 1.188 |
      | 20 | 1   | 0.8   | 1.193 |
      | 1  | 2   | 0.8   | 1.000 |
      | 2  | 2   | 0.8   | 1.028 |
      | 3  | 2   | 0.8   | 1.045 |
      | 4  | 2   | 0.8   | 1.056 |
      | 5  | 2   | 0.8   | 1.063 |
      | 6  | 2   | 0.8   | 1.069 |
      | 7  | 2   | 0.8   | 1.073 |
      | 8  | 2   | 0.8   | 1.076 |
      | 9  | 2   | 0.8   | 1.079 |
      | 10 | 2   | 0.8   | 1.081 |
      | 11 | 2   | 0.8   | 1.083 |
      | 12 | 2   | 0.8   | 1.085 |
      | 15 | 2   | 0.8   | 1.088 |
      | 20 | 2   | 0.8   | 1.092 |
      | 1  | 3   | 0.8   | 1.000 |
      | 2  | 3   | 0.8   | 1.010 |
      | 3  | 3   | 0.8   | 1.020 |
      | 4  | 3   | 0.8   | 1.027 |
      | 5  | 3   | 0.8   | 1.032 |
      | 6  | 3   | 0.8   | 1.036 |
      | 7  | 3   | 0.8   | 1.039 |
      | 8  | 3   | 0.8   | 1.041 |
      | 9  | 3   | 0.8   | 1.043 |
      | 10 | 3   | 0.8   | 1.045 |
      | 11 | 3   | 0.8   | 1.046 |
      | 12 | 3   | 0.8   | 1.048 |
      | 15 | 3   | 0.8   | 1.050 |
      | 20 | 3   | 0.8   | 1.054 |

    Examples: 1-beta=0.9
      | K  | rho | power | R_LD  |
      | 1  | 1   | 0.9   | 1.000 |
      | 2  | 1   | 0.9   | 1.075 |
      | 3  | 1   | 0.9   | 1.107 |
      | 4  | 1   | 0.9   | 1.124 |
      | 5  | 1   | 0.9   | 1.136 |
      | 6  | 1   | 0.9   | 1.144 |
      | 7  | 1   | 0.9   | 1.150 |
      | 8  | 1   | 0.9   | 1.155 |
      | 9  | 1   | 0.9   | 1.159 |
      | 10 | 1   | 0.9   | 1.162 |
      | 11 | 1   | 0.9   | 1.164 |
      | 12 | 1   | 0.9   | 1.166 |
      | 15 | 1   | 0.9   | 1.171 |
      | 20 | 1   | 0.9   | 1.176 |
      | 1  | 2   | 0.9   | 1.000 |
      | 2  | 2   | 0.9   | 1.025 |
      | 3  | 2   | 0.9   | 1.041 |
      | 4  | 2   | 0.9   | 1.051 |
      | 5  | 2   | 0.9   | 1.058 |
      | 6  | 2   | 0.9   | 1.063 |
      | 7  | 2   | 0.9   | 1.067 |
      | 8  | 2   | 0.9   | 1.070 |
      | 9  | 2   | 0.9   | 1.073 |
      | 10 | 2   | 0.9   | 1.075 |
      | 11 | 2   | 0.9   | 1.077 |
      | 12 | 2   | 0.9   | 1.078 |
      | 15 | 2   | 0.9   | 1.082 |
      | 20 | 2   | 0.9   | 1.085 |
      | 1  | 3   | 0.9   | 1.000 |
      | 2  | 3   | 0.9   | 1.009 |
      | 3  | 3   | 0.9   | 1.018 |
      | 4  | 3   | 0.9   | 1.025 |
      | 5  | 3   | 0.9   | 1.030 |
      | 6  | 3   | 0.9   | 1.033 |
      | 7  | 3   | 0.9   | 1.036 |
      | 8  | 3   | 0.9   | 1.039 |
      | 9  | 3   | 0.9   | 1.040 |
      | 10 | 3   | 0.9   | 1.042 |
      | 11 | 3   | 0.9   | 1.043 |
      | 12 | 3   | 0.9   | 1.044 |
      | 15 | 3   | 0.9   | 1.047 |
      | 20 | 3   | 0.9   | 1.050 |

  Scenario: Properties of maximum information tests (Table 7.2)
    Given a two-sided maximum information test with alpha 0.05
    And a target power 0.8 at some effect size
    And a maximum of <K> looks with rho-family spending <rho>
    When I evaluate the expected sample size relative to fixed design
    Then the maximum information (R_LD) should be <R_LD_pct> percent with 2.0 precision
    And the expected sample size at theta=0 should be <ASN_0> percent with 2.0 precision
    And the expected sample size at theta=0.5δ should be <ASN_05delta> percent with 2.0 precision
    And the expected sample size at theta=δ should be <ASN_delta> percent with 2.0 precision
    And the expected sample size at theta=1.5δ should be <ASN_15delta> percent with 2.0 precision

    Examples: rho=1
      | K  | rho | R_LD_pct | ASN_0 | ASN_05delta | ASN_delta | ASN_15delta |
      | 1  | 1   | 100.0    | 100.0 | 100.0       | 100.0     | 100.0       |
      | 2  | 1   | 108.2    | 106.9 | 102.1       | 85.0      | 64.8        |
      | 3  | 1   | 111.7    | 109.9 | 103.4       | 81.2      | 56.5        |
      | 4  | 1   | 113.7    | 111.6 | 104.2       | 79.5      | 53.0        |
      | 5  | 1   | 115.0    | 112.7 | 104.8       | 78.5      | 51.0        |
      | 10 | 1   | 117.8    | 115.1 | 106.1       | 76.6      | 47.4        |
      | 20 | 1   | 119.3    | 116.5 | 106.9       | 75.8      | 45.8        |

    Examples: rho=2
      | K  | rho | R_LD_pct | ASN_0 | ASN_05delta | ASN_delta | ASN_15delta |
      | 1  | 2   | 100.0    | 100.0 | 100.0       | 100.0     | 100.0       |
      | 2  | 2   | 102.8    | 102.1 | 99.3        | 86.7      | 67.0        |
      | 3  | 2   | 104.5    | 103.5 | 99.2        | 82.3      | 60.4        |
      | 4  | 2   | 105.6    | 104.4 | 99.2        | 80.1      | 57.1        |
      | 5  | 2   | 106.3    | 105.1 | 99.3        | 78.8      | 55.0        |
      | 10 | 2   | 108.1    | 106.6 | 99.7        | 76.2      | 51.0        |
      | 20 | 2   | 109.2    | 107.5 | 100.0       | 75.1      | 49.1        |

    Examples: rho=3
      | K  | rho | R_LD_pct | ASN_0 | ASN_05delta | ASN_delta | ASN_15delta |
      | 1  | 3   | 100.0    | 100.0 | 100.0       | 100.0     | 100.0       |
      | 2  | 3   | 101.0    | 100.7 | 98.9        | 89.5      | 70.7        |
      | 3  | 3   | 102.0    | 101.4 | 98.3        | 84.6      | 64.7        |
      | 4  | 3   | 102.7    | 102.0 | 98.0        | 82.1      | 61.0        |
      | 5  | 3   | 103.2    | 102.4 | 97.9        | 80.6      | 58.8        |
      | 10 | 3   | 104.5    | 103.4 | 97.9        | 77.8      | 54.6        |
      | 20 | 3   | 105.4    | 104.2 | 98.0        | 76.5      | 52.7        |

  Scenario: Properties of maximum information tests (Table 7.3)
    # References: Table 7.3. alpha=0.05, 1-beta=0.9
    Given a two-sided maximum information test with alpha 0.05
    And a target power 0.9 at some effect size
    And a maximum of <K> looks with rho-family spending <rho>
    When I evaluate the expected sample size relative to fixed design
    Then the maximum information (R_LD) should be <R_LD_pct> percent with 2.0 precision
    And the expected sample size at theta=0 should be <ASN_0> percent with 2.0 precision
    And the expected sample size at theta=0.5δ should be <ASN_05delta> percent with 2.0 precision
    And the expected sample size at theta=δ should be <ASN_delta> percent with 2.0 precision
    And the expected sample size at theta=1.5δ should be <ASN_15delta> percent with 2.0 precision

    Examples: rho=1
      | K  | rho | R_LD_pct | ASN_0 | ASN_05delta | ASN_delta | ASN_15delta |
      | 1  | 1   | 100.0    | 100.0 | 100.0       | 100.0     | 100.0       |
      | 2  | 1   | 107.5    | 106.1 | 99.6        | 77.7      | 58.7        |
      | 3  | 1   | 110.7    | 108.8 | 100.1       | 72.2      | 48.5        |
      | 4  | 1   | 112.4    | 110.3 | 100.5       | 69.8      | 44.5        |
      | 5  | 1   | 113.6    | 111.3 | 100.7       | 68.4      | 42.3        |
      | 10 | 1   | 116.2    | 113.6 | 101.5       | 65.7      | 38.4        |
      | 15 | 1   | 117.1    | 114.4 | 101.8       | 64.9      | 37.3        |
      | 20 | 1   | 117.6    | 114.8 | 102.0       | 64.5      | 36.7        |

    Examples: rho=2
      | K  | rho | R_LD_pct | ASN_0 | ASN_05delta | ASN_delta | ASN_15delta |
      | 1  | 2   | 100.0    | 100.0 | 100.0       | 100.0     | 100.0       |
      | 2  | 2   | 102.5    | 101.9 | 97.9        | 80.5      | 59.6        |
      | 3  | 2   | 104.1    | 103.2 | 97.1        | 75.0      | 52.3        |
      | 4  | 2   | 105.1    | 104.0 | 96.8        | 72.2      | 48.9        |
      | 5  | 2   | 105.8    | 104.6 | 96.7        | 70.5      | 46.8        |
      | 10 | 2   | 107.5    | 106.0 | 96.6        | 67.2      | 42.6        |
      | 15 | 2   | 108.2    | 106.5 | 96.6        | 66.2      | 41.3        |
      | 20 | 2   | 108.5    | 106.8 | 96.6        | 65.7      | 40.7        |

    Examples: rho=3
      | K  | rho | R_LD_pct | ASN_0 | ASN_05delta | ASN_delta | ASN_15delta |
      | 1  | 3   | 100.0    | 100.0 | 100.0       | 100.0     | 100.0       |
      | 2  | 3   | 100.9    | 100.6 | 98.1        | 84.1      | 62.4        |
      | 3  | 3   | 101.8    | 101.3 | 96.8        | 78.2      | 56.7        |
      | 4  | 3   | 102.5    | 101.8 | 96.2        | 75.1      | 53.2        |
      | 5  | 3   | 103.0    | 102.1 | 95.9        | 73.3      | 50.9        |
      | 10 | 3   | 104.2    | 103.1 | 95.4        | 69.8      | 46.5        |
      | 15 | 3   | 104.7    | 103.6 | 95.4        | 68.7      | 45.1        |
      | 20 | 3   | 105.0    | 103.8 | 95.3        | 68.2      | 44.5        |

  Scenario: Group sequential design for parallel group comparison (Subsection 7.2.2)
    Given a two-sided A/B test with alpha 0.05
    And a target power 0.9 at effect size 1.0
    And a known variance (sigma squared) 4.0
    When I analyze the Lan-DeMets design from section 7.2.2 with K=10 and rho=2
    Then the fixed sample information (I_f) should be 10.51 with 0.01 precision
    And the fixed sample size per group should be 85 with 1.5 precision
    And the maximum information (I_max) should be 11.30 with 0.5 precision
    And the maximum sample size per group should be 91 with 1.5 precision

  Scenario: Power evaluation with constrained information (Subsection 7.2.2)
    Given a two-sided A/B test with alpha 0.05
    And a maximum information (I_max) constrained to 10.0
    When I evaluate the required effect size for power 0.9 using K=10 and rho=2
    Then the required effect size (delta) should be 1.06 with 0.01 precision
    When I evaluate the required effect size for power 0.8 using K=10 and rho=2
    Then the inflation factor R_LD should be 1.081 with 0.05 precision
    And the required effect size (delta) should be 0.92 with 0.05 precision

  Scenario: Under-running in Subsection 7.2.2
    Given a two-sided A/B test with alpha 0.05
    And a planned maximum information 11.25
    And a rho-family spending function with rho 2.0
    When I perform a trial with actual information sequence 1.125, 2.25, 3.375, 4.5, 5.625, 6.75, 7.875, 9.0, 10.125, 10.6
    Then the power at delta 1.0 should be 0.88 with 0.01 precision

  Scenario: Over-running in Subsection 7.2.2
    Given a two-sided A/B test with alpha 0.05
    And a planned maximum information 11.25
    And a rho-family spending function with rho 2.0
    And a known variance (sigma squared) 4.0
    When I perform a trial with group size 12 per stage until I_max 11.25 is reached
    Then the trial should stop at look 8
    And the final maximum information should be 12.0 with 0.5 precision
    And the power at delta 1.0 should be 0.92 with 0.02 precision

  Scenario: Power under mismatched information schedules (Table 7.4)
    Given a two-sided maximum information test with alpha 0.05
    And a target power 0.9 at effect size 1.0
    When I evaluate Table 7.4 robustness with K_tilde=<K>, rho=<rho>, r=<r>, and pi=<pi>
    Then the resulting power should be <power> with 0.02 precision

    Examples: rho=1
      | K  | rho | r    | pi  | power |
      | 2  | 1   | 0.80 | 0.9 | 0.870 |
      | 2  | 1   | 1.00 | 1.0 | 0.900 |
      | 2  | 1   | 1.25 | 1.1 | 0.925 |
      | 5  | 1   | 0.80 | 0.9 | 0.875 |
      | 5  | 1   | 1.00 | 1.0 | 0.900 |
      | 5  | 1   | 1.25 | 1.1 | 0.920 |
      | 10 | 1   | 0.80 | 0.9 | 0.878 |
      | 10 | 1   | 1.00 | 1.0 | 0.900 |
      | 10 | 1   | 1.25 | 1.1 | 0.912 |
      | 15 | 1   | 1.25 | 1.1 | 0.902 |

    Examples: rho=2
      | K  | rho | r    | pi  | power |
      | 2  | 2   | 0.80 | 0.9 | 0.869 |
      | 2  | 2   | 1.00 | 1.1 | 0.923 |
      | 5  | 2   | 1.25 | 1.0 | 0.901 |
      | 10 | 2   | 0.80 | 1.0 | 0.899 |
      | 15 | 2   | 1.00 | 0.9 | 0.877 |

    Examples: rho=3
      | K  | rho | r    | pi  | power |
      | 2  | 3   | 0.80 | 0.9 | 0.868 |
      | 5  | 3   | 1.25 | 1.1 | 0.921 |
      | 10 | 3   | 1.00 | 1.0 | 0.900 |
      | 15 | 3   | 0.80 | 1.1 | 0.904 |

  Scenario: Power when number of analyses differs from plan (Table 7.5)
    Given a two-sided maximum information test with alpha 0.05
    And a target power 0.9 at effect size 1.0
    When I evaluate Table 7.5 robustness with K_tilde=<K_tilde>, K=<K>, and rho=<rho>
    Then the resulting power should be <power> with 0.02 precision

    Examples: rho=1
      | K_tilde | K  | rho | power |
      | 2       | 5  | 1   | 0.883 |
      | 5       | 2  | 1   | 0.915 |
      | 10      | 2  | 1   | 0.921 |
      | 2       | 10 | 1   | 0.875 |
      | 15      | 5  | 1   | 0.909 |

    Examples: rho=2
      | K_tilde | K  | rho | power |
      | 5       | 2  | 2   | 0.909 |
      | 2       | 5  | 2   | 0.891 |
      | 10      | 15 | 2   | 0.898 |
      | 15      | 10 | 2   | 0.902 |

    Examples: rho=3
      | K_tilde | K  | rho | power |
      | 2       | 15 | 3   | 0.889 |
      | 15      | 2  | 3   | 0.910 |
      | 5       | 10 | 3   | 0.897 |
      | 10      | 5  | 3   | 0.903 |

  Scenario: Beta-Blocker Heart Attack Trial (BHAT) verification (Subsection 7.2.3)
    Given a two-sided maximum information test with alpha 0.05
    And the BHAT trial setup with total duration 48 months
    And rho-family spending rho 1.0 based on calendar time
    And information estimated as deaths divided by 4.0
    When I analyze the BHAT trial with calendar months 11, 16, 21, 28, 34, 40
    And the observed death counts are 56, 77, 126, 177, 247, 318
    Then the boundaries should be "2.53, 2.59, 2.64, 2.50, 2.51, 2.47" with 0.08 precision
    And the observed Z-statistics 1.68, 2.24, 2.37, 2.30, 2.34, 2.82 should reject H0 at look 6

  Scenario: Verification of maximum information test constant R_OS (Table 7.6)
    Given a one-sided maximum information test with alpha 0.05 and beta <beta>
    And a maximum of <K> looks with rho-family spending <rho> for both errors
    When I compute the inflation factor R_OS
    Then the result should be <R_OS> with 0.10 precision

    Examples: 1-beta=0.8
      | K  | beta | rho | R_OS  |
      | 1  | 0.2  | 2   | 1.000 |
      | 2  | 0.2  | 2   | 1.043 |
      | 3  | 0.2  | 2   | 1.070 |
      | 4  | 0.2  | 2   | 1.087 |
      | 5  | 0.2  | 2   | 1.098 |
      | 6  | 0.2  | 2   | 1.106 |
      | 7  | 0.2  | 2   | 1.111 |
      | 8  | 0.2  | 2   | 1.116 |
      | 9  | 0.2  | 2   | 1.120 |
      | 10 | 0.2  | 2   | 1.123 |
      | 11 | 0.2  | 2   | 1.125 |
      | 12 | 0.2  | 2   | 1.127 |
      | 15 | 0.2  | 2   | 1.132 |
      | 20 | 0.2  | 2   | 1.137 |
      | 1  | 0.2  | 3   | 1.000 |
      | 2  | 0.2  | 3   | 1.014 |
      | 3  | 0.2  | 3   | 1.028 |
      | 4  | 0.2  | 3   | 1.038 |
      | 5  | 0.2  | 3   | 1.045 |
      | 6  | 0.2  | 3   | 1.050 |
      | 7  | 0.2  | 3   | 1.054 |
      | 8  | 0.2  | 3   | 1.058 |
      | 9  | 0.2  | 3   | 1.060 |
      | 10 | 0.2  | 3   | 1.062 |
      | 11 | 0.2  | 3   | 1.064 |
      | 12 | 0.2  | 3   | 1.066 |
      | 15 | 0.2  | 3   | 1.069 |
      | 20 | 0.2  | 3   | 1.073 |

    Examples: 1-beta=0.9
      | K  | beta | rho | R_OS  |
      | 1  | 0.1  | 2   | 1.000 |
      | 2  | 0.1  | 2   | 1.044 |
      | 3  | 0.1  | 2   | 1.072 |
      | 4  | 0.1  | 2   | 1.089 |
      | 5  | 0.1  | 2   | 1.100 |
      | 6  | 0.1  | 2   | 1.108 |
      | 7  | 0.1  | 2   | 1.114 |
      | 8  | 0.1  | 2   | 1.119 |
      | 9  | 0.1  | 2   | 1.123 |
      | 10 | 0.1  | 2   | 1.126 |
      | 11 | 0.1  | 2   | 1.128 |
      | 12 | 0.1  | 2   | 1.131 |
      | 15 | 0.1  | 2   | 1.135 |
      | 20 | 0.1  | 2   | 1.140 |
      | 1  | 0.1  | 3   | 1.000 |
      | 2  | 0.1  | 3   | 1.015 |
      | 3  | 0.1  | 3   | 1.030 |
      | 4  | 0.1  | 3   | 1.040 |
      | 5  | 0.1  | 3   | 1.048 |
      | 6  | 0.1  | 3   | 1.053 |
      | 7  | 0.1  | 3   | 1.058 |
      | 8  | 0.1  | 3   | 1.061 |
      | 9  | 0.1  | 3   | 1.064 |
      | 10 | 0.1  | 3   | 1.066 |
      | 11 | 0.1  | 3   | 1.068 |
      | 12 | 0.1  | 3   | 1.070 |
      | 15 | 0.1  | 3   | 1.073 |
      | 20 | 0.1  | 3   | 1.077 |

    Examples: 1-beta=0.95
      | K  | beta | rho | R_OS  |
      | 1  | 0.05 | 2   | 1.000 |
      | 2  | 0.05 | 2   | 1.045 |
      | 3  | 0.05 | 2   | 1.073 |
      | 4  | 0.05 | 2   | 1.090 |
      | 5  | 0.05 | 2   | 1.101 |
      | 6  | 0.05 | 2   | 1.109 |
      | 7  | 0.05 | 2   | 1.115 |
      | 8  | 0.05 | 2   | 1.120 |
      | 9  | 0.05 | 2   | 1.124 |
      | 10 | 0.05 | 2   | 1.127 |
      | 11 | 0.05 | 2   | 1.130 |
      | 12 | 0.05 | 2   | 1.132 |
      | 15 | 0.05 | 2   | 1.137 |
      | 20 | 0.05 | 2   | 1.142 |
      | 1  | 0.05 | 3   | 1.000 |
      | 2  | 0.05 | 3   | 1.016 |
      | 3  | 0.05 | 3   | 1.031 |
      | 4  | 0.05 | 3   | 1.042 |
      | 5  | 0.05 | 3   | 1.050 |
      | 6  | 0.05 | 3   | 1.055 |
      | 7  | 0.05 | 3   | 1.060 |
      | 8  | 0.05 | 3   | 1.063 |
      | 9  | 0.05 | 3   | 1.066 |
      | 10 | 0.05 | 3   | 1.069 |
      | 11 | 0.05 | 3   | 1.071 |
      | 12 | 0.05 | 3   | 1.072 |
      | 15 | 0.05 | 3   | 1.076 |
      | 20 | 0.05 | 3   | 1.080 |

  Scenario: Properties of one-sided asymmetric maximum information tests (Table 7.7)
    # References: Table 7.7. alpha=0.05, 1-beta=0.8
    Given a one-sided maximum information test with alpha 0.05 and beta 0.2
    And a maximum of <K> looks with rho-family spending <rho> for both errors
    When I evaluate the one-sided expected sample size relative to fixed design
    Then the maximum information (R_OS) should be <R_OS_pct> percent with 10.0 precision
    And the expected sample size at theta=0 should be <ASN_0> percent with 4.0 precision
    And the expected sample size at theta=0.5δ should be <ASN_05delta> percent with 4.0 precision
    And the expected sample size at theta=δ should be <ASN_delta> percent with 4.0 precision

    Examples: rho=2
      | K  | rho | R_OS_pct | ASN_0 | ASN_05delta | ASN_delta |
      | 1  | 2   | 100.0    | 100.0 | 100.0       | 100.0     |
      | 2  | 2   | 104.3    | 74.5  | 87.8        | 84.6      |
      | 3  | 2   | 107.0    | 68.0  | 82.8        | 79.2      |
      | 4  | 2   | 108.7    | 64.7  | 80.2        | 76.4      |
      | 5  | 2   | 109.8    | 62.7  | 78.5        | 74.6      |
      | 10 | 2   | 112.3    | 58.7  | 75.2        | 71.1      |
      | 15 | 2   | 113.2    | 57.4  | 74.0        | 69.9      |
      | 20 | 2   | 113.7    | 56.8  | 73.5        | 69.3      |

    Examples: rho=3
      | K  | rho | R_OS_pct | ASN_0 | ASN_05delta | ASN_delta |
      | 1  | 3   | 100.0    | 100.0 | 100.0       | 100.0     |
      | 2  | 3   | 101.4    | 79.6  | 91.6        | 88.3      |
      | 3  | 3   | 102.8    | 73.1  | 86.4        | 82.6      |
      | 4  | 3   | 103.8    | 69.6  | 83.7        | 79.6      |
      | 5  | 3   | 104.5    | 67.6  | 82.0        | 77.8      |
      | 10 | 3   | 106.2    | 63.5  | 78.5        | 74.2      |
      | 15 | 3   | 106.9    | 62.2  | 77.3        | 73.0      |
      | 20 | 3   | 107.3    | 61.6  | 76.8        | 72.4      |

  Scenario: Properties of one-sided asymmetric maximum information tests (Table 7.8)
    # References: Table 7.8. alpha=0.05, 1-beta=0.9
    Given a one-sided maximum information test with alpha 0.05 and beta 0.1
    And a maximum of <K> looks with rho-family spending <rho> for both errors
    When I evaluate the one-sided expected sample size relative to fixed design
    Then the maximum information (R_OS) should be <R_OS_pct> percent with 10.0 precision
    And the expected sample size at theta=0 should be <ASN_0> percent with 4.0 precision
    And the expected sample size at theta=0.5δ should be <ASN_05delta> percent with 4.0 precision
    And the expected sample size at theta=δ should be <ASN_delta> percent with 4.0 precision

    Examples: rho=2
      | K  | rho | R_OS_pct | ASN_0 | ASN_05delta | ASN_delta |
      | 1  | 2   | 100.0    | 100.0 | 100.0       | 100.0     |
      | 2  | 2   | 104.4    | 74.5  | 88.7        | 79.7      |
      | 3  | 2   | 107.2    | 68.1  | 83.9        | 73.8      |
      | 4  | 2   | 108.9    | 64.9  | 81.3        | 70.7      |
      | 5  | 2   | 110.0    | 62.9  | 79.7        | 68.8      |
      | 10 | 2   | 112.6    | 58.9  | 76.4        | 65.1      |
      | 15 | 2   | 113.5    | 57.7  | 75.3        | 63.8      |
      | 20 | 2   | 114.0    | 57.0  | 74.8        | 63.2      |

    Examples: rho=3
      | K  | rho | R_OS_pct | ASN_0 | ASN_05delta | ASN_delta |
      | 1  | 3   | 100.0    | 100.0 | 100.0       | 100.0     |
      | 2  | 3   | 101.5    | 79.0  | 92.0        | 83.6      |
      | 3  | 3   | 103.0    | 72.6  | 86.9        | 77.5      |
      | 4  | 3   | 104.0    | 69.2  | 84.2        | 74.2      |
      | 5  | 3   | 104.8    | 67.1  | 82.6        | 72.2      |
      | 10 | 3   | 106.6    | 63.1  | 79.2        | 68.4      |
      | 15 | 3   | 107.3    | 61.8  | 78.0        | 67.1      |
      | 20 | 3   | 107.7    | 61.1  | 77.4        | 66.5      |

  Scenario: Properties of one-sided asymmetric maximum information tests (Table 7.9)
    # References: Table 7.9. alpha=0.05, 1-beta=0.95
    Given a one-sided maximum information test with alpha 0.05 and beta 0.05
    And a maximum of <K> looks with rho-family spending <rho> for both errors
    When I evaluate the one-sided expected sample size relative to fixed design
    Then the maximum information (R_OS) should be <R_OS_pct> percent with 10.0 precision
    And the expected sample size at theta=0 should be <ASN_0> percent with 4.0 precision
    And the expected sample size at theta=0.5δ should be <ASN_05delta> percent with 4.0 precision
    And the expected sample size at theta=δ should be <ASN_delta> percent with 4.0 precision

    Examples: rho=2
      | K  | rho | R_OS_pct | ASN_0 | ASN_05delta | ASN_delta |
      | 1  | 2   | 100.0    | 100.0 | 100.0       | 100.0     |
      | 2  | 2   | 104.5    | 74.9  | 89.2        | 74.9      |
      | 3  | 2   | 107.3    | 68.7  | 84.5        | 68.7      |
      | 4  | 2   | 109.0    | 65.4  | 81.9        | 65.4      |
      | 5  | 2   | 110.1    | 63.4  | 80.3        | 63.4      |
      | 10 | 2   | 112.7    | 59.5  | 77.1        | 59.5      |
      | 15 | 2   | 113.7    | 58.3  | 76.0        | 58.3      |
      | 20 | 2   | 114.2    | 57.7  | 75.5        | 57.7      |

    Examples: rho=3
      | K  | rho | R_OS_pct | ASN_0 | ASN_05delta | ASN_delta |
      | 1  | 3   | 100.0    | 100.0 | 100.0       | 100.0     |
      | 2  | 3   | 101.6    | 79.0  | 92.2        | 79.0      |
      | 3  | 3   | 103.1    | 72.7  | 87.2        | 72.7      |
      | 4  | 3   | 104.2    | 69.2  | 84.5        | 69.2      |
      | 5  | 3   | 105.0    | 67.1  | 82.8        | 67.1      |
      | 10 | 3   | 106.9    | 63.1  | 79.5        | 63.1      |
      | 15 | 3   | 107.6    | 61.8  | 78.3        | 61.8      |
      | 20 | 3   | 108.0    | 61.2  | 77.8        | 61.2      |

