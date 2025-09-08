"""
earlysign.api - User-Friendly Facade
====================================

This module provides an off-the-shelf usage interface for using this package,
organizing functionality by business use cases and ubiquitous language from the domain.
In terms of the design patterns, this is the facade pattern.

The API is designed to be:
- **Business-Oriented**: Organized by what users want to accomplish
- **Intuitive**: Use familiar terminology from statistics and experimentation
- **Discoverable**: Clear naming that maps to common experimental scenarios
- **Consistent**: Uniform interface patterns across different methods

Examples
--------
>>> # A/B Testing with interim analysis
>>> from earlysign.api.ab_test import interim_analysis
>>> experiment = interim_analysis("checkout_test", alpha=0.05, looks=4)
>>>
>>> # Guardrail monitoring
>>> from earlysign.api.ab_test import guardrail_monitoring
>>> monitor = guardrail_monitoring("conversion_safety", sensitivity="balanced")
>>>
>>> # Continuous monitoring
>>> from earlysign.api.ab_test import continuous_monitoring
>>> tracker = continuous_monitoring("feature_impact", baseline_assumption="no_effect")

Unified Interface
-----------------
All A/B testing functionality is consolidated in `earlysign.api.ab_test`:
- `interim_analysis()`: A/B tests with planned interim looks
- `fixed_sample_test()`: Traditional fixed-sample A/B tests
- `guardrail_monitoring()`: Safety monitoring with e-values
- `continuous_monitoring()`: Long-term feature impact tracking

Architecture
------------
This facade delegates to the underlying framework components:
- earlysign.core: Basic infrastructure and components
- earlysign.schemes: Specific experimental designs
- earlysign.runners: Execution environments
- earlysign.backends: Data storage solutions

The facade pattern allows users to focus on their business objectives
without needing to understand the internal framework architecture.
"""
