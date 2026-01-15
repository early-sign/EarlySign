"""
earlysign.v0.templates - User-Friendly Facade
====================================

This module provides an off-the-shelf usage interface for using this package,
organizing functionality by business use cases and ubiquitous language from the domain.
In terms of the design patterns, this is the facade pattern.

The API is designed to be:
- **Business-Oriented**: Organized by what users want to accomplish
- **Intuitive**: Use familiar terminology from statistics and experimentation
- **Discoverable**: Clear naming that maps to common experimental scenarios
- **Consistent**: Uniform interface patterns across different methods

Unified Interface
-----------------
All A/B testing functionality is consolidated in `earlysign.v0.templates.ab_test`:
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
- earlysign.v0.methods: Statistical methods, operators, and reporting helpers

The facade pattern allows users to focus on their business objectives
without needing to understand the internal framework architecture.
"""
