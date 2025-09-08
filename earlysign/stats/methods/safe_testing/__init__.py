"""
Safe testing methods based on e-values and e-processes for anytime-valid inference.

This module implements various safe testing approaches that provide anytime-valid statistical
inference without traditional alpha-spending or multiple comparison corrections.

The core mathematical functions for safe testing including beta-binomial e-value
calculations and optimization are available in the `core` module.

Key concepts:
- E-values: Evidence measures that provide anytime-valid testing
- Anytime validity: Type I error control regardless of stopping time
- No multiple testing correction required

References:
- Grünwald, P., de Heide, R., & Koolen, W. (2020). Safe testing. arXiv:1906.07801.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference.
"""
