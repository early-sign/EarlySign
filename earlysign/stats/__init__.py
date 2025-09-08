"""
Statistical methods and theories for sequential testing.

This module provides a 2-layer architecture for statistical methods:

1. **Methods** (earlysign.stats.methods):
   Statistical theories and methodologies like group sequential testing,
   safe testing, and other sequential analysis approaches. Contains the
   mathematical foundations and core algorithms.

2. **Schemes** (earlysign.stats.schemes):
   Problem-specific implementations that combine methods and data handling
   for particular experimental scenarios (A/B testing, survival analysis, etc.).

This architecture separates statistical theory from problem-domain specifics,
enabling flexible composition and reuse of statistical methods across different
experimental contexts.
"""
