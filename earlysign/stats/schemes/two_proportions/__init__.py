"""
Two-proportion testing schemes for A/B experiments.

This module provides complete implementations for comparing two binomial
proportions using different statistical methodologies:

- Group Sequential Testing (GST) implementations with error spending
- Safe Testing implementations using beta-binomial e-processes
- Data ingestion and validation for two-proportion experiments
- Experiment orchestration and result extraction

Main components:
- `gst.py`: Group sequential testing experiment modules
- `safe.py`: Safe testing experiment modules
- `gst_components.py`: GST-specific statistical components
- `safe_components.py`: Safe testing statistical components
- `ingest.py`: Data validation and ingestion
- `model.py`: Type definitions and payload schemas
- `reduce.py`: Data aggregation utilities
"""
