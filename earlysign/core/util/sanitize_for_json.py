"""
Utilities for normalising Python and NumPy values into JSON-serialisable
representations.

The helpers in this module perform a best-effort conversion of nested
structures so that data can be persisted to JSON payloads without
raising ``TypeError`` for unsupported types.

Examples
--------
>>> import numpy as np
>>> sanitize_for_json({"value": np.float64(1.5), "items": [np.int64(2), float("inf")]})
{'value': 1.5, 'items': [2, None]}
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _is_numpy_scalar(value: Any) -> bool:
    return isinstance(value, (np.generic,))


def _convert_numpy_scalar(value: np.generic) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        casted = float(value)
        return casted if np.isfinite(casted) else None
    if isinstance(value, (np.bytes_, np.str_)):
        return str(value)
    return str(value)


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-friendly structure."""
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8") if isinstance(value, bytes) else value
    if _is_numpy_scalar(value):
        return _convert_numpy_scalar(value)
    if isinstance(value, np.ndarray):
        return [sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, set):
        return [sanitize_for_json(item) for item in sorted(value, key=repr)]
    return str(value)
