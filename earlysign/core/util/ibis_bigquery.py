"""BigQuery-specific Ibis utilities."""

import ibis
import ibis.expr.datatypes as dt


@ibis.udf.scalar.builtin(name="PARSE_JSON")  # type: ignore
def bq_parse_json(json_str: dt.string) -> dt.json:
    """Parse JSON string to JSON type in BigQuery.

    This maps to the native BigQuery `PARSE_JSON` function.
    """
    ...
