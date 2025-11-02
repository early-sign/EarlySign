"""Helpers to cache ibis expression compilation and reuse compiled SQL."""

import hashlib
import json
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import (
    Any,
    Callable,
    Literal,
    MutableMapping,
    Optional,
    Protocol,
    Tuple,
    cast,
    runtime_checkable,
)

import ibis
import pandas as pd
from ibis.expr.types import Column, Scalar, Table


@runtime_checkable
class CompilableExpr(Protocol):
    """Subset of the ibis expression API needed for caching."""

    def compile(self) -> str:  # pragma: no cover - signature only
        ...

    def op(self) -> Any:  # pragma: no cover - signature only
        ...

    def __str__(self) -> str: ...


@runtime_checkable
class IbisLikeConnection(Protocol):
    """Protocol for ibis-like backend connections."""

    def execute(self, expr: Any) -> Any: ...

    def raw_sql(self, statement: str) -> Any:  # pragma: no cover - optional
        ...


@dataclass
class CacheEntry:
    """Cached compilation result."""

    value: str
    timestamp: float
    mode: str


def default_fingerprint(expr: Any) -> str:
    """Return a lightweight fingerprint for an ibis expression."""
    # Prefer hashing the operator graph directly – inexpensive and stable
    try:
        op_fn = getattr(expr, "op", None)
        if callable(op_fn):
            op_obj = op_fn()
            return f"ophash:{hash(op_obj)}"
    except Exception:
        pass

    try:
        op = getattr(expr, "op", None)
        if callable(op):
            op_obj = op()
            tree_repr = getattr(op_obj, "_repr", None)
            if callable(tree_repr):
                return hashlib.sha256(tree_repr().encode("utf-8")).hexdigest()
    except Exception:
        pass

    try:
        compile_fn = getattr(expr, "compile", None)
        if callable(compile_fn):
            return hashlib.sha256(compile_fn().encode("utf-8")).hexdigest()
    except Exception:
        pass

    return hashlib.sha256(str(expr).encode("utf-8")).hexdigest()


class IbisCache(AbstractContextManager[Callable[[CompilableExpr], Any]]):
    """Context-managed cache that memoises ibis compilation."""

    def __init__(
        self,
        conn: IbisLikeConnection,
        *,
        mode: str = "execute",
        cache: Optional[MutableMapping[str, CacheEntry]] = None,
        ttl: Optional[float] = None,
        max_entries: Optional[int] = None,
        fingerprint: Callable[[Any], str] = default_fingerprint,
    ) -> None:
        if mode not in ("compile", "execute"):
            raise ValueError('mode must be "compile" or "execute"')
        self._conn = conn
        self._mode = mode
        self._cache = cache if cache is not None else {}
        self._ttl = ttl
        self._max_entries = max_entries
        self._fingerprint = fingerprint

    def __enter__(self) -> Callable[[CompilableExpr], Any]:
        def cached_call(expr: CompilableExpr) -> Any:
            key = self._fingerprint(expr)
            now = time.time()
            entry = self._cache.get(key)

            sql: Optional[str]
            if entry is not None and entry.mode == self._mode:
                if self._ttl is None or now - entry.timestamp <= self._ttl:
                    sql = entry.value
                else:
                    sql = None
            else:
                sql = None

            if sql is None:
                sql = self._compile(expr)
                self._cache[key] = CacheEntry(value=sql, timestamp=now, mode=self._mode)
                self._maybe_evict()

            if self._mode == "compile":
                return sql
            return self._execute(expr, sql)

        return cached_call

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Literal[False]:
        return False

    def _compile(self, expr: CompilableExpr) -> str:
        compile_fn = getattr(expr, "compile", None)
        if callable(compile_fn):
            return cast(str, compile_fn())
        if isinstance(expr, str):
            return expr
        return str(expr)

    def _execute(self, expr: CompilableExpr, sql: str) -> Any:
        raw_sql = getattr(self._conn, "raw_sql", None)
        if callable(raw_sql):
            cursor = raw_sql(sql)
            return self._cursor_to_result(cursor, expr)
        return self._conn.execute(sql)

    def _cursor_to_result(self, cursor: Any, expr: CompilableExpr) -> Any:
        if ibis is not None:
            if isinstance(expr, Table):
                if hasattr(cursor, "df"):
                    df = cursor.df()
                    return self._decode_dataframe(df)
                rows = cursor.fetchall()
                return self._decode_sequence(rows)
            if isinstance(expr, Column):
                if hasattr(cursor, "df"):
                    df = cursor.df()
                    series = df.iloc[:, 0]
                    return self._decode_series(series)
                rows = cursor.fetchall()
                return self._decode_series(
                    pd.Series(
                        [row[0] if isinstance(row, tuple) else row for row in rows]
                    )
                )
            if isinstance(expr, Scalar):
                row = cursor.fetchone()
                if row is None:
                    return None
                value = row[0] if isinstance(row, tuple) else row
                return self._maybe_decode_json(value)
        rows = cursor.fetchall()
        return self._decode_sequence(rows)

    def _maybe_evict(self) -> None:
        if self._max_entries is None:
            return
        excess = len(self._cache) - self._max_entries
        if excess <= 0:
            return
        items: Tuple[Tuple[str, CacheEntry], ...] = tuple(self._cache.items())
        for key, _ in sorted(items, key=lambda item: item[1].timestamp)[:excess]:
            self._cache.pop(key, None)

    def _decode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in result.columns:
            result[column] = self._decode_series(result[column])
        return result

    def _decode_series(self, series: pd.Series) -> pd.Series:
        if series.dtype != object:
            return series

        return series.map(self._maybe_decode_json)

    def _decode_sequence(self, rows: Any) -> Any:
        try:
            return [self._decode_row(row) for row in rows]
        except TypeError:
            return rows

    def _decode_row(self, row: Any) -> Any:
        if isinstance(row, tuple):
            return tuple(self._maybe_decode_json(value) for value in row)
        return self._maybe_decode_json(row)

    def _maybe_decode_json(self, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return value
        return value
