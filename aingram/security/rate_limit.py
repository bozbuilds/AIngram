"""Per-session token bucket rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aingram.exceptions import RateLimitError

_WRITE_TOOLS = frozenset({'remember', 'reference', 'review_memory', 'sync'})
_READ_TOOLS = frozenset(
    {
        'recall',
        'get_related',
        'get_experiment_context',
        'verify',
        'get_due_reviews',
        'get_surprise',
    }
)

_DEFAULT_WRITE_CAPACITY = 100
_DEFAULT_WRITE_RATE = 100 / 60
_DEFAULT_READ_CAPACITY = 300
_DEFAULT_READ_RATE = 300 / 60
_EVICTION_INTERVAL = 100  # prune stale buckets every N checks
_BUCKET_TTL = 600  # evict buckets idle for 10 minutes


@dataclass
class _Bucket:
    capacity: float
    rate: float
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)

    def try_consume(self) -> float | None:
        """Try to consume 1 token. Returns None on success, retry_after on failure."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return None
        return (1.0 - self.tokens) / self.rate if self.rate > 0 else 60.0


class RateLimiter:
    """Per-session token bucket rate limiter for MCP tools."""

    def __init__(
        self,
        *,
        write_capacity: float = _DEFAULT_WRITE_CAPACITY,
        write_rate: float = _DEFAULT_WRITE_RATE,
        read_capacity: float = _DEFAULT_READ_CAPACITY,
        read_rate: float = _DEFAULT_READ_RATE,
    ) -> None:
        self._write_capacity = write_capacity
        self._write_rate = write_rate
        self._read_capacity = read_capacity
        self._read_rate = read_rate
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._check_count = 0

    def check(self, session_id: str, tool_name: str, *, is_admin: bool = False) -> None:
        if is_admin:
            return

        self._check_count += 1
        if self._check_count % _EVICTION_INTERVAL == 0:
            self._evict_stale()

        if tool_name in _WRITE_TOOLS:
            bucket_type = 'write'
            capacity, rate = self._write_capacity, self._write_rate
        elif tool_name in _READ_TOOLS:
            bucket_type = 'read'
            capacity, rate = self._read_capacity, self._read_rate
        else:
            return

        key = (session_id, bucket_type)
        if key not in self._buckets:
            self._buckets[key] = _Bucket(
                capacity=capacity,
                rate=rate,
                tokens=capacity,
            )

        retry_after = self._buckets[key].try_consume()
        if retry_after is not None:
            raise RateLimitError(retry_after_seconds=retry_after)

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale = [k for k, b in self._buckets.items() if now - b.last_refill > _BUCKET_TTL]
        for k in stale:
            del self._buckets[k]
