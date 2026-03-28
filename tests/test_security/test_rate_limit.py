"""Per-session token bucket rate limiter tests."""

import time

import pytest


class TestTokenBucket:
    def test_allows_within_capacity(self):
        from aingram.security.rate_limit import RateLimiter

        limiter = RateLimiter()
        limiter.check('session-1', 'remember')

    def test_exhaustion_raises(self):
        from aingram.exceptions import RateLimitError
        from aingram.security.rate_limit import RateLimiter

        limiter = RateLimiter(write_capacity=3, write_rate=0.05)
        limiter.check('s', 'remember')
        limiter.check('s', 'remember')
        limiter.check('s', 'remember')
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check('s', 'remember')
        assert exc_info.value.retry_after_seconds > 0

    def test_separate_sessions_independent(self):
        from aingram.security.rate_limit import RateLimiter

        limiter = RateLimiter(write_capacity=1, write_rate=0.01)
        limiter.check('session-a', 'remember')
        limiter.check('session-b', 'remember')

    def test_read_and_write_separate_buckets(self):
        from aingram.security.rate_limit import RateLimiter

        limiter = RateLimiter(write_capacity=1, write_rate=0.01, read_capacity=100, read_rate=10.0)
        limiter.check('s', 'remember')
        limiter.check('s', 'recall')

    def test_admin_exempt(self):
        from aingram.security.rate_limit import RateLimiter

        limiter = RateLimiter(write_capacity=1, write_rate=0.01)
        limiter.check('s', 'remember')
        limiter.check('s', 'remember', is_admin=True)

    def test_refill_over_time(self):
        from aingram.security.rate_limit import RateLimiter

        limiter = RateLimiter(write_capacity=1, write_rate=100.0)
        limiter.check('s', 'remember')
        time.sleep(0.02)
        limiter.check('s', 'remember')
