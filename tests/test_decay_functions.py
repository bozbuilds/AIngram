# tests/test_decay_functions.py — pure-function tests for decay math
"""Tests for compute_stability, compute_decay, and MINIMUM_IMPORTANCE.

These are pure functions that don't touch the database. Ported from
legacy test_consolidation/test_decay.py (TestComputeStability,
TestComputeDecay classes) to survive the v2→v3 schema migration.
"""

from aingram.consolidation.decay import (
    MINIMUM_IMPORTANCE,
    compute_decay,
    compute_stability,
)


class TestComputeStability:
    def test_zero_access_count(self):
        stability = compute_stability(access_count=0, importance=0.5)
        assert stability > 0
        # Base stability ~168 hours (1 week)
        assert 100 < stability < 300

    def test_higher_access_count_increases_stability(self):
        low = compute_stability(access_count=1, importance=0.5)
        high = compute_stability(access_count=10, importance=0.5)
        assert high > low

    def test_higher_importance_increases_stability(self):
        low = compute_stability(access_count=0, importance=0.3)
        high = compute_stability(access_count=0, importance=0.9)
        assert high > low

    def test_well_accessed_memory_has_long_stability(self):
        stability = compute_stability(access_count=10, importance=0.8)
        # Should be several weeks
        assert stability > 500  # > 20 days in hours


class TestComputeDecay:
    def test_zero_hours_no_decay(self):
        result = compute_decay(importance=0.5, hours_since_access=0, access_count=0)
        assert abs(result - 0.5) < 0.001

    def test_decay_reduces_importance(self):
        result = compute_decay(importance=0.5, hours_since_access=168, access_count=0)
        assert result < 0.5
        assert result > 0

    def test_more_hours_more_decay(self):
        week1 = compute_decay(importance=0.5, hours_since_access=168, access_count=0)
        week2 = compute_decay(importance=0.5, hours_since_access=336, access_count=0)
        assert week2 < week1

    def test_higher_access_count_slows_decay(self):
        low_access = compute_decay(importance=0.5, hours_since_access=168, access_count=0)
        high_access = compute_decay(importance=0.5, hours_since_access=168, access_count=10)
        assert high_access > low_access

    def test_never_below_minimum(self):
        result = compute_decay(importance=0.5, hours_since_access=100000, access_count=0)
        assert result >= MINIMUM_IMPORTANCE

    def test_negative_hours_treated_as_zero(self):
        result = compute_decay(importance=0.5, hours_since_access=-10, access_count=0)
        assert abs(result - 0.5) < 0.001

    def test_minimum_importance_constant(self):
        assert MINIMUM_IMPORTANCE > 0
        assert MINIMUM_IMPORTANCE < 0.1
