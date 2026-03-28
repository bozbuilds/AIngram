# tests/test_rrf.py — Reciprocal Rank Fusion algorithm tests
"""Tests for the reciprocal_rank_fusion utility.

This function is used by the 4-way retrieval pipeline. Ported from
legacy test_storage/test_queries.py coverage gap.
"""

from aingram.storage.queries import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self):
        scores = reciprocal_rank_fusion([['a', 'b', 'c']])
        assert 'a' in scores
        assert 'b' in scores
        assert 'c' in scores
        # First rank should have highest score
        assert scores['a'] > scores['b'] > scores['c']

    def test_two_lists_boost_overlap(self):
        scores = reciprocal_rank_fusion(
            [
                ['a', 'b', 'c'],
                ['b', 'a', 'd'],
            ]
        )
        # 'a' and 'b' appear in both lists, should score higher than 'c' or 'd'
        assert scores['a'] > scores['c']
        assert scores['b'] > scores['d']

    def test_empty_lists(self):
        scores = reciprocal_rank_fusion([[], []])
        assert scores == {}

    def test_single_item(self):
        scores = reciprocal_rank_fusion([['only']])
        assert len(scores) == 1
        assert scores['only'] > 0

    def test_custom_k(self):
        scores_low_k = reciprocal_rank_fusion([['a', 'b']], k=1)
        scores_high_k = reciprocal_rank_fusion([['a', 'b']], k=100)
        # Higher k spreads scores more evenly (smaller gap between ranks)
        gap_low = scores_low_k['a'] - scores_low_k['b']
        gap_high = scores_high_k['a'] - scores_high_k['b']
        assert gap_low > gap_high

    def test_no_lists(self):
        scores = reciprocal_rank_fusion([])
        assert scores == {}

    def test_three_way_fusion(self):
        scores = reciprocal_rank_fusion(
            [
                ['a', 'b'],
                ['c', 'a'],
                ['a', 'c'],
            ]
        )
        # 'a' appears in all three lists — highest score
        assert scores['a'] > scores['b']
        assert scores['a'] > scores['c']
