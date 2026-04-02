"""QJL 1-bit vector compression — pure math tests."""

import numpy as np
import pytest


class TestCreateProjection:
    def test_correct_shape(self):
        from aingram.processing.qjl import create_projection

        s = create_projection(dim=64, num_projections=64, seed=42)
        assert s.shape == (64, 64)
        assert s.dtype == np.float32

    def test_orthonormal_rows(self):
        from aingram.processing.qjl import create_projection

        s = create_projection(dim=64, num_projections=64, seed=42)
        product = s @ s.T
        np.testing.assert_allclose(product, np.eye(64), atol=1e-5)

    def test_deterministic_from_seed(self):
        from aingram.processing.qjl import create_projection

        s1 = create_projection(dim=64, num_projections=64, seed=42)
        s2 = create_projection(dim=64, num_projections=64, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_different_matrix(self):
        from aingram.processing.qjl import create_projection

        s1 = create_projection(dim=64, num_projections=64, seed=42)
        s2 = create_projection(dim=64, num_projections=64, seed=99)
        assert not np.array_equal(s1, s2)

    def test_fewer_projections_than_dim(self):
        from aingram.processing.qjl import create_projection

        s = create_projection(dim=64, num_projections=32, seed=42)
        assert s.shape == (32, 64)
        product = s @ s.T
        np.testing.assert_allclose(product, np.eye(32), atol=1e-5)

    def test_more_projections_than_dim(self):
        from aingram.processing.qjl import create_projection

        s = create_projection(dim=32, num_projections=64, seed=42)
        assert s.shape == (64, 32)
        top = s[:32]
        product = top @ top.T
        np.testing.assert_allclose(product, np.eye(32), atol=1e-5)

    def test_768_dim_default(self):
        from aingram.processing.qjl import create_projection

        s = create_projection(dim=768, num_projections=768, seed=42)
        assert s.shape == (768, 768)


class TestEncode:
    def test_returns_bytes_and_norm(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=8, num_projections=8, seed=42)
        vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        packed, norm = encode(vec, s)
        assert isinstance(packed, bytes)
        assert isinstance(norm, float)

    def test_packed_size(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=64, num_projections=64, seed=42)
        vec = np.random.RandomState(7).randn(64).tolist()
        packed, _ = encode(vec, s)
        assert len(packed) == 8

    def test_768_dim_produces_96_bytes(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=768, num_projections=768, seed=42)
        vec = np.random.RandomState(8).randn(768).tolist()
        packed, _ = encode(vec, s)
        assert len(packed) == 96

    def test_norm_matches_numpy(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=16, num_projections=16, seed=42)
        vec = [float(i) for i in range(16)]
        _, norm = encode(vec, s)
        expected = float(np.linalg.norm(vec))
        assert abs(norm - expected) < 1e-4

    def test_all_positive_vector(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=8, num_projections=8, seed=42)
        vec = [1.0] * 8
        packed, norm = encode(vec, s)
        assert len(packed) == 1
        assert norm > 0

    def test_all_negative_vector(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=8, num_projections=8, seed=42)
        vec = [-1.0] * 8
        packed, _ = encode(vec, s)
        assert len(packed) == 1

    def test_zero_vector(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=8, num_projections=8, seed=42)
        vec = [0.0] * 8
        _, norm = encode(vec, s)
        assert norm == 0.0

    def test_single_dimension(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=1, num_projections=1, seed=42)
        packed, norm = encode([3.0], s)
        assert len(packed) == 1
        assert abs(norm - 3.0) < 1e-4

    def test_very_large_vector(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=64, num_projections=64, seed=42)
        vec = [1e6] * 64
        packed, norm = encode(vec, s)
        assert len(packed) == 8
        assert norm > 0

    def test_near_duplicate_distinguishable(self):
        from aingram.processing.qjl import create_projection, encode

        s = create_projection(dim=768, num_projections=768, seed=42)
        v1 = np.random.RandomState(1).randn(768).tolist()
        v2 = [x + 1e-3 for x in v1]
        p1, _ = encode(v1, s)
        p2, _ = encode(v2, s)
        bits1 = np.unpackbits(np.frombuffer(p1, dtype=np.uint8))[:768]
        bits2 = np.unpackbits(np.frombuffer(p2, dtype=np.uint8))[:768]
        hamming = int(np.sum(bits1 != bits2))
        assert hamming < 768 * 0.1


class TestEncodeBatch:
    def test_matches_individual_encoding(self):
        from aingram.processing.qjl import create_projection, encode, encode_batch

        s = create_projection(dim=16, num_projections=16, seed=42)
        rng = np.random.RandomState(9)
        vecs = [rng.randn(16).tolist() for _ in range(5)]
        batch_results = encode_batch(vecs, s)

        for i, vec in enumerate(vecs):
            individual_packed, individual_norm = encode(vec, s)
            batch_packed, batch_norm = batch_results[i]
            assert individual_packed == batch_packed
            assert abs(individual_norm - batch_norm) < 1e-5

    def test_empty_batch(self):
        from aingram.processing.qjl import create_projection, encode_batch

        s = create_projection(dim=8, num_projections=8, seed=42)
        assert encode_batch([], s) == []


class TestAsymmetricScore:
    def test_unbiased_estimator(self):
        from aingram.processing.qjl import asymmetric_score, create_projection, encode

        dim = 64
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)
        true_ip = float(np.dot(q, k))

        estimates = []
        for seed in range(100):
            s = create_projection(dim=dim, num_projections=dim, seed=seed)
            packed, norm = encode(k.tolist(), s)
            q_proj = s @ q
            est = asymmetric_score(q_proj, packed, norm, dim)
            estimates.append(est)

        mean_est = np.mean(estimates)
        assert abs(mean_est - true_ip) < max(abs(true_ip), 1.0) * 2.0

    def test_identical_vectors_positive_score(self):
        from aingram.processing.qjl import asymmetric_score, create_projection, encode

        s = create_projection(dim=16, num_projections=16, seed=42)
        vec = np.random.randn(16).astype(np.float32)
        packed, norm = encode(vec.tolist(), s)
        q_proj = s @ vec
        score = asymmetric_score(q_proj, packed, norm, 16)
        assert score > 0


class TestRankingFidelity:
    def test_hamming_ranking_correlates_with_cosine(self):
        from aingram.processing.qjl import create_projection, encode

        dim = 64
        s = create_projection(dim=dim, num_projections=dim, seed=42)

        rng = np.random.RandomState(123)
        query = rng.randn(dim).astype(np.float32)
        query /= np.linalg.norm(query)
        db_vecs = [rng.randn(dim).astype(np.float32) for _ in range(50)]
        for i in range(len(db_vecs)):
            db_vecs[i] /= np.linalg.norm(db_vecs[i])

        true_scores = [float(np.dot(query, v)) for v in db_vecs]
        true_top10 = set(sorted(range(50), key=lambda i: -true_scores[i])[:10])

        query_packed, _ = encode(query.tolist(), s)
        query_bits = np.unpackbits(np.frombuffer(query_packed, dtype=np.uint8))[:dim]
        hamming_dists = []
        for v in db_vecs:
            packed, _ = encode(v.tolist(), s)
            bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:dim]
            dist = int(np.sum(query_bits != bits))
            hamming_dists.append(dist)
        hamming_top10 = set(sorted(range(50), key=lambda i: hamming_dists[i])[:10])

        overlap = len(true_top10 & hamming_top10)
        assert overlap >= 5, f'Recall@10 too low: {overlap}/10'
