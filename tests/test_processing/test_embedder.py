# tests/test_processing/test_embedder.py
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from aingram.processing.embedder import NomicEmbedder
from aingram.processing.protocols import Embedder


class TestNomicEmbedderProtocol:
    def test_satisfies_protocol(self, tmp_path):
        """NomicEmbedder should satisfy the Embedder protocol
        (checked structurally, not by instantiation)."""
        assert hasattr(NomicEmbedder, 'embed')
        assert hasattr(NomicEmbedder, 'embed_batch')


class TestMockEmbedder:
    def test_satisfies_protocol(self, mock_embedder):
        assert isinstance(mock_embedder, Embedder)

    def test_embed_returns_correct_dim(self, mock_embedder):
        result = mock_embedder.embed('hello world')
        assert len(result) == 768

    def test_embed_deterministic(self, mock_embedder):
        r1 = mock_embedder.embed('test')
        r2 = mock_embedder.embed('test')
        assert r1 == r2

    def test_embed_different_texts_differ(self, mock_embedder):
        r1 = mock_embedder.embed('hello')
        r2 = mock_embedder.embed('goodbye')
        assert r1 != r2

    def test_embed_batch(self, mock_embedder):
        results = mock_embedder.embed_batch(['hello', 'world'])
        assert len(results) == 2
        assert len(results[0]) == 768


class TestNomicEmbedderMatryoshkaNorm:
    """Truncated embedding must be L2-unit so it matches sqlite-vec / compact() storage."""

    def test_truncated_embedding_is_unit_norm(self):
        seq_len, hidden = 4, 768
        fake_hidden = np.ones((seq_len, hidden), dtype=np.float32)
        enc = Mock()
        enc.ids = [1, 2, 3, 4]
        tok = Mock()
        tok.encode = Mock(return_value=enc)

        sess = Mock()
        sess.run = Mock(return_value=[fake_hidden])
        sess.get_inputs = Mock(
            return_value=[
                SimpleNamespace(name='input_ids'),
                SimpleNamespace(name='attention_mask'),
                SimpleNamespace(name='token_type_ids'),
            ]
        )

        e = NomicEmbedder(dim=256)
        e._ensure_loaded = lambda: None
        e._tokenizer = tok
        e._session = sess

        out = e.embed('x')
        assert len(out) == 256
        n = float(np.linalg.norm(np.array(out, dtype=np.float32)))
        assert abs(n - 1.0) < 1e-4
