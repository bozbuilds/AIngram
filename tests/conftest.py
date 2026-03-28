# tests/conftest.py
import re

import pytest

from aingram.processing.protocols import Embedder  # noqa: F401 — re-exported for test use
from aingram.storage.engine import StorageEngine
from aingram.types import ExtractedEntity


@pytest.fixture
def engine(tmp_path):
    db_path = tmp_path / 'test.db'
    eng = StorageEngine(str(db_path))
    yield eng
    eng.close()


class MockEmbedder:
    """Deterministic fake embedder for testing."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        import math

        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) / max(len(text), 1)
        return [math.sin(seed * (i + 1)) for i in range(self.dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


class MockExtractor:
    """Deterministic fake entity extractor for testing.

    Finds capitalized words (2+ chars) that appear after whitespace.
    Intentionally skips text-initial words (e.g., "Meeting" at position 0)
    to avoid false positives on sentence-start capitalization.
    Test texts should be crafted with this behavior in mind.
    """

    def extract(self, text: str, entity_types: list[str]) -> list[ExtractedEntity]:
        default_type = entity_types[0] if entity_types else 'unknown'
        matches = re.findall(r'(?<=\s)[A-Z][a-z]+', text)
        seen: set[str] = set()
        result: list[ExtractedEntity] = []
        for name in matches:
            if name not in seen:
                seen.add(name)
                result.append(ExtractedEntity(name=name, entity_type=default_type, score=0.9))
        return result


@pytest.fixture
def mock_extractor():
    return MockExtractor()


class MockLLM:
    """Deterministic fake LLM for testing."""

    def __init__(self, response: str = '[]'):
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.response


@pytest.fixture
def mock_llm():
    return MockLLM()


class ClusterTestEmbedder:
    """Test embedder that maps keywords to predetermined vectors for deterministic clustering.
    Texts containing 'pool' cluster together, 'cache'/'redis' cluster together, others default."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vector_for(self, text: str) -> list[float]:
        lower = text.lower()
        if 'pool' in lower:
            base = [1.0, 0.1, 0.0]
        elif 'cache' in lower or 'redis' in lower:
            base = [0.0, 0.1, 1.0]
        else:
            base = [0.5, 0.5, 0.5]
        seed = sum(ord(c) for c in text) % 100 / 10000
        return [base[i % 3] + seed * (i + 1) / self.dim for i in range(self.dim)]

    def embed(self, text: str) -> list[float]:
        return self._vector_for(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]


def ensure_test_session(engine: StorageEngine, session_id: str = 'test-session') -> None:
    """Insert a test session into agent_sessions if it doesn't exist.
    Shared helper for tests that need FK-compliant session references."""
    engine._conn.execute(
        'INSERT OR IGNORE INTO agent_sessions '
        '(session_id, agent_name, public_key, created_at) VALUES (?, ?, ?, ?)',
        (session_id, 'test', 'ab' * 32, '2026-01-01T00:00:00+00:00'),
    )
    engine._conn.commit()
