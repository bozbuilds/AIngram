# tests/test_consolidation/test_contradiction.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aingram.config import AIngramConfig, load_merged_config
from aingram.processing.protocols import ContradictionClassifier
from aingram.storage.engine import StorageEngine
from aingram.types import AgentSession, ContradictionVerdict
from tests.conftest import MockLLM


class TestContradictionVerdict:
    def test_basic_construction(self):
        v = ContradictionVerdict(contradicts=True, confidence=0.85)
        assert v.contradicts is True
        assert v.confidence == 0.85
        assert v.superseded_index is None

    def test_with_superseded_index(self):
        v = ContradictionVerdict(contradicts=True, confidence=1.0, superseded_index=0)
        assert v.superseded_index == 0

    def test_no_contradiction(self):
        v = ContradictionVerdict(contradicts=False, confidence=0.2)
        assert v.contradicts is False


class TestContradictionClassifierProtocol:
    def test_mock_satisfies_protocol(self):
        class MockClassifier:
            def classify(self, text_a: str, text_b: str) -> ContradictionVerdict:
                return ContradictionVerdict(contradicts=False, confidence=0.1)

        assert isinstance(MockClassifier(), ContradictionClassifier)

    def test_non_conforming_rejects(self):
        class BadClassifier:
            def judge(self, a: str, b: str) -> bool:
                return False

        assert not isinstance(BadClassifier(), ContradictionClassifier)


class TestLLMContradictionClassifier:
    def test_satisfies_protocol(self):
        from aingram.consolidation.contradiction import LLMContradictionClassifier

        classifier = LLMContradictionClassifier(MockLLM('{"contradicts": false}'))
        assert isinstance(classifier, ContradictionClassifier)

    def test_no_contradiction(self):
        from aingram.consolidation.contradiction import LLMContradictionClassifier

        llm = MockLLM('{"contradicts": false}')
        classifier = LLMContradictionClassifier(llm)
        verdict = classifier.classify('sky is blue', 'grass is green')
        assert verdict.contradicts is False
        assert len(llm.calls) == 1

    def test_contradiction_with_superseded_index(self):
        from aingram.consolidation.contradiction import LLMContradictionClassifier

        response = json.dumps({'contradicts': True, 'superseded_index': 0})
        llm = MockLLM(response)
        classifier = LLMContradictionClassifier(llm)
        verdict = classifier.classify('Python 3.9 is latest', 'Python 3.12 is latest')
        assert verdict.contradicts is True
        assert verdict.superseded_index == 0
        assert verdict.confidence == 1.0

    def test_invalid_json_returns_no_contradiction(self):
        from aingram.consolidation.contradiction import LLMContradictionClassifier

        llm = MockLLM('not valid json')
        classifier = LLMContradictionClassifier(llm)
        verdict = classifier.classify('a', 'b')
        assert verdict.contradicts is False

    def test_missing_contradicts_field_returns_no_contradiction(self):
        from aingram.consolidation.contradiction import LLMContradictionClassifier

        llm = MockLLM('{"answer": true}')
        classifier = LLMContradictionClassifier(llm)
        verdict = classifier.classify('a', 'b')
        assert verdict.contradicts is False


@pytest.fixture
def engine_with_entity_pair(tmp_path):
    """Engine with two entries linked to the same entity — minimal contradiction scenario."""
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    session = AgentSession(
        session_id='s1',
        agent_name='test',
        public_key='a' * 64,
        created_at='2026-01-01T00:00:00+00:00',
    )
    eng.store_session(session)
    eng.store_entry(
        entry_id='older',
        content_hash='ch_old',
        entry_type='observation',
        content='{"text":"Python 3.9 is the latest version"}',
        session_id='s1',
        sequence_num=1,
        prev_entry_id=None,
        signature='sig_old',
        created_at='2025-01-01T00:00:00+00:00',
        embedding=[0.1] * 768,
    )
    eng.store_entry(
        entry_id='newer',
        content_hash='ch_new',
        entry_type='observation',
        content='{"text":"Python 3.12 is the latest version"}',
        session_id='s1',
        sequence_num=2,
        prev_entry_id='older',
        signature='sig_new',
        created_at='2025-06-01T00:00:00+00:00',
        embedding=[0.1] * 768,
    )
    eid = eng.upsert_entity(name='python', entity_type='technology')
    eng.link_entity_to_mention(eid, 'older')
    eng.link_entity_to_mention(eid, 'newer')
    yield eng
    eng.close()


class MockContradictionClassifier:
    """Configurable mock for testing the orchestrator."""

    def __init__(self, verdict: ContradictionVerdict):
        self._verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def classify(self, text_a: str, text_b: str) -> ContradictionVerdict:
        self.calls.append((text_a, text_b))
        return self._verdict


class TestContradictionDetectorOrchestration:
    def test_no_op_without_classifier(self, engine_with_entity_pair):
        from aingram.consolidation.contradiction import ContradictionDetector

        detector = ContradictionDetector(engine_with_entity_pair)
        result = detector.detect_and_resolve()
        assert result.contradictions_found == 0

    def test_delegates_to_classifier(self, engine_with_entity_pair):
        from aingram.consolidation.contradiction import ContradictionDetector

        classifier = MockContradictionClassifier(
            ContradictionVerdict(contradicts=False, confidence=0.3)
        )
        detector = ContradictionDetector(engine_with_entity_pair, classifier=classifier)
        detector.detect_and_resolve()
        assert len(classifier.calls) == 1

    def test_recency_fallback_when_superseded_index_none(self, engine_with_entity_pair):
        from aingram.consolidation.contradiction import ContradictionDetector

        classifier = MockContradictionClassifier(
            ContradictionVerdict(contradicts=True, confidence=0.9, superseded_index=None)
        )
        detector = ContradictionDetector(engine_with_entity_pair, classifier=classifier)
        result = detector.detect_and_resolve()
        assert result.contradictions_found == 1
        assert result.contradictions_resolved == 1
        older = engine_with_entity_pair.get_entry('older')
        newer = engine_with_entity_pair.get_entry('newer')
        assert older.importance < 0.5
        assert newer.importance == 0.5

    def test_uses_superseded_index_when_provided(self, engine_with_entity_pair):
        from aingram.consolidation.contradiction import ContradictionDetector

        classifier = MockContradictionClassifier(
            ContradictionVerdict(contradicts=True, confidence=1.0, superseded_index=1)
        )
        detector = ContradictionDetector(engine_with_entity_pair, classifier=classifier)
        result = detector.detect_and_resolve()
        assert result.contradictions_found == 1


class TestDeBERTaContradictionClassifier:
    def test_satisfies_protocol(self):
        from aingram.consolidation.deberta import DeBERTaContradictionClassifier

        classifier = DeBERTaContradictionClassifier()
        assert isinstance(classifier, ContradictionClassifier)

    def test_lazy_loading_not_called_on_init(self):
        from aingram.consolidation.deberta import DeBERTaContradictionClassifier

        classifier = DeBERTaContradictionClassifier()
        assert classifier._session is None
        assert classifier._tokenizer is None

    def test_contradiction_above_threshold(self):
        from aingram.consolidation.deberta import DeBERTaContradictionClassifier

        classifier = DeBERTaContradictionClassifier(threshold=0.7)

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.2, 3.0]])]

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [0, 1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1, 1]
        mock_encoding.type_ids = [0, 0, 1, 1]
        mock_tokenizer.encode.return_value = mock_encoding

        classifier._session = mock_session
        classifier._tokenizer = mock_tokenizer

        verdict = classifier.classify('sky is blue', 'sky is red')
        assert verdict.contradicts is True
        assert verdict.confidence > 0.7
        assert verdict.superseded_index is None

    def test_no_contradiction_below_threshold(self):
        from aingram.consolidation.deberta import DeBERTaContradictionClassifier

        classifier = DeBERTaContradictionClassifier(threshold=0.7)

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[3.0, 0.2, 0.1]])]

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [0, 1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1, 1]
        mock_encoding.type_ids = [0, 0, 1, 1]
        mock_tokenizer.encode.return_value = mock_encoding

        classifier._session = mock_session
        classifier._tokenizer = mock_tokenizer

        verdict = classifier.classify('sky is blue', 'sky is blue')
        assert verdict.contradicts is False
        assert verdict.confidence < 0.7
        assert verdict.superseded_index is None

    def test_ensure_loaded_raises_model_not_found_on_failure(self):
        from aingram.consolidation.deberta import DeBERTaContradictionClassifier
        from aingram.exceptions import ModelNotFoundError

        classifier = DeBERTaContradictionClassifier()

        with patch('huggingface_hub.hf_hub_download', side_effect=OSError('network')):
            with pytest.raises(ModelNotFoundError, match='DeBERTa NLI model'):
                classifier.classify('a', 'b')


class TestContradictionConfig:
    def test_default_values(self):
        config = AIngramConfig()
        assert config.contradiction_backend == 'none'
        assert config.contradiction_threshold == 0.7

    def test_valid_backends(self):
        for backend in ('deberta', 'llm', 'none'):
            config = AIngramConfig(contradiction_backend=backend)
            assert config.contradiction_backend == backend

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match='contradiction_backend'):
            AIngramConfig(contradiction_backend='invalid')

    def test_env_var_override(self):
        env = {
            'AINGRAM_CONTRADICTION_BACKEND': 'deberta',
            'AINGRAM_CONTRADICTION_THRESHOLD': '0.85',
        }
        config = load_merged_config(env=env)
        assert config.contradiction_backend == 'deberta'
        assert config.contradiction_threshold == 0.85

    def test_threshold_coercion_from_string(self):
        env = {'AINGRAM_CONTRADICTION_THRESHOLD': '0.6'}
        config = load_merged_config(env=env)
        assert config.contradiction_threshold == 0.6
        assert isinstance(config.contradiction_threshold, float)


class TestBuildContradictionClassifier:
    def test_none_backend(self):
        config = AIngramConfig(contradiction_backend='none')
        from aingram.store import _build_contradiction_classifier

        assert _build_contradiction_classifier(config) is None

    def test_llm_backend(self):
        config = AIngramConfig(contradiction_backend='llm')
        from aingram.consolidation.contradiction import LLMContradictionClassifier
        from aingram.store import _build_contradiction_classifier

        with patch('aingram.processing.llm.OllamaLLM'):
            result = _build_contradiction_classifier(config)
        assert isinstance(result, LLMContradictionClassifier)

    def test_deberta_backend(self):
        config = AIngramConfig(contradiction_backend='deberta', contradiction_threshold=0.8)
        from aingram.consolidation.deberta import DeBERTaContradictionClassifier
        from aingram.store import _build_contradiction_classifier

        result = _build_contradiction_classifier(config)
        assert isinstance(result, DeBERTaContradictionClassifier)
        assert result._threshold == 0.8
