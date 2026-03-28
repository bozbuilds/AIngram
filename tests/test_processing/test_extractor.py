# tests/test_processing/test_extractor.py
from aingram.processing.extractor import GlinerExtractor
from aingram.processing.protocols import EntityExtractor
from aingram.types import ExtractedEntity


class TestMockExtractor:
    def test_satisfies_protocol(self, mock_extractor):
        assert isinstance(mock_extractor, EntityExtractor)

    def test_extracts_entities(self, mock_extractor):
        result = mock_extractor.extract(
            'Meeting with Alice at Google',
            ['person', 'organization'],
        )
        assert len(result) >= 1
        assert all(isinstance(e, ExtractedEntity) for e in result)
        names = {e.name for e in result}
        assert 'Alice' in names

    def test_deterministic(self, mock_extractor):
        r1 = mock_extractor.extract('Talk to Alice about Google', ['person'])
        r2 = mock_extractor.extract('Talk to Alice about Google', ['person'])
        assert r1 == r2

    def test_empty_text(self, mock_extractor):
        result = mock_extractor.extract('', ['person'])
        assert result == []

    def test_no_entities(self, mock_extractor):
        result = mock_extractor.extract('the quick brown fox', ['person'])
        assert result == []


class TestGlinerExtractorProtocol:
    def test_has_correct_methods(self):
        """GlinerExtractor should satisfy the EntityExtractor protocol
        (checked structurally, not by instantiation)."""
        assert hasattr(GlinerExtractor, 'extract')
