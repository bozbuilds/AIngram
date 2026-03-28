import pytest

from aingram.types import ExtractedRelationship, ExtractionResult


def test_extraction_result_defaults():
    r = ExtractionResult(entry_type='observation', confidence=0.8, relevance=0.9)
    assert r.entities == []
    assert r.relationships == []


def test_extraction_result_with_entities():
    from aingram.types import ExtractedEntity

    r = ExtractionResult(
        entry_type='hypothesis',
        confidence=0.9,
        relevance=0.85,
        entities=[ExtractedEntity(name='pool', entity_type='component', score=0.95)],
        relationships=[
            ExtractedRelationship(
                source='pool',
                target='gateway',
                relation_type='causes_issue',
                fact='pool exhaustion causes latency',
            )
        ],
    )
    assert len(r.entities) == 1
    assert len(r.relationships) == 1
    assert r.relationships[0].fact == 'pool exhaustion causes latency'


def test_extracted_relationship_defaults():
    r = ExtractedRelationship(source='a', target='b', relation_type='related')
    assert r.fact is None


def test_extraction_result_validates_confidence():
    with pytest.raises(ValueError, match='confidence'):
        ExtractionResult(entry_type='observation', confidence=1.5, relevance=0.5)


def test_extraction_result_validates_relevance():
    with pytest.raises(ValueError, match='relevance'):
        ExtractionResult(entry_type='observation', confidence=0.5, relevance=-0.1)


def test_memory_extractor_protocol():
    from aingram.extraction.protocol import MemoryExtractor

    class FakeExtractor:
        def extract(self, text: str) -> ExtractionResult:
            return ExtractionResult(entry_type='observation', confidence=0.5, relevance=0.5)

    assert isinstance(FakeExtractor(), MemoryExtractor)
