"""SonnetExtractor tests — all API calls mocked."""

from unittest.mock import MagicMock

import pytest

from aingram.types import ExtractionResult


@pytest.fixture
def mock_anthropic():
    """Mock the anthropic module."""
    mock_client = MagicMock()
    # Simulate a tool-use response
    tool_input = {
        'entry_type': 'hypothesis',
        'confidence': 0.85,
        'relevance': 0.9,
        'entities': [{'name': 'connection pool', 'type': 'component'}],
        'relationships': [
            {
                'source': 'connection pool',
                'target': 'API gateway',
                'type': 'causes_issue',
                'fact': 'pool exhaustion causes latency',
            }
        ],
    }
    tool_block = MagicMock()
    tool_block.type = 'tool_use'
    tool_block.input = tool_input

    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = 'tool_use'
    mock_client.messages.create.return_value = response
    return mock_client


def test_sonnet_extract_basic(mock_anthropic):
    from aingram.extraction.sonnet import SonnetExtractor

    ext = SonnetExtractor(client=mock_anthropic, model='claude-sonnet-4-20250514')
    result = ext.extract('The connection pool is exhausted causing latency spikes')
    assert isinstance(result, ExtractionResult)
    assert result.entry_type == 'hypothesis'
    assert result.confidence == 0.85
    assert len(result.entities) == 1
    assert result.entities[0].name == 'connection pool'
    assert len(result.relationships) == 1


def test_sonnet_extract_no_entities(mock_anthropic):
    """When API returns no entities/relationships."""
    tool_input = {
        'entry_type': 'observation',
        'confidence': 0.7,
        'relevance': 0.6,
    }
    tool_block = MagicMock()
    tool_block.type = 'tool_use'
    tool_block.input = tool_input
    mock_anthropic.messages.create.return_value.content = [tool_block]

    from aingram.extraction.sonnet import SonnetExtractor

    ext = SonnetExtractor(client=mock_anthropic)
    result = ext.extract('Simple observation text')
    assert result.entities == []
    assert result.relationships == []


def test_sonnet_extract_handles_text_response(mock_anthropic):
    """When API returns text instead of tool_use (edge case)."""
    text_block = MagicMock()
    text_block.type = 'text'
    text_block.text = 'I cannot extract metadata from this.'
    mock_anthropic.messages.create.return_value.content = [text_block]
    mock_anthropic.messages.create.return_value.stop_reason = 'end_turn'

    from aingram.extraction.sonnet import SonnetExtractor

    ext = SonnetExtractor(client=mock_anthropic)
    result = ext.extract('Ambiguous text')
    # Should return a safe default
    assert result.entry_type == 'observation'
    assert result.confidence == 0.5


def test_sonnet_extract_validates_entry_type(mock_anthropic):
    """Invalid entry_type from API should fall back to observation."""
    tool_input = {
        'entry_type': 'unknown_type',
        'confidence': 0.8,
        'relevance': 0.7,
    }
    tool_block = MagicMock()
    tool_block.type = 'tool_use'
    tool_block.input = tool_input
    mock_anthropic.messages.create.return_value.content = [tool_block]

    from aingram.extraction.sonnet import SonnetExtractor

    ext = SonnetExtractor(client=mock_anthropic)
    result = ext.extract('Some text')
    assert result.entry_type == 'observation'  # fallback


def test_sonnet_extract_clamps_confidence(mock_anthropic):
    """Out-of-range confidence should be clamped."""
    tool_input = {
        'entry_type': 'result',
        'confidence': 1.5,
        'relevance': -0.1,
    }
    tool_block = MagicMock()
    tool_block.type = 'tool_use'
    tool_block.input = tool_input
    mock_anthropic.messages.create.return_value.content = [tool_block]

    from aingram.extraction.sonnet import SonnetExtractor

    ext = SonnetExtractor(client=mock_anthropic)
    result = ext.extract('Some result')
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.relevance <= 1.0
