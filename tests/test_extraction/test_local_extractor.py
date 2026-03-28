"""LocalExtractor tests — Ollama-backed extraction with JSON mode."""

import json
from unittest.mock import MagicMock, patch


class TestLocalExtractor:
    def _make_ollama_response(self, data: dict) -> MagicMock:
        """Create a mock httpx response with JSON data."""
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {'response': json.dumps(data)}
        mock.raise_for_status = MagicMock()
        return mock

    def test_extract_valid_response(self):
        from aingram.extraction.local import LocalExtractor

        response_data = {
            'entry_type': 'observation',
            'confidence': 0.8,
            'relevance': 0.7,
            'entities': [{'name': 'pool', 'type': 'component'}],
            'relationships': [
                {'source': 'pool', 'target': 'gateway', 'type': 'uses', 'fact': 'pool uses gw'}
            ],
        }
        mock_response = self._make_ollama_response(response_data)

        with patch('httpx.post', return_value=mock_response) as mock_post:
            extractor = LocalExtractor(model='test-model')
            result = extractor.extract('Connection pool causes latency')

        assert result.entry_type == 'observation'
        assert result.confidence == 0.8
        assert result.relevance == 0.7
        assert len(result.entities) == 1
        assert result.entities[0].name == 'pool'
        assert result.entities[0].entity_type == 'component'
        assert len(result.relationships) == 1
        assert result.relationships[0].source == 'pool'

        # Verify Ollama was called with JSON format
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get('json') or call_kwargs[1].get('json')
        assert payload['format'] == 'json'
        assert payload['stream'] is False

    def test_extract_invalid_entry_type_defaults(self):
        from aingram.extraction.local import LocalExtractor

        response_data = {
            'entry_type': 'bogus_type',
            'confidence': 0.5,
            'relevance': 0.5,
        }
        mock_response = self._make_ollama_response(response_data)

        with patch('httpx.post', return_value=mock_response):
            extractor = LocalExtractor(model='test')
            result = extractor.extract('test')

        assert result.entry_type == 'observation'

    def test_extract_clamps_confidence(self):
        from aingram.extraction.local import LocalExtractor

        response_data = {
            'entry_type': 'observation',
            'confidence': 5.0,
            'relevance': -1.0,
        }
        mock_response = self._make_ollama_response(response_data)

        with patch('httpx.post', return_value=mock_response):
            extractor = LocalExtractor(model='test')
            result = extractor.extract('test')

        assert result.confidence == 1.0
        assert result.relevance == 0.0

    def test_extract_malformed_json_returns_default(self):
        from aingram.extraction.local import LocalExtractor

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'not valid json{{{'}
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response):
            extractor = LocalExtractor(model='test')
            result = extractor.extract('test')

        assert result.entry_type == 'observation'
        assert result.confidence == 0.5

    def test_extract_http_error_returns_default(self):
        from aingram.extraction.local import LocalExtractor

        with patch('httpx.post', side_effect=Exception('connection refused')):
            extractor = LocalExtractor(model='test')
            result = extractor.extract('test')

        assert result.entry_type == 'observation'
        assert result.confidence == 0.5

    def test_extract_empty_entities_ok(self):
        from aingram.extraction.local import LocalExtractor

        response_data = {
            'entry_type': 'method',
            'confidence': 0.9,
            'relevance': 0.8,
            'entities': [],
            'relationships': [],
        }
        mock_response = self._make_ollama_response(response_data)

        with patch('httpx.post', return_value=mock_response):
            extractor = LocalExtractor(model='test')
            result = extractor.extract('test')

        assert result.entry_type == 'method'
        assert result.entities == []

    def test_implements_memory_extractor_protocol(self):
        from aingram.extraction.local import LocalExtractor
        from aingram.extraction.protocol import MemoryExtractor

        assert isinstance(LocalExtractor(model='test'), MemoryExtractor)

    def test_uses_system_prompt(self):
        from aingram.extraction.local import LocalExtractor

        response_data = {
            'entry_type': 'observation',
            'confidence': 0.5,
            'relevance': 0.5,
        }
        mock_response = self._make_ollama_response(response_data)

        with patch('httpx.post', return_value=mock_response) as mock_post:
            extractor = LocalExtractor(model='test')
            extractor.extract('test text')

        payload = mock_post.call_args.kwargs.get('json') or mock_post.call_args[1].get('json')
        assert 'system' in payload
        assert len(payload['system']) > 20  # has a real system prompt

    def test_custom_base_url(self):
        from aingram.extraction.local import LocalExtractor

        response_data = {'entry_type': 'observation', 'confidence': 0.5, 'relevance': 0.5}
        mock_response = self._make_ollama_response(response_data)

        with patch('httpx.post', return_value=mock_response) as mock_post:
            extractor = LocalExtractor(model='test', base_url='http://custom:9999')
            extractor.extract('test')

        url = mock_post.call_args[0][0]
        assert url.startswith('http://custom:9999')
