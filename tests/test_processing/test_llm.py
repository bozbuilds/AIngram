from unittest.mock import Mock, patch

import pytest

from aingram.config import AIngramConfig
from aingram.processing.llm import OllamaLLM
from aingram.processing.protocols import LLMProcessor


class TestLLMProtocol:
    def test_mock_implements_protocol(self):
        from tests.conftest import MockLLM

        llm = MockLLM()
        assert isinstance(llm, LLMProcessor)

    def test_mock_returns_response(self):
        from tests.conftest import MockLLM

        llm = MockLLM(response='hello')
        result = llm.complete('prompt')
        assert result == 'hello'

    def test_mock_records_calls(self):
        from tests.conftest import MockLLM

        llm = MockLLM()
        llm.complete('prompt1', system='sys')
        llm.complete('prompt2')
        assert len(llm.calls) == 2
        assert llm.calls[0] == ('prompt1', 'sys')
        assert llm.calls[1] == ('prompt2', None)


class TestOllamaLLM:
    def test_implements_protocol(self):
        llm = OllamaLLM()
        assert isinstance(llm, LLMProcessor)

    @patch('aingram.processing.llm.httpx')
    def test_complete_sends_request(self, mock_httpx):
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'Hello world'}
        mock_response.raise_for_status = Mock()
        mock_httpx.post.return_value = mock_response

        llm = OllamaLLM(model='mistral')
        result = llm.complete('Say hello')

        assert result == 'Hello world'
        mock_httpx.post.assert_called_once()
        call_kwargs = mock_httpx.post.call_args
        assert call_kwargs[0][0] == 'http://localhost:11434/api/generate'
        payload = call_kwargs[1]['json']
        assert payload['model'] == 'mistral'
        assert payload['prompt'] == 'Say hello'

    @patch('aingram.processing.llm.httpx')
    def test_complete_with_system_prompt(self, mock_httpx):
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'ok'}
        mock_response.raise_for_status = Mock()
        mock_httpx.post.return_value = mock_response

        llm = OllamaLLM()
        llm.complete('prompt', system='be helpful')

        payload = mock_httpx.post.call_args[1]['json']
        assert payload['system'] == 'be helpful'

    @patch('aingram.processing.llm.httpx')
    def test_complete_without_system_prompt(self, mock_httpx):
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'ok'}
        mock_response.raise_for_status = Mock()
        mock_httpx.post.return_value = mock_response

        llm = OllamaLLM()
        llm.complete('prompt')

        payload = mock_httpx.post.call_args[1]['json']
        assert 'system' not in payload

    def test_custom_base_url(self):
        llm = OllamaLLM(base_url='http://gpu-server:11434')
        assert llm._base_url == 'http://gpu-server:11434'

    def test_trailing_slash_stripped(self):
        llm = OllamaLLM(base_url='http://localhost:11434/')
        assert llm._base_url == 'http://localhost:11434'

    def test_uses_explicit_config_when_passed(self):
        cfg = AIngramConfig(llm_url='http://ollama.local:11434', llm_model='phi3')
        llm = OllamaLLM(config=cfg)
        assert llm._base_url == 'http://ollama.local:11434'
        assert llm._model == 'phi3'

    def test_constructor_overrides_config(self):
        cfg = AIngramConfig(llm_url='http://ignored:1', llm_model='ignored')
        llm = OllamaLLM(config=cfg, model='mistral', base_url='http://keep:11434')
        assert llm._base_url == 'http://keep:11434'
        assert llm._model == 'mistral'

    def test_loads_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv('AINGRAM_LLM_URL', 'http://from-env:11434')
        monkeypatch.setenv('AINGRAM_LLM_MODEL', 'gemma2')
        llm = OllamaLLM()
        assert llm._base_url == 'http://from-env:11434'
        assert llm._model == 'gemma2'

    @patch('aingram.processing.llm.httpx')
    def test_complete_raises_on_missing_response_key(self, mock_httpx):
        mock_response = Mock()
        mock_response.json.return_value = {'unexpected_key': 'value'}
        mock_response.raise_for_status = Mock()
        mock_httpx.post.return_value = mock_response

        llm = OllamaLLM()
        with pytest.raises(KeyError):
            llm.complete('prompt')
