# aingram/processing/llm.py
from __future__ import annotations

import logging

import httpx

from aingram.config import AIngramConfig, load_merged_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'mistral'
DEFAULT_BASE_URL = 'http://localhost:11434'


class OllamaLLM:
    """LLM processor that connects to a local Ollama instance via HTTP.

    ``model`` and ``base_url`` default from ``config`` if given, else from
    ``load_merged_config()`` (env / ``~/.aingram/config.toml`` / built-ins).
    Explicit constructor arguments always win.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        config: AIngramConfig | None = None,
    ) -> None:
        cfg = config or load_merged_config()
        self._model = model if model is not None else cfg.llm_model
        resolved_url = base_url if base_url is not None else cfg.llm_url
        self._base_url = resolved_url.rstrip('/')
        self._timeout = timeout

    def complete(self, prompt: str, system: str | None = None) -> str:
        payload: dict[str, object] = {
            'model': self._model,
            'prompt': prompt,
            'stream': False,
        }
        if system is not None:
            payload['system'] = system

        response = httpx.post(
            f'{self._base_url}/api/generate',
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        try:
            return response.json()['response']
        except (KeyError, ValueError) as e:
            logger.warning('Unexpected Ollama response: %s', e)
            raise
