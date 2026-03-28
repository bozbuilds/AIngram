"""LocalExtractor — structured extraction via Ollama with JSON format mode.

Uses httpx directly against Ollama's /api/generate endpoint with the `format`
parameter for constrained JSON decoding. This bypasses the LLMProcessor protocol
which lacks grammar/format support.
"""

from __future__ import annotations

import json
import logging

import httpx

from aingram.types import (
    EntryType,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

_VALID_ENTRY_TYPES = {e.value for e in EntryType}

_SYSTEM_PROMPT = (
    "You are an AI memory extraction system. Given text from an agent's reasoning, "
    'extract structured metadata as JSON with these fields:\n'
    '- entry_type: one of observation, hypothesis, method, result, lesson, decision, meta\n'
    '- confidence: float 0-1, how confident the content is\n'
    '- relevance: float 0-1, how important for long-term memory\n'
    '- entities: array of {name, type} objects for named entities\n'
    '- relationships: array of {source, target, type, fact} objects\n'
    'Output ONLY valid JSON.'
)

_DEFAULT_BASE_URL = 'http://localhost:11434'


class LocalExtractor:
    """Extract memory metadata using a local Ollama model with JSON format mode.

    Uses Ollama's `format: "json"` parameter for constrained JSON output,
    which is enforced via llama.cpp's grammar mode.
    """

    def __init__(
        self,
        *,
        model: str = 'aingram-extractor',
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    def extract(self, text: str) -> ExtractionResult:
        """Extract structured metadata from text via local Ollama model."""
        try:
            response = httpx.post(
                f'{self._base_url}/api/generate',
                json={
                    'model': self._model,
                    'prompt': text,
                    'system': _SYSTEM_PROMPT,
                    'format': 'json',
                    'stream': False,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json().get('response', '')
            data = json.loads(raw)
        except Exception:
            logger.warning('Local extraction failed', exc_info=True)
            return self._default_result()

        return self._parse_response(data)

    def _parse_response(self, data: dict) -> ExtractionResult:
        """Parse Ollama JSON response into ExtractionResult with validation."""
        entry_type = data.get('entry_type', 'observation')
        if entry_type not in _VALID_ENTRY_TYPES:
            entry_type = 'observation'

        confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))
        relevance = max(0.0, min(1.0, float(data.get('relevance', 0.5))))

        entities = []
        for e in data.get('entities', []):
            if isinstance(e, dict) and 'name' in e and 'type' in e:
                entities.append(
                    ExtractedEntity(
                        name=e['name'],
                        entity_type=e['type'],
                        score=confidence,
                    )
                )

        relationships = []
        for r in data.get('relationships', []):
            if isinstance(r, dict) and 'source' in r and 'target' in r and 'type' in r:
                relationships.append(
                    ExtractedRelationship(
                        source=r['source'],
                        target=r['target'],
                        relation_type=r['type'],
                        fact=r.get('fact'),
                    )
                )

        return ExtractionResult(
            entry_type=entry_type,
            confidence=confidence,
            relevance=relevance,
            entities=entities,
            relationships=relationships,
        )

    @staticmethod
    def _default_result() -> ExtractionResult:
        return ExtractionResult(entry_type='observation', confidence=0.5, relevance=0.5)
