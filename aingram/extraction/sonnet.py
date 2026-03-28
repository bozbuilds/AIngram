"""SonnetExtractor — structured extraction via Anthropic tool use."""

from __future__ import annotations

import logging
from typing import Any

from aingram.types import (
    EntryType,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

_EXTRACTION_TOOL = {
    'name': 'extract_memory_metadata',
    'description': (
        'Extract structured metadata from agent memory content. '
        'Classify the text, assess confidence and relevance, '
        'and identify entities and relationships.'
    ),
    'input_schema': {
        'type': 'object',
        'properties': {
            'entry_type': {
                'type': 'string',
                'enum': [e.value for e in EntryType],
                'description': (
                    'observation=raw fact, hypothesis=testable claim, method=procedure, '
                    'result=experiment outcome, lesson=synthesized principle, '
                    'decision=chosen action, meta=system entry'
                ),
            },
            'confidence': {
                'type': 'number',
                'minimum': 0,
                'maximum': 1,
                'description': 'How confident the content statement is (0=uncertain, 1=certain)',
            },
            'relevance': {
                'type': 'number',
                'minimum': 0,
                'maximum': 1,
                'description': (
                    'How important/relevant this is for long-term memory (0=trivial, 1=critical)'
                ),
            },
            'entities': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'type': {'type': 'string'},
                    },
                    'required': ['name', 'type'],
                },
                'description': 'Named entities mentioned in the text',
            },
            'relationships': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'source': {'type': 'string'},
                        'target': {'type': 'string'},
                        'type': {'type': 'string'},
                        'fact': {'type': 'string'},
                    },
                    'required': ['source', 'target', 'type'],
                },
                'description': 'Relationships between entities',
            },
        },
        'required': ['entry_type', 'confidence', 'relevance'],
    },
}

_SYSTEM_PROMPT = (
    'You are an AI memory extraction system for an agent reasoning platform. '
    "Given text from an agent's reasoning or observation, extract structured metadata. "
    'Always use the extract_memory_metadata tool to return your analysis.'
)

_VALID_ENTRY_TYPES = {e.value for e in EntryType}


class SonnetExtractor:
    """Extract memory metadata using Anthropic Claude via tool use."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = 'claude-sonnet-4-6',
        api_key: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    'anthropic package required for SonnetExtractor. '
                    'Install with: pip install "aingram[api]"'
                ) from e
            self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def extract(self, text: str) -> ExtractionResult:
        """Extract structured metadata from text via Anthropic tool use."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=[_EXTRACTION_TOOL],
                tool_choice={'type': 'tool', 'name': 'extract_memory_metadata'},
                messages=[{'role': 'user', 'content': text}],
            )
        except Exception:
            logger.warning('Extraction API call failed', exc_info=True)
            return self._default_result()

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> ExtractionResult:
        """Parse Anthropic response into ExtractionResult."""
        for block in response.content:
            if block.type == 'tool_use':
                return self._parse_tool_input(block.input)
        return self._default_result()

    def _parse_tool_input(self, data: dict) -> ExtractionResult:
        """Parse tool input dict into ExtractionResult with validation."""
        entry_type = data.get('entry_type', 'observation')
        if entry_type not in _VALID_ENTRY_TYPES:
            entry_type = 'observation'

        confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))
        relevance = max(0.0, min(1.0, float(data.get('relevance', 0.5))))

        entities = [
            ExtractedEntity(name=e['name'], entity_type=e['type'], score=confidence)
            for e in data.get('entities', [])
            if 'name' in e and 'type' in e
        ]

        relationships = [
            ExtractedRelationship(
                source=r['source'],
                target=r['target'],
                relation_type=r['type'],
                fact=r.get('fact'),
            )
            for r in data.get('relationships', [])
            if 'source' in r and 'target' in r and 'type' in r
        ]

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
