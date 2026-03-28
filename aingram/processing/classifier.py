# aingram/processing/classifier.py
from __future__ import annotations

import re

from aingram.types import MemoryType

_TEMPORAL_PATTERNS = re.compile(
    r'\b('
    r'yesterday|today|tomorrow|last\s+(?:week|month|year|night|time)'
    r'|this\s+(?:morning|afternoon|evening|week|month)'
    r'|(?:on|in|at)\s+\w+\s+\d+'
    r'|\d{4}-\d{2}-\d{2}'
    r'|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+'
    r'|ago\b'
    r'|\bjust\s+(?:now|happened)'
    r')',
    re.IGNORECASE,
)

_PROCEDURAL_PATTERNS = re.compile(
    r'(?:'
    r'^how\s+to\b'
    r'|^to\s+\w+.*,\s*(?:first|then|next)'
    r'|^step\s+\d+'
    r'|^first,?\s+.*(?:then|next|after)'
    r')',
    re.IGNORECASE,
)

# Multi-word proper noun sequences: "Alice Chen", "Acme Corp"
_PROPER_NOUN_SEQUENCE = re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+')
# Single capitalized words that appear mid-sentence (not at position 0)
_MID_SENTENCE_PROPER = re.compile(r'(?<=\s)[A-Z][a-z]+')


class HeuristicClassifier:
    def classify(self, text: str) -> MemoryType:
        if not text:
            return MemoryType.SEMANTIC

        if _PROCEDURAL_PATTERNS.search(text):
            return MemoryType.PROCEDURAL

        if _TEMPORAL_PATTERNS.search(text):
            return MemoryType.EPISODIC

        # Count distinct proper noun entities: multi-word sequences count as one each,
        # plus standalone capitalized words appearing mid-sentence
        sequences = _PROPER_NOUN_SEQUENCE.findall(text)
        # Remove characters already consumed by sequences before counting singles
        text_remainder = _PROPER_NOUN_SEQUENCE.sub('', text)
        singles = _MID_SENTENCE_PROPER.findall(text_remainder)
        if len(sequences) + len(singles) >= 2:
            return MemoryType.ENTITY

        return MemoryType.SEMANTIC
