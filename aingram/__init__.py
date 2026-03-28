# aingram/__init__.py
"""AIngram — local-first, privacy-first agent memory (Lite, Apache-2.0)."""

__version__ = '1.0.0'

from aingram.config import AIngramConfig, load_merged_config
from aingram.exceptions import (
    AIngramError,
    DatabaseError,
    EmbeddingError,
    ModelNotFoundError,
    TrustError,
    VerificationError,
)
from aingram.store import MemoryStore
from aingram.types import (
    AgentSession,
    ConsolidationResult,
    Entity,
    EntrySearchResult,
    EntryType,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    Memory,
    MemoryEntry,
    MemoryType,
    ReasoningChain,
    Relationship,
    SearchResult,
    VerificationResult,
)

__all__ = [
    'AgentSession',
    'AIngramConfig',
    'AIngramError',
    'ConsolidationResult',
    'DatabaseError',
    'EmbeddingError',
    'Entity',
    'EntrySearchResult',
    'EntryType',
    'ExtractedEntity',
    'ExtractedRelationship',
    'ExtractionResult',
    'load_merged_config',
    'Memory',
    'MemoryEntry',
    'MemoryStore',
    'MemoryType',
    'ModelNotFoundError',
    'ReasoningChain',
    'Relationship',
    'SearchResult',
    'TrustError',
    'VerificationError',
    'VerificationResult',
]
