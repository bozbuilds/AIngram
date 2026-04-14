# aingram/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class MemoryType(StrEnum):
    EPISODIC = 'episodic'
    SEMANTIC = 'semantic'
    PROCEDURAL = 'procedural'
    ENTITY = 'entity'


@dataclass
class Memory:
    id: str
    content: str
    summary: str | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.5
    agent_id: str = 'default'
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime | None = None
    access_count: int = 0


@dataclass
class SearchResult:
    memory: Memory
    score: float


class TaskStatus(StrEnum):
    PENDING = 'pending'
    CLAIMED = 'claimed'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass
class Entity:
    entity_id: str
    name: str
    entity_type: str
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    mention_count: int = 1


@dataclass
class Relationship:
    id: str
    source_id: str
    target_id: str
    relation_type: str
    fact: str | None = None
    weight: float = 1.0
    t_valid: datetime | None = None
    t_invalid: datetime | None = None
    source_memory: str | None = None


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    score: float = 1.0


@dataclass
class ContradictionVerdict:
    """Result of a pairwise contradiction classification."""

    contradicts: bool
    confidence: float
    superseded_index: int | None = None  # 0 or 1; None = orchestrator decides


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    memories_decayed: int = 0
    contradictions_found: int = 0
    contradictions_resolved: int = 0
    memories_merged: int = 0
    summaries_created: int = 0
    knowledge_synthesized: int = 0
    chains_analyzed: int = 0
    knowledge_reviewed: int = 0


# --- Trust / v3 entries ---


class EntryType(StrEnum):
    OBSERVATION = 'observation'
    HYPOTHESIS = 'hypothesis'
    METHOD = 'method'
    RESULT = 'result'
    LESSON = 'lesson'
    DECISION = 'decision'
    META = 'meta'


@dataclass
class MemoryEntry:
    entry_id: str
    content_hash: str
    entry_type: EntryType
    content: str  # canonical JSON
    session_id: str
    sequence_num: int
    prev_entry_id: str | None
    signature: str
    created_at: str
    reasoning_chain_id: str | None = None
    parent_entry_id: str | None = None
    tags: list | None = None
    metadata: dict | None = None
    confidence: float | None = None
    importance: float = 0.5
    accessed_at: str | None = None
    access_count: int = 0
    surprise: float | None = None
    consolidated: int = 0


@dataclass
class AgentSession:
    session_id: str
    agent_name: str
    public_key: str  # 64-char hex Ed25519 pubkey
    created_at: str
    parent_session_id: str | None = None
    metadata: dict | None = None


@dataclass
class ReasoningChain:
    chain_id: str
    title: str
    created_by_session: str
    created_at: str
    status: str = 'active'
    outcome: str | None = None


@dataclass
class EntrySearchResult:
    entry: MemoryEntry
    score: float
    verified: bool | None = None


@dataclass
class VerificationResult:
    valid: bool
    session_id: str
    entries_checked: int
    errors: list[str] = field(default_factory=list)


@dataclass
class ExtractedRelationship:
    source: str
    target: str
    relation_type: str
    fact: str | None = None


@dataclass
class ExtractionResult:
    entry_type: str
    confidence: float
    relevance: float
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f'confidence must be 0-1, got {self.confidence}')
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f'relevance must be 0-1, got {self.relevance}')
