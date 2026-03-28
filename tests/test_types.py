# tests/test_types.py
from datetime import UTC, datetime

import pytest

from aingram.types import (
    Entity,
    ExtractedEntity,
    Memory,
    MemoryType,
    Relationship,
    SearchResult,
    TaskStatus,
)


class TestMemoryType:
    def test_enum_values(self):
        assert MemoryType.EPISODIC == 'episodic'
        assert MemoryType.SEMANTIC == 'semantic'
        assert MemoryType.PROCEDURAL == 'procedural'
        assert MemoryType.ENTITY == 'entity'

    def test_from_string(self):
        assert MemoryType('episodic') == MemoryType.EPISODIC

    def test_all_types_present(self):
        assert len(MemoryType) == 4


class TestMemory:
    def test_creation_with_defaults(self):
        mem = Memory(
            id='abc123',
            content='User prefers dark mode',
        )
        assert mem.id == 'abc123'
        assert mem.content == 'User prefers dark mode'
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.importance == 0.5
        assert mem.agent_id == 'default'
        assert mem.metadata == {}
        assert mem.summary is None
        assert mem.access_count == 0

    def test_creation_with_all_fields(self):
        now = datetime.now(UTC)
        mem = Memory(
            id='xyz789',
            content='Met with Alice yesterday',
            summary='Meeting with Alice',
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
            agent_id='agent-1',
            metadata={'project': 'atlas'},
            created_at=now,
            updated_at=now,
            accessed_at=now,
            access_count=3,
        )
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.importance == 0.8
        assert mem.metadata == {'project': 'atlas'}

    def test_metadata_is_dict(self):
        mem = Memory(id='a', content='test')
        assert isinstance(mem.metadata, dict)


class TestSearchResult:
    def test_creation(self):
        mem = Memory(id='a', content='test content')
        result = SearchResult(memory=mem, score=0.95)
        assert result.memory.content == 'test content'
        assert result.score == 0.95

    def test_ordering_by_score(self):
        m1 = Memory(id='a', content='first')
        m2 = Memory(id='b', content='second')
        r1 = SearchResult(memory=m1, score=0.7)
        r2 = SearchResult(memory=m2, score=0.9)
        results = sorted([r1, r2], key=lambda r: r.score, reverse=True)
        assert results[0].memory.id == 'b'


class TestEntity:
    def test_creation_with_defaults(self):
        entity = Entity(entity_id='e1', name='Alice', entity_type='person')
        assert entity.name == 'Alice'
        assert entity.entity_type == 'person'
        assert entity.mention_count == 1

    def test_creation_with_all_fields(self):
        now = datetime.now(UTC)
        entity = Entity(
            entity_id='e1',
            name='Alice Chen',
            entity_type='person',
            first_seen=now,
            last_seen=now,
            mention_count=5,
        )
        assert entity.entity_type == 'person'
        assert entity.mention_count == 5


class TestRelationship:
    def test_creation_with_defaults(self):
        rel = Relationship(
            id='r1',
            source_id='e1',
            target_id='e2',
            relation_type='works_at',
        )
        assert rel.relation_type == 'works_at'
        assert rel.weight == 1.0
        assert rel.fact is None
        assert rel.t_valid is None
        assert rel.source_memory is None

    def test_creation_with_all_fields(self):
        now = datetime.now(UTC)
        rel = Relationship(
            id='r1',
            source_id='e1',
            target_id='e2',
            relation_type='works_at',
            fact='Alice works at Acme Corp',
            weight=0.9,
            t_valid=now,
            source_memory='mem1',
        )
        assert rel.fact == 'Alice works at Acme Corp'
        assert rel.weight == 0.9


class TestExtractedEntity:
    def test_creation_with_defaults(self):
        e = ExtractedEntity(name='Alice', entity_type='person')
        assert e.name == 'Alice'
        assert e.entity_type == 'person'
        assert e.score == 1.0

    def test_creation_with_score(self):
        e = ExtractedEntity(name='Acme', entity_type='organization', score=0.95)
        assert e.score == 0.95


class TestTaskStatus:
    def test_enum_values(self):
        assert TaskStatus.PENDING == 'pending'
        assert TaskStatus.CLAIMED == 'claimed'
        assert TaskStatus.COMPLETED == 'completed'
        assert TaskStatus.FAILED == 'failed'

    def test_all_statuses(self):
        assert len(TaskStatus) == 4


# --- Phase 1 Trust Foundation Types ---


class TestEntryType:
    def test_values(self):
        from aingram.types import EntryType

        assert EntryType.OBSERVATION == 'observation'
        assert EntryType.HYPOTHESIS == 'hypothesis'
        assert EntryType.METHOD == 'method'
        assert EntryType.RESULT == 'result'
        assert EntryType.LESSON == 'lesson'
        assert EntryType.DECISION == 'decision'
        assert EntryType.META == 'meta'

    def test_from_string(self):
        from aingram.types import EntryType

        assert EntryType('observation') == EntryType.OBSERVATION
        with pytest.raises(ValueError):
            EntryType('invalid')


class TestMemoryEntry:
    def test_defaults(self):
        from aingram.types import EntryType, MemoryEntry

        entry = MemoryEntry(
            entry_id='abc',
            content_hash='def',
            entry_type=EntryType.OBSERVATION,
            content='{"text":"hello"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig',
            created_at='2026-01-01T00:00:00+00:00',
        )
        assert entry.importance == 0.5
        assert entry.access_count == 0
        assert entry.surprise is None
        assert entry.confidence is None
        assert entry.tags is None
        assert entry.metadata is None


class TestAgentSession:
    def test_defaults(self):
        from aingram.types import AgentSession

        session = AgentSession(
            session_id='s1',
            agent_name='test',
            public_key='a' * 64,
            created_at='2026-01-01T00:00:00+00:00',
        )
        assert session.parent_session_id is None
        assert session.metadata is None


class TestReasoningChain:
    def test_defaults(self):
        from aingram.types import ReasoningChain

        chain = ReasoningChain(
            chain_id='c1',
            title='test',
            created_by_session='s1',
            created_at='2026-01-01T00:00:00+00:00',
        )
        assert chain.status == 'active'
        assert chain.outcome is None


class TestEntrySearchResult:
    def test_verified_is_none_by_default(self):
        from aingram.types import EntrySearchResult, EntryType, MemoryEntry

        entry = MemoryEntry(
            entry_id='abc',
            content_hash='def',
            entry_type=EntryType.OBSERVATION,
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig',
            created_at='2026-01-01T00:00:00+00:00',
        )
        result = EntrySearchResult(entry=entry, score=0.95)
        assert result.verified is None


class TestVerificationResult:
    def test_defaults(self):
        from aingram.types import VerificationResult

        vr = VerificationResult(valid=True, session_id='s1', entries_checked=5)
        assert vr.errors == []
