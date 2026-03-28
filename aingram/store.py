# aingram/store.py — Lite MemoryStore
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aingram.security.auth import CallerContext

from aingram.config import AIngramConfig
from aingram.exceptions import DatabaseError
from aingram.storage.engine import StorageEngine
from aingram.storage.queries import reciprocal_rank_fusion
from aingram.trust import (
    canonicalize_content,
    compute_content_hash,
    compute_entry_id,
    verify_signature,
)
from aingram.trust.session import SessionManager
from aingram.types import EntrySearchResult, MemoryEntry, ReasoningChain, VerificationResult

logger = logging.getLogger(__name__)

_VALID_REFERENCE_TYPES = frozenset(
    {
        'builds_on',
        'contradicts',
        'supports',
        'refines',
        'supersedes',
    }
)


class MemoryStore:
    def __init__(
        self,
        db_path: str,
        *,
        agent_name: str = 'default',
        embedder=None,
        embedding_dim: int | None = None,
        models_dir: Path | str | None = None,
        extractor=None,
        config: AIngramConfig | None = None,
    ) -> None:
        self._db_path = db_path
        self._engine = StorageEngine(db_path, embedding_dim=embedding_dim or 768)

        from aingram.config import load_merged_config

        merged = config if config is not None else load_merged_config()
        self._config = merged

        if embedder is None:
            from aingram.processing.embedder import NomicEmbedder

            assert merged is not None
            embedder = NomicEmbedder(
                dim=self._engine.get_embedding_dim(),
                models_dir=Path(models_dir) if models_dir else None,
                preferred_provider=merged.onnx_provider,
            )
        self._embedder = embedder

        self._session = SessionManager(agent_name=agent_name)
        self._engine.store_session(self._session.to_agent_session())
        self._extractor = None
        if extractor is not None:
            self._extractor = extractor
        else:
            if merged is None:
                from aingram.config import load_merged_config

                merged = load_merged_config()
            if merged.extractor_mode == 'local':
                from aingram.extraction.local import LocalExtractor

                self._extractor = LocalExtractor(
                    model=merged.extractor_model,
                    base_url=merged.llm_url,
                )
            elif merged.extractor_mode == 'sonnet':
                from aingram.extraction.sonnet import SonnetExtractor

                self._extractor = SonnetExtractor(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    def set_extractor(self, extractor) -> None:
        """Attach an extractor for type inference (optional)."""
        self._extractor = extractor

    def remember(
        self,
        content: str | dict,
        *,
        entry_type: str = 'observation',
        chain_id: str | None = None,
        parent_entry_id: str | None = None,
        parent_ids: list[str] | None = None,
        tags: list | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        from aingram.types import EntryType

        EntryType(entry_type)  # validate before doing any work

        if isinstance(content, str):
            content_payload = {'text': content}
            embed_text = content
        else:
            content_payload = content
            embed_text = json.dumps(content, sort_keys=True)

        if self._extractor is not None:
            try:
                extraction_result = self._extractor.extract(embed_text)
                if entry_type == 'observation':
                    entry_type = extraction_result.entry_type
                if confidence is None:
                    confidence = extraction_result.confidence
                EntryType(entry_type)  # re-validate
            except Exception:
                logger.warning('Extraction failed, using defaults', exc_info=True)

        content_hash = compute_content_hash(content_payload, entry_type)
        content_data = canonicalize_content(content_payload)
        seq_num = self._session.next_sequence_num
        prev_id = self._session.prev_entry_id
        created_at = datetime.now(UTC).isoformat()

        dag_parent_ids = (
            parent_ids if parent_ids is not None else ([prev_id] if prev_id is not None else [])
        )
        entry_id = compute_entry_id(
            content_data=content_data,
            parent_ids=dag_parent_ids,
            pubkey_hex=self._session.public_key_hex,
        )
        signature = self._session.sign(entry_id)
        embedding = self._embedder.embed(embed_text)

        canonical_str = content_data.decode('utf-8')
        self._engine.store_entry(
            entry_id=entry_id,
            content_hash=content_hash,
            entry_type=entry_type,
            content=canonical_str,
            session_id=self._session.session_id,
            sequence_num=seq_num,
            prev_entry_id=prev_id,
            signature=signature,
            created_at=created_at,
            embedding=embedding,
            reasoning_chain_id=chain_id,
            parent_entry_id=parent_entry_id,
            tags=tags,
            metadata=metadata,
            confidence=confidence,
            surprise=None,
        )

        self._session.advance(entry_id)

        try:
            self._engine.enqueue_task(
                task_type='extract_entities_v3',
                payload={'entry_id': entry_id},
            )
        except Exception:
            logger.warning('Failed to enqueue extraction task', exc_info=True)

        return entry_id

    def recall(
        self,
        query: str | None = None,
        *,
        entry_type: str | None = None,
        chain_id: str | None = None,
        session_id: str | None = None,
        entry_id: str | None = None,
        limit: int = 20,
        verify: bool = True,
    ) -> list[EntrySearchResult]:
        if entry_id is not None:
            entry = self._engine.get_entry(entry_id)
            if entry is None:
                return []
            verified = self._verify_entry(entry) if verify else None
            return [EntrySearchResult(entry=entry, score=1.0, verified=verified)]

        if chain_id is not None and query is None:
            entries = self._engine.get_entries_by_chain(chain_id, limit=limit)
            return [
                EntrySearchResult(
                    entry=e,
                    score=1.0,
                    verified=self._verify_entry(e) if verify else None,
                )
                for e in entries
            ]

        if query is None:
            raise ValueError('query is required unless entry_id or chain_id is provided')

        from aingram.graph.traversal import GraphTraversal

        embedding = self._embedder.embed(query)
        candidate_limit = limit * 3

        vec_results = self._engine.search_vectors(embedding, limit=candidate_limit)
        vec_ids = [eid for eid, _ in vec_results]

        fts_results = self._engine.search_fts(query, limit=candidate_limit)
        fts_ids = [eid for eid, _ in fts_results]

        try:
            traversal = GraphTraversal(self._engine)
            graph_ids = traversal.search(query, limit=candidate_limit)
        except Exception:
            graph_ids = []

        chain_ids_list: list[str] = []
        if chain_id:
            chain_entries = self._engine.get_entries_by_chain(chain_id, limit=candidate_limit)
            chain_ids_list = [e.entry_id for e in chain_entries]

        ranked_lists = [lst for lst in [vec_ids, fts_ids, graph_ids, chain_ids_list] if lst]
        rrf_scores = reciprocal_rank_fusion(ranked_lists, k=60)

        all_ids = list(rrf_scores.keys())
        entries_map = {e.entry_id: e for e in self._engine.get_entries_by_ids(all_ids)}

        results = []
        session_cache: dict = {}
        for eid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = entries_map.get(eid)
            if entry is None:
                continue
            if entry_type and entry.entry_type != entry_type:
                continue
            if chain_id and entry.reasoning_chain_id != chain_id:
                continue
            if session_id and entry.session_id != session_id:
                continue

            created = datetime.fromisoformat(entry.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            hours_ago = max(
                (datetime.now(UTC) - created).total_seconds() / 3600,
                0.001,
            )
            recency = math.exp(-0.001 * hours_ago)
            conf = entry.confidence if entry.confidence is not None else 0.5
            composite = rrf_score * entry.importance * conf * recency

            verified = self._verify_entry(entry, _session_cache=session_cache) if verify else None
            results.append(EntrySearchResult(entry=entry, score=composite, verified=verified))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_context(self, query: str, *, max_tokens: int = 2000) -> str:
        results = self.recall(query, limit=20, verify=False)
        budget = max_tokens
        parts: list[str] = []
        for r in results:
            tokens = len(r.entry.content) // 4
            if tokens > budget:
                break
            parts.append(r.entry.content)
            budget -= tokens
        return '\n\n'.join(parts)

    def reference(
        self,
        *,
        source_id: str,
        target_id: str,
        reference_type: str,
        caller: CallerContext | None = None,
    ) -> None:
        if reference_type not in _VALID_REFERENCE_TYPES:
            raise ValueError(
                f'Invalid reference_type: {reference_type}. '
                f'Must be one of: {", ".join(sorted(_VALID_REFERENCE_TYPES))}'
            )
        source_entry = self._engine.get_entry(source_id)
        if source_entry is None:
            raise ValueError(f'Source entry {source_id} not found')
        if self._engine.get_entry(target_id) is None:
            raise ValueError(f'Target entry {target_id} not found')

        if caller is not None:
            from aingram.exceptions import AuthorizationError
            from aingram.security.roles import Role

            if caller.role != Role.ADMIN:
                if source_entry.session_id != caller.session_id:
                    raise AuthorizationError(
                        'Contributors can only create references from their own entries'
                    )

        sig_payload = f'{source_id}:{target_id}:{reference_type}'
        signature = self._session.sign(sig_payload)

        self._engine.store_cross_reference(
            source_entry_id=source_id,
            target_entry_id=target_id,
            reference_type=reference_type,
            session_id=self._session.session_id,
            signature=signature,
        )

    @property
    def entities(self) -> list:
        return self._engine.get_entities(limit=10000)

    def create_chain(self, title: str) -> str:
        chain = ReasoningChain(
            chain_id=uuid.uuid4().hex,
            title=title,
            created_by_session=self._session.session_id,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._engine.create_chain(chain)
        return chain.chain_id

    def complete_chain(self, chain_id: str, *, outcome: str) -> None:
        self._engine.complete_chain(chain_id, outcome=outcome)

    @property
    def knowledge_items(self) -> list[dict]:
        return self._engine.get_knowledge_items()

    def verify(
        self, session_id: str | None = None, *, caller: CallerContext | None = None
    ) -> VerificationResult:
        if caller is not None and session_id is not None:
            from aingram.exceptions import AuthorizationError
            from aingram.security.roles import Role

            if caller.role != Role.ADMIN and session_id != caller.session_id:
                raise AuthorizationError('Non-admin callers can only verify their own session')
        if caller is not None and session_id is None:
            session_id = caller.session_id
        sid = session_id or self._session.session_id
        entries = self._engine.get_entries_by_session(sid)

        errors: list[str] = []
        prev_id: str | None = None
        expected_seq = 1
        session_cache: dict = {}
        parents_cache = self._engine.get_verify_parents_batch([e.entry_id for e in entries])

        for entry in entries:
            if entry.sequence_num != expected_seq:
                errors.append(f'Sequence gap: expected {expected_seq}, got {entry.sequence_num}')
            if entry.prev_entry_id != prev_id:
                errors.append(
                    f'Chain break at seq {entry.sequence_num}: '
                    f'expected prev={prev_id}, got prev={entry.prev_entry_id}'
                )
            if not self._verify_entry(
                entry, _session_cache=session_cache, _parents_cache=parents_cache
            ):
                errors.append(f'Integrity check failed for entry {entry.entry_id[:16]}...')
            prev_id = entry.entry_id
            expected_seq = entry.sequence_num + 1

        return VerificationResult(
            valid=len(errors) == 0,
            session_id=sid,
            entries_checked=len(entries),
            errors=errors,
        )

    def _verify_entry(
        self,
        entry: MemoryEntry,
        *,
        _session_cache: dict | None = None,
        _parents_cache: dict[str, list[str]] | None = None,
    ) -> bool:
        content_dict = json.loads(entry.content)
        expected_hash = compute_content_hash(content_dict, entry.entry_type)
        if expected_hash != entry.content_hash:
            return False

        if _session_cache is not None and entry.session_id in _session_cache:
            session = _session_cache[entry.session_id]
        else:
            session = self._engine.get_session(entry.session_id)
            if _session_cache is not None and session is not None:
                _session_cache[entry.session_id] = session
        if session is None:
            return False

        if _parents_cache is not None and entry.entry_id in _parents_cache:
            parents = _parents_cache[entry.entry_id]
        else:
            parents = self._engine.get_verify_parents_batch([entry.entry_id]).get(
                entry.entry_id, []
            )
        expected_id = compute_entry_id(
            content_data=entry.content.encode('utf-8'),
            parent_ids=parents,
            pubkey_hex=session.public_key,
        )
        if expected_id != entry.entry_id:
            return False

        return verify_signature(session.public_key, entry.entry_id, entry.signature)

    def consolidate(self, *, llm=None, force: bool = True):
        from aingram.consolidation.contradiction import ContradictionDetector
        from aingram.consolidation.decay import apply_decay
        from aingram.consolidation.knowledge import KnowledgeSynthesizer
        from aingram.consolidation.merger import MemoryMerger
        from aingram.types import ConsolidationResult

        del force  # Lite: no daemon convergence; always run steps when called

        decayed = apply_decay(self._engine)

        detector = ContradictionDetector(self._engine, llm=llm)
        contradiction_result = detector.detect_and_resolve()

        merger = MemoryMerger(
            self._engine,
            embedder=self._embedder,
            llm=llm,
            session=self._session,
        )
        merge_result = merger.merge_similar()

        synthesizer = KnowledgeSynthesizer(
            self._engine,
            llm=llm,
            embedder=self._embedder,
            session_id=self._session.session_id,
            require_quorum=True,
        )
        synthesis_result = synthesizer.synthesize()

        return ConsolidationResult(
            memories_decayed=decayed,
            contradictions_found=contradiction_result.contradictions_found,
            contradictions_resolved=contradiction_result.contradictions_resolved,
            memories_merged=merge_result.memories_merged,
            summaries_created=merge_result.summaries_created,
            knowledge_synthesized=synthesis_result.knowledge_synthesized,
            chains_analyzed=synthesis_result.chains_analyzed,
            knowledge_reviewed=0,
        )

    def compact(self, *, confirm: bool = False, target_dim: int = 256) -> None:
        """One-way truncate all stored embeddings to target_dim (Matryoshka slice)."""
        if not confirm:
            raise ValueError('compact() is destructive; call with confirm=True')
        old = self._engine.get_embedding_dim()
        if target_dim >= old:
            raise ValueError(f'target_dim ({target_dim}) must be less than current dim ({old})')
        self._engine.truncate_all_embeddings_to_dim(target_dim)
        self._embedder.dim = target_dim

    def export_json(self, path: str | Path, *, agent_id: str | None = None) -> None:
        """Export sessions, chains, entries, graph, and vectors to JSON."""
        import base64

        p = Path(path)
        eng = self._engine
        with eng._lock:
            sess_rows = eng._conn.execute(
                'SELECT session_id, agent_name, public_key, parent_session_id, '
                'created_at, metadata FROM agent_sessions'
            ).fetchall()
            if agent_id is not None:
                want = {r[0] for r in sess_rows if r[1] == agent_id}
                if not want:
                    raise ValueError(f'No session with agent_name {agent_id!r}')
            else:
                want = None

            def row_filter_session(
                session_col: str, rows: list[tuple], idx: int = 0
            ) -> list[tuple]:
                if want is None:
                    return rows
                return [r for r in rows if r[idx] in want]

            mem_rows = eng._conn.execute(
                'SELECT entry_id, content_hash, entry_type, content, session_id, sequence_num, '
                'prev_entry_id, signature, created_at, reasoning_chain_id, parent_entry_id, '
                'tags, metadata, confidence, importance, accessed_at, access_count, surprise, '
                'consolidated FROM memory_entries ORDER BY created_at ASC'
            ).fetchall()
            mem_rows = row_filter_session('session_id', mem_rows, 4)
            entry_ids = {r[0] for r in mem_rows}

            chain_rows = eng._conn.execute(
                'SELECT chain_id, title, status, outcome, created_by_session, created_at '
                'FROM reasoning_chains'
            ).fetchall()
            chain_rows = row_filter_session('created_by_session', chain_rows, 4)

            entity_rows = eng._conn.execute(
                'SELECT entity_id, name, entity_type, first_seen, last_seen, mention_count '
                'FROM entities'
            ).fetchall()
            rel_rows = eng._conn.execute(
                'SELECT id, source_id, target_id, relation_type, fact, weight, '
                't_valid, t_invalid, source_memory FROM relationships'
            ).fetchall()
            mention_rows = eng._conn.execute(
                'SELECT entity_id, entry_id, confidence FROM entity_mentions'
            ).fetchall()
            if entry_ids:
                mention_rows = [m for m in mention_rows if m[1] in entry_ids]

            vec_rows = eng._conn.execute('SELECT entry_id, embedding FROM vec_entries').fetchall()
            if entry_ids:
                vec_rows = [v for v in vec_rows if v[0] in entry_ids]

        sessions_out = [
            {
                'session_id': r[0],
                'agent_name': r[1],
                'public_key': r[2],
                'parent_session_id': r[3],
                'created_at': r[4],
                'metadata': r[5],
            }
            for r in (sess_rows if want is None else [s for s in sess_rows if s[0] in want])
        ]
        payload = {
            'aingram_lite_export_version': 1,
            'embedding_dim': eng.get_embedding_dim(),
            'agent_sessions': sessions_out,
            'reasoning_chains': [
                {
                    'chain_id': r[0],
                    'title': r[1],
                    'status': r[2],
                    'outcome': r[3],
                    'created_by_session': r[4],
                    'created_at': r[5],
                }
                for r in chain_rows
            ],
            'memory_entries': [
                {
                    'entry_id': r[0],
                    'content_hash': r[1],
                    'entry_type': r[2],
                    'content': r[3],
                    'session_id': r[4],
                    'sequence_num': r[5],
                    'prev_entry_id': r[6],
                    'signature': r[7],
                    'created_at': r[8],
                    'reasoning_chain_id': r[9],
                    'parent_entry_id': r[10],
                    'tags': r[11],
                    'metadata': r[12],
                    'confidence': r[13],
                    'importance': r[14],
                    'accessed_at': r[15],
                    'access_count': r[16],
                    'surprise': r[17],
                    'consolidated': r[18],
                }
                for r in mem_rows
            ],
            'entities': [
                {
                    'entity_id': r[0],
                    'name': r[1],
                    'entity_type': r[2],
                    'first_seen': r[3],
                    'last_seen': r[4],
                    'mention_count': r[5],
                }
                for r in entity_rows
            ],
            'relationships': [
                {
                    'id': r[0],
                    'source_id': r[1],
                    'target_id': r[2],
                    'relation_type': r[3],
                    'fact': r[4],
                    'weight': r[5],
                    't_valid': r[6],
                    't_invalid': r[7],
                    'source_memory': r[8],
                }
                for r in rel_rows
            ],
            'entity_mentions': [
                {'entity_id': r[0], 'entry_id': r[1], 'confidence': r[2]} for r in mention_rows
            ],
            'vec_entries': [
                {'entry_id': eid, 'embedding_b64': base64.b64encode(blob).decode('ascii')}
                for eid, blob in vec_rows
            ],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def import_json(self, path: str | Path, *, merge: bool = False) -> None:
        """Import a Lite export. When merge=False, database must be empty."""
        import base64

        p = Path(path)
        raw = json.loads(p.read_text(encoding='utf-8'))
        if raw.get('aingram_lite_export_version') != 1:
            raise ValueError('Unsupported export version (expected aingram_lite_export_version 1)')

        eng = self._engine
        if not merge and eng.get_entry_count() > 0:
            raise ValueError('Target database is not empty; pass merge=True to import anyway')
        export_dim = int(raw['embedding_dim'])
        if export_dim != eng.get_embedding_dim():
            got = eng.get_embedding_dim()
            raise DatabaseError(f'Export embedding_dim {export_dim} does not match database {got}')

        with eng._lock:
            if not merge:
                for table in (
                    'entity_mentions',
                    'relationships',
                    'entities',
                    'cross_references',
                    'entries_fts',
                    'vec_entries',
                    'memory_entries',
                    'reasoning_chains',
                    'agent_sessions',
                ):
                    try:
                        eng._conn.execute(f'DELETE FROM {table}')
                    except Exception:
                        pass
                eng._conn.execute("DELETE FROM task_queue WHERE status IN ('pending', 'claimed')")

            for s in raw['agent_sessions']:
                eng._conn.execute(
                    'INSERT OR REPLACE INTO agent_sessions '
                    '(session_id, agent_name, public_key, parent_session_id, created_at, metadata) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        s['session_id'],
                        s['agent_name'],
                        s['public_key'],
                        s.get('parent_session_id'),
                        s['created_at'],
                        s.get('metadata'),
                    ),
                )

            for c in raw['reasoning_chains']:
                eng._conn.execute(
                    'INSERT OR REPLACE INTO reasoning_chains '
                    '(chain_id, title, status, outcome, created_by_session, created_at) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        c['chain_id'],
                        c['title'],
                        c['status'],
                        c.get('outcome'),
                        c['created_by_session'],
                        c['created_at'],
                    ),
                )

            existing_ids = set()
            if merge:
                existing_ids = {
                    r[0]
                    for r in eng._conn.execute('SELECT entry_id FROM memory_entries').fetchall()
                }

            for e in raw['memory_entries']:
                if merge and e['entry_id'] in existing_ids:
                    continue
                cols = (
                    'entry_id, content_hash, entry_type, content, session_id, sequence_num, '
                    'prev_entry_id, signature, created_at, reasoning_chain_id, parent_entry_id, '
                    'tags, metadata, confidence, importance, accessed_at, access_count, surprise, '
                    'consolidated'
                )
                eng._conn.execute(
                    f'INSERT OR REPLACE INTO memory_entries ({cols}) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        e['entry_id'],
                        e['content_hash'],
                        e['entry_type'],
                        e['content'],
                        e['session_id'],
                        e['sequence_num'],
                        e['prev_entry_id'],
                        e['signature'],
                        e['created_at'],
                        e.get('reasoning_chain_id'),
                        e.get('parent_entry_id'),
                        e.get('tags'),
                        e.get('metadata'),
                        e.get('confidence'),
                        e.get('importance', 0.5),
                        e.get('accessed_at'),
                        e.get('access_count', 0),
                        e.get('surprise'),
                        e.get('consolidated', 0),
                    ),
                )
                fts_text = e['content']
                try:
                    parsed = json.loads(e['content'])
                    if isinstance(parsed, dict) and 'text' in parsed:
                        fts_text = parsed['text']
                except (json.JSONDecodeError, TypeError):
                    pass
                eng._conn.execute(
                    'INSERT OR REPLACE INTO entries_fts (content, entry_id) VALUES (?, ?)',
                    (fts_text, e['entry_id']),
                )

            for ent in raw['entities']:
                eng._conn.execute(
                    'INSERT OR REPLACE INTO entities '
                    '(entity_id, name, entity_type, first_seen, last_seen, mention_count) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        ent['entity_id'],
                        ent['name'],
                        ent['entity_type'],
                        ent['first_seen'],
                        ent['last_seen'],
                        ent.get('mention_count', 1),
                    ),
                )

            for rel in raw['relationships']:
                eng._conn.execute(
                    'INSERT OR REPLACE INTO relationships '
                    '(id, source_id, target_id, relation_type, fact, weight, t_valid, t_invalid, '
                    'source_memory) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        rel['id'],
                        rel['source_id'],
                        rel['target_id'],
                        rel['relation_type'],
                        rel.get('fact'),
                        rel.get('weight', 1.0),
                        rel.get('t_valid'),
                        rel.get('t_invalid'),
                        rel.get('source_memory'),
                    ),
                )

            for m in raw['entity_mentions']:
                eng._conn.execute(
                    'INSERT OR IGNORE INTO entity_mentions '
                    '(entity_id, entry_id, confidence) VALUES (?, ?, ?)',
                    (m['entity_id'], m['entry_id'], m.get('confidence', 1.0)),
                )

            dim = eng.get_embedding_dim()
            for v in raw['vec_entries']:
                blob = base64.b64decode(v['embedding_b64'])
                if len(blob) != dim * 4:
                    raise DatabaseError(
                        f'vec_entries row {v["entry_id"][:16]}… length mismatch for dim {dim}'
                    )
                eng._conn.execute(
                    'INSERT OR REPLACE INTO vec_entries (entry_id, embedding) VALUES (?, ?)',
                    (v['entry_id'], blob),
                )

            eng._conn.commit()

    @property
    def stats(self) -> dict:
        db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0
        return {
            'entry_count': self._engine.get_entry_count(),
            'db_size_bytes': db_size,
            'embedding_dim': self._engine.get_embedding_dim(),
        }

    def close(self) -> None:
        self._engine.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()
