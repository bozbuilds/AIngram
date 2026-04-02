# aingram/consolidation/merger.py
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from aingram.processing.protocols import Embedder, LLMProcessor
from aingram.storage.engine import StorageEngine
from aingram.trust import canonicalize_content, compute_content_hash, compute_entry_id
from aingram.trust.session import SessionManager

logger = logging.getLogger(__name__)

_MIN_CLUSTER_SIZE = 3
_MERGED_IMPORTANCE_FACTOR = 0.3
_SUMMARY_INITIAL_IMPORTANCE = 0.7

MERGE_SYSTEM_PROMPT = (
    'You are a memory summarizer. Given multiple related memories, '
    'write ONE concise summary that captures the key information. '
    'Output ONLY the summary text, nothing else.'
)

MERGE_USER_PROMPT = 'Summarize these related memories:\n\n{memories}'


@dataclass
class MergeResult:
    memories_merged: int
    summaries_created: int


class MemoryMerger:
    def __init__(
        self,
        engine: StorageEngine,
        *,
        embedder: Embedder,
        llm: LLMProcessor | None = None,
        session: SessionManager | None = None,
    ) -> None:
        self._engine = engine
        self._embedder = embedder
        self._llm = llm
        self._session = session
        self._qjl_projection = None

    def _get_qjl_projection(self):
        """Lazy-load and cache the QJL projection matrix."""
        if self._qjl_projection is None:
            from aingram.processing.qjl import NUM_PROJECTIONS, SEED, create_projection

            dim = self._engine.get_embedding_dim()
            with self._engine._lock:
                row = self._engine._conn.execute(
                    "SELECT value FROM db_metadata WHERE key = 'qjl_seed'"
                ).fetchone()
            seed = int(row[0]) if row else SEED
            self._qjl_projection = create_projection(dim, NUM_PROJECTIONS, seed)
        return self._qjl_projection

    def merge_similar(self, *, min_cluster_size: int = _MIN_CLUSTER_SIZE) -> MergeResult:
        if self._llm is None or self._session is None:
            return MergeResult(memories_merged=0, summaries_created=0)

        entity_pairs = self._engine.get_entity_entry_pairs()
        if not entity_pairs:
            return MergeResult(memories_merged=0, summaries_created=0)

        # Group entry_ids by entity
        entity_entries: dict[str, set[str]] = defaultdict(set)
        for entity_id, entry_id in entity_pairs:
            entity_entries[entity_id].add(entry_id)

        # Build clusters — each entry assigned to at most one cluster
        assigned: set[str] = set()
        clusters: list[tuple[list[str], list]] = []

        for entity_id in sorted(
            entity_entries, key=lambda eid: len(entity_entries[eid]), reverse=True
        ):
            entry_ids = entity_entries[entity_id] - assigned
            if len(entry_ids) >= min_cluster_size:
                cluster = list(entry_ids)
                # Exclude entries that are in a reasoning chain — never merge within a chain
                entries = self._engine.get_entries_by_ids(cluster)
                chain_entries = {e.entry_id for e in entries if e.reasoning_chain_id is not None}
                cluster = [eid for eid in cluster if eid not in chain_entries]
                # Keep only entries that survived chain filtering
                entries = [e for e in entries if e.entry_id in set(cluster)]
                if len(cluster) >= min_cluster_size:
                    clusters.append((cluster, entries))
                    assigned.update(cluster)

        merged_count = 0
        summaries_created = 0

        for cluster, entries in clusters:
            memories_text = '\n'.join(f'- {e.content}' for e in entries)
            prompt = MERGE_USER_PROMPT.format(memories=memories_text)

            try:
                summary = self._llm.complete(prompt, system=MERGE_SYSTEM_PROMPT)
            except Exception:
                logger.warning('Merge LLM call failed', exc_info=True)
                continue

            content_payload = {'text': summary}
            entry_type = 'lesson'
            content_hash = compute_content_hash(content_payload, entry_type)
            content_data = canonicalize_content(content_payload)
            seq_num = self._session.next_sequence_num
            prev_id = self._session.prev_entry_id
            created_at = datetime.now(UTC).isoformat()

            dag_parents = [prev_id] if prev_id else []
            entry_id = compute_entry_id(
                content_data=content_data,
                parent_ids=dag_parents,
                pubkey_hex=self._session.public_key_hex,
            )
            signature = self._session.sign(entry_id)
            canonical_str = content_data.decode('utf-8')
            embedding = self._embedder.embed(summary)

            from aingram.processing.qjl import encode

            qjl_bits = encode(embedding, self._get_qjl_projection())[0]

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
                importance=_SUMMARY_INITIAL_IMPORTANCE,
                metadata={'merged_from': cluster},
                qjl_bits=qjl_bits,
            )
            if dag_parents:
                self._engine.insert_dag_parents(entry_id, dag_parents)
            self._session.advance(entry_id)

            # Reduce original importance
            entry_map = {e.entry_id: e for e in entries}
            updates = [
                (eid, entry_map[eid].importance * _MERGED_IMPORTANCE_FACTOR)
                for eid in cluster
                if eid in entry_map
            ]
            self._engine.batch_update_entry_importance(updates)

            merged_count += len(cluster)
            summaries_created += 1

        return MergeResult(memories_merged=merged_count, summaries_created=summaries_created)
