# aingram/pipeline.py
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from aingram.consolidation.contradiction import ContradictionDetector, LLMContradictionClassifier
from aingram.consolidation.decay import apply_decay
from aingram.consolidation.merger import MemoryMerger
from aingram.graph.traversal import GraphTraversal
from aingram.processing.capabilities import Capabilities
from aingram.processing.classifier import HeuristicClassifier
from aingram.processing.protocols import Embedder, LLMProcessor, MemoryClassifier
from aingram.storage.engine import StorageEngine
from aingram.types import ConsolidationResult, Memory, MemoryType, SearchResult

logger = logging.getLogger(__name__)

# Recency decay: half-life ~29 days. Memories >3 months old contribute minimally.
_RECENCY_DECAY_RATE = 0.001


class MemoryPipeline:
    def __init__(
        self,
        *,
        engine: StorageEngine,
        embedder: Embedder,
        classifier: MemoryClassifier | None = None,
        has_extractor: bool = False,
        has_llm: bool = False,
    ) -> None:
        self._engine = engine
        self._embedder = embedder
        self._classifier = classifier or HeuristicClassifier()
        self._has_extractor = has_extractor
        self._has_llm = has_llm
        self._traversal = GraphTraversal(engine)

    def sync_embedder_dimension(self, dim: int) -> None:
        if hasattr(self._embedder, 'dim'):
            setattr(self._embedder, 'dim', dim)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            has_embedder=True,
            has_classifier=True,
            has_extractor=self._has_extractor,
            has_llm=self._has_llm,
        )

    def add(
        self,
        content: str,
        *,
        metadata: dict | None = None,
        agent_id: str = 'default',
    ) -> str:
        memory_type = self._classifier.classify(content)
        importance = self._compute_importance(content)
        embedding = self._embedder.embed(content)

        memory_id = self._engine.store_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            agent_id=agent_id,
            metadata=metadata or {},
            embedding=embedding,
        )

        if self._has_extractor:
            self._engine.enqueue_task(
                task_type='extract_entities',
                payload={'memory_id': memory_id},
            )

        logger.debug(
            'Stored memory %s (type=%s, importance=%.2f)',
            memory_id,
            memory_type,
            importance,
        )
        return memory_id

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        agent_id: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[SearchResult]:
        from aingram.storage.queries import reciprocal_rank_fusion

        query_embedding = self._embedder.embed(query)
        wider = limit * 5

        # Three-way search
        vec_results = self._engine.search_vectors(query_embedding, limit=wider)
        fts_results = self._engine.search_fts(query, limit=wider)
        graph_ids = self._traversal.search(query, limit=wider)

        # Build ranked lists for RRF
        ranked_lists = [
            [mid for mid, _dist in vec_results],
            [mid for mid, _score in fts_results],
        ]
        if graph_ids:
            ranked_lists.append(graph_ids)

        rrf_scores = reciprocal_rank_fusion(ranked_lists)

        # Filter by agent_id and memory_type
        if agent_id is not None or memory_type is not None:
            valid = self._engine.filter_memory_ids(
                list(rrf_scores.keys()),
                agent_id=agent_id,
                memory_type=memory_type,
            )
            rrf_scores = {mid: s for mid, s in rrf_scores.items() if mid in valid}

        # Fetch memories for reranking
        candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: limit * 2]
        memories = self._engine.get_memories_batch(candidate_ids)

        # Composite reranking
        reranked = self._rerank(
            {mid: rrf_scores[mid] for mid in candidate_ids},
            memories,
        )

        return [
            SearchResult(memory=memories[mid], score=score)
            for mid, score in reranked[:limit]
            if mid in memories
        ]

    @staticmethod
    def _recency_boost(memory: Memory) -> float:
        """Exponential decay based on age. Recent memories score closer to 1.0."""
        reference = memory.accessed_at or memory.created_at
        age_hours = (datetime.now(UTC) - reference).total_seconds() / 3600
        return math.exp(-_RECENCY_DECAY_RATE * max(age_hours, 0))

    def _rerank(
        self, rrf_scores: dict[str, float], memories: dict[str, Memory]
    ) -> list[tuple[str, float]]:
        """Composite reranking: RRF × importance × recency."""
        reranked = []
        for mid, rrf_score in rrf_scores.items():
            memory = memories.get(mid)
            if not memory:
                continue
            importance = memory.importance or 0.5
            recency = self._recency_boost(memory)
            composite = rrf_score * importance * recency
            reranked.append((mid, composite))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def get_context(
        self,
        query: str,
        *,
        max_tokens: int = 2000,
        limit: int = 20,
    ) -> str:
        """Search for relevant memories and pack into a token-budgeted context string."""
        results = self.search(query, limit=limit)
        if not results:
            return ''

        lines: list[str] = []
        tokens_used = 0

        for result in results:
            entry = f'[Relevance: {result.score:.2f}]\n{result.memory.content}'
            entry_tokens = self._estimate_tokens(entry)

            if tokens_used + entry_tokens > max_tokens:
                break

            lines.append(entry)
            tokens_used += entry_tokens

        return '\n'.join(lines)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count (1 token ~ 4 chars for English text)."""
        return max(1, len(text) // 4)

    def delete(self, memory_id: str) -> bool:
        return self._engine.delete_memory(memory_id)

    def consolidate(self, *, llm: LLMProcessor | None = None) -> ConsolidationResult:
        """Run all consolidation operations: decay, contradiction detection, merging.
        Contradiction detection and merging require an LLM and are no-ops without one."""
        # Step 1: Temporal decay (always runs)
        decayed = apply_decay(self._engine)

        # Step 2: Contradiction detection (LLM-dependent)
        classifier = LLMContradictionClassifier(llm) if llm is not None else None
        detector = ContradictionDetector(self._engine, classifier=classifier)
        contradiction_result = detector.detect_and_resolve()

        # Step 3: Memory merging (LLM-dependent)
        merger = MemoryMerger(self._engine, embedder=self._embedder, llm=llm)
        merge_result = merger.merge_similar()

        return ConsolidationResult(
            memories_decayed=decayed,
            contradictions_found=contradiction_result.contradictions_found,
            contradictions_resolved=contradiction_result.contradictions_resolved,
            memories_merged=merge_result.memories_merged,
            summaries_created=merge_result.summaries_created,
            knowledge_reviewed=0,
        )

    def _compute_importance(self, content: str) -> float:
        words = content.split()
        word_count = len(words)
        if word_count < 3:
            return 0.3
        if word_count < 10:
            return 0.5
        if word_count < 30:
            return 0.6
        return 0.7
