"""Knowledge synthesis — distill durable principles from completed reasoning chains."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aingram.processing.protocols import Embedder, LLMProcessor
    from aingram.storage.engine import StorageEngine

from aingram.security.bounds import sanitize_for_prompt

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = (
    'You are a knowledge distillation system. Given the key findings from multiple '
    'completed experiments, synthesize ONE concise, durable principle that captures '
    'the most important insight. Output ONLY the principle text, nothing else.'
)

SYNTHESIS_USER_PROMPT = (
    'The following experiments have been completed:\n\n{chain_summaries}\n\n'
    'Synthesize the key principle learned from these experiments.'
)

_MIN_CHAINS_FOR_SYNTHESIS = 2
_SYNTHESIS_CONFIDENCE = 0.8
_CLUSTER_SIMILARITY_THRESHOLD = 0.75


def extract_text(content: str) -> str:
    """Extract human-readable text from canonical JSON content."""
    try:
        return json.loads(content).get('text', content)
    except (json.JSONDecodeError, AttributeError):
        return content


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero-norm vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    dot = float(np.dot(a_arr, b_arr))
    norm = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return dot / norm if norm > 0 else 0.0


def _cluster_chains(
    embeddings: list[tuple[str, list[float]]],
) -> list[list[str]]:
    """Group chain IDs by embedding cosine similarity.

    Returns list of clusters, each a list of chain IDs.
    Chains are expected in created_at ASC order for determinism.
    """
    clusters: list[tuple[list[str], list[float]]] = []

    for chain_id, embedding in embeddings:
        best_cluster = None
        best_sim = -1.0
        for i, (_, centroid) in enumerate(clusters):
            sim = _cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = i

        if best_cluster is not None and best_sim >= _CLUSTER_SIMILARITY_THRESHOLD:
            cluster_ids, centroid = clusters[best_cluster]
            old_n = len(cluster_ids)
            new_centroid = [(c * old_n + e) / (old_n + 1) for c, e in zip(centroid, embedding)]
            cluster_ids.append(chain_id)
            clusters[best_cluster] = (cluster_ids, new_centroid)
        else:
            clusters.append(([chain_id], list(embedding)))

    return [ids for ids, _ in clusters if len(ids) >= _MIN_CHAINS_FOR_SYNTHESIS]


@dataclass
class SynthesisResult:
    knowledge_synthesized: int
    chains_analyzed: int
    touched_knowledge_ids: list[str] = field(default_factory=list)


class KnowledgeSynthesizer:
    """Scan completed reasoning chains, cluster by topic, and distill knowledge items."""

    def __init__(
        self,
        engine: StorageEngine,
        *,
        llm: LLMProcessor | None = None,
        embedder: Embedder,
        session_id: str,
        fallback_to_single_cluster: bool = False,
        require_quorum: bool = True,
    ) -> None:
        self._engine = engine
        self._llm = llm
        self._embedder = embedder
        self._session_id = session_id
        self._fallback_to_single_cluster = fallback_to_single_cluster
        self._require_quorum = require_quorum
        self._touched_ids: list[str] = []

    def synthesize(self) -> SynthesisResult:
        """Embed completed chains, cluster by similarity, synthesize per cluster.

        No-op without LLM. When fallback_to_single_cluster=True and no pair
        exceeds the similarity threshold, all chains are grouped into one cluster.
        """
        self._touched_ids = []
        if self._llm is None:
            return SynthesisResult(knowledge_synthesized=0, chains_analyzed=0)

        completed = self._engine.get_completed_chains()
        if len(completed) < _MIN_CHAINS_FOR_SYNTHESIS:
            return SynthesisResult(knowledge_synthesized=0, chains_analyzed=0)

        chain_texts: list[tuple[str, str]] = []
        for chain in completed:
            entries = self._engine.get_entries_by_chain(chain.chain_id)
            if not entries:
                continue
            text = self._build_chain_text(chain, entries)
            chain_texts.append((chain.chain_id, text))

        if len(chain_texts) < _MIN_CHAINS_FOR_SYNTHESIS:
            return SynthesisResult(
                knowledge_synthesized=0,
                chains_analyzed=len(chain_texts),
            )

        texts = [text for _, text in chain_texts]
        vectors = self._embedder.embed_batch(texts)
        embeddings = [(cid, vec) for (cid, _), vec in zip(chain_texts, vectors)]

        clusters = _cluster_chains(embeddings)
        if (
            not clusters
            and self._fallback_to_single_cluster
            and len(embeddings) >= _MIN_CHAINS_FOR_SYNTHESIS
        ):
            # No pairs exceeded threshold — fall back to grouping all chains together.
            # Only used when explicitly opted in (e.g. test embedders with poor similarity).
            clusters = [[cid for cid, _ in embeddings]]

        if not clusters:
            return SynthesisResult(
                knowledge_synthesized=0,
                chains_analyzed=len(chain_texts),
            )

        synthesized = 0
        for cluster_chain_ids in clusters:
            if self._require_quorum:
                sessions: set[str] = set()
                for cid in cluster_chain_ids:
                    chain = self._engine.get_chain(cid)
                    if chain is not None:
                        sessions.add(chain.created_by_session)
                if len(sessions) < 2:
                    logger.debug(
                        'Skipping cluster with %d chains — only %d session(s)',
                        len(cluster_chain_ids),
                        len(sessions),
                    )
                    continue
            if self._synthesize_cluster(cluster_chain_ids, self._touched_ids):
                synthesized += 1

        return SynthesisResult(
            knowledge_synthesized=synthesized,
            chains_analyzed=len(chain_texts),
            touched_knowledge_ids=list(dict.fromkeys(self._touched_ids)),
        )

    def _build_chain_text(self, chain, entries) -> str:
        """Build the embedding representation string for a chain."""
        last_relevant = None
        for entry in reversed(entries):
            if str(entry.entry_type) in ('result', 'lesson'):
                last_relevant = entry
                break
        if last_relevant is None:
            last_relevant = entries[-1]

        content_text = extract_text(last_relevant.content)
        return f'{chain.title} — outcome: {chain.outcome}. {content_text}'

    def _find_matching_item(self, cluster_chain_ids: list[str]) -> dict | None:
        """Find an existing knowledge item whose supporting_chains overlaps the cluster.
        O(n) scan over all items — acceptable at current scale (< 100 items).
        If this becomes a bottleneck, add engine.get_knowledge_items_by_chain()."""
        items = self._engine.get_knowledge_items()
        cluster_set = set(cluster_chain_ids)
        for item in items:
            if cluster_set & set(item['supporting_chains']):
                return item
        return None

    def _synthesize_cluster(self, cluster_chain_ids: list[str], touched_ids: list[str]) -> bool:
        """Synthesize or update a knowledge item for one cluster. Returns True on success.
        Note: O(n) queries per chain (get_entries_by_chain + get_chain). Acceptable at
        current scale (< 20 chains per cluster). Batch methods can optimize if needed."""
        existing = self._find_matching_item(cluster_chain_ids)

        if existing:
            merged_chains = sorted(set(existing['supporting_chains']) | set(cluster_chain_ids))
        else:
            merged_chains = cluster_chain_ids

        chain_summaries = []
        for chain_id in merged_chains:
            entries = self._engine.get_entries_by_chain(chain_id)
            if not entries:
                continue
            chain = self._engine.get_chain(chain_id)
            if chain is None:
                continue
            safe_entries = '\n'.join(
                f'  - [{e.entry_type}] {extract_text(e.content)}' for e in entries[:10]
            )
            summary = f'**{chain.title}** (outcome: {chain.outcome}):\n'
            summary += sanitize_for_prompt(safe_entries, max_length=2000)
            chain_summaries.append(summary)

        if not chain_summaries:
            return False

        prompt = SYNTHESIS_USER_PROMPT.format(
            chain_summaries='\n\n'.join(chain_summaries),
        )

        try:
            principle = self._llm.complete(prompt, system=SYNTHESIS_SYSTEM_PROMPT)
        except Exception:
            logger.warning('Knowledge synthesis LLM call failed', exc_info=True)
            return False

        if not principle or not principle.strip():
            return False
        principle = principle.strip()
        if len(principle) > 500:
            principle = principle[:500]
        if principle.startswith('{') or principle.startswith('['):
            logger.warning('Synthesis returned JSON-like output, rejecting')
            return False
        lower = principle.lower()
        if any(
            marker in lower for marker in ('system:', 'assistant:', 'ignore previous', 'ignore all')
        ):
            logger.warning('Synthesis output contains injection markers, rejecting')
            return False

        if existing:
            kid = existing['knowledge_id']
            self._engine.update_knowledge_item(
                kid,
                principle=principle.strip(),
                supporting_chains=merged_chains,
                confidence=_SYNTHESIS_CONFIDENCE,
            )
            touched_ids.append(kid)
        else:
            kid = self._engine.store_knowledge_item(
                principle=principle.strip(),
                supporting_chains=merged_chains,
                confidence=_SYNTHESIS_CONFIDENCE,
                session_id=self._session_id,
            )
            touched_ids.append(kid)

        logger.info(
            'Synthesized knowledge from %d chains: %s',
            len(merged_chains),
            principle.strip()[:80],
        )
        return True

    def synthesize_entry_cluster(self, entry_ids: list[str]) -> bool:
        """Synthesize one knowledge item from entry-level cluster IDs (HDBSCAN path)."""
        if self._llm is None:
            return False

        entries = self._engine.get_entries_by_ids(entry_ids)
        if not entries:
            return False

        entry_texts: list[str] = []
        for e in entries[:10]:
            text = extract_text(e.content)
            entry_texts.append(f'  - [{e.entry_type}] {sanitize_for_prompt(text, max_length=500)}')

        if not entry_texts:
            return False

        prompt = SYNTHESIS_USER_PROMPT.format(
            chain_summaries='Cluster of related observations:\n' + '\n'.join(entry_texts),
        )

        try:
            principle = self._llm.complete(prompt, system=SYNTHESIS_SYSTEM_PROMPT)
        except Exception:
            logger.warning('Entry cluster synthesis LLM call failed', exc_info=True)
            return False

        if not principle or not principle.strip():
            return False
        principle = principle.strip()
        if len(principle) > 500:
            principle = principle[:500]
        if principle.startswith('{') or principle.startswith('['):
            return False
        lower = principle.lower()
        if any(m in lower for m in ('system:', 'assistant:', 'ignore previous', 'ignore all')):
            return False

        tagged_ids = [f'entry:{eid}' for eid in entry_ids]
        kid = self._engine.store_knowledge_item(
            principle=principle,
            supporting_chains=tagged_ids,
            confidence=_SYNTHESIS_CONFIDENCE,
            session_id=self._session_id,
        )
        self._touched_ids.append(kid)
        logger.info('Synthesized knowledge from entry cluster (%d ids)', len(entry_ids))
        return True
