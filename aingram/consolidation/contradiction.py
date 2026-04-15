# aingram/consolidation/contradiction.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations

from aingram.processing.protocols import ContradictionClassifier, LLMProcessor
from aingram.security.bounds import sanitize_for_prompt
from aingram.storage.engine import StorageEngine
from aingram.types import ContradictionVerdict, MemoryEntry

logger = logging.getLogger(__name__)

_SUPERSEDED_IMPORTANCE_FACTOR = 0.3
_MAX_PAIRS_PER_ENTITY = 50
_MAX_TOTAL_PAIRS = 200

# Entry type pairs where supersession is natural (not a contradiction)
# A result naturally supersedes a hypothesis — that's the scientific method.
_NATURAL_SUPERSESSION_PAIRS = {
    frozenset({'hypothesis', 'result'}),
    frozenset({'hypothesis', 'lesson'}),
    frozenset({'method', 'result'}),
    frozenset({'decision', 'result'}),
}

CONTRADICTION_SYSTEM_PROMPT = (
    'You are a fact-checker analyzing statements for contradictions. '
    'Compare the two statements and determine if they contradict each other. '
    'Output ONLY valid JSON with keys: contradicts (bool), superseded_index (0 or 1, '
    'which statement is outdated). If no contradiction, just: {"contradicts": false}'
)

CONTRADICTION_USER_PROMPT = (
    'Statement 0: "{text_a}"\nStatement 1: "{text_b}"\n\nDo these statements contradict each other?'
)


@dataclass
class ContradictionResult:
    contradictions_found: int
    contradictions_resolved: int


class LLMContradictionClassifier:
    """Contradiction classifier using an LLM via the LLMProcessor protocol."""

    def __init__(self, llm: LLMProcessor) -> None:
        self._llm = llm

    def classify(self, text_a: str, text_b: str) -> ContradictionVerdict:
        prompt = CONTRADICTION_USER_PROMPT.format(text_a=text_a, text_b=text_b)
        try:
            raw = self._llm.complete(prompt, system=CONTRADICTION_SYSTEM_PROMPT)
            data = json.loads(raw)
        except Exception:
            return ContradictionVerdict(contradicts=False, confidence=0.0)

        if not isinstance(data, dict):
            return ContradictionVerdict(contradicts=False, confidence=0.0)
        if not isinstance(data.get('contradicts'), bool):
            return ContradictionVerdict(contradicts=False, confidence=0.0)

        return ContradictionVerdict(
            contradicts=data['contradicts'],
            confidence=1.0,
            superseded_index=data.get('superseded_index'),
        )


class ContradictionDetector:
    def __init__(
        self,
        engine: StorageEngine,
        *,
        classifier: ContradictionClassifier | None = None,
    ) -> None:
        self._engine = engine
        self._classifier = classifier

    def detect_and_resolve(self) -> ContradictionResult:
        if self._classifier is None:
            return ContradictionResult(contradictions_found=0, contradictions_resolved=0)

        entity_pairs = self._engine.get_entity_entry_pairs()
        if not entity_pairs:
            return ContradictionResult(contradictions_found=0, contradictions_resolved=0)

        # Group entry_ids by entity
        entity_entries: dict[str, list[str]] = {}
        for entity_id, entry_id in entity_pairs:
            entity_entries.setdefault(entity_id, []).append(entry_id)

        found = 0
        resolved = 0
        checked: set[frozenset[str]] = set()
        total_checked = 0

        for entry_ids in entity_entries.values():
            if total_checked >= _MAX_TOTAL_PAIRS:
                break
            pairs_checked = 0
            for id_a, id_b in combinations(entry_ids, 2):
                if total_checked >= _MAX_TOTAL_PAIRS:
                    break
                pair_key = frozenset({id_a, id_b})
                if pair_key in checked or pairs_checked >= _MAX_PAIRS_PER_ENTITY:
                    continue
                checked.add(pair_key)
                pairs_checked += 1
                total_checked += 1

                result = self._check_pair(id_a, id_b)
                if result is not None:
                    superseded_id, superseded_entry = result
                    found += 1
                    self._engine.batch_update_entry_importance(
                        [
                            (
                                superseded_id,
                                superseded_entry.importance * _SUPERSEDED_IMPORTANCE_FACTOR,
                            )
                        ]
                    )
                    resolved += 1

        if total_checked >= _MAX_TOTAL_PAIRS:
            logger.info(
                'Contradiction detection hit global cap (%d pairs); '
                'remaining pairs deferred to next consolidation run',
                _MAX_TOTAL_PAIRS,
            )

        return ContradictionResult(contradictions_found=found, contradictions_resolved=resolved)

    def _check_pair(self, id_a: str, id_b: str) -> tuple[str, MemoryEntry] | None:
        entries = self._engine.get_entries_by_ids([id_a, id_b])
        if len(entries) != 2:
            return None

        by_id = {e.entry_id: e for e in entries}
        entry_a, entry_b = by_id[id_a], by_id[id_b]

        # Skip entry-type pairs where supersession is natural
        type_pair = frozenset({str(entry_a.entry_type), str(entry_b.entry_type)})
        if type_pair in _NATURAL_SUPERSESSION_PAIRS:
            return None

        verdict = self._classifier.classify(
            sanitize_for_prompt(entry_a.content),
            sanitize_for_prompt(entry_b.content),
        )

        if not verdict.contradicts:
            return None

        if verdict.superseded_index is not None:
            if verdict.superseded_index not in (0, 1):
                return None
            ordered = [entry_a, entry_b]
            superseded = ordered[verdict.superseded_index]
            return superseded.entry_id, superseded

        # Recency fallback: older entry is superseded
        if entry_a.created_at <= entry_b.created_at:
            return entry_a.entry_id, entry_a
        return entry_b.entry_id, entry_b
