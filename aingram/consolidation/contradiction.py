# aingram/consolidation/contradiction.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations

from aingram.processing.protocols import LLMProcessor
from aingram.security.bounds import sanitize_for_prompt
from aingram.storage.engine import StorageEngine
from aingram.types import MemoryEntry

logger = logging.getLogger(__name__)

_SUPERSEDED_IMPORTANCE_FACTOR = 0.3
_MAX_PAIRS_PER_ENTITY = 50

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


class ContradictionDetector:
    def __init__(
        self,
        engine: StorageEngine,
        *,
        llm: LLMProcessor | None = None,
    ) -> None:
        self._engine = engine
        self._llm = llm

    def detect_and_resolve(self) -> ContradictionResult:
        if self._llm is None:
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

        for entry_ids in entity_entries.values():
            pairs_checked = 0
            for id_a, id_b in combinations(entry_ids, 2):
                pair_key = frozenset({id_a, id_b})
                if pair_key in checked or pairs_checked >= _MAX_PAIRS_PER_ENTITY:
                    continue
                checked.add(pair_key)
                pairs_checked += 1

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

        return ContradictionResult(contradictions_found=found, contradictions_resolved=resolved)

    def _check_pair(self, id_a: str, id_b: str) -> tuple[str, MemoryEntry] | None:
        entries = self._engine.get_entries_by_ids([id_a, id_b])
        if len(entries) != 2:
            return None

        # Build a dict for reliable lookup (get_entries_by_ids has no ORDER BY)
        by_id = {e.entry_id: e for e in entries}
        entry_a, entry_b = by_id[id_a], by_id[id_b]

        # Skip entry-type pairs where supersession is natural
        type_pair = frozenset({str(entry_a.entry_type), str(entry_b.entry_type)})
        if type_pair in _NATURAL_SUPERSESSION_PAIRS:
            return None

        prompt = CONTRADICTION_USER_PROMPT.format(
            text_a=sanitize_for_prompt(entry_a.content),
            text_b=sanitize_for_prompt(entry_b.content),
        )
        try:
            raw = self._llm.complete(prompt, system=CONTRADICTION_SYSTEM_PROMPT)
            data = json.loads(raw)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None
        if not isinstance(data.get('contradicts'), bool):
            return None
        if not data['contradicts']:
            return None

        idx = data.get('superseded_index')
        if idx not in (0, 1):
            return None
        ordered = [entry_a, entry_b]
        return ordered[idx].entry_id, ordered[idx]
