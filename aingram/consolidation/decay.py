# aingram/consolidation/decay.py
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aingram.storage.engine import StorageEngine

logger = logging.getLogger(__name__)

MINIMUM_IMPORTANCE = 0.01
_BASE_STABILITY = 168.0
_REPETITION_BONUS = 0.4
_DECAY_THRESHOLD = 0.001


def compute_stability(access_count: int, importance: float) -> float:
    """Compute stability factor (in hours) for the Ebbinghaus curve."""
    repetition_factor = 1.0 + _REPETITION_BONUS * access_count
    importance_factor = 0.5 + importance
    return _BASE_STABILITY * repetition_factor * importance_factor


def compute_decay(
    importance: float,
    hours_since_access: float,
    access_count: int,
) -> float:
    """Apply Ebbinghaus forgetting curve to an importance score."""
    hours = max(hours_since_access, 0)
    if hours == 0:
        return importance
    stability = compute_stability(access_count, importance)
    retention = math.exp(-hours / stability)
    return max(MINIMUM_IMPORTANCE, importance * retention)


def apply_decay(engine: StorageEngine, *, limit: int = 5000) -> int:
    """Apply decay to entry importance scores. Returns count updated."""
    entries = engine.get_entries_for_decay(limit=limit)
    if not entries:
        return 0

    now = datetime.now(UTC)
    updates: list[tuple[str, float]] = []

    for entry in entries:
        ref_str = entry.accessed_at or entry.created_at
        reference = datetime.fromisoformat(ref_str)
        age_hours = (now - reference).total_seconds() / 3600

        new_importance = compute_decay(
            importance=entry.importance,
            hours_since_access=age_hours,
            access_count=entry.access_count,
        )

        if abs(new_importance - entry.importance) > _DECAY_THRESHOLD:
            updates.append((entry.entry_id, new_importance))

    if not updates:
        return 0
    return engine.batch_update_entry_importance(updates)
