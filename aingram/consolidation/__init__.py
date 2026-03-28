# aingram/consolidation/__init__.py
from aingram.consolidation.contradiction import ContradictionDetector
from aingram.consolidation.decay import apply_decay
from aingram.consolidation.knowledge import KnowledgeSynthesizer
from aingram.consolidation.merger import MemoryMerger

__all__ = [
    'ContradictionDetector',
    'KnowledgeSynthesizer',
    'MemoryMerger',
    'apply_decay',
]
