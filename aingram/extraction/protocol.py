from __future__ import annotations

from typing import Protocol, runtime_checkable

from aingram.types import ExtractionResult


@runtime_checkable
class MemoryExtractor(Protocol):
    """Protocol for memory metadata extraction."""

    def extract(self, text: str) -> ExtractionResult: ...
