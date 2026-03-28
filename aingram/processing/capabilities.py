# aingram/processing/capabilities.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    has_embedder: bool = False
    has_classifier: bool = False
    has_extractor: bool = False
    has_llm: bool = False

    def summary(self) -> dict[str, bool]:
        return {
            'embedder': self.has_embedder,
            'classifier': self.has_classifier,
            'extractor': self.has_extractor,
            'llm': self.has_llm,
        }
