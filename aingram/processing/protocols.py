# aingram/processing/protocols.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from aingram.types import ContradictionVerdict, ExtractedEntity, MemoryType


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class MemoryClassifier(Protocol):
    def classify(self, text: str) -> MemoryType: ...


@runtime_checkable
class EntityExtractor(Protocol):
    def extract(self, text: str, entity_types: list[str]) -> list[ExtractedEntity]: ...


@runtime_checkable
class LLMProcessor(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...


@runtime_checkable
class ContradictionClassifier(Protocol):
    def classify(self, text_a: str, text_b: str) -> ContradictionVerdict: ...
