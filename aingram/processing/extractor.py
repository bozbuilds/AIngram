# aingram/processing/extractor.py
from __future__ import annotations

import logging

from aingram.exceptions import ModelNotFoundError
from aingram.types import ExtractedEntity

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = ['person', 'organization', 'location', 'project', 'technology']
DEFAULT_MODEL = 'knowledgator/gliner-multitask-large-v0.5'


class GlinerExtractor:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        entity_types: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self.default_entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.threshold = threshold
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from gliner import GLiNER

            logger.info('Loading GLiNER model %s...', self._model_name)
            self._model = GLiNER.from_pretrained(self._model_name)
            logger.info('GLiNER model loaded')
        except ImportError as e:
            raise ModelNotFoundError(
                'gliner not installed. Install with: pip install aingram[extraction]'
            ) from e
        except Exception as e:
            raise ModelNotFoundError(f'Failed to load GLiNER model {self._model_name}: {e}') from e

    def extract(self, text: str, entity_types: list[str]) -> list[ExtractedEntity]:
        if not text.strip():
            return []
        self._ensure_loaded()
        entities = self._model.predict_entities(text, entity_types, threshold=self.threshold)
        # Deduplicate by (lowercased name, label)
        seen: set[tuple[str, str]] = set()
        result: list[ExtractedEntity] = []
        for e in entities:
            key = (e['text'].lower(), e['label'])
            if key not in seen:
                seen.add(key)
                result.append(
                    ExtractedEntity(
                        name=e['text'],
                        entity_type=e['label'],
                        score=e['score'],
                    )
                )
        return result
