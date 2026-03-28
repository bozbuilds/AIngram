# aingram/models/manager.py
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / '.aingram' / 'models'


class ModelManager:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, model_name: str) -> Path:
        return self.cache_dir / model_name

    def is_downloaded(self, model_name: str) -> bool:
        model_dir = self.model_path(model_name)
        if not model_dir.exists():
            return False
        # Check for the required ONNX model file, not just any file
        return (model_dir / 'model.onnx').exists()
