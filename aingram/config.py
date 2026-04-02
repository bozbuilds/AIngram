# aingram/config.py
from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = Path.home() / '.aingram' / 'models'
_DEFAULT_CONFIG_PATH = Path.home() / '.aingram' / 'config.toml'


@dataclass
class AIngramConfig:
    """Layered configuration defaults. Merge order: file → env → MemoryStore kwargs."""

    models_dir: Path = _DEFAULT_MODELS_DIR
    embedding_dim: int = 768
    worker_enabled: bool = True
    llm_url: str = 'http://localhost:11434'
    llm_model: str = 'mistral'
    log_level: str = 'INFO'
    consolidation_interval: int | None = None
    extractor_mode: str = 'none'  # 'none', 'sonnet', 'local'
    extractor_model: str = 'aingram-extractor'
    onnx_provider: str | None = None  # None=auto, 'cuda', 'npu', 'cpu'
    telemetry_enabled: bool = True  # opt-out: set false to disable anonymous usage telemetry
    fts_prefilter_threshold: int = 50

    def __post_init__(self) -> None:
        if self.extractor_mode not in ('none', 'sonnet', 'local'):
            raise ValueError(
                f'extractor_mode must be none, sonnet, or local — got {self.extractor_mode!r}'
            )

    def apply_log_level(self) -> None:
        level_name = self.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.getLogger('aingram').setLevel(level)


def _coerce_value(field_name: str, raw: Any) -> Any:
    if field_name == 'models_dir':
        if isinstance(raw, Path):
            return raw.expanduser()
        if isinstance(raw, str):
            return Path(raw).expanduser()
    if field_name == 'embedding_dim':
        return int(raw)
    if field_name == 'fts_prefilter_threshold' and raw is not None:
        return int(raw)
    if field_name in ('worker_enabled', 'telemetry_enabled'):
        return bool(raw)
    if field_name == 'consolidation_interval' and raw is not None:
        return int(raw)
    if field_name in ('extractor_mode', 'extractor_model'):
        return str(raw)
    if field_name == 'onnx_provider':
        if raw is None or raw == 'None' or raw == '':
            return None
        return str(raw)
    return raw


def _merge_toml_into(config: AIngramConfig, data: dict[str, Any]) -> AIngramConfig:
    allowed = {f.name for f in fields(AIngramConfig)}
    updates: dict[str, Any] = {}
    for key, raw in data.items():
        if key not in allowed:
            logger.debug('Ignoring unknown config key %r', key)
            continue
        updates[key] = _coerce_value(key, raw)
    return replace(config, **updates)


def _merge_env_into(config: AIngramConfig, env: dict[str, str]) -> AIngramConfig:
    updates: dict[str, Any] = {}
    if v := env.get('AINGRAM_MODELS_DIR'):
        updates['models_dir'] = Path(v).expanduser()
    if v := env.get('AINGRAM_EMBEDDING_DIM'):
        updates['embedding_dim'] = int(v)
    if v := env.get('AINGRAM_LLM_URL'):
        updates['llm_url'] = v
    if v := env.get('AINGRAM_LLM_MODEL'):
        updates['llm_model'] = v
    if v := env.get('AINGRAM_LOG_LEVEL'):
        updates['log_level'] = v
    if v := env.get('AINGRAM_WORKER_ENABLED'):
        updates['worker_enabled'] = v.strip().lower() in ('1', 'true', 'yes', 'on')
    if v := env.get('AINGRAM_CONSOLIDATION_INTERVAL'):
        updates['consolidation_interval'] = int(v)
    if v := env.get('AINGRAM_EXTRACTOR_MODE'):
        updates['extractor_mode'] = v
    if v := env.get('AINGRAM_EXTRACTOR_MODEL'):
        updates['extractor_model'] = v
    if v := env.get('AINGRAM_ONNX_PROVIDER'):
        updates['onnx_provider'] = v if v.lower() != 'none' else None
    if v := env.get('AINGRAM_TELEMETRY_ENABLED'):
        updates['telemetry_enabled'] = v.strip().lower() in ('1', 'true', 'yes', 'on')
    if v := env.get('AINGRAM_FTS_PREFILTER_THRESHOLD'):
        updates['fts_prefilter_threshold'] = int(v)
    return replace(config, **updates) if updates else config


def _merge_overrides(config: AIngramConfig, overrides: dict[str, Any]) -> AIngramConfig:
    if not overrides:
        return config
    cleaned: dict[str, Any] = {}
    allowed = {f.name for f in fields(AIngramConfig)}
    for key, raw in overrides.items():
        if key not in allowed or raw is None:
            continue
        cleaned[key] = _coerce_value(key, raw)
    return replace(config, **cleaned)


def load_merged_config(
    *,
    config_file: Path | None = None,
    env: dict[str, str] | None = None,
    **overrides: Any,
) -> AIngramConfig:
    """Load config: defaults → optional TOML file → env → explicit overrides (highest)."""
    config = AIngramConfig()

    paths: list[Path] = []
    if config_file is not None:
        paths.append(config_file)
    elif _DEFAULT_CONFIG_PATH.exists():
        paths.append(_DEFAULT_CONFIG_PATH)

    for path in paths:
        if not path.exists():
            continue
        try:
            raw = path.read_bytes()
            data = tomllib.loads(raw.decode())
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
            logger.warning('Could not read config file %s: %s', path, e)
            continue
        config = _merge_toml_into(config, data)

    env_map = env if env is not None else os.environ
    config = _merge_env_into(config, env_map)
    config = _merge_overrides(config, overrides)
    return config
