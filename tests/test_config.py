# tests/test_config.py
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from aingram.config import AIngramConfig, load_merged_config


def test_defaults():
    c = AIngramConfig()
    assert c.embedding_dim == 768
    assert c.worker_enabled is True
    assert c.telemetry_enabled is True
    assert c.models_dir is not None
    assert c.llm_url == 'http://localhost:11434'
    assert c.llm_model == 'mistral'
    assert c.log_level == 'INFO'
    assert c.consolidation_interval is None


def test_env_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv('AINGRAM_MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setenv('AINGRAM_LLM_URL', 'http://example:11434')
    monkeypatch.setenv('AINGRAM_LLM_MODEL', 'llama3')
    monkeypatch.setenv('AINGRAM_LOG_LEVEL', 'DEBUG')
    monkeypatch.delenv('AINGRAM_EMBEDDING_DIM', raising=False)
    c = load_merged_config(config_file=None, env=os_environ_only())
    assert c.models_dir == Path(tmp_path / 'models').expanduser().resolve()
    assert c.llm_url == 'http://example:11434'
    assert c.llm_model == 'llama3'
    assert c.log_level == 'DEBUG'


def test_env_embedding_dim(monkeypatch):
    monkeypatch.setenv('AINGRAM_EMBEDDING_DIM', '256')
    c = load_merged_config(config_file=None, env=os_environ_only())
    assert c.embedding_dim == 256


def test_telemetry_disabled_via_env(monkeypatch):
    monkeypatch.setenv('AINGRAM_TELEMETRY_ENABLED', 'false')
    c = load_merged_config(config_file=None, env=os_environ_only())
    assert c.telemetry_enabled is False


def test_fts_prefilter_threshold_default():
    c = AIngramConfig()
    assert c.fts_prefilter_threshold == 50


def test_file_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv('AINGRAM_MODELS_DIR', raising=False)
    models_path = tmp_path / 'my-models'
    cfg = tmp_path / 'config.toml'
    cfg.write_text(
        f'models_dir = {json.dumps(str(models_path))}\n'
        'embedding_dim = 256\n'
        'worker_enabled = false\n',
        encoding='utf-8',
    )
    c = load_merged_config(config_file=cfg, env={})
    assert c.models_dir == models_path
    assert c.embedding_dim == 256
    assert c.worker_enabled is False


def test_env_overrides_file(tmp_path, monkeypatch):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('embedding_dim = 256\n', encoding='utf-8')
    monkeypatch.setenv('AINGRAM_EMBEDDING_DIM', '768')
    c = load_merged_config(config_file=cfg, env=os_environ_only())
    assert c.embedding_dim == 768


def test_models_dir_path_override_expands_tilde(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    c = load_merged_config(config_file=None, env={}, models_dir=Path('~/models'))
    assert c.models_dir == home / 'models'


def test_constructor_overrides_all(tmp_path, monkeypatch):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('embedding_dim = 256\n', encoding='utf-8')
    monkeypatch.setenv('AINGRAM_EMBEDDING_DIM', '512')
    c = load_merged_config(
        config_file=cfg,
        env=os_environ_only(),
        embedding_dim=128,
    )
    assert c.embedding_dim == 128


def test_missing_config_file_ignored(tmp_path):
    missing = tmp_path / 'nope.toml'
    c = load_merged_config(config_file=missing, env={})
    assert c.embedding_dim == 768


def test_unknown_toml_keys_ignored(tmp_path):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('future_option = true\nembedding_dim = 256\n', encoding='utf-8')
    c = load_merged_config(config_file=cfg, env={})
    assert c.embedding_dim == 256


def test_apply_log_level_sets_aingram_namespace():
    logging.getLogger('aingram').setLevel(logging.WARNING)
    c = AIngramConfig(log_level='DEBUG')
    c.apply_log_level()
    assert logging.getLogger('aingram').level == logging.DEBUG
    logging.getLogger('aingram').setLevel(logging.NOTSET)


def os_environ_only() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if k.startswith('AINGRAM_')}


@pytest.fixture(autouse=True)
def clear_aingram_env(monkeypatch):
    import os

    for key in list(os.environ.keys()):
        if key.startswith('AINGRAM_'):
            monkeypatch.delenv(key, raising=False)
