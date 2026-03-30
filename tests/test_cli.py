# tests/test_cli.py
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def disable_cli_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AINGRAM_TELEMETRY_ENABLED', 'false')


def test_cli_help(runner: CliRunner) -> None:
    from aingram.cli import app

    r = runner.invoke(app, ['--help'])
    assert r.exit_code == 0
    assert 'AIngram' in r.stdout


def test_add_uses_memory_store(runner: CliRunner, tmp_path, monkeypatch) -> None:
    calls: list = []

    class MS:
        def __init__(self, db: str, **kw) -> None:
            calls.append(('init', db))

        def remember(self, content: str, **kw) -> str:
            calls.append(('remember', content))
            return 'entry-1'

        def close(self) -> None:
            calls.append('close')

    monkeypatch.setattr('aingram.MemoryStore', MS)
    from aingram.cli import app

    db = tmp_path / 'c.db'
    r = runner.invoke(app, ['--db', str(db), 'add', 'hello'])
    assert r.exit_code == 0
    assert 'entry-1' in r.stdout
    assert ('remember', 'hello') in calls


def test_status_json(runner: CliRunner, tmp_path, monkeypatch) -> None:
    class MS:
        def __init__(self, db: str, **kw) -> None:
            self._db = db

        @property
        def stats(self) -> dict:
            return {
                'entry_count': 0,
                'db_size_bytes': 0,
                'embedding_dim': 768,
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr('aingram.MemoryStore', MS)
    from aingram.cli import app

    r = runner.invoke(app, ['--db', str(tmp_path / 's.db'), 'status'])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert data['entry_count'] == 0
    assert data['embedding_dim'] == 768


def test_compact_without_confirm_exits_error(runner: CliRunner, tmp_path, monkeypatch) -> None:
    class MS:
        def __init__(self, db: str, **kw) -> None:
            pass

        def compact(self, **kw) -> None:
            raise ValueError('compact() is destructive; call with confirm=True')

        def close(self) -> None:
            pass

    monkeypatch.setattr('aingram.MemoryStore', MS)
    from aingram.cli import app

    r = runner.invoke(app, ['--db', str(tmp_path / 'x.db'), 'compact'])
    assert r.exit_code == 1
    assert 'confirm' in r.stderr or 'confirm' in r.stdout
