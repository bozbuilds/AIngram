# tests/test_telemetry.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aingram.telemetry import maybe_send_cli_telemetry


def test_maybe_send_skips_when_disabled():
    with patch('aingram.telemetry.httpx.Client') as client_cls:
        maybe_send_cli_telemetry(command='add', enabled=False)
    client_cls.assert_not_called()


def test_maybe_send_skips_when_no_command():
    with patch('aingram.telemetry.httpx.Client') as client_cls:
        maybe_send_cli_telemetry(command=None, enabled=True)
    client_cls.assert_not_called()


def test_maybe_send_posts_json(tmp_path, monkeypatch):
    monkeypatch.setenv('AINGRAM_TELEMETRY_ENDPOINT', 'http://test.local/ingest')
    home = tmp_path / 'h'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post = MagicMock(return_value=mock_response)

    with patch('aingram.telemetry.httpx.Client', return_value=mock_client):
        maybe_send_cli_telemetry(command='status', enabled=True)

    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert args[0] == 'http://test.local/ingest'
    body = kwargs['json']
    assert body['kind'] == 'cli_invocation'
    assert body['command'] == 'status'
    assert body['schema_version'] == 1
    assert 'install_id' in body
    assert 'aingram_version' in body
