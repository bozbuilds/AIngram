# aingram/telemetry.py
"""Anonymous CLI usage telemetry (opt-out). Never sends memory content or file paths."""

from __future__ import annotations

import logging
import os
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TELEMETRY_URL = 'https://api.aingram.dev/v1/telemetry'
TELEMETRY_ID_PATH = Path.home() / '.aingram' / 'telemetry_id'


def _package_version() -> str:
    try:
        return version('aingram')
    except PackageNotFoundError:
        return '0.0.0'


def _read_or_create_install_id() -> str:
    try:
        if TELEMETRY_ID_PATH.exists():
            raw = TELEMETRY_ID_PATH.read_text(encoding='utf-8').strip()
            if raw:
                return raw
    except OSError:
        pass
    install_id = str(uuid.uuid4())
    try:
        TELEMETRY_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        TELEMETRY_ID_PATH.write_text(install_id + '\n', encoding='utf-8')
    except OSError as e:
        logger.debug('Could not persist telemetry_id: %s', e)
    return install_id


def _telemetry_url() -> str:
    raw = os.environ.get('AINGRAM_TELEMETRY_ENDPOINT', DEFAULT_TELEMETRY_URL).strip()
    return raw or DEFAULT_TELEMETRY_URL


def maybe_send_cli_telemetry(*, command: str | None, enabled: bool) -> None:
    """POST a single anonymous CLI event. Swallows all errors; never raises."""
    if not enabled or not command:
        return
    payload = {
        'schema_version': 1,
        'kind': 'cli_invocation',
        'install_id': _read_or_create_install_id(),
        'command': command,
        'aingram_version': _package_version(),
    }
    url = _telemetry_url()
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.post(url, json=payload, headers={'Content-Type': 'application/json'})
            r.raise_for_status()
    except Exception as e:
        logger.debug('Telemetry send skipped: %s', e)
