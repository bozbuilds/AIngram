from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache

from aingram.capture.config import CaptureConfig, resolve_container_tag
from aingram.capture.types import CaptureRecord


@lru_cache(maxsize=64)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def apply_filters(record: CaptureRecord, config: CaptureConfig) -> CaptureRecord | None:
    if '@nocapture' in (record.user_prompt or ''):
        return None

    user_prompt = record.user_prompt or ''
    assistant_response = record.assistant_response or ''
    for pattern in config.redaction_patterns:
        compiled = _compile_pattern(pattern)
        user_prompt = compiled.sub('[REDACTED]', user_prompt)
        assistant_response = compiled.sub('[REDACTED]', assistant_response)

    container_tag = resolve_container_tag(record, config)

    return replace(
        record,
        user_prompt=user_prompt,
        assistant_response=assistant_response if record.assistant_response else None,
        container_tag=container_tag,
    )
