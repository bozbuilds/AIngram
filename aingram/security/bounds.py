"""Input bounds enforcement and prompt sanitization."""

from __future__ import annotations

import re

from aingram.exceptions import InputBoundsError

_MAX_CONTENT_BYTES = 64 * 1024
_MAX_TITLE_CHARS = 1000
_MAX_LIMIT = 100
_MAX_DEPTH = 10
_MAX_TOKENS = 50_000

# Tool -> parameter -> (cap_value, behavior)
_BOUNDS: dict[str, dict[str, tuple[float, str]]] = {
    'recall': {'limit': (_MAX_LIMIT, 'clamp')},
    'get_related': {'depth': (_MAX_DEPTH, 'clamp')},
    'get_experiment_context': {'max_tokens': (_MAX_TOKENS, 'clamp')},
    'get_due_reviews': {'limit': (_MAX_LIMIT, 'clamp')},
    'remember': {'confidence': (1.0, 'clamp')},
    'create_chain': {},
    'sync': {},
}


class InputBoundsChecker:
    """Validate and clamp MCP tool parameters to safe bounds."""

    def validate(self, tool_name: str, params: dict) -> None:
        _text_key = None
        if tool_name == 'remember' and 'content' in params:
            _text_key = 'content'
        elif tool_name == 'get_surprise' and 'text' in params:
            _text_key = 'text'

        if _text_key is not None:
            content = params[_text_key]
            if isinstance(content, str):
                char_len = len(content)
                if char_len > _MAX_CONTENT_BYTES and content[:_MAX_CONTENT_BYTES]:
                    raise InputBoundsError(
                        f'{_text_key} exceeds maximum size of {_MAX_CONTENT_BYTES} bytes'
                    )
                if char_len > _MAX_CONTENT_BYTES // 4:
                    if len(content.encode('utf-8')) > _MAX_CONTENT_BYTES:
                        raise InputBoundsError(
                            f'{_text_key} exceeds maximum size of {_MAX_CONTENT_BYTES} bytes'
                        )

        if tool_name == 'create_chain' and 'title' in params:
            title = params['title']
            if isinstance(title, str) and len(title) > _MAX_TITLE_CHARS:
                raise InputBoundsError(
                    f'title exceeds maximum length of {_MAX_TITLE_CHARS} characters'
                )

        bounds = _BOUNDS.get(tool_name, {})
        for param_name, (cap, behavior) in bounds.items():
            if param_name not in params:
                continue
            val = params[param_name]
            if val is None:
                continue
            if behavior == 'clamp':
                if param_name == 'confidence':
                    params[param_name] = max(0.0, min(float(val), cap))
                else:
                    params[param_name] = min(int(val), int(cap))


_INJECTION_PATTERNS = [
    re.compile(r'^\s*(system|assistant)\s*:', re.IGNORECASE),
    re.compile(r'^\s*ignore\s+(previous|all)', re.IGNORECASE),
    re.compile(r'^\s*disregard', re.IGNORECASE),
    re.compile(r'^\s*you\s+are\s+now', re.IGNORECASE),
    re.compile(r'^\s*new\s+instructions', re.IGNORECASE),
]


def sanitize_for_prompt(text: str, *, max_length: int = 2000) -> str:
    """Sanitize stored content before injecting into LLM prompts."""
    truncated = text[:max_length]
    lines = truncated.split('\n')
    clean_lines = [line for line in lines if not any(p.match(line) for p in _INJECTION_PATTERNS)]
    cleaned = '\n'.join(clean_lines)
    return f'<user-content>\n{cleaned}\n</user-content>'
