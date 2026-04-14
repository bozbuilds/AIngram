from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aingram.capture.types import CaptureRecord


def _default_redaction_patterns() -> list[str]:
    return [
        r'sk-[a-zA-Z0-9]{20,}',
        r'ghp_[a-zA-Z0-9]{36}',
        r'AKIA[A-Z0-9]{16}',
        r'xoxb-[0-9]+-[a-zA-Z0-9]+',
    ]


@dataclass
class ToolConfig:
    enabled: bool = True
    container_tag: str = ''


@dataclass
class AiderToolConfig(ToolConfig):
    history_file_pattern: str = '**/.aider.llm.history'
    watch_directories: list[str] = field(default_factory=lambda: ['~/projects'])


@dataclass
class ChatGPTToolConfig(ToolConfig):
    enabled: bool = False
    import_directory: str = '~/.aingram/chatgpt-imports/'


def _default_tool_configs() -> dict[str, ToolConfig]:
    return {
        'claude_code': ToolConfig(container_tag='claude_code'),
        'cursor': ToolConfig(container_tag='cursor'),
        'gemini': ToolConfig(container_tag='gemini'),
        'aider': AiderToolConfig(container_tag='aider'),
        'copilot': ToolConfig(container_tag='copilot'),
        'cline': ToolConfig(container_tag='cline'),
        'chatgpt': ChatGPTToolConfig(container_tag='chatgpt'),
    }


@dataclass
class CaptureConfig:
    enabled: bool = False
    host: str = '127.0.0.1'
    port: int = 7749
    queue_db_path: str = '~/.aingram/capture_queue.db'
    memory_mode: str = 'unified'
    poll_interval: float = 0.5
    drain_batch_size: int = 10
    redaction_patterns: list[str] = field(default_factory=_default_redaction_patterns)
    tools: dict[str, ToolConfig] = field(default_factory=_default_tool_configs)


def resolve_container_tag(record: CaptureRecord, config: CaptureConfig) -> str:
    tool_cfg = config.tools.get(record.source_tool)
    if config.memory_mode == 'unified':
        return 'aingram:unified'
    return tool_cfg.container_tag if tool_cfg else record.source_tool
