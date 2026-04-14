from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaptureRecord:
    source_tool: str
    session_id: str
    user_prompt: str
    timestamp: float
    turn_number: int | None = None
    assistant_response: str | None = None
    tool_calls: str | None = None
    model: str | None = None
    project_path: str | None = None
    metadata: str | None = None
    container_tag: str | None = None
    state: str = 'pending'


@dataclass
class ToolHealth:
    tool_name: str
    connected: bool
    last_capture: float | None = None
    error: str | None = None
    tier: int = 1
