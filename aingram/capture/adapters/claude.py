from __future__ import annotations

import json
import time

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth


class ClaudeCodeAdapter(ToolAdapter):
    # Read-only tool calls produce lookup artifacts, not knowledge worth remembering.
    _SKIP_TOOLS = frozenset({'Read', 'Glob', 'Grep', 'ListDir', 'LS', 'View'})

    tool_name = 'claude_code'

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        session_id = raw.get('session_id', '')
        ts = raw.get('timestamp', time.time())
        event_name = raw.get('hook_event_name', '')
        # Legacy format support: "type" field from direct API calls
        legacy_type = raw.get('type', '')

        if event_name == 'UserPromptSubmit':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt=raw.get('prompt', ''),
                    project_path=raw.get('cwd'),
                    timestamp=ts,
                )
            ]
        if event_name == 'PostToolUse':
            tool_name = raw.get('tool_name', '')
            if tool_name in self._SKIP_TOOLS:
                return []
            tool_input = raw.get('tool_input', {})
            tool_response = raw.get('tool_response', {})
            summary = f'[{tool_name}] {json.dumps(tool_input, default=str)[:500]}'
            response_str = json.dumps(tool_response, default=str)[:2000] if tool_response else ''
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt=summary,
                    assistant_response=response_str,
                    tool_calls=json.dumps({
                        'tool_name': tool_name,
                        'tool_use_id': raw.get('tool_use_id', ''),
                        'tool_input': tool_input,
                    }),
                    project_path=raw.get('cwd'),
                    timestamp=ts,
                )
            ]

        # Legacy direct-POST format
        message = raw.get('message', '')
        if legacy_type == 'user_prompt':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt=message,
                    timestamp=ts,
                )
            ]
        if legacy_type == 'assistant_response':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt='',
                    assistant_response=message,
                    timestamp=ts,
                )
            ]
        return []

    def get_installation_instructions(self) -> str:
        return (
            'Add to ~/.claude/settings.json under "hooks":\n\n'
            '{\n'
            '  "hooks": {\n'
            '    "UserPromptSubmit": [{\n'
            '      "matcher": "",\n'
            '      "hooks": [{\n'
            '        "type": "command",\n'
            '        "command": "curl -s -X POST http://localhost:7749/capture/claude-code/hook '
            "-H 'Content-Type: application/json' -d @-\"\n"
            '      }]\n'
            '    }],\n'
            '    "PostToolUse": [{\n'
            '      "matcher": "",\n'
            '      "hooks": [{\n'
            '        "type": "command",\n'
            '        "command": "curl -s -X POST http://localhost:7749/capture/claude-code/hook '
            "-H 'Content-Type: application/json' -d @-\"\n"
            '      }]\n'
            '    }]\n'
            '  }\n'
            '}'
        )

    def health_check(self) -> ToolHealth:
        return ToolHealth(tool_name=self.tool_name, connected=True, tier=1)
