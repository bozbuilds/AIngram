from __future__ import annotations

import time

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth


class ClaudeCodeAdapter(ToolAdapter):
    tool_name = 'claude_code'

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        event_type = raw.get('type', '')
        session_id = raw.get('session_id', '')
        ts = raw.get('timestamp', time.time())
        message = raw.get('message', '')

        if event_type == 'user_prompt':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt=message,
                    timestamp=ts,
                )
            ]
        if event_type == 'assistant_response':
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
