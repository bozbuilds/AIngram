from __future__ import annotations

import time

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth


class GeminiAdapter(ToolAdapter):
    tool_name = 'gemini'

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        event = raw.get('hook_event_name', '')
        session_id = raw.get('session_id', '')
        ts = raw.get('timestamp', time.time())

        if event == 'BeforeAgent':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt=raw.get('prompt', ''),
                    project_path=raw.get('cwd'),
                    timestamp=ts,
                )
            ]
        if event == 'AfterAgent':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=session_id,
                    user_prompt=raw.get('prompt', ''),
                    assistant_response=raw.get('prompt_response', ''),
                    timestamp=ts,
                )
            ]
        return []

    def get_installation_instructions(self) -> str:
        return (
            'Add to ~/.gemini/settings.json:\n\n'
            '{\n'
            '  "hooks": {\n'
            '    "BeforeAgent": [{\n'
            '      "matcher": "*",\n'
            '      "hooks": [{\n'
            '        "name": "capture-prompt",\n'
            '        "type": "command",\n'
            '        "command": "curl -s -X POST http://localhost:7749/capture/gemini/hook '
            "-H 'Content-Type: application/json' -d @-\",\n"
            '        "timeout": 5000\n'
            '      }]\n'
            '    }],\n'
            '    "AfterAgent": [{\n'
            '      "matcher": "*",\n'
            '      "hooks": [{\n'
            '        "name": "capture-response",\n'
            '        "type": "command",\n'
            '        "command": "curl -s -X POST http://localhost:7749/capture/gemini/hook '
            "-H 'Content-Type: application/json' -d @-\",\n"
            '        "timeout": 5000\n'
            '      }]\n'
            '    }]\n'
            '  }\n'
            '}'
        )

    def health_check(self) -> ToolHealth:
        return ToolHealth(tool_name=self.tool_name, connected=True, tier=1)
