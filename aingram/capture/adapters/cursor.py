from __future__ import annotations

import json
import time

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth


class CursorAdapter(ToolAdapter):
    tool_name = 'cursor'

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        event = raw.get('hook_event_name', '')
        conversation_id = raw.get('conversation_id', '')
        generation_id = raw.get('generation_id', '')
        ts = raw.get('timestamp', time.time())

        if event == 'beforeSubmitPrompt':
            workspace_roots = raw.get('workspace_roots', [])
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=conversation_id,
                    user_prompt=raw.get('prompt', ''),
                    project_path=workspace_roots[0] if workspace_roots else None,
                    timestamp=ts,
                    metadata=json.dumps(
                        {
                            'generation_id': generation_id,
                            'attachments': raw.get('attachments', []),
                        }
                    ),
                )
            ]
        if event == 'afterAgentResponse':
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=conversation_id,
                    user_prompt='',
                    assistant_response=raw.get('response', ''),
                    timestamp=ts,
                    metadata=json.dumps({'generation_id': generation_id}),
                )
            ]
        return []

    def get_installation_instructions(self) -> str:
        return (
            'Add to .cursor/hooks.json (project) or ~/.cursor/hooks.json (global):\n\n'
            '{\n'
            '  "version": 1,\n'
            '  "hooks": {\n'
            '    "beforeSubmitPrompt": [{\n'
            '      "command": "curl -s -X POST http://localhost:7749/capture/cursor/hook '
            "-H 'Content-Type: application/json' -d @-\"\n"
            '    }],\n'
            '    "afterAgentResponse": [{\n'
            '      "command": "curl -s -X POST http://localhost:7749/capture/cursor/hook '
            "-H 'Content-Type: application/json' -d @-\"\n"
            '    }]\n'
            '  }\n'
            '}\n\n'
            'Requires Cursor >= 1.7 (hooks beta).'
        )

    def health_check(self) -> ToolHealth:
        return ToolHealth(tool_name=self.tool_name, connected=True, tier=2)
