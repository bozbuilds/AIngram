from __future__ import annotations

import json
import logging
import os
import sys
import time

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE_BASE = {
    'darwin': '~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/tasks/',
    'linux': '~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/tasks/',
    'win32': os.path.expandvars('%APPDATA%/Code/User/globalStorage/saoudrizwan.claude-dev/tasks/'),
}


class ClineAdapter(ToolAdapter):
    tool_name = 'cline'

    def __init__(self, config) -> None:
        super().__init__(config)
        self._storage_base: str | None = None

    def set_storage_base(self, path: str) -> None:
        self._storage_base = path

    def _get_storage_base(self) -> str:
        if self._storage_base:
            return self._storage_base
        default = _DEFAULT_STORAGE_BASE.get(sys.platform, '')
        return os.path.expanduser(default)

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        hook_name = raw.get('hookName', '')
        task_id = raw.get('taskId', '')
        ts = raw.get('timestamp', time.time() * 1000) / 1000

        if hook_name == 'UserPromptSubmit':
            prompt_data = raw.get('userPromptSubmit', {})
            model_data = raw.get('model', {})
            workspace_roots = raw.get('workspaceRoots', [])
            return [
                CaptureRecord(
                    source_tool=self.tool_name,
                    session_id=task_id,
                    user_prompt=prompt_data.get('prompt', ''),
                    model=model_data.get('slug', ''),
                    project_path=workspace_roots[0] if workspace_roots else None,
                    timestamp=ts,
                )
            ]
        if hook_name in ('TaskComplete', 'TaskCancel', 'PreCompact'):
            return self._read_conversation_file(task_id)

        return []

    def _read_conversation_file(self, task_id: str) -> list[CaptureRecord]:
        base = self._get_storage_base()
        conv_path = os.path.join(base, task_id, 'api_conversation_history.json')
        if not os.path.exists(conv_path):
            logger.debug('Cline conversation file not found: %s', conv_path)
            return []

        try:
            with open(conv_path, encoding='utf-8') as f:
                messages = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning('Failed to read Cline conversation: %s', conv_path)
            return []

        records = []
        for i in range(0, len(messages) - 1, 2):
            if messages[i].get('role') == 'user' and messages[i + 1].get('role') == 'assistant':
                user_text = self._extract_text(messages[i].get('content', []))
                asst_text = self._extract_text(messages[i + 1].get('content', []))
                records.append(
                    CaptureRecord(
                        source_tool=self.tool_name,
                        session_id=task_id,
                        turn_number=i // 2,
                        user_prompt=user_text,
                        assistant_response=asst_text,
                        timestamp=time.time(),
                    )
                )
        return records

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    parts.append(block.get('text', ''))
            return '\n'.join(parts)
        return ''

    def get_installation_instructions(self) -> str:
        return (
            'Create an executable script at ~/Documents/Cline/Hooks/aingram-capture:\n\n'
            '  #!/bin/bash\n'
            '  INPUT=$(cat)\n'
            '  curl -s -X POST http://localhost:7749/capture/cline/hook \\\n'
            '    -H "Content-Type: application/json" \\\n'
            '    -d "$INPUT" > /dev/null 2>&1\n'
            '  echo \'{"cancel":false}\'\n\n'
            'Make it executable: chmod +x ~/Documents/Cline/Hooks/aingram-capture'
        )

    def health_check(self) -> ToolHealth:
        base = self._get_storage_base()
        connected = os.path.isdir(base)
        return ToolHealth(tool_name=self.tool_name, connected=connected, tier=2)
