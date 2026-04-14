from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth

logger = logging.getLogger(__name__)


class AiderAdapter(ToolAdapter):
    tool_name = 'aider'

    def __init__(self, config) -> None:
        super().__init__(config)
        self._history_path: str | None = None
        self._last_position: int = 0

    def set_history_path(self, path: str) -> None:
        self._history_path = path
        self._last_position = 0

    def poll_new_entries(self) -> list[CaptureRecord]:
        if not self._history_path or not os.path.exists(self._history_path):
            return []
        try:
            with open(self._history_path, encoding='utf-8') as f:
                f.seek(self._last_position)
                new_content = f.read()
                self._last_position = f.tell()
        except OSError:
            logger.warning('Failed to read aider history file: %s', self._history_path)
            return []

        if not new_content.strip():
            return []

        return self._parse_history(new_content)

    def _parse_history(self, content: str) -> list[CaptureRecord]:
        records = []
        blocks = re.split(r'\n(?=TO LLM |LLM RESPONSE )', content)
        prompt_text = None

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if block.startswith('TO LLM '):
                lines = block.split('\n')
                user_lines = [line[5:] for line in lines[1:] if line.startswith('USER ')]
                prompt_text = '\n'.join(user_lines) if user_lines else None

            elif block.startswith('LLM RESPONSE '):
                lines = block.split('\n')
                response_ts = lines[0].replace('LLM RESPONSE ', '').strip()
                asst_lines = [line[10:] for line in lines[1:] if line.startswith('ASSISTANT ')]
                response_text = '\n'.join(asst_lines) if asst_lines else None

                if prompt_text and response_text:
                    ts = self._parse_timestamp(response_ts)
                    hist_dir = os.path.dirname(self._history_path or '')
                    sid = f'aider-{os.path.basename(hist_dir)}'
                    records.append(
                        CaptureRecord(
                            source_tool=self.tool_name,
                            session_id=sid,
                            user_prompt=prompt_text,
                            assistant_response=response_text,
                            timestamp=ts,
                            project_path=hist_dir,
                        )
                    )
                    prompt_text = None

        return records

    @staticmethod
    def _parse_timestamp(ts_str: str) -> float:
        try:
            dt = datetime.fromisoformat(ts_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            return time.time()

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        path = raw.get('history_path', '')
        if path:
            if path != self._history_path:
                self.set_history_path(path)
            return self.poll_new_entries()
        return []

    def get_installation_instructions(self) -> str:
        return (
            'Add to your project .aider.conf.yml (or ~/.aider.conf.yml):\n\n'
            '  llm-history-file: .aider.llm.history\n\n'
            'Or run aider with: aider --llm-history-file .aider.llm.history\n\n'
            'Then configure watch directories in ~/.aingram/config.toml:\n\n'
            '  [capture.tools.aider]\n'
            '  watch_directories = ["~/projects"]\n'
            '  history_file_pattern = "**/.aider.llm.history"'
        )

    def health_check(self) -> ToolHealth:
        connected = self._history_path is not None and os.path.exists(self._history_path)
        return ToolHealth(tool_name=self.tool_name, connected=connected, tier=1)
