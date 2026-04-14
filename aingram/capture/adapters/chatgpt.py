from __future__ import annotations

import json
import logging
import os
import zipfile
from collections import deque

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth

logger = logging.getLogger(__name__)


class ChatGPTAdapter(ToolAdapter):
    tool_name = 'chatgpt'

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        zip_path = raw.get('zip_path', '')
        if not zip_path or not os.path.exists(zip_path):
            return []
        return self._process_export(zip_path)

    def _process_export(self, zip_path: str) -> list[CaptureRecord]:
        try:
            with zipfile.ZipFile(zip_path) as z:
                with z.open('conversations.json') as f:
                    conversations = json.load(f)
        except (OSError, KeyError, json.JSONDecodeError):
            logger.warning('Failed to read ChatGPT export: %s', zip_path)
            return []

        records = []
        for conv in conversations:
            conv_id = conv.get('id', '')
            title = conv.get('title', '')
            turns = self._flatten_mapping(conv.get('mapping', {}))
            pairs = self._pair_turns(turns)
            for i, (user_msg, asst_msg) in enumerate(pairs):
                records.append(
                    CaptureRecord(
                        source_tool=self.tool_name,
                        session_id=conv_id,
                        turn_number=i,
                        user_prompt=self._extract_parts(user_msg),
                        assistant_response=self._extract_parts(asst_msg),
                        model=asst_msg.get('metadata', {}).get('model_slug', ''),
                        timestamp=asst_msg.get('create_time', 0) or 0,
                        metadata=json.dumps({'title': title}),
                    )
                )
        return records

    @staticmethod
    def _flatten_mapping(mapping: dict) -> list[dict]:
        ordered = []
        root_id = None
        for node_id, node in mapping.items():
            if node.get('parent') is None:
                root_id = node_id
                break
        if root_id is None:
            return []
        visit_queue = deque([root_id])
        while visit_queue:
            current = visit_queue.popleft()
            node = mapping.get(current, {})
            msg = node.get('message')
            if msg and msg.get('author', {}).get('role') in ('user', 'assistant'):
                ordered.append(msg)
            visit_queue.extend(node.get('children', []))
        return ordered

    @staticmethod
    def _pair_turns(messages: list[dict]) -> list[tuple[dict, dict]]:
        pairs = []
        i = 0
        while i < len(messages) - 1:
            if (
                messages[i].get('author', {}).get('role') == 'user'
                and messages[i + 1].get('author', {}).get('role') == 'assistant'
            ):
                pairs.append((messages[i], messages[i + 1]))
                i += 2
            else:
                i += 1
        return pairs

    @staticmethod
    def _extract_parts(msg: dict) -> str:
        parts = msg.get('content', {}).get('parts', [])
        return '\n'.join(str(p) for p in parts if isinstance(p, str))

    def get_installation_instructions(self) -> str:
        return (
            'ChatGPT Desktop integration is EXPERIMENTAL (degraded, non-real-time).\n\n'
            '1. In ChatGPT: Settings > Data Controls > Export Data\n'
            '2. Download the ZIP file\n'
            '3. Run: aingram capture import-chatgpt /path/to/export.zip\n'
            '   Or place in ~/.aingram/chatgpt-imports/ for automatic pickup.\n\n'
            'Limitations: manual export only, 24-48h delay, no real-time capture.'
        )

    def health_check(self) -> ToolHealth:
        return ToolHealth(
            tool_name=self.tool_name,
            connected=False,
            tier=3,
            error='ChatGPT requires manual export (no real-time capture)',
        )
