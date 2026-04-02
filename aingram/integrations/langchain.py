# aingram/integrations/langchain.py
from __future__ import annotations

import json

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aingram import MemoryStore


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(str(block.get('text', '')))
                else:
                    parts.append(json.dumps(block))
            else:
                parts.append(str(block))
        return '\n'.join(parts) if parts else ''
    return str(content)


class AIngramChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by AIngram."""

    def __init__(
        self,
        db_path: str = 'agent_memory.db',
        *,
        session_id: str = 'langchain',
        store: MemoryStore | None = None,
    ):
        self._session_id = session_id
        self._store = store or MemoryStore(db_path)

    def add_message(self, message: BaseMessage) -> None:
        entry_type = 'observation'
        if getattr(message, 'type', None) == 'ai':
            entry_type = 'result'
        self._store.remember(_message_text(message), entry_type=entry_type)

    @property
    def messages(self) -> list[BaseMessage]:
        with self._store._engine._lock:
            rows = self._store._engine._conn.execute(
                'SELECT entry_type, content FROM memory_entries '
                'WHERE session_id = ? ORDER BY created_at DESC LIMIT 100',
                (self._store._session.session_id,),
            ).fetchall()
        msgs: list[BaseMessage] = []
        for entry_type, content in reversed(rows):
            text = content
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    text = str(parsed.get('text', text))
            except (json.JSONDecodeError, TypeError):
                pass
            if entry_type in ('observation', 'hypothesis'):
                msgs.append(HumanMessage(content=text))
            else:
                msgs.append(AIMessage(content=text))
        return msgs

    def clear(self) -> None:
        pass
