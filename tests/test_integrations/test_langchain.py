import importlib.util
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('langchain_core') is None,
    reason='needs langchain-core (pip install -e ".[langchain]" or ".[dev]")',
)


def test_langchain_adapter_add_message(tmp_path):
    from aingram.integrations.langchain import AIngramChatMessageHistory

    with patch('aingram.integrations.langchain.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem

        history = AIngramChatMessageHistory(db_path=str(tmp_path / 'test.db'))

        from langchain_core.messages import HumanMessage

        msg = HumanMessage(content='Hello world')
        history.add_message(msg)
        mock_mem.remember.assert_called_once()
        call_args = mock_mem.remember.call_args
        assert 'Hello world' in call_args[0][0]


def test_langchain_adapter_messages_returns_list(tmp_path):
    from aingram.integrations.langchain import AIngramChatMessageHistory

    with patch('aingram.integrations.langchain.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem
        mock_mem._engine._lock.__enter__ = MagicMock(return_value=None)
        mock_mem._engine._lock.__exit__ = MagicMock(return_value=False)
        mock_mem._engine._conn.execute.return_value.fetchall.return_value = [
            ('observation', '{"text":"hello"}'),
            ('result', '{"text":"world"}'),
        ]

        history = AIngramChatMessageHistory(db_path=str(tmp_path / 'test.db'))
        msgs = history.messages
        assert len(msgs) == 2


def test_langchain_adapter_clear(tmp_path):
    from aingram.integrations.langchain import AIngramChatMessageHistory

    with patch('aingram.integrations.langchain.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem

        history = AIngramChatMessageHistory(db_path=str(tmp_path / 'test.db'))
        history.clear()
