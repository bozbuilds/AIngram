from unittest.mock import MagicMock, patch


def test_langgraph_put_calls_remember(tmp_path):
    from aingram.integrations.langgraph import AIngramLangGraphStore

    with patch('aingram.integrations.langgraph.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem
        store = AIngramLangGraphStore(db_path=str(tmp_path / 'g.db'))
        store.put(('user',), 'k1', {'text': 'value'})
        mock_mem.remember.assert_called_once()
        args, kwargs = mock_mem.remember.call_args
        assert kwargs.get('entry_type') == 'meta'


def test_langgraph_search_calls_recall(tmp_path):
    from aingram.integrations.langgraph import AIngramLangGraphStore

    with patch('aingram.integrations.langgraph.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_mem.recall.return_value = []
        mock_store_cls.return_value = mock_mem
        store = AIngramLangGraphStore(db_path=str(tmp_path / 'h.db'))
        store.search(('user',), query='hello', limit=5)
        mock_mem.recall.assert_called_once()
