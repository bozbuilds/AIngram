from unittest.mock import MagicMock, patch


def test_smolagents_write_memory_calls_remember(tmp_path):
    from aingram.integrations.smolagents import AIngramSmolagentsMemory

    with patch('aingram.integrations.smolagents.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem
        m = AIngramSmolagentsMemory(db_path=str(tmp_path / 's.db'))
        m.write_memory('note')
        mock_mem.remember.assert_called_once_with('note')


def test_smolagents_retrieve_calls_recall(tmp_path):
    from aingram.integrations.smolagents import AIngramSmolagentsMemory

    with patch('aingram.integrations.smolagents.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_mem.recall.return_value = []
        mock_store_cls.return_value = mock_mem
        m = AIngramSmolagentsMemory(db_path=str(tmp_path / 't.db'))
        m.retrieve_memory('topic', limit=7)
        mock_mem.recall.assert_called_once_with('topic', limit=7, verify=False)
