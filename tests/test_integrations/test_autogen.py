from unittest.mock import MagicMock, patch


def test_autogen_add_calls_remember(tmp_path):
    from aingram.integrations.autogen import AIngramAutogenMemory

    with patch('aingram.integrations.autogen.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem
        m = AIngramAutogenMemory(db_path=str(tmp_path / 'x.db'))
        m.add('remembered text')
        mock_mem.remember.assert_called_once_with('remembered text')


def test_autogen_query_calls_recall(tmp_path):
    from aingram.integrations.autogen import AIngramAutogenMemory

    with patch('aingram.integrations.autogen.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_mem.recall.return_value = []
        mock_store_cls.return_value = mock_mem
        m = AIngramAutogenMemory(db_path=str(tmp_path / 'y.db'))
        m.query('q', limit=4)
        mock_mem.recall.assert_called_once_with('q', limit=4, verify=False)
