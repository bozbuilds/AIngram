from unittest.mock import MagicMock, patch


def test_crewai_save_calls_remember(tmp_path):
    from aingram.integrations.crewai import AIngramCrewMemory

    with patch('aingram.integrations.crewai.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_store_cls.return_value = mock_mem
        mem = AIngramCrewMemory(db_path=str(tmp_path / 'a.db'))
        mem.save('store this fact')
        mock_mem.remember.assert_called_once()


def test_crewai_search_calls_recall(tmp_path):
    from aingram.integrations.crewai import AIngramCrewMemory

    with patch('aingram.integrations.crewai.MemoryStore') as mock_store_cls:
        mock_mem = MagicMock()
        mock_mem.recall.return_value = []
        mock_store_cls.return_value = mock_mem
        mem = AIngramCrewMemory(db_path=str(tmp_path / 'b.db'))
        mem.search('query', limit=3)
        mock_mem.recall.assert_called_once_with('query', limit=3, verify=False)
