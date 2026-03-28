# tests/test_models/test_manager.py
from aingram.models.manager import ModelManager


class TestModelManager:
    def test_default_cache_dir(self):
        mgr = ModelManager()
        assert mgr.cache_dir.name == 'models'
        assert '.aingram' in str(mgr.cache_dir)

    def test_custom_cache_dir(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path / 'custom')
        assert mgr.cache_dir == tmp_path / 'custom'

    def test_ensures_cache_dir_exists(self, tmp_path):
        cache = tmp_path / 'new_dir' / 'models'
        ModelManager(cache_dir=cache)
        assert cache.exists()

    def test_model_path(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        path = mgr.model_path('nomic-embed-text-v1.5')
        assert path == tmp_path / 'nomic-embed-text-v1.5'

    def test_is_downloaded_false(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        assert mgr.is_downloaded('nomic-embed-text-v1.5') is False

    def test_is_downloaded_true(self, tmp_path):
        mgr = ModelManager(cache_dir=tmp_path)
        model_dir = tmp_path / 'nomic-embed-text-v1.5'
        model_dir.mkdir()
        (model_dir / 'model.onnx').touch()
        assert mgr.is_downloaded('nomic-embed-text-v1.5') is True
