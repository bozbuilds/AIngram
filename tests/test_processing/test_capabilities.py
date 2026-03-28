# tests/test_processing/test_capabilities.py
from aingram.processing.capabilities import Capabilities


class TestCapabilities:
    def test_tracks_embedder(self):
        caps = Capabilities(has_embedder=True, has_classifier=True)
        assert caps.has_embedder is True

    def test_tracks_classifier(self):
        caps = Capabilities(has_embedder=True, has_classifier=True)
        assert caps.has_classifier is True

    def test_tracks_extractor(self):
        caps = Capabilities(has_embedder=True, has_classifier=True, has_extractor=False)
        assert caps.has_extractor is False

    def test_tracks_llm(self):
        caps = Capabilities(has_embedder=True, has_classifier=True, has_llm=False)
        assert caps.has_llm is False

    def test_summary(self):
        caps = Capabilities(
            has_embedder=True,
            has_classifier=True,
            has_extractor=False,
            has_llm=False,
        )
        summary = caps.summary()
        assert 'embedder' in summary
        assert 'classifier' in summary
