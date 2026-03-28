# tests/test_processing/test_classifier.py
from aingram.processing.classifier import HeuristicClassifier
from aingram.processing.protocols import MemoryClassifier
from aingram.types import MemoryType


class TestHeuristicClassifier:
    def setup_method(self):
        self.classifier = HeuristicClassifier()

    def test_satisfies_protocol(self):
        assert isinstance(self.classifier, MemoryClassifier)

    def test_episodic_with_temporal_markers(self):
        assert self.classifier.classify('Yesterday I met with Alice') == MemoryType.EPISODIC
        assert self.classifier.classify('On March 5th we discussed the plan') == MemoryType.EPISODIC
        assert self.classifier.classify('Last week the server went down') == MemoryType.EPISODIC
        assert self.classifier.classify('Today I learned about SQLite') == MemoryType.EPISODIC

    def test_procedural_with_how_to_patterns(self):
        assert self.classifier.classify('How to deploy the application') == MemoryType.PROCEDURAL
        result = self.classifier.classify('To fix this bug, first restart the service')
        assert result == MemoryType.PROCEDURAL
        assert self.classifier.classify('Step 1: open the terminal') == MemoryType.PROCEDURAL
        result = self.classifier.classify('First, install Python. Then run pip install')
        assert result == MemoryType.PROCEDURAL

    def test_entity_with_dense_proper_nouns(self):
        assert self.classifier.classify('Alice Chen is the CTO of Acme Corp') == MemoryType.ENTITY
        assert (
            self.classifier.classify('Bob works at Google on the Search team') == MemoryType.ENTITY
        )

    def test_semantic_as_default(self):
        assert self.classifier.classify('User prefers dark mode') == MemoryType.SEMANTIC
        assert self.classifier.classify('Python is a programming language') == MemoryType.SEMANTIC
        assert self.classifier.classify('The API returns JSON') == MemoryType.SEMANTIC

    def test_empty_string(self):
        assert self.classifier.classify('') == MemoryType.SEMANTIC

    def test_case_insensitive(self):
        assert self.classifier.classify('YESTERDAY we had a meeting') == MemoryType.EPISODIC
        assert self.classifier.classify('HOW TO fix the bug') == MemoryType.PROCEDURAL
