import pytest
from unittest.mock import patch
from bridgenlp.base import BridgeBase
from bridgenlp.config import BridgeConfig
from bridgenlp.result import BridgeResult

class DummyAdapter(BridgeBase):
    def __init__(self):
        super().__init__(BridgeConfig(collect_metrics=True))
        self._model = None

    def _load_model(self):
        self._model = object()

    def from_text(self, text: str) -> BridgeResult:
        with self._measure_performance():
            if self._model is None:
                self._load_model()
            return BridgeResult(tokens=text.split())

    def from_tokens(self, tokens):
        return BridgeResult(tokens=tokens)

    def from_spacy(self, doc):
        return BridgeResult(tokens=[t.text for t in doc]).attach_to_spacy(doc)


def test_memory_metric_recorded():
    adapter = DummyAdapter()
    with patch('bridgenlp.base.get_model_memory_usage', return_value=12.5) as mem:
        adapter.from_text("hello world")
        metrics = adapter.get_metrics()
        assert metrics['memory_mb'] == 12.5
        assert mem.called
