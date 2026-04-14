# aingram/consolidation/deberta.py
from __future__ import annotations

import numpy as np

from aingram.exceptions import ModelNotFoundError
from aingram.types import ContradictionVerdict

_HF_REPO = 'aingram/deberta-v3-base-nli-onnx'


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


class DeBERTaContradictionClassifier:
    """Contradiction classifier using DeBERTa-v3-base NLI via ONNX."""

    def __init__(self, *, threshold: float = 0.7, onnx_provider: str | None = None) -> None:
        self._threshold = threshold
        self._onnx_provider = onnx_provider
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(_HF_REPO, 'model.onnx')
            tokenizer_path = hf_hub_download(_HF_REPO, 'tokenizer.json')
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            import onnxruntime as ort

            from aingram.processing.embedder import _select_providers

            providers = _select_providers(
                set(ort.get_available_providers()),
                self._onnx_provider,
            )
            self._session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            raise ModelNotFoundError(
                f'Failed to load DeBERTa NLI model from {_HF_REPO}: {exc}'
            ) from exc

    def classify(self, text_a: str, text_b: str) -> ContradictionVerdict:
        self._ensure_loaded()
        encoding = self._tokenizer.encode(text_a, text_b)
        available = {
            'input_ids': np.array([encoding.ids], dtype=np.int64),
            'attention_mask': np.array([encoding.attention_mask], dtype=np.int64),
            'token_type_ids': np.array([encoding.type_ids], dtype=np.int64),
        }
        expected = {inp.name for inp in self._session.get_inputs()}
        feed = {k: v for k, v in available.items() if k in expected}
        logits = self._session.run(None, feed)[0][0]
        probs = _softmax(logits)
        contradiction_prob = float(probs[2])  # index 2 = contradiction
        return ContradictionVerdict(
            contradicts=contradiction_prob > self._threshold,
            confidence=contradiction_prob,
            superseded_index=None,
        )
