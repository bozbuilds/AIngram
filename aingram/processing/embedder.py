# aingram/processing/embedder.py
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import numpy as np

from aingram.exceptions import EmbeddingError, ModelNotFoundError
from aingram.models.manager import ModelManager

logger = logging.getLogger(__name__)

MODEL_NAME = 'nomic-embed-text-v1.5'
HUGGINGFACE_REPO = 'nomic-ai/nomic-embed-text-v1.5'
DEFAULT_DIM = 768
MAX_TOKENS = 8192

# Prefer loading order for NVIDIA pip wheels (cuDNN/cuFFT need runtime visible early on Windows).
_NVIDIA_BIN_PRIORITY = (
    'cuda_runtime',
    'cublas',
    'cufft',
    'curand',
    'cusolver',
    'cusparse',
    'cudnn',
    'nvjitlink',
    'cufile',
)


def _prepend_nvidia_wheel_bins_to_path() -> None:
    """Put all ``site-packages/nvidia/*/bin`` dirs on PATH so ORT can load cuFFT/cuDNN on Windows."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    site_pkgs = Path(ort.__file__).resolve().parent.parent
    nvidia_root = site_pkgs / 'nvidia'
    if not nvidia_root.is_dir():
        return
    seen: set[str] = set()
    bins: list[str] = []
    for name in _NVIDIA_BIN_PRIORITY:
        b = nvidia_root / name / 'bin'
        if b.is_dir():
            r = str(b.resolve())
            if r not in seen:
                seen.add(r)
                bins.append(r)
    try:
        for child in sorted(nvidia_root.iterdir()):
            if not child.is_dir():
                continue
            b = child / 'bin'
            if b.is_dir():
                r = str(b.resolve())
                if r not in seen:
                    seen.add(r)
                    bins.append(r)
    except OSError:
        pass
    if bins:
        os.environ['PATH'] = os.pathsep.join(bins) + os.pathsep + os.environ.get('PATH', '')


def _select_providers(available: set[str], preferred_provider: str | None = None) -> list[str]:
    """Select ONNX Runtime execution providers based on preference and availability.

    Provider preference orders:
      None (auto):  CUDA > VitisAI > DirectML > CPU
      'cuda':       CUDA > CPU
      'npu':        VitisAI > DirectML > CPU
      'cpu':        CPU only
    """
    if preferred_provider == 'cpu':
        return ['CPUExecutionProvider']

    if preferred_provider == 'cuda':
        order = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif preferred_provider == 'npu':
        order = ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        # Auto: prefer GPU, then NPU, then CPU (unknown strings use this path)
        order = [
            'CUDAExecutionProvider',
            'VitisAIExecutionProvider',
            'DmlExecutionProvider',
            'CPUExecutionProvider',
        ]

    chosen = [p for p in order if p in available]
    if not chosen and available:
        chosen = list(available)
    return chosen


class NomicEmbedder:
    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        models_dir: Path | None = None,
        preferred_provider: str | None = None,
    ) -> None:
        self.dim = dim
        self._manager = ModelManager(cache_dir=models_dir)
        self._session = None
        self._tokenizer = None
        self._preferred_provider = preferred_provider

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        if not self._manager.is_downloaded(MODEL_NAME):
            logger.info('Downloading %s...', MODEL_NAME)
            self._download()

        model_path = self._manager.model_path(MODEL_NAME)
        onnx_path = model_path / 'onnx' / 'model.onnx'

        if not onnx_path.exists():
            raise ModelNotFoundError(
                f'ONNX model not found at {onnx_path}. '
                f'Run `aingram setup` or delete {model_path} and retry.'
            )

        try:
            import onnxruntime as ort

            if not callable(getattr(ort, 'InferenceSession', None)):
                raise EmbeddingError(
                    'onnxruntime is incomplete (no InferenceSession). '
                    'Often caused by a partial uninstall. Run: '
                    'pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu'
                )

            get_providers = getattr(ort, 'get_available_providers', None)
            session_kw: dict = {}
            cuda_listed = False
            if callable(get_providers):
                try:
                    available = set(get_providers())
                except Exception:
                    available = set()
                chosen = _select_providers(available, preferred_provider=self._preferred_provider)
                cuda_listed = 'CUDAExecutionProvider' in chosen
                if chosen:
                    session_kw['providers'] = chosen

            # onnxruntime-gpu lists CUDA even when cuFFT/cuDNN DLLs are not on PATH. Prepend
            # every nvidia-* wheel bin dir, then ORT's preload_dlls() when available.
            if cuda_listed:
                _prepend_nvidia_wheel_bins_to_path()
                preload = getattr(ort, 'preload_dlls', None)
                if callable(preload):
                    with contextlib.suppress(Exception):
                        preload()

            if os.environ.get('ORT_LOG_SEVERITY_LEVEL') is None and os.environ.get(
                'AINGRAM_ORT_VERBOSE', ''
            ).strip().lower() not in ('1', 'true', 'yes'):
                set_sev = getattr(ort, 'set_default_logger_severity', None)
                if callable(set_sev):
                    with contextlib.suppress(Exception):
                        set_sev(4)

            self._session = ort.InferenceSession(str(onnx_path), **session_kw)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f'Failed to load embedding model: {e}') from e

        try:
            from tokenizers import Tokenizer

            tokenizer_path = model_path / 'tokenizer.json'
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as e:
            raise EmbeddingError(f'Failed to load tokenizer: {e}') from e

    def _download(self) -> None:
        model_path = self._manager.model_path(MODEL_NAME)
        model_path.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download

            for filename in ['onnx/model.onnx', 'tokenizer.json', 'config.json']:
                hf_hub_download(
                    repo_id=HUGGINGFACE_REPO,
                    filename=filename,
                    local_dir=str(model_path),
                )
            logger.info('Downloaded %s to %s', MODEL_NAME, model_path)
        except ImportError as e:
            raise ModelNotFoundError(
                f'huggingface_hub not installed. Install it with: '
                f'pip install huggingface_hub\n'
                f'Or manually download {HUGGINGFACE_REPO} to {model_path}'
            ) from e
        except Exception as e:
            raise ModelNotFoundError(f'Failed to download {MODEL_NAME}: {e}') from e

    def _session_input_feed(
        self, input_ids: list[int], attention_mask: list[int]
    ) -> dict[str, np.ndarray]:
        """Build ONNX Runtime input feed; nomic ONNX exports may require token_type_ids."""
        n = len(input_ids)
        candidates: dict[str, np.ndarray] = {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64),
            'token_type_ids': np.zeros((1, n), dtype=np.int64),
        }
        wanted = {inp.name for inp in self._session.get_inputs()}
        return {k: v for k, v in candidates.items() if k in wanted}

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        try:
            encoded = self._tokenizer.encode(text)
            input_ids = encoded.ids[:MAX_TOKENS]
            attention_mask = [1] * len(input_ids)

            outputs = self._session.run(None, self._session_input_feed(input_ids, attention_mask))

            # Mean pooling over token embeddings
            embeddings = outputs[0][0]  # (seq_len, hidden_dim)
            mask = np.array(attention_mask, dtype=np.float32)
            pooled = np.sum(embeddings * mask[:, np.newaxis], axis=0) / np.sum(mask)

            # Matryoshka truncation, then L2-normalize the slice (matches compact_embeddings)
            pooled = pooled[: self.dim]
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

            return pooled.tolist()
        except Exception as e:
            raise EmbeddingError(f'Embedding failed: {e}') from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def active_execution_providers(self) -> list[str]:
        self._ensure_loaded()
        assert self._session is not None
        return list(self._session.get_providers())
