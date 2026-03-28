"""ONNX Runtime execution provider selection tests."""

from aingram.config import AIngramConfig


class TestProviderSelection:
    def test_default_prefers_cuda(self):
        from aingram.processing.embedder import _select_providers

        available = {'CUDAExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider=None)
        assert chosen[0] == 'CUDAExecutionProvider'

    def test_npu_vitisai_when_requested(self):
        from aingram.processing.embedder import _select_providers

        available = {'VitisAIExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider='npu')
        assert chosen[0] == 'VitisAIExecutionProvider'

    def test_npu_directml_fallback(self):
        from aingram.processing.embedder import _select_providers

        available = {'DmlExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider='npu')
        assert chosen[0] == 'DmlExecutionProvider'

    def test_npu_falls_back_to_cpu(self):
        from aingram.processing.embedder import _select_providers

        available = {'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider='npu')
        assert chosen == ['CPUExecutionProvider']

    def test_explicit_cuda_provider(self):
        from aingram.processing.embedder import _select_providers

        available = {'CUDAExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider='cuda')
        assert chosen[0] == 'CUDAExecutionProvider'

    def test_explicit_cpu_provider(self):
        from aingram.processing.embedder import _select_providers

        available = {'CUDAExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider='cpu')
        assert chosen == ['CPUExecutionProvider']

    def test_none_preferred_uses_default_order(self):
        from aingram.processing.embedder import _select_providers

        available = {'CUDAExecutionProvider', 'VitisAIExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider=None)
        assert chosen[0] == 'CUDAExecutionProvider'

    def test_unknown_provider_falls_back_to_auto(self):
        from aingram.processing.embedder import _select_providers

        available = {'CUDAExecutionProvider', 'CPUExecutionProvider'}
        chosen = _select_providers(available, preferred_provider='tensorrt')
        assert 'CUDAExecutionProvider' in chosen or 'CPUExecutionProvider' in chosen

    def test_config_onnx_provider_field(self):
        cfg = AIngramConfig()
        assert cfg.onnx_provider is None

        cfg2 = AIngramConfig(onnx_provider='npu')
        assert cfg2.onnx_provider == 'npu'
