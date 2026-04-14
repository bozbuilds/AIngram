from aingram.capture.config import CaptureConfig, ToolConfig
from aingram.capture.filters import apply_filters
from aingram.capture.types import CaptureRecord


def _make_record(**overrides):
    defaults = dict(
        source_tool='claude_code',
        session_id='sess-1',
        user_prompt='hello world',
        timestamp=1000.0,
    )
    defaults.update(overrides)
    return CaptureRecord(**defaults)


class TestNocaptureOptOut:
    def test_nocapture_in_prompt_drops_record(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='@nocapture this is private')
        assert apply_filters(record, config) is None

    def test_nocapture_absent_keeps_record(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='just a normal prompt')
        assert apply_filters(record, config) is not None


class TestSecretRedaction:
    def test_redacts_openai_key(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='my key is sk-abcdefghijklmnopqrstuvwxyz')
        result = apply_filters(record, config)
        assert 'sk-abcdef' not in result.user_prompt
        assert '[REDACTED]' in result.user_prompt

    def test_redacts_github_token(self):
        config = CaptureConfig()
        record = _make_record(assistant_response='use ghp_abcdefghijklmnopqrstuvwxyz0123456789')
        result = apply_filters(record, config)
        assert 'ghp_' not in result.assistant_response
        assert '[REDACTED]' in result.assistant_response

    def test_redacts_aws_key(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='key AKIAIOSFODNN7EXAMPLE here')
        result = apply_filters(record, config)
        assert 'AKIA' not in result.user_prompt

    def test_no_redaction_on_clean_text(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='just normal text')
        result = apply_filters(record, config)
        assert result.user_prompt == 'just normal text'


class TestContainerTagResolution:
    def test_unified_mode_sets_tag(self):
        config = CaptureConfig(memory_mode='unified')
        record = _make_record()
        result = apply_filters(record, config)
        assert result.container_tag == 'aingram:unified'

    def test_isolated_mode_sets_tool_tag(self):
        config = CaptureConfig(
            memory_mode='isolated',
            tools={'claude_code': ToolConfig(container_tag='my-claude')},
        )
        record = _make_record(source_tool='claude_code')
        result = apply_filters(record, config)
        assert result.container_tag == 'my-claude'
