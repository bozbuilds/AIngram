from aingram.capture.config import CaptureConfig, ToolConfig
from aingram.capture.filters import apply_filters
from aingram.capture.types import CaptureRecord


def _make_record(**overrides):
    defaults = dict(
        source_tool='claude_code',
        session_id='sess-1',
        user_prompt='refactor the authentication module to use JWT tokens',
        timestamp=1000.0,
    )
    defaults.update(overrides)
    return CaptureRecord(**defaults)


class TestNocaptureOptOut:
    def test_nocapture_in_prompt_drops_record(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='@nocapture this is private and should not be stored')
        assert apply_filters(record, config) is None

    def test_nocapture_absent_keeps_record(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='refactor the database connection pooling logic')
        assert apply_filters(record, config) is not None

    def test_short_prompt_dropped(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='looks good')
        assert apply_filters(record, config) is None

    def test_short_prompt_with_tool_calls_kept(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='yes', tool_calls='{"tool_name": "Edit"}')
        assert apply_filters(record, config) is not None


class TestSecretRedaction:
    def test_redacts_openai_key(self):
        config = CaptureConfig()
        record = _make_record(
            user_prompt='set the OpenAI API key to sk-abcdefghijklmnopqrstuvwxyz in the env config',
        )
        result = apply_filters(record, config)
        assert 'sk-abcdef' not in result.user_prompt
        assert '[REDACTED]' in result.user_prompt

    def test_redacts_github_token(self):
        config = CaptureConfig()
        record = _make_record(
            assistant_response='use ghp_abcdefghijklmnopqrstuvwxyz0123456789 for authentication',
        )
        result = apply_filters(record, config)
        assert 'ghp_' not in result.assistant_response
        assert '[REDACTED]' in result.assistant_response

    def test_redacts_aws_key(self):
        config = CaptureConfig()
        record = _make_record(user_prompt='the AWS key AKIAIOSFODNN7EXAMPLE is used in production')
        result = apply_filters(record, config)
        assert 'AKIA' not in result.user_prompt

    def test_no_redaction_on_clean_text(self):
        config = CaptureConfig()
        record = _make_record(
            user_prompt='implement the vector search pipeline for semantic recall',
        )
        result = apply_filters(record, config)
        assert result.user_prompt == 'implement the vector search pipeline for semantic recall'


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
