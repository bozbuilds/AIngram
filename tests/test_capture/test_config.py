from aingram.capture.config import (
    AiderToolConfig,
    CaptureConfig,
    ChatGPTToolConfig,
    ToolConfig,
    resolve_container_tag,
)
from aingram.capture.types import CaptureRecord


class TestCaptureConfigDefaults:
    def test_disabled_by_default(self):
        c = CaptureConfig()
        assert c.enabled is False

    def test_default_port(self):
        c = CaptureConfig()
        assert c.port == 7749

    def test_default_host(self):
        c = CaptureConfig()
        assert c.host == '127.0.0.1'

    def test_default_memory_mode(self):
        c = CaptureConfig()
        assert c.memory_mode == 'unified'

    def test_default_redaction_patterns_not_empty(self):
        c = CaptureConfig()
        assert len(c.redaction_patterns) >= 4

    def test_default_tools_include_all_seven(self):
        c = CaptureConfig()
        expected = {'claude_code', 'cursor', 'gemini', 'aider', 'copilot', 'cline', 'chatgpt'}
        assert set(c.tools.keys()) == expected

    def test_chatgpt_disabled_by_default(self):
        c = CaptureConfig()
        assert c.tools['chatgpt'].enabled is False

    def test_other_tools_enabled_by_default(self):
        c = CaptureConfig()
        for name in ('claude_code', 'cursor', 'gemini', 'aider', 'copilot', 'cline'):
            assert c.tools[name].enabled is True, f'{name} should be enabled'


class TestToolConfigSubclasses:
    def test_aider_has_watch_directories(self):
        c = CaptureConfig()
        aider = c.tools['aider']
        assert isinstance(aider, AiderToolConfig)
        assert aider.history_file_pattern == '**/.aider.llm.history'
        assert len(aider.watch_directories) >= 1

    def test_chatgpt_has_import_directory(self):
        c = CaptureConfig()
        chatgpt = c.tools['chatgpt']
        assert isinstance(chatgpt, ChatGPTToolConfig)
        assert 'chatgpt-imports' in chatgpt.import_directory


class TestContainerTagResolution:
    def test_unified_mode_returns_unified_tag(self):
        config = CaptureConfig(memory_mode='unified')
        record = CaptureRecord(source_tool='cursor', session_id='s', user_prompt='p', timestamp=0)
        assert resolve_container_tag(record, config) == 'aingram:unified'

    def test_isolated_mode_returns_tool_tag(self):
        config = CaptureConfig(
            memory_mode='isolated',
            tools={'cursor': ToolConfig(container_tag='my-cursor')},
        )
        record = CaptureRecord(source_tool='cursor', session_id='s', user_prompt='p', timestamp=0)
        assert resolve_container_tag(record, config) == 'my-cursor'

    def test_isolated_mode_unknown_tool_returns_source_tool(self):
        config = CaptureConfig(memory_mode='isolated', tools={})
        record = CaptureRecord(source_tool='newtool', session_id='s', user_prompt='p', timestamp=0)
        assert resolve_container_tag(record, config) == 'newtool'


class TestConsolidationIntervalRecords:
    def test_default_value(self):
        c = CaptureConfig()
        assert c.consolidation_interval_records == 50

    def test_zero_is_valid(self):
        c = CaptureConfig(consolidation_interval_records=0)
        assert c.consolidation_interval_records == 0

    def test_custom_value(self):
        c = CaptureConfig(consolidation_interval_records=100)
        assert c.consolidation_interval_records == 100

    def test_negative_raises(self):
        import pytest

        with pytest.raises(ValueError, match='consolidation_interval_records'):
            CaptureConfig(consolidation_interval_records=-1)
