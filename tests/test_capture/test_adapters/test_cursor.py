from aingram.capture.adapters.cursor import CursorAdapter
from aingram.capture.config import CaptureConfig


class TestCursorAdapter:
    def setup_method(self):
        self.adapter = CursorAdapter(CaptureConfig())

    def test_tool_name(self):
        assert self.adapter.tool_name == 'cursor'

    def test_parse_before_submit_prompt(self):
        payload = {
            'hook_event_name': 'beforeSubmitPrompt',
            'conversation_id': 'conv-1',
            'generation_id': 'gen-1',
            'prompt': 'fix the null pointer exception',
            'workspace_roots': ['/home/user/myproject'],
            'attachments': ['file.py'],
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].user_prompt == 'fix the null pointer exception'
        assert records[0].session_id == 'conv-1'
        assert records[0].project_path == '/home/user/myproject'

    def test_parse_after_agent_response(self):
        payload = {
            'hook_event_name': 'afterAgentResponse',
            'conversation_id': 'conv-1',
            'generation_id': 'gen-1',
            'text': 'Fixed the null check in line 42.',
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].assistant_response == 'Fixed the null check in line 42.'

    def test_parse_session_start_returns_empty(self):
        payload = {
            'hook_event_name': 'sessionStart',
            'conversation_id': 'conv-1',
        }
        assert self.adapter.parse_payload(payload) == []

    def test_metadata_includes_generation_id(self):
        payload = {
            'hook_event_name': 'beforeSubmitPrompt',
            'conversation_id': 'conv-1',
            'generation_id': 'gen-42',
            'prompt': 'test',
        }
        records = self.adapter.parse_payload(payload)
        assert 'gen-42' in (records[0].metadata or '')

    def test_health_check(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'cursor'
        assert h.tier == 2
