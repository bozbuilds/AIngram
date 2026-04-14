import time

from aingram.capture.adapters.gemini import GeminiAdapter
from aingram.capture.config import CaptureConfig


class TestGeminiAdapter:
    def setup_method(self):
        self.adapter = GeminiAdapter(CaptureConfig())

    def test_tool_name(self):
        assert self.adapter.tool_name == 'gemini'

    def test_parse_before_agent(self):
        payload = {
            'hook_event_name': 'BeforeAgent',
            'prompt': 'explain this code',
            'session_id': 'gem-sess-1',
            'cwd': '/home/user/proj',
            'timestamp': time.time(),
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].user_prompt == 'explain this code'
        assert records[0].project_path == '/home/user/proj'

    def test_parse_after_agent(self):
        payload = {
            'hook_event_name': 'AfterAgent',
            'prompt': 'explain this code',
            'prompt_response': 'This code does X...',
            'session_id': 'gem-sess-1',
            'timestamp': time.time(),
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].assistant_response == 'This code does X...'
        assert records[0].user_prompt == 'explain this code'

    def test_parse_session_start(self):
        payload = {
            'hook_event_name': 'SessionStart',
            'session_id': 'gem-sess-1',
            'timestamp': time.time(),
        }
        records = self.adapter.parse_payload(payload)
        assert records == []

    def test_installation_instructions(self):
        instructions = self.adapter.get_installation_instructions()
        assert 'gemini' in instructions
        assert 'localhost:7749' in instructions

    def test_health_check(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'gemini'
        assert h.tier == 1
