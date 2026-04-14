import time

from aingram.capture.adapters.claude import ClaudeCodeAdapter
from aingram.capture.config import CaptureConfig


class TestClaudeCodeAdapter:
    def setup_method(self):
        self.adapter = ClaudeCodeAdapter(CaptureConfig())

    def test_tool_name(self):
        assert self.adapter.tool_name == 'claude_code'

    def test_parse_user_message(self):
        payload = {
            'session_id': 'sess-abc',
            'type': 'user_prompt',
            'message': 'fix the bug in auth.py',
            'timestamp': time.time(),
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].source_tool == 'claude_code'
        assert records[0].user_prompt == 'fix the bug in auth.py'
        assert records[0].session_id == 'sess-abc'

    def test_parse_assistant_response(self):
        payload = {
            'session_id': 'sess-abc',
            'type': 'assistant_response',
            'message': 'I fixed the bug by...',
            'timestamp': time.time(),
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].assistant_response == 'I fixed the bug by...'

    def test_parse_unknown_type_returns_empty(self):
        payload = {
            'session_id': 'sess-abc',
            'type': 'unknown_event',
            'timestamp': time.time(),
        }
        assert self.adapter.parse_payload(payload) == []

    def test_parse_hook_user_prompt_submit(self):
        payload = {
            'session_id': 'sess-hook',
            'hook_event_name': 'UserPromptSubmit',
            'prompt': 'explain how recall works',
            'cwd': '/home/user/project',
            'transcript_path': '/tmp/transcript.json',
            'permission_mode': 'default',
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].source_tool == 'claude_code'
        assert records[0].user_prompt == 'explain how recall works'
        assert records[0].session_id == 'sess-hook'
        assert records[0].project_path == '/home/user/project'

    def test_parse_hook_post_tool_use(self):
        payload = {
            'session_id': 'sess-hook',
            'hook_event_name': 'PostToolUse',
            'tool_name': 'Read',
            'tool_input': {'file_path': '/src/main.py'},
            'tool_response': {'content': 'print("hello")'},
            'tool_use_id': 'tu-123',
            'cwd': '/home/user/project',
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].source_tool == 'claude_code'
        assert '[Read]' in records[0].user_prompt
        assert records[0].session_id == 'sess-hook'
        assert records[0].project_path == '/home/user/project'
        assert records[0].tool_calls is not None
        assert 'Read' in records[0].tool_calls

    def test_installation_instructions_contains_curl(self):
        instructions = self.adapter.get_installation_instructions()
        assert 'localhost:7749' in instructions
        assert 'claude-code' in instructions

    def test_health_check_returns_tool_health(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'claude_code'
        assert h.tier == 1
