from aingram.capture.adapters.aider import AiderAdapter
from aingram.capture.config import AiderToolConfig, CaptureConfig

SAMPLE_HISTORY = """\
TO LLM 2026-04-12T14:23:45

SYSTEM You are an AI coding assistant.
USER Fix the authentication bug in auth.py

LLM RESPONSE 2026-04-12T14:23:52

ASSISTANT Here is the fix for the auth bug.
ASSISTANT It changes the token validation.

TO LLM 2026-04-12T14:25:00

SYSTEM You are an AI coding assistant.
USER Now add tests for the fix

LLM RESPONSE 2026-04-12T14:25:10

ASSISTANT Here are the tests.
"""


class TestAiderAdapter:
    def setup_method(self):
        self.config = CaptureConfig(tools={'aider': AiderToolConfig(container_tag='aider')})
        self.adapter = AiderAdapter(self.config)

    def test_tool_name(self):
        assert self.adapter.tool_name == 'aider'

    def test_parse_history_file(self, tmp_path):
        history_file = tmp_path / '.aider.llm.history'
        history_file.write_text(SAMPLE_HISTORY)
        self.adapter.set_history_path(str(history_file))

        records = self.adapter.poll_new_entries()
        assert len(records) == 2

        assert records[0].user_prompt == 'Fix the authentication bug in auth.py'
        assert 'fix for the auth bug' in records[0].assistant_response
        assert 'token validation' in records[0].assistant_response

        assert records[1].user_prompt == 'Now add tests for the fix'
        assert records[1].assistant_response == 'Here are the tests.'

    def test_incremental_read(self, tmp_path):
        history_file = tmp_path / '.aider.llm.history'
        history_file.write_text(SAMPLE_HISTORY)
        self.adapter.set_history_path(str(history_file))

        records1 = self.adapter.poll_new_entries()
        assert len(records1) == 2

        with open(history_file, 'a') as f:
            f.write(
                '\nTO LLM 2026-04-12T14:30:00\n\n'
                'USER Third question\n\n'
                'LLM RESPONSE 2026-04-12T14:30:05\n\n'
                'ASSISTANT Third answer.\n'
            )

        records2 = self.adapter.poll_new_entries()
        assert len(records2) == 1
        assert records2[0].user_prompt == 'Third question'

    def test_empty_file_returns_empty(self, tmp_path):
        history_file = tmp_path / '.aider.llm.history'
        history_file.write_text('')
        self.adapter.set_history_path(str(history_file))
        assert self.adapter.poll_new_entries() == []

    def test_missing_file_returns_empty(self):
        self.adapter.set_history_path('/nonexistent/path')
        assert self.adapter.poll_new_entries() == []

    def test_health_check(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'aider'
        assert h.tier == 1
