import json

from aingram.capture.adapters.cline import ClineAdapter
from aingram.capture.config import CaptureConfig


class TestClineAdapter:
    def setup_method(self):
        self.adapter = ClineAdapter(CaptureConfig())

    def test_tool_name(self):
        assert self.adapter.tool_name == 'cline'

    def test_parse_user_prompt_submit(self):
        payload = {
            'hookName': 'UserPromptSubmit',
            'taskId': 'task-1',
            'userPromptSubmit': {'prompt': 'refactor this function'},
            'model': {'slug': 'claude-sonnet-4-20250514'},
            'workspaceRoots': ['/home/user/proj'],
            'timestamp': 1713000000000,
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].user_prompt == 'refactor this function'
        assert records[0].session_id == 'task-1'
        assert records[0].model == 'claude-sonnet-4-20250514'

    def test_parse_task_complete_reads_conversation_file(self, tmp_path):
        conversation = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'do X'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'did X'}]},
        ]
        task_dir = tmp_path / 'task-2'
        task_dir.mkdir()
        conv_file = task_dir / 'api_conversation_history.json'
        conv_file.write_text(json.dumps(conversation))

        self.adapter.set_storage_base(str(tmp_path))
        payload = {
            'hookName': 'TaskComplete',
            'taskId': 'task-2',
            'timestamp': 1713000000000,
        }
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].user_prompt == 'do X'
        assert records[0].assistant_response == 'did X'

    def test_parse_unknown_hook_returns_empty(self):
        payload = {'hookName': 'SomethingElse', 'taskId': 'task-3'}
        assert self.adapter.parse_payload(payload) == []

    def test_health_check(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'cline'
        assert h.tier == 2
