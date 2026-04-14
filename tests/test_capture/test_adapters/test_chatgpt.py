import io
import json
import zipfile

from aingram.capture.adapters.chatgpt import ChatGPTAdapter
from aingram.capture.config import CaptureConfig


def _make_export_zip(conversations, tmp_path):
    zip_path = tmp_path / 'export.zip'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('conversations.json', json.dumps(conversations))
    zip_path.write_bytes(buf.getvalue())
    return str(zip_path)


SAMPLE_CONVERSATION = {
    'id': 'conv-abc',
    'title': 'Test Chat',
    'mapping': {
        'msg-1': {
            'id': 'msg-1',
            'parent': None,
            'children': ['msg-2'],
            'message': {
                'author': {'role': 'user'},
                'content': {'parts': ['What is Python?']},
                'create_time': 1713000000.0,
            },
        },
        'msg-2': {
            'id': 'msg-2',
            'parent': 'msg-1',
            'children': [],
            'message': {
                'author': {'role': 'assistant'},
                'content': {'parts': ['Python is a programming language.']},
                'create_time': 1713000005.0,
                'metadata': {'model_slug': 'gpt-4o'},
            },
        },
    },
}


class TestChatGPTAdapter:
    def setup_method(self):
        self.adapter = ChatGPTAdapter(CaptureConfig())

    def test_tool_name(self):
        assert self.adapter.tool_name == 'chatgpt'

    def test_parse_export_zip(self, tmp_path):
        zip_path = _make_export_zip([SAMPLE_CONVERSATION], tmp_path)
        payload = {'zip_path': zip_path}
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].user_prompt == 'What is Python?'
        assert records[0].assistant_response == 'Python is a programming language.'
        assert records[0].session_id == 'conv-abc'
        assert records[0].model == 'gpt-4o'

    def test_parse_empty_conversations(self, tmp_path):
        zip_path = _make_export_zip([], tmp_path)
        records = self.adapter.parse_payload({'zip_path': zip_path})
        assert records == []

    def test_missing_zip_returns_empty(self):
        records = self.adapter.parse_payload({'zip_path': '/nonexistent.zip'})
        assert records == []

    def test_health_check(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'chatgpt'
        assert h.tier == 3
