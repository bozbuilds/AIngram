import pytest
from starlette.testclient import TestClient

from aingram.capture.config import CaptureConfig
from aingram.capture.daemon import create_app
from aingram.capture.queue import CaptureQueue


@pytest.fixture
def app_and_queue(tmp_path):
    queue_db = str(tmp_path / 'queue.db')
    config = CaptureConfig()
    queue = CaptureQueue(queue_db)
    queue.init_toggles({name: cfg.enabled for name, cfg in config.tools.items()})
    app = create_app(config=config, queue=queue)
    yield app, queue
    queue.close()


class TestCaptureRoutes:
    def test_claude_code_hook(self, app_and_queue):
        app, queue = app_and_queue
        client = TestClient(app)
        payload = {
            'session_id': 'sess-1',
            'type': 'user_prompt',
            'message': 'hello from claude',
            'timestamp': 1000.0,
        }
        resp = client.post('/capture/claude-code/hook', json=payload)
        assert resp.status_code == 200
        assert queue.pending_count() == 1

    def test_cursor_hook(self, app_and_queue):
        app, queue = app_and_queue
        client = TestClient(app)
        payload = {
            'hook_event_name': 'beforeSubmitPrompt',
            'conversation_id': 'conv-1',
            'generation_id': 'gen-1',
            'prompt': 'hello from cursor',
        }
        resp = client.post('/capture/cursor/hook', json=payload)
        assert resp.status_code == 200
        assert queue.pending_count() == 1

    def test_gemini_hook(self, app_and_queue):
        app, queue = app_and_queue
        client = TestClient(app)
        payload = {
            'hook_event_name': 'BeforeAgent',
            'user_prompt': 'hello from gemini',
            'session_id': 'gem-1',
            'timestamp': 1000.0,
        }
        resp = client.post('/capture/gemini/hook', json=payload)
        assert resp.status_code == 200
        assert queue.pending_count() == 1

    def test_toggle_off_drops_silently(self, app_and_queue):
        app, queue = app_and_queue
        queue.set_toggle('claude_code', 'off')
        client = TestClient(app)
        payload = {
            'session_id': 'sess-1',
            'type': 'user_prompt',
            'message': 'should be dropped',
            'timestamp': 1000.0,
        }
        resp = client.post('/capture/claude-code/hook', json=payload)
        assert resp.status_code == 200
        assert queue.pending_count() == 0


class TestStatusAndToggle:
    def test_status_endpoint(self, app_and_queue):
        app, _ = app_and_queue
        client = TestClient(app)
        resp = client.get('/status')
        assert resp.status_code == 200
        data = resp.json()
        assert 'tools' in data
        assert 'queue_depth' in data

    def test_toggle_endpoint(self, app_and_queue):
        app, queue = app_and_queue
        client = TestClient(app)
        resp = client.post('/toggle', json={'tool': 'cursor', 'state': 'off'})
        assert resp.status_code == 200
        assert queue.get_toggle('cursor') == 'off'

    def test_toggle_all(self, app_and_queue):
        app, queue = app_and_queue
        client = TestClient(app)
        resp = client.post('/toggle', json={'tool': 'all', 'state': 'off'})
        assert resp.status_code == 200
        assert queue.get_toggle('claude_code') == 'off'
        assert queue.get_toggle('cursor') == 'off'
