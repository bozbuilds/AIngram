from aingram.capture.types import CaptureRecord, ToolHealth


class TestCaptureRecord:
    def test_minimal_construction(self):
        r = CaptureRecord(
            source_tool='claude_code',
            session_id='sess-1',
            user_prompt='hello',
            timestamp=1000.0,
        )
        assert r.source_tool == 'claude_code'
        assert r.state == 'pending'
        assert r.assistant_response is None
        assert r.turn_number is None

    def test_full_construction(self):
        r = CaptureRecord(
            source_tool='cursor',
            session_id='sess-2',
            user_prompt='fix bug',
            timestamp=2000.0,
            turn_number=3,
            assistant_response='done',
            tool_calls='[{"name":"edit"}]',
            model='gpt-4o',
            project_path='/home/user/proj',
            metadata='{"key":"val"}',
            container_tag='coding',
            state='done',
        )
        assert r.turn_number == 3
        assert r.model == 'gpt-4o'
        assert r.state == 'done'


class TestToolHealth:
    def test_defaults(self):
        h = ToolHealth(tool_name='cursor', connected=True)
        assert h.tier == 1
        assert h.last_capture is None
        assert h.error is None

    def test_error_state(self):
        h = ToolHealth(tool_name='aider', connected=False, error='file not found', tier=2)
        assert not h.connected
        assert h.tier == 2
