from aingram.capture.adapters.copilot import CopilotAdapter
from aingram.capture.config import CaptureConfig


def _make_otlp_payload(spans):
    return {
        'resourceSpans': [
            {
                'scopeSpans': [
                    {
                        'spans': spans,
                    }
                ]
            }
        ]
    }


def _make_chat_span(*, trace_id='trace-1', prompt='hello', completion='world', model='gpt-4o'):
    return {
        'name': 'chat',
        'traceId': trace_id,
        'spanId': 'span-1',
        'startTimeUnixNano': '1713000000000000000',
        'endTimeUnixNano': '1713000003000000000',
        'attributes': [
            {'key': 'gen_ai.prompt.0.content', 'value': {'stringValue': 'System prompt'}},
            {'key': 'gen_ai.prompt.1.content', 'value': {'stringValue': prompt}},
            {'key': 'gen_ai.completion.0.content', 'value': {'stringValue': completion}},
            {'key': 'gen_ai.request.model', 'value': {'stringValue': model}},
        ],
    }


class TestCopilotAdapter:
    def setup_method(self):
        self.adapter = CopilotAdapter(CaptureConfig())

    def test_tool_name(self):
        assert self.adapter.tool_name == 'copilot'

    def test_parse_single_chat_span(self):
        payload = _make_otlp_payload([_make_chat_span()])
        records = self.adapter.parse_payload(payload)
        assert len(records) == 1
        assert records[0].user_prompt == 'hello'
        assert records[0].assistant_response == 'world'
        assert records[0].model == 'gpt-4o'

    def test_skips_non_chat_spans(self):
        span = _make_chat_span()
        span['name'] = 'execute_tool'
        payload = _make_otlp_payload([span])
        assert self.adapter.parse_payload(payload) == []

    def test_extracts_last_prompt_as_user_message(self):
        span = _make_chat_span(prompt='the actual user question')
        payload = _make_otlp_payload([span])
        records = self.adapter.parse_payload(payload)
        assert records[0].user_prompt == 'the actual user question'

    def test_handles_empty_resource_spans(self):
        assert self.adapter.parse_payload({'resourceSpans': []}) == []
        assert self.adapter.parse_payload({}) == []

    def test_health_check(self):
        h = self.adapter.health_check()
        assert h.tool_name == 'copilot'
        assert h.tier == 2
