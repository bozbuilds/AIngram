from __future__ import annotations

import json

from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.types import CaptureRecord, ToolHealth


class CopilotAdapter(ToolAdapter):
    tool_name = 'copilot'

    def __init__(self, config) -> None:
        super().__init__(config)
        self._seen_traces: set[str] = set()

    def parse_payload(self, raw: dict) -> list[CaptureRecord]:
        records = []
        for resource_span in raw.get('resourceSpans', []):
            for scope_span in resource_span.get('scopeSpans', []):
                for span in scope_span.get('spans', []):
                    if span.get('name') != 'chat':
                        continue
                    trace_id = span.get('traceId', '')
                    if trace_id in self._seen_traces:
                        continue
                    self._seen_traces.add(trace_id)
                    record = self._parse_chat_span(span)
                    if record:
                        records.append(record)
        return records

    def _parse_chat_span(self, span: dict) -> CaptureRecord | None:
        attrs = {a['key']: a['value'] for a in span.get('attributes', [])}

        prompts = []
        i = 0
        while True:
            key = f'gen_ai.prompt.{i}.content'
            if key not in attrs:
                break
            prompts.append(attrs[key].get('stringValue', ''))
            i += 1

        completion = attrs.get('gen_ai.completion.0.content', {}).get('stringValue', '')
        model = attrs.get('gen_ai.request.model', {}).get('stringValue', '')
        user_prompt = prompts[-1] if prompts else ''

        start_nano = int(span.get('startTimeUnixNano', 0))
        end_nano = int(span.get('endTimeUnixNano', 0))

        return CaptureRecord(
            source_tool=self.tool_name,
            session_id=span.get('traceId', ''),
            user_prompt=user_prompt,
            assistant_response=completion,
            model=model,
            timestamp=start_nano / 1e9,
            metadata=json.dumps(
                {
                    'span_id': span.get('spanId'),
                    'duration_ms': (end_nano - start_nano) / 1e6,
                }
            ),
        )

    def get_installation_instructions(self) -> str:
        return (
            'Add to VSCode settings.json:\n\n'
            '{\n'
            '  "github.copilot.chat.otel.enabled": true,\n'
            '  "github.copilot.chat.otel.otlpEndpoint": '
            '"http://localhost:7749/capture/copilot/otlp",\n'
            '  "github.copilot.chat.otel.otlpExporterType": "otlp-http",\n'
            '  "github.copilot.chat.otel.captureContent": true\n'
            '}\n\n'
            'IMPORTANT: captureContent must be true for prompt/response capture.'
        )

    def health_check(self) -> ToolHealth:
        return ToolHealth(tool_name=self.tool_name, connected=True, tier=2)
