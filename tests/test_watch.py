import json
import re

from aingram.watch import format_entry_color, format_entry_json


def test_format_entry_color_result():
    row = {
        'created_at': '2026-04-01T14:23:01+00:00',
        'entry_type': 'result',
        'confidence': 0.91,
        'content': '{"text":"Reducing LR below 1e-5 eliminated loss oscillation"}',
        'entry_id': 'abc123',
    }
    output = format_entry_color(row, width=120)
    assert '14:23:01' in output
    assert 'RESULT' in output
    assert '0.91' in output
    assert 'Reducing LR' in output


def test_format_entry_color_no_confidence():
    row = {
        'created_at': '2026-04-01T14:23:01+00:00',
        'entry_type': 'hypothesis',
        'confidence': None,
        'content': '{"text":"Some hypothesis"}',
        'entry_id': 'def456',
    }
    output = format_entry_color(row, width=120)
    assert '--' in output


def test_format_entry_json():
    row = {
        'created_at': '2026-04-01T14:23:01+00:00',
        'entry_type': 'result',
        'confidence': 0.91,
        'content': '{"text":"test content"}',
        'entry_id': 'abc123',
    }
    output = format_entry_json(row)
    parsed = json.loads(output)
    assert parsed['type'] == 'result'
    assert parsed['confidence'] == 0.91
    assert parsed['entry_id'] == 'abc123'


def test_format_entry_color_truncates_long_content():
    row = {
        'created_at': '2026-04-01T14:23:01+00:00',
        'entry_type': 'observation',
        'confidence': 0.5,
        'content': '{"text":"' + 'x' * 500 + '"}',
        'entry_id': 'ghi789',
    }
    output = format_entry_color(row, width=80)
    plain = re.sub(r'\033\[[0-9;]*m', '', output)
    assert len(plain) <= 80 + 5
