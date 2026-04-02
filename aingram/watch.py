# aingram/watch.py
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

_TYPE_COLORS = {
    'hypothesis': '\033[34m',
    'method': '\033[37m',
    'result': '\033[32m',
    'lesson': '\033[33m',
    'observation': '\033[36m',
    'decision': '\033[0m',
    'meta': '\033[0m',
}
_RESET = '\033[0m'


def _extract_text(content: str) -> str:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return str(parsed.get('text', content))
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def format_entry_color(row: dict, *, width: int = 120) -> str:
    ts = row['created_at']
    time_part = ts[11:19] if len(ts) >= 19 else ts
    entry_type = (row.get('entry_type') or 'unknown').upper()
    conf = row.get('confidence')
    conf_str = f'{conf:.2f}' if conf is not None else '--'
    text = _extract_text(row.get('content', ''))

    color = _TYPE_COLORS.get(entry_type.lower(), _RESET)
    prefix = f'[{time_part}] {color}{entry_type:<10}{_RESET} confidence={conf_str}  '
    prefix_len = len(f'[{time_part}] {entry_type:<10} confidence={conf_str}  ')
    max_text = max(width - prefix_len - 2, 20)

    if len(text) > max_text:
        text = text[: max_text - 1] + '\u2026'
    return f'{prefix}"{text}"'


def format_entry_json(row: dict) -> str:
    text = _extract_text(row.get('content', ''))
    return json.dumps({
        'timestamp': row['created_at'],
        'type': row.get('entry_type', 'unknown'),
        'confidence': row.get('confidence'),
        'content': text,
        'entry_id': row.get('entry_id', ''),
    })


def watch_loop(db_path: str, *, json_output: bool = False) -> None:
    """Poll for new entries and print them. Blocks until KeyboardInterrupt."""
    if not os.path.exists(db_path):
        print(f'Error: database not found: {db_path}', file=sys.stderr, flush=True)
        raise SystemExit(1)

    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row

    row = conn.execute('SELECT MAX(rowid) FROM memory_entries').fetchone()
    last_rowid = row[0] if row[0] is not None else 0

    if last_rowid == 0:
        print('Waiting for entries...', flush=True)

    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 120

    try:
        while True:
            cursor = conn.execute(
                'SELECT rowid, entry_id, entry_type, content, confidence, created_at '
                'FROM memory_entries WHERE rowid > ? ORDER BY rowid ASC',
                (last_rowid,),
            )
            for r in cursor:
                last_rowid = r['rowid']
                entry = dict(r)
                if json_output:
                    print(format_entry_json(entry), flush=True)
                else:
                    print(format_entry_color(entry, width=width), flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
