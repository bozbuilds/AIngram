"""Recall with an optional entry_type filter."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aingram import MemoryStore


def main() -> None:
    path = Path(tempfile.mkdtemp()) / 'filter.db'
    with MemoryStore(str(path)) as mem:
        mem.remember('Observation: the default timeout is 30 seconds.', entry_type='observation')
        mem.remember('Lesson: validate untrusted input at the boundary.', entry_type='lesson')
        print('--- observations matching "timeout" ---')
        for result in mem.recall('timeout', entry_type='observation', limit=5):
            print(result.entry.entry_type, '\t', result.entry.content)


if __name__ == '__main__':
    main()
