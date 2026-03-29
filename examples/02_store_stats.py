"""Print aggregate stats after writing a few entries."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aingram import MemoryStore


def main() -> None:
    path = Path(tempfile.mkdtemp()) / 'stats.db'
    with MemoryStore(str(path)) as mem:
        mem.remember('Session note: document the public API.')
        mem.remember('Reminder: run ruff and pytest before tagging a release.', tags=['process'])
        print(mem.stats)


if __name__ == '__main__':
    main()
