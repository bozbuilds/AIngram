"""Store two observations and run a hybrid recall query."""

from __future__ import annotations

import tempfile
from pathlib import Path

from aingram import MemoryStore


def main() -> None:
    path = Path(tempfile.mkdtemp()) / 'demo.db'
    with MemoryStore(str(path)) as mem:
        mem.remember('User prefers dark mode and concise answers.')
        mem.remember('User is building agent tooling in Python.')
        for result in mem.recall('What does the user prefer?', limit=5):
            print(f'{result.score:.4f}\t{result.entry.content}')


if __name__ == '__main__':
    main()
