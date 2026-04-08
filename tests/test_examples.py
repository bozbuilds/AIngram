"""Smoke tests for files under examples/.

Protects example scripts from bit-rot as AIngram evolves.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / 'examples'


def test_multi_agent_shared_memory_example_runs_without_errors(capsys):
    """The 05_multi_agent_shared_memory.py example must run to completion."""
    path = _EXAMPLES_DIR / '05_multi_agent_shared_memory.py'
    assert path.exists(), f'example missing at {path}'

    spec = importlib.util.spec_from_file_location('_ex05', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'main')

    async def _run_with_timeout():
        return await asyncio.wait_for(module.main(), timeout=30.0)

    asyncio.run(_run_with_timeout())

    captured = capsys.readouterr()
    assert 'agents × 3 iterations' in captured.out or 'findings' in captured.out
    assert 'aingram-AR' in captured.out
