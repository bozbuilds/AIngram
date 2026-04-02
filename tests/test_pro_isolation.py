"""Verify no Pro-only references leaked into Lite codebase."""

import glob
import os

PRO_PATTERNS = [
    'partition_id',
    'partition_predictor',
    'PartitionPredictor',
    'predict_topk',
    'nprobe',
    'check_staleness',
    'train_predictor',
    'train_partitions',
    'partition_stale',
    'scikit-learn',
    'sklearn',
    'from torch',
    'import torch',
    'onnx.export',
    'CRDT',
    'causal_graph',
    'CausalGraph',
    'aingram.search.predictor',
    'Aingram-pro',
]

# Dynamically discover all Python files touched by the QJL port.
_QJL_GLOBS = [
    'aingram/processing/qjl.py',
    'aingram/storage/schema.py',
    'aingram/storage/engine.py',
    'aingram/storage/migration.py',
    'aingram/consolidation/merger.py',
    'aingram/store.py',
    'tests/**/test_qjl*.py',
]

# This file is excluded from scanning — it necessarily contains Pro patterns as test data.
_EXCLUDE = {'tests/test_pro_isolation.py', 'tests\\test_pro_isolation.py'}


def _collect_files() -> list[str]:
    paths: set[str] = set()
    for pattern in _QJL_GLOBS:
        paths.update(glob.glob(pattern, recursive=True))
    return sorted(p for p in paths if os.path.isfile(p) and p.replace('/', os.sep) not in _EXCLUDE and p not in _EXCLUDE)


def test_no_pro_references_in_any_modified_file():
    """All ported and modified files must not reference Pro-only concepts."""
    files = _collect_files()
    assert files, 'No files matched — glob patterns may be stale'
    for filepath in files:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
        for pattern in PRO_PATTERNS:
            assert pattern not in content, f'Pro reference "{pattern}" found in {filepath}'
