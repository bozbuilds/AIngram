# AIngram Lite Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement five independent improvements to AIngram Lite: FTS5 pre-filter, watch command, local visualization, framework integrations, and int8 quantization.

**Architecture:** Each improvement is a self-contained subsystem. They share no dependencies and can be built in any order. All changes follow existing patterns in the codebase: Typer CLI commands, `StorageEngine` for data access, `MemoryStore` for orchestration, pytest + `MockEmbedder` for testing.

**Tech Stack:** Python 3.11+, SQLite + sqlite-vec + FTS5, Typer CLI, D3.js (bundled), stdlib `http.server`

**Spec:** `docs/superpowers/specs/2026-04-01-lite-improvements-design.md`

---

## Chunk 1: FTS5 Pre-Filter

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `aingram/storage/engine.py` | Modify | Add `search_vectors_filtered()` method |
| `aingram/config.py` | Modify | Add `fts_prefilter_threshold` field |
| `aingram/store.py` | Modify | Reorder search pipeline in `recall()` |
| `tests/test_storage/test_engine_v3.py` | Modify | Unit tests for filtered vector search |
| `tests/test_store_v3.py` | Modify | Integration test for pre-filter recall |

### Task 1: Add `fts_prefilter_threshold` to config

**Files:**
- Modify: `aingram/config.py:18` (AIngramConfig dataclass)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py`, add:

```python
def test_fts_prefilter_threshold_default():
    from aingram.config import AIngramConfig
    cfg = AIngramConfig()
    assert cfg.fts_prefilter_threshold == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_fts_prefilter_threshold_default -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add the field to AIngramConfig**

In `aingram/config.py`, add to the `AIngramConfig` dataclass (after `telemetry_enabled`):

```python
fts_prefilter_threshold: int = 50
```

Also add env var support in `_merge_env_into()`:

```python
if v := env.get('AINGRAM_FTS_PREFILTER_THRESHOLD'):
    updates['fts_prefilter_threshold'] = int(v)
```

And coercion in `_coerce_value()`:

```python
if field_name == 'fts_prefilter_threshold' and raw is not None:
    return int(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_fts_prefilter_threshold_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/config.py tests/test_config.py
git commit -m "feat: add fts_prefilter_threshold config field (default 50)"
```

### Task 2: Implement `search_vectors_filtered()` in engine

**Files:**
- Modify: `aingram/storage/engine.py:316` (after `search_vectors`)
- Test: `tests/test_storage/test_engine_v3.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_storage/test_engine_v3.py`, add inside an existing test class or a new `class TestFilteredVectorSearch:` to match the file's class-based organization:

```python
class TestFilteredVectorSearch:
  def test_returns_subset(self, engine_with_session):
    engine = engine_with_session
    # Store 3 entries with different embeddings
    import struct, math
    dim = engine.get_embedding_dim()
    for i, eid in enumerate(['e1', 'e2', 'e3']):
        vec = [math.sin(i + j) for j in range(dim)]
        engine.store_entry(
            entry_id=eid, content_hash=f'h-{eid}', entry_type='observation',
            content=f'{{"text":"entry {eid}"}}', session_id='s1', sequence_num=i,
            prev_entry_id=None, signature='sig', created_at='2026-01-01T00:00:00+00:00',
            embedding=vec,
        )
    # Query with candidate_ids=['e1', 'e3'] — should never return e2
    query_vec = [math.sin(j) for j in range(dim)]  # close to e1's embedding (i=0)
    results = engine.search_vectors_filtered(query_vec, ['e1', 'e3'], limit=10)
    result_ids = [eid for eid, _ in results]
    assert 'e2' not in result_ids
    assert len(results) <= 2
    # Results should be sorted by similarity (descending, so lowest distance first)
    if len(results) == 2:
        assert results[0][1] <= results[1][1]


  def test_empty_candidates(self, engine_with_session):
    results = engine_with_session.search_vectors_filtered([0.0] * 768, [], limit=10)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage/test_engine_v3.py::TestFilteredVectorSearch::test_returns_subset tests/test_storage/test_engine_v3.py::TestFilteredVectorSearch::test_empty_candidates -v`
Expected: FAIL with `AttributeError: 'StorageEngine' object has no attribute 'search_vectors_filtered'`

- [ ] **Step 3: Implement the method**

In `aingram/storage/engine.py`, add after the `search_vectors` method (after line 316):

```python
def search_vectors_filtered(
    self, query_embedding: list[float], candidate_ids: list[str], *, limit: int = 10
) -> list[tuple[str, float]]:
    """Cosine similarity search over a pre-filtered candidate set.

    Fetches embedding blobs for candidate_ids, computes cosine similarity
    in Python, and returns top results sorted by ascending distance.
    Returns empty list if candidate_ids is empty.
    """
    self._check_open()
    if not candidate_ids:
        return []

    query_norm = math.sqrt(sum(x * x for x in query_embedding))
    if query_norm == 0:
        return []

    results: list[tuple[str, float]] = []
    # Note: vec0 virtual tables support WHERE clauses that are not MATCH.
    # If IN() is unsupported, fall back to individual lookups:
    #   rows = [self._conn.execute('SELECT entry_id, embedding FROM vec_entries WHERE entry_id = ?', (cid,)).fetchone() for cid in candidate_ids]
    #   rows = [r for r in rows if r]
    placeholders = ','.join('?' * len(candidate_ids))
    with self._lock:
        rows = self._conn.execute(
            f'SELECT entry_id, embedding FROM vec_entries '
            f'WHERE entry_id IN ({placeholders})',
            candidate_ids,
        ).fetchall()

    dim = len(query_embedding)
    for entry_id, blob in rows:
        vec = struct.unpack(f'{dim}f', blob)
        dot = sum(a * b for a, b in zip(query_embedding, vec))
        vec_norm = math.sqrt(sum(x * x for x in vec))
        if vec_norm == 0:
            continue
        cosine_sim = dot / (query_norm * vec_norm)
        distance = 1.0 - cosine_sim  # convert to distance (lower = more similar)
        results.append((entry_id, distance))

    results.sort(key=lambda x: x[1])
    return results[:limit]
```

Add `import math` to the imports at the top of `engine.py` (it is not currently imported).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage/test_engine_v3.py::TestFilteredVectorSearch::test_returns_subset tests/test_storage/test_engine_v3.py::TestFilteredVectorSearch::test_empty_candidates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/storage/engine.py tests/test_storage/test_engine_v3.py
git commit -m "feat: add search_vectors_filtered() — cosine similarity over candidate set"
```

### Task 3: Wire FTS5 pre-filter into `recall()`

**Files:**
- Modify: `aingram/store.py:182-270` (recall method)
- Test: `tests/test_store_v3.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_store_v3.py`, add:

```python
def test_recall_uses_fts_prefilter_when_above_threshold(tmp_path, mock_embedder):
    """When FTS returns >= threshold candidates, filtered vector search is used."""
    from unittest.mock import patch
    from aingram import MemoryStore

    db = str(tmp_path / 'test.db')
    mem = MemoryStore(db, embedder=mock_embedder)

    # Add enough entries with keyword "gradient" to exceed threshold
    for i in range(60):
        mem.remember(f'gradient descent step {i} in training loop')

    # Patch search_vectors_filtered to track calls
    with patch.object(mem._engine, 'search_vectors_filtered', wraps=mem._engine.search_vectors_filtered) as mock_filtered:
        with patch.object(mem._engine, 'search_vectors', wraps=mem._engine.search_vectors) as mock_full:
            mem.recall('gradient descent', limit=10)
            # With 60 FTS hits (above default threshold of 50), filtered should be called
            assert mock_filtered.call_count == 1
            assert mock_full.call_count == 0

    mem.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store_v3.py::test_recall_uses_fts_prefilter_when_above_threshold -v`
Expected: FAIL (current recall doesn't call `search_vectors_filtered`)

- [ ] **Step 3: Modify `recall()` to use adaptive FTS pre-filter**

In `aingram/store.py`, modify the `recall()` method. Replace the section from `embedding = self._embedder.embed(query)` through the RRF merge (lines ~216-237) with:

```python
embedding = self._embedder.embed(query)
candidate_limit = limit * 3
threshold = self._config.fts_prefilter_threshold
fts_candidate_limit = max(candidate_limit, threshold * 2)

fts_results = self._engine.search_fts(query, limit=fts_candidate_limit)
fts_ids = [eid for eid, _ in fts_results]

# Adaptive: FTS pre-filter or full KNN scan
if len(fts_ids) >= threshold:
    vec_results = self._engine.search_vectors_filtered(
        embedding, fts_ids, limit=candidate_limit
    )
else:
    vec_results = self._engine.search_vectors(embedding, limit=candidate_limit)
vec_ids = [eid for eid, _ in vec_results]

try:
    traversal = GraphTraversal(self._engine)
    graph_ids = traversal.search(query, limit=candidate_limit)
except Exception:
    graph_ids = []

chain_ids_list: list[str] = []
if chain_id:
    chain_entries = self._engine.get_entries_by_chain(chain_id, limit=candidate_limit)
    chain_ids_list = [e.entry_id for e in chain_entries]

ranked_lists = [lst for lst in [vec_ids, fts_ids, graph_ids, chain_ids_list] if lst]
rrf_scores = reciprocal_rank_fusion(ranked_lists, k=60)
```

Note: `self._config` is already available on the `MemoryStore` instance. **Task 3 depends on Tasks 1 and 2** — the config field and engine method must exist before wiring them into `recall()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store_v3.py -v`
Expected: ALL PASS (both new test and existing tests)

- [ ] **Step 5: Commit**

```bash
git add aingram/store.py tests/test_store_v3.py
git commit -m "feat: wire FTS5 pre-filter into recall() with adaptive fallback"
```

---

## Chunk 2: `aingram watch`

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `aingram/watch.py` | Create | Watch loop: polling, formatting, color output |
| `aingram/cli.py` | Modify | Add `watch` subcommand |
| `tests/test_watch.py` | Create | Unit tests for formatting + integration |

### Task 4: Implement watch formatting functions

**Files:**
- Create: `aingram/watch.py`
- Create: `tests/test_watch.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_watch.py`:

```python
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
    import json
    output = format_entry_json(row)
    parsed = json.loads(output)
    assert parsed['type'] == 'result'
    assert parsed['confidence'] == 0.91
    assert parsed['entry_id'] == 'abc123'


def test_format_entry_color_truncates_long_content():
    import re
    row = {
        'created_at': '2026-04-01T14:23:01+00:00',
        'entry_type': 'observation',
        'confidence': 0.5,
        'content': '{"text":"' + 'x' * 500 + '"}',
        'entry_id': 'ghi789',
    }
    output = format_entry_color(row, width=80)
    plain = re.sub(r'\033\[[0-9;]*m', '', output)
    assert len(plain) <= 80 + 5  # width + margin for quotes and rounding
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_watch.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `aingram/watch.py` with formatting functions**

```python
# aingram/watch.py
from __future__ import annotations

import json
import time

_TYPE_COLORS = {
    'hypothesis': '\033[34m',   # blue
    'method': '\033[37m',       # gray
    'result': '\033[32m',       # green
    'lesson': '\033[33m',       # gold/yellow
    'observation': '\033[36m',  # cyan
    'decision': '\033[0m',      # white (default)
    'meta': '\033[0m',          # white (default)
}
_RESET = '\033[0m'


def _extract_text(content: str) -> str:
    """Pull human-readable text from JSON content or return raw."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed.get('text', content)
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def format_entry_color(row: dict, *, width: int = 120) -> str:
    ts = row['created_at']
    # Extract HH:MM:SS from ISO timestamp
    time_part = ts[11:19] if len(ts) >= 19 else ts
    entry_type = (row.get('entry_type') or 'unknown').upper()
    conf = row.get('confidence')
    conf_str = f'{conf:.2f}' if conf is not None else '--'
    text = _extract_text(row.get('content', ''))

    color = _TYPE_COLORS.get(entry_type.lower(), _RESET)
    prefix = f'[{time_part}] {color}{entry_type:<10}{_RESET} confidence={conf_str}  '
    # Rough prefix length without ANSI
    prefix_len = len(f'[{time_part}] {entry_type:<10} confidence={conf_str}  ')
    max_text = max(width - prefix_len - 2, 20)

    if len(text) > max_text:
        text = text[:max_text - 1] + '\u2026'
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
    import os
    import sqlite3

    import sys
    if not os.path.exists(db_path):
        print(f'Error: database not found: {db_path}', file=sys.stderr, flush=True)
        raise SystemExit(1)

    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row

    # Get current max rowid as high-water mark
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
            for row in cursor:
                last_rowid = row['rowid']
                entry = dict(row)
                if json_output:
                    print(format_entry_json(entry), flush=True)
                else:
                    print(format_entry_color(entry, width=width), flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_watch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/watch.py tests/test_watch.py
git commit -m "feat: add watch module — entry formatting and polling loop"
```

### Task 5: Add `watch` CLI subcommand

**Files:**
- Modify: `aingram/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py`, add:

```python
def test_watch_missing_db(tmp_path):
    from typer.testing import CliRunner
    from aingram.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ['--db', str(tmp_path / 'nonexistent.db'), 'watch'])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_watch_missing_db -v`
Expected: FAIL with `No such command 'watch'`

- [ ] **Step 3: Add the watch subcommand to `cli.py`**

In `aingram/cli.py`, add before the `agent_app` section (before line 162):

```python
@app.command()
def watch(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, '--json', help='Output as JSONL'),
) -> None:
    """Live tail of new memory entries."""
    from aingram.watch import watch_loop
    watch_loop(ctx.obj['db'], json_output=json_output)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_watch_missing_db -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/cli.py tests/test_cli.py
git commit -m "feat: add 'aingram watch' CLI subcommand with --json flag"
```

---

## Chunk 3: Local Visualization

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `aingram/viz/__init__.py` | Create | Package init |
| `aingram/viz/server.py` | Create | HTTP server with JSON API endpoints |
| `aingram/viz/static/index.html` | Create | SPA shell with tabs + sidebar |
| `aingram/viz/static/d3.min.js` | Create | Bundled D3.js for offline use |
| `aingram/viz/static/app.js` | Create | D3 rendering + fetch logic |
| `aingram/viz/static/style.css` | Create | Layout and color coding |
| `aingram/cli.py` | Modify | Add `viz` subcommand |
| `tests/test_viz.py` | Create | API endpoint unit tests |

### Task 6: Implement viz API server

**Files:**
- Create: `aingram/viz/__init__.py`
- Create: `aingram/viz/server.py`
- Create: `tests/test_viz.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_viz.py`:

```python
import json
import pytest
from aingram.viz.server import VizHandler, create_server
from aingram.storage.engine import StorageEngine
from tests.conftest import ensure_test_session


@pytest.fixture
def viz_engine(tmp_path):
    db = tmp_path / 'viz_test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 'test-session')
    yield eng
    eng.close()


def test_api_stats(viz_engine):
    stats = VizHandler._get_stats(viz_engine)
    assert 'entry_count' in stats
    assert 'entity_count' in stats
    assert 'chain_count' in stats


def test_api_entities(viz_engine):
    data = VizHandler._get_entities(viz_engine)
    assert 'nodes' in data
    assert 'edges' in data


def test_api_entry_not_found(viz_engine):
    result = VizHandler._get_entry(viz_engine, 'nonexistent-id')
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_viz.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the viz package and server**

Create `aingram/viz/__init__.py`:

```python
# aingram/viz — local visualization server
```

Create `aingram/viz/server.py`:

```python
# aingram/viz/server.py
from __future__ import annotations

import json
import os
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aingram.storage.engine import StorageEngine

_STATIC_DIR = Path(__file__).parent / 'static'


class VizHandler(SimpleHTTPRequestHandler):
    engine: StorageEngine

    def __init__(self, *args, engine: StorageEngine, **kwargs):
        self.engine = engine
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith('/api/'):
            self._handle_api(path, parsed.query)
        else:
            super().do_GET()

    def _handle_api(self, path: str, query_string: str):
        self._set_cors()
        params = parse_qs(query_string)

        if path == '/api/stats':
            self._json_response(self._get_stats(self.engine))
        elif path == '/api/entities':
            self._json_response(self._get_entities(self.engine))
        elif path == '/api/chains':
            self._json_response(self._get_chains(self.engine))
        elif path == '/api/entry':
            entry_id = params.get('id', [None])[0]
            if not entry_id:
                self._json_error(400, 'Missing required parameter: id')
                return
            result = self._get_entry(self.engine, entry_id)
            if result is None:
                self._json_error(404, f'Entry not found: {entry_id}')
                return
            self._json_response(result)
        else:
            self._json_error(404, f'Unknown API endpoint: {path}')

    def _set_cors(self):
        pass  # Headers set in _json_response

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status, message):
        self._json_response({'error': message}, status=status)

    @staticmethod
    def _get_stats(engine: StorageEngine) -> dict:
        with engine._lock:
            entry_count = engine._conn.execute(
                'SELECT COUNT(*) FROM memory_entries'
            ).fetchone()[0]
            entity_count = engine._conn.execute(
                'SELECT COUNT(*) FROM entities'
            ).fetchone()[0]
            chain_count = engine._conn.execute(
                'SELECT COUNT(DISTINCT chain_id) FROM reasoning_chains'
            ).fetchone()[0]
        return {
            'entry_count': entry_count,
            'entity_count': entity_count,
            'chain_count': chain_count,
        }

    @staticmethod
    def _get_entities(engine: StorageEngine) -> dict:
        with engine._lock:
            entities = engine._conn.execute(
                'SELECT entity_id, name, entity_type FROM entities '
                'ORDER BY last_seen DESC LIMIT 500'
            ).fetchall()
            relationships = engine._conn.execute(
                'SELECT source_id, target_id, relation_type, weight '
                'FROM relationships LIMIT 2000'
            ).fetchall()
        nodes = [
            {'id': r[0], 'name': r[1], 'type': r[2]}
            for r in entities
        ]
        edges = [
            {'source': r[0], 'target': r[1], 'type': r[2], 'weight': r[3]}
            for r in relationships
        ]
        return {'nodes': nodes, 'edges': edges}

    @staticmethod
    def _get_chains(engine: StorageEngine) -> list:
        with engine._lock:
            chains = engine._conn.execute(
                'SELECT chain_id, title, status, created_at '
                'FROM reasoning_chains ORDER BY created_at DESC LIMIT 100'
            ).fetchall()
            result = []
            for chain in chains:
                entries = engine._conn.execute(
                    'SELECT entry_id, entry_type, content, confidence, created_at '
                    'FROM memory_entries WHERE reasoning_chain_id = ? '
                    'ORDER BY sequence_num ASC',
                    (chain[0],),
                ).fetchall()
                result.append({
                    'chain_id': chain[0],
                    'title': chain[1],
                    'status': chain[2],
                    'created_at': chain[3],
                    'entries': [
                        {
                            'entry_id': e[0],
                            'type': e[1],
                            'content': e[2],
                            'confidence': e[3],
                            'created_at': e[4],
                        }
                        for e in entries
                    ],
                })
        return result

    @staticmethod
    def _get_entry(engine: StorageEngine, entry_id: str) -> dict | None:
        entry = engine.get_entry(entry_id)
        if entry is None:
            return None

        with engine._lock:
            mentions = engine._conn.execute(
                'SELECT e.entity_id, e.name, e.entity_type '
                'FROM entity_mentions em JOIN entities e ON em.entity_id = e.entity_id '
                'WHERE em.entry_id = ?',
                (entry_id,),
            ).fetchall()

        # Note: verification requires session crypto context that the read-only
        # viz server does not have. We expose the hash and signature so the
        # frontend can display them, but mark verified as null.
        return {
            'entry_id': entry.entry_id,
            'content_hash': entry.content_hash,
            'entry_type': entry.entry_type,
            'content': entry.content,
            'confidence': entry.confidence,
            'surprise': entry.surprise,
            'importance': entry.importance,
            'created_at': entry.created_at,
            'signature': entry.signature,
            'verified': None,
            'reasoning_chain_id': entry.reasoning_chain_id,
            'entities': [
                {'id': m[0], 'name': m[1], 'type': m[2]}
                for m in mentions
            ],
        }

    def log_message(self, format, *args):
        pass  # Suppress request logs


def create_server(
    db_path: str, *, port: int = 8420
) -> tuple[HTTPServer, StorageEngine]:
    engine = StorageEngine(db_path)
    handler = partial(VizHandler, engine=engine)
    server = HTTPServer(('127.0.0.1', port), handler)
    return server, engine


def run_viz(db_path: str, *, port: int = 8420, open_browser: bool = True) -> None:
    server, engine = create_server(db_path, port=port)
    url = f'http://127.0.0.1:{port}'
    print(f'AIngram Viz running at {url}')
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        engine.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_viz.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/viz/__init__.py aingram/viz/server.py tests/test_viz.py
git commit -m "feat: add viz API server — stats, entities, chains, entry endpoints"
```

### Task 7: Create frontend static files

**Files:**
- Create: `aingram/viz/static/index.html`
- Create: `aingram/viz/static/style.css`
- Create: `aingram/viz/static/app.js`

Note: `d3.min.js` should be downloaded from `https://d3js.org/d3.v7.min.js` and saved to `aingram/viz/static/d3.min.js`. This is a manual step — download the file and place it there.

- [ ] **Step 1: Create `index.html`**

Create `aingram/viz/static/index.html` — a single-page app shell with:
- Tab bar (Graph | Chains)
- Main panel div for the active view
- Right sidebar div for entry inspector
- Script/style includes for `d3.min.js`, `app.js`, `style.css`
- A loading indicator and stats summary in the header

- [ ] **Step 2: Create `style.css`**

Create `aingram/viz/static/style.css` with:
- Flexbox layout: main panel (flex: 1) + sidebar (300px fixed width)
- Tab styling with active state
- Entry type color classes matching watch colors
- Node/edge styling for the D3 graph
- Timeline entry cards for chain view
- Sidebar inspector detail formatting
- Responsive collapse for sidebar on narrow viewports

- [ ] **Step 3: Create `app.js`**

Create `aingram/viz/static/app.js` with:
- `fetchStats()` → update header summary
- `fetchEntities()` → D3 force-directed graph in main panel
- `fetchChains()` → vertical timeline rendering in main panel
- `fetchEntry(id)` → populate sidebar inspector
- Tab switching logic (Graph ↔ Chains)
- Node click → `fetchEntry()` → sidebar
- D3 force simulation with node colors by entity type, edge weights
- Timeline entry cards with type colors, click-to-inspect

- [ ] **Step 4: Download D3.js (pinned version)**

Run: `curl -o aingram/viz/static/d3.min.js https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js`

Verify the download succeeded and the file is non-empty. Pin to D3 v7.9.0 for reproducibility.

- [ ] **Step 5: Manual QA — test the viz**

Run: `python -c "from aingram.viz.server import run_viz; run_viz('agent_memory.db')"`
Verify: browser opens, graph renders, clicking nodes populates sidebar, chains tab works.

- [ ] **Step 6: Commit**

```bash
git add aingram/viz/static/
git commit -m "feat: add viz frontend — D3 graph, chain timeline, entry inspector"
```

### Task 8: Add `viz` CLI subcommand

**Files:**
- Modify: `aingram/cli.py`

- [ ] **Step 1: Add the viz subcommand**

In `aingram/cli.py`, add before the `agent_app` section:

```python
@app.command()
def viz(
    ctx: typer.Context,
    port: int = typer.Option(8420, '--port', help='Server port'),
    no_open: bool = typer.Option(False, '--no-open', help='Do not open browser'),
) -> None:
    """Start local visualization server."""
    from aingram.viz.server import run_viz
    run_viz(ctx.obj['db'], port=port, open_browser=not no_open)
```

- [ ] **Step 2: Run existing CLI tests to verify no regressions**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add aingram/cli.py
git commit -m "feat: add 'aingram viz' CLI subcommand with --port and --no-open"
```

---

## Chunk 4: Framework Integrations

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `aingram/integrations/__init__.py` | Create | Package init |
| `aingram/integrations/langchain.py` | Create | LangChain `BaseChatMessageHistory` adapter |
| `aingram/integrations/crewai.py` | Create | CrewAI `Memory` adapter |
| `aingram/integrations/langgraph.py` | Create | LangGraph `BaseStore` adapter |
| `aingram/integrations/autogen.py` | Create | AutoGen `MemoryStore` adapter |
| `aingram/integrations/smolagents.py` | Create | smolagents adapter |
| `pyproject.toml` | Modify | Add optional dependency groups |
| `tests/test_integrations/` | Create | Mock-based adapter contract tests |

### Task 9: Implement LangChain adapter

**Files:**
- Create: `aingram/integrations/__init__.py`
- Create: `aingram/integrations/langchain.py`
- Create: `tests/test_integrations/__init__.py`
- Create: `tests/test_integrations/test_langchain.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_integrations/__init__.py` (empty).

Create `tests/test_integrations/test_langchain.py`:

```python
from unittest.mock import MagicMock, patch


def test_langchain_adapter_add_message(tmp_path):
    """Test that add_message delegates to MemoryStore.remember()."""
    from aingram.integrations.langchain import AIngramChatMessageHistory

    with patch('aingram.integrations.langchain.MemoryStore') as MockStore:
        mock_mem = MagicMock()
        MockStore.return_value = mock_mem

        history = AIngramChatMessageHistory(db_path=str(tmp_path / 'test.db'))

        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content='Hello world')
        history.add_message(msg)
        mock_mem.remember.assert_called_once()
        call_args = mock_mem.remember.call_args
        assert 'Hello world' in call_args[0][0]


def test_langchain_adapter_messages_returns_list(tmp_path):
    """Test that messages property returns a list of BaseMessage."""
    from aingram.integrations.langchain import AIngramChatMessageHistory

    with patch('aingram.integrations.langchain.MemoryStore') as MockStore:
        mock_mem = MagicMock()
        MockStore.return_value = mock_mem
        # Mock the SQL query path
        mock_conn = MagicMock()
        mock_mem._engine._lock.__enter__ = MagicMock(return_value=None)
        mock_mem._engine._lock.__exit__ = MagicMock(return_value=False)
        mock_mem._engine._conn.execute.return_value.fetchall.return_value = [
            ('observation', '{"text":"hello"}'),
            ('result', '{"text":"world"}'),
        ]

        history = AIngramChatMessageHistory(db_path=str(tmp_path / 'test.db'))
        msgs = history.messages
        assert len(msgs) == 2


def test_langchain_adapter_clear(tmp_path):
    from aingram.integrations.langchain import AIngramChatMessageHistory

    with patch('aingram.integrations.langchain.MemoryStore') as MockStore:
        mock_mem = MagicMock()
        MockStore.return_value = mock_mem

        history = AIngramChatMessageHistory(db_path=str(tmp_path / 'test.db'))
        # clear() should not raise (AIngram is append-only)
        history.clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integrations/test_langchain.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the integrations package and LangChain adapter**

Create `aingram/integrations/__init__.py`:

```python
# aingram/integrations — framework adapter packages
```

Create `aingram/integrations/langchain.py`:

```python
# aingram/integrations/langchain.py
from __future__ import annotations

import json
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, messages_from_dict

from aingram import MemoryStore


class AIngramChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by AIngram."""

    def __init__(
        self,
        db_path: str = 'agent_memory.db',
        *,
        session_id: str = 'langchain',
        store: MemoryStore | None = None,
    ):
        self._store = store or MemoryStore(db_path)
        self._session_id = session_id

    def add_message(self, message: BaseMessage) -> None:
        entry_type = 'observation'
        if hasattr(message, 'type'):
            if message.type == 'human':
                entry_type = 'observation'
            elif message.type == 'ai':
                entry_type = 'result'
        self._store.remember(
            message.content,
            entry_type=entry_type,
        )

    @property
    def messages(self) -> list[BaseMessage]:
        # Query recent entries directly via engine SQL
        with self._store._engine._lock:
            rows = self._store._engine._conn.execute(
                'SELECT entry_type, content FROM memory_entries '
                'ORDER BY created_at DESC LIMIT 100'
            ).fetchall()
        msgs: list[BaseMessage] = []
        for entry_type, content in reversed(rows):  # oldest first
            text = content
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    text = parsed.get('text', text)
            except (json.JSONDecodeError, TypeError):
                pass
            if entry_type in ('observation', 'hypothesis'):
                msgs.append(HumanMessage(content=text))
            else:
                msgs.append(AIMessage(content=text))
        return msgs

    def clear(self) -> None:
        pass  # AIngram is append-only; clear is a no-op
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_integrations/test_langchain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/integrations/__init__.py aingram/integrations/langchain.py tests/test_integrations/
git commit -m "feat: add LangChain BaseChatMessageHistory adapter"
```

### Task 10: Implement remaining framework adapters

**Files:**
- Create: `aingram/integrations/crewai.py`
- Create: `aingram/integrations/langgraph.py`
- Create: `aingram/integrations/autogen.py`
- Create: `aingram/integrations/smolagents.py`
- Create: `tests/test_integrations/test_crewai.py`
- Create: `tests/test_integrations/test_langgraph.py`
- Create: `tests/test_integrations/test_autogen.py`
- Create: `tests/test_integrations/test_smolagents.py`

Each adapter follows the same pattern as LangChain: wrap `MemoryStore`, implement the framework's interface, write a mock-based contract test. The key interfaces are:

| Adapter | Base Class | `remember()` maps to | `recall()` maps to |
|---------|-----------|---------------------|-------------------|
| CrewAI | `Memory` | `save()` | `search()` |
| LangGraph | `BaseStore` | `put()` | `get()`, `search()` |
| AutoGen | Protocol | `add()` | `query()` |
| smolagents | Pattern | `write_memory()` | `retrieve_memory()` |

- [ ] **Step 1: Write failing tests for all 4 adapters**

Each test file follows the LangChain test pattern: mock `MemoryStore`, verify the adapter calls `remember()` on write and `recall()` on read.

- [ ] **Step 2: Run all tests to verify they fail**

Run: `pytest tests/test_integrations/ -v`
Expected: FAIL

- [ ] **Step 3: Implement all 4 adapters**

Each adapter is a thin module (~30-50 lines) that wraps `MemoryStore` and translates between the framework's protocol and `remember()`/`recall()`. Check the framework's base class docs at implementation time for the exact method signatures required — they may have evolved since this plan was written.

For LangGraph specifically: target `BaseStore` from `langgraph >= 0.2`. If `BaseStore` is not available at the target version, fall back to a tool-node pattern (a plain function callable as a LangGraph tool).

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest tests/test_integrations/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/integrations/ tests/test_integrations/
git commit -m "feat: add CrewAI, LangGraph, AutoGen, smolagents adapters"
```

### Task 11: Add pip extras to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional dependency groups**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
langchain = ["langchain-core>=0.2"]
crewai = ["crewai>=0.40"]
langgraph = ["langgraph>=0.2"]
autogen = ["autogen-agentchat>=0.4"]
smolagents = ["smolagents>=1.0"]
```

Update the `all` extra to include them if desired, or keep `all` as the core extras only.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pip extras for framework integrations"
```

---

## Chunk 5: int8 Embedding Quantization

### File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `aingram/storage/schema.py` | Modify | Add `vec_entries_int8` table, version metadata |
| `aingram/storage/engine.py` | Modify | Quantize/dequantize methods, staleness detection |
| `aingram/store.py` | Modify | `quantize()` method, dual-write in `remember()` |
| `aingram/cli.py` | Modify | Add `quantize` subcommand |
| `tests/test_storage/test_quantize.py` | Create | Round-trip accuracy, storage size, search quality |

### Task 12: Add `vec_entries_int8` table to schema

**Files:**
- Modify: `aingram/storage/schema.py`
- Test: `tests/test_storage/test_schema.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_storage/test_schema.py`, add:

```python
def test_schema_has_vec_entries_int8_table(tmp_path):
    import sqlite3
    from aingram.storage.schema import apply_schema

    conn = sqlite3.connect(str(tmp_path / 'test.db'))
    apply_schema(conn, enable_vec=False)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert 'vec_entries_int8' in tables
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage/test_schema.py::test_schema_has_vec_entries_int8_table -v`
Expected: FAIL

- [ ] **Step 3: Add the table to schema**

In `aingram/storage/schema.py`, add the table DDL:

```python
VEC_ENTRIES_INT8_TABLE = """
CREATE TABLE IF NOT EXISTS vec_entries_int8 (
    entry_id TEXT PRIMARY KEY,
    quantized BLOB NOT NULL,
    scale REAL NOT NULL,
    min_val REAL NOT NULL
)
"""
```

Add it to the `apply_schema()` function:

```python
conn.execute(VEC_ENTRIES_INT8_TABLE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage/test_schema.py::test_schema_has_vec_entries_int8_table -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/storage/schema.py tests/test_storage/test_schema.py
git commit -m "feat: add vec_entries_int8 table to schema"
```

### Task 13: Implement quantize/dequantize in engine

**Files:**
- Modify: `aingram/storage/engine.py`
- Create: `tests/test_storage/test_quantize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage/test_quantize.py`:

```python
import math
import struct
import pytest
from aingram.storage.engine import StorageEngine
from tests.conftest import ensure_test_session


@pytest.fixture
def engine_with_entries(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 's1')
    # Store 5 entries with known embeddings
    dim = eng.get_embedding_dim()
    for i in range(5):
        vec = [math.sin(i * 0.1 + j * 0.01) for j in range(dim)]
        eng.store_entry(
            entry_id=f'e{i}', content_hash=f'h{i}', entry_type='observation',
            content=f'{{"text":"entry {i}"}}', session_id='s1', sequence_num=i,
            prev_entry_id=None, signature='sig',
            created_at='2026-01-01T00:00:00+00:00', embedding=vec,
        )
    yield eng
    eng.close()


def test_quantize_round_trip_accuracy(engine_with_entries):
    eng = engine_with_entries
    dim = eng.get_embedding_dim()

    # Get original embeddings
    originals = {}
    for i in range(5):
        blob = eng.get_entry_embedding(f'e{i}')
        originals[f'e{i}'] = struct.unpack(f'{dim}f', blob)

    # Quantize
    eng.quantize_all_embeddings()

    # Rebuild vec0 from int8
    eng.rebuild_vec_from_int8()

    # Check round-trip accuracy
    for eid, original in originals.items():
        blob = eng.get_entry_embedding(eid)
        rebuilt = struct.unpack(f'{dim}f', blob)
        for j in range(dim):
            assert abs(original[j] - rebuilt[j]) < 0.01, (
                f'Dimension {j} of {eid}: {original[j]:.6f} vs {rebuilt[j]:.6f}'
            )


def test_quantize_is_quantized(engine_with_entries):
    eng = engine_with_entries
    assert not eng.is_quantized()
    eng.quantize_all_embeddings()
    assert eng.is_quantized()


def test_quantize_zero_range_vector(tmp_path):
    """A vector with all identical values should quantize without error."""
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 's1')
    dim = eng.get_embedding_dim()
    vec = [0.42] * dim  # all identical
    eng.store_entry(
        entry_id='e-zero', content_hash='h-zero', entry_type='observation',
        content='{"text":"zero range"}', session_id='s1', sequence_num=0,
        prev_entry_id=None, signature='sig',
        created_at='2026-01-01T00:00:00+00:00', embedding=vec,
    )
    eng.quantize_all_embeddings()
    eng.rebuild_vec_from_int8()
    blob = eng.get_entry_embedding('e-zero')
    rebuilt = struct.unpack(f'{dim}f', blob)
    for j in range(dim):
        assert abs(rebuilt[j] - 0.42) < 0.01
    eng.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage/test_quantize.py -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement quantize/dequantize methods**

In `aingram/storage/engine.py`, add the following methods (before `close()`):

```python
def is_quantized(self) -> bool:
    """Check if the database has quantized embeddings."""
    self._check_open()
    with self._lock:
        row = self._conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'quantized_version'"
        ).fetchone()
    return row is not None and int(row[0]) > 0

def quantize_all_embeddings(self) -> int:
    """Quantize all embeddings from float32 to uint8 with per-vector min/max scaling."""
    self._check_open()
    dim = self.get_embedding_dim()
    count = 0
    with self._lock:
        rows = self._conn.execute(
            'SELECT entry_id, embedding FROM vec_entries'
        ).fetchall()
        for entry_id, blob in rows:
            vec = struct.unpack(f'{dim}f', blob)
            min_val = min(vec)
            max_val = max(vec)
            if max_val == min_val:
                scale = 0.0
                quantized = bytes(dim)  # all zeros
            else:
                scale = (max_val - min_val) / 255.0
                quantized = bytes(
                    min(255, max(0, round((v - min_val) / scale)))
                    for v in vec
                )
            self._conn.execute(
                'INSERT OR REPLACE INTO vec_entries_int8 '
                '(entry_id, quantized, scale, min_val) VALUES (?, ?, ?, ?)',
                (entry_id, quantized, scale, min_val),
            )
            count += 1
        now = datetime.now(UTC).isoformat()
        # Increment quantized version counter
        qv_row = self._conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'quantized_version'"
        ).fetchone()
        new_qv = str(int(qv_row[0]) + 1) if qv_row else '1'
        self._conn.execute(
            'INSERT OR REPLACE INTO db_metadata (key, value, updated_at) '
            'VALUES (?, ?, ?)',
            ('quantized_version', new_qv, now),
        )
        self._conn.execute(
            'INSERT OR REPLACE INTO db_metadata (key, value, updated_at) '
            'VALUES (?, ?, ?)',
            ('vec_cache_version', new_qv, now),
        )
        self._conn.commit()
    return count

def rebuild_vec_from_int8(self) -> int:
    """Rebuild float32 vec_entries from int8 source of truth."""
    self._check_open()
    dim = self.get_embedding_dim()
    count = 0
    with self._lock:
        rows = self._conn.execute(
            'SELECT entry_id, quantized, scale, min_val FROM vec_entries_int8'
        ).fetchall()
        for entry_id, quantized_blob, scale, min_val in rows:
            uint8_vals = list(quantized_blob)
            vec = [v * scale + min_val for v in uint8_vals]
            blob = struct.pack(f'{dim}f', *vec)
            # DELETE + INSERT because vec0 may not support UPDATE
            self._conn.execute(
                'DELETE FROM vec_entries WHERE entry_id = ?', (entry_id,)
            )
            self._conn.execute(
                'INSERT INTO vec_entries (entry_id, embedding) VALUES (?, ?)',
                (entry_id, blob),
            )
            count += 1
        now = datetime.now(UTC).isoformat()
        # Update cache version to match quantized version
        qv_row = self._conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'quantized_version'"
        ).fetchone()
        qv = qv_row[0] if qv_row else '1'
        self._conn.execute(
            'INSERT OR REPLACE INTO db_metadata (key, value, updated_at) '
            'VALUES (?, ?, ?)',
            ('vec_cache_version', qv, now),
        )
        self._conn.commit()
    return count

def store_entry_int8(
    self, entry_id: str, embedding: list[float]
) -> None:
    """Store a single embedding as int8 (for dual-write after quantize).

    Commits per call — designed for single-entry use from remember().
    For batch operations, use quantize_all_embeddings() instead.
    """
    self._check_open()
    min_val = min(embedding)
    max_val = max(embedding)
    if max_val == min_val:
        scale = 0.0
        quantized = bytes(len(embedding))
    else:
        scale = (max_val - min_val) / 255.0
        quantized = bytes(
            min(255, max(0, round((v - min_val) / scale)))
            for v in embedding
        )
    with self._lock:
        self._conn.execute(
            'INSERT OR REPLACE INTO vec_entries_int8 '
            '(entry_id, quantized, scale, min_val) VALUES (?, ?, ?, ?)',
            (entry_id, quantized, scale, min_val),
        )
        self._conn.commit()
```

Make sure `from datetime import UTC, datetime` is imported at the top of the file (it likely already is).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage/test_quantize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/storage/engine.py tests/test_storage/test_quantize.py
git commit -m "feat: add quantize/dequantize engine methods with uint8 per-vector scaling"
```

### Task 14: Add `quantize()` to MemoryStore and dual-write

**Files:**
- Modify: `aingram/store.py`
- Test: `tests/test_store_v3.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_store_v3.py`, add:

```python
def test_quantize_requires_confirm(tmp_path, mock_embedder):
    from aingram import MemoryStore
    mem = MemoryStore(str(tmp_path / 'test.db'), embedder=mock_embedder)
    with pytest.raises(ValueError, match='destructive'):
        mem.quantize()
    mem.close()


def test_quantize_and_remember_dual_writes(tmp_path, mock_embedder):
    from aingram import MemoryStore
    mem = MemoryStore(str(tmp_path / 'test.db'), embedder=mock_embedder)

    mem.remember('before quantize')
    assert not mem._engine.is_quantized()

    mem.quantize(confirm=True)
    assert mem._engine.is_quantized()

    # New entries after quantize should dual-write
    mem.remember('after quantize')

    # Verify int8 table has both entries
    with mem._engine._lock:
        count = mem._engine._conn.execute(
            'SELECT COUNT(*) FROM vec_entries_int8'
        ).fetchone()[0]
    assert count == 2
    mem.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store_v3.py::test_quantize_requires_confirm tests/test_store_v3.py::test_quantize_and_remember_dual_writes -v`
Expected: FAIL

- [ ] **Step 3: Add `quantize()` method and dual-write logic**

In `aingram/store.py`, add after `compact()`:

```python
def quantize(self, *, confirm: bool = False) -> None:
    """One-way quantize all stored embeddings to uint8 (4x storage reduction)."""
    if not confirm:
        raise ValueError('quantize() is destructive; call with confirm=True')
    self._engine.quantize_all_embeddings()
```

In the `remember()` method, after the `self._engine.store_entry(...)` call, add dual-write logic:

```python
if self._engine.is_quantized():
    self._engine.store_entry_int8(entry_id, embedding)
```

Where `embedding` is the embedding list computed earlier in the method.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store_v3.py::test_quantize_requires_confirm tests/test_store_v3.py::test_quantize_and_remember_dual_writes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aingram/store.py tests/test_store_v3.py
git commit -m "feat: add quantize() to MemoryStore with dual-write on remember()"
```

### Task 15: Add `quantize` CLI subcommand

**Files:**
- Modify: `aingram/cli.py`

- [ ] **Step 1: Add the quantize subcommand**

In `aingram/cli.py`, add after the `compact` command:

```python
@app.command()
def quantize(
    ctx: typer.Context,
    yes: bool = typer.Option(False, '--yes', help='Confirm one-way uint8 quantization'),
    rebuild: bool = typer.Option(False, '--rebuild', help='Rebuild float32 cache from int8 data'),
) -> None:
    """Quantize embeddings to uint8 for ~4x storage reduction."""
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        if rebuild:
            count = mem._engine.rebuild_vec_from_int8()
            typer.echo(f'Rebuilt {count} float32 embeddings from int8 source.')
        else:
            try:
                mem.quantize(confirm=yes)
            except ValueError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(code=1) from e
            typer.echo('Embeddings quantized to uint8.')
    finally:
        mem.close()
```

- [ ] **Step 2: Run existing CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add aingram/cli.py
git commit -m "feat: add 'aingram quantize' CLI subcommand with --yes and --rebuild"
```

---

## Summary

| Chunk | Tasks | Commits | Key Files |
|-------|-------|---------|-----------|
| 1. FTS5 Pre-Filter | 1-3 | 3 | engine.py, store.py, config.py |
| 2. Watch | 4-5 | 2 | watch.py, cli.py |
| 3. Visualization | 6-8 | 3 | viz/server.py, viz/static/*, cli.py |
| 4. Integrations | 9-11 | 3 | integrations/*.py, pyproject.toml |
| 5. Quantization | 12-15 | 4 | engine.py, schema.py, store.py, cli.py |

**Total: 15 tasks, 15 commits**

All 5 chunks are independent of each other and can be executed in parallel by separate agents. Within each chunk, tasks must be executed in order (e.g., Chunk 1: Task 1 → Task 2 → Task 3).
