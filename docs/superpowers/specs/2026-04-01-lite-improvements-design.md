# AIngram Lite Improvements Design Spec

Five independent improvements to AIngram Lite that maximize traction and performance without encroaching on Pro territory.

## 1. FTS5 Pre-Filter for Vector Search

### Problem

Hybrid recall runs FTS5 and vector search independently, then merges via RRF. Vector search scans the full `vec_entries` table via sqlite-vec KNN, which is O(all entries). At 100K entries this takes ~320ms and dominates recall latency.

### Design

Use FTS5 results as a candidate set for vector search. Fetch embeddings for the FTS5 candidates and compute cosine similarity in Python, bypassing sqlite-vec's KNN for the filtered path.

**Note on sqlite-vec limitation:** The `vec0` virtual table does not support combining `entry_id IN (...)` with the KNN `MATCH` operator. True SQL-level pre-filtering is not possible. Instead, the filtered path fetches candidate embedding blobs directly from `vec_entries` by ID and scores them with cosine similarity in Python. This is still fast because the candidate set is small (typically 50-500 entries).

**New method in `engine.py`:**

```python
def search_vectors_filtered(
    self, query_embedding: list[float], candidate_ids: list[str], *, limit: int = 10
) -> list[tuple[str, float]]:
    """Cosine similarity search over a pre-filtered candidate set.
    Returns empty list immediately if candidate_ids is empty."""
```

Fetches embedding blobs for `candidate_ids` from `vec_entries`, unpacks to float32, computes cosine similarity against `query_embedding`, returns top `limit` results sorted by similarity. Returns `[]` immediately if `candidate_ids` is empty.

**Modified `recall()` pipeline in `store.py`:**

```
query -> FTS5 (limit = max(limit * 3, threshold * 2)) -> N candidates
  if N >= threshold:  filtered vector search (N candidates, Python cosine) -> RRF merge
  if N < threshold:   full vector scan (sqlite-vec KNN) -> RRF merge
```

Adaptive fallback: when FTS5 returns fewer than `fts_prefilter_threshold` candidates (default 50), the query is purely semantic and FTS5 has no useful signal. Fall back to full KNN scan. Never runs both.

The FTS candidate limit is set to `max(limit * 3, fts_prefilter_threshold * 2)` to ensure enough candidates are fetched for the pre-filter to activate at default settings.

**Config:**

`fts_prefilter_threshold: int = 50` on `AIngramConfig`.

### Files Changed

- `aingram/storage/engine.py` -- new `search_vectors_filtered()` method
- `aingram/store.py` -- reorder search pipeline in `recall()`
- `aingram/config.py` -- new `fts_prefilter_threshold` field

### Testing

- Unit test for `search_vectors_filtered()` returning correct subset
- Integration test comparing filtered vs full scan recall quality
- Benchmark test showing speedup at 10K+ entries

---

## 2. `aingram watch` Live Command

### Problem

Agent memory is invisible during execution. Users can't see what their agent is learning in real time without querying the database manually.

### Design

A live-tail CLI command that polls for new entries every 1 second and prints them with colorized output. New CLI commands use Typer (`@app.command()` with `typer.Option`), consistent with existing commands like `compact`.

**New module `aingram/watch.py`:**

```python
def watch_loop(db_path: str, *, json_output: bool = False) -> None:
    """Poll for new entries and print them. Blocks until KeyboardInterrupt."""
```

- Opens a read-only SQLite connection
- Tracks high-water mark via `rowid` (monotonically increasing, immune to timestamp collisions)
- Polls `SELECT * FROM memory_entries WHERE rowid > ? ORDER BY rowid ASC` every 1 second
- Catches `KeyboardInterrupt` for clean exit

**Default output format (colorized):**

```
[14:23:01] RESULT     confidence=0.91  "Reducing LR below 1e-5 eliminated loss oscillation"
[14:23:04] HYPOTHESIS confidence=0.74  "Warmup steps may interact with the LR effect"
```

Entry type colors: hypothesis=blue, method=gray, result=green, lesson=gold, observation=cyan, decision=white, meta=white. All other/unknown types render in white. Confidence shown as decimal, `--` if absent. Content truncated to terminal width with ellipsis.

**JSON output (`--json` flag):**

```json
{"timestamp":"2026-04-01T14:23:01","type":"result","confidence":0.91,"content":"...","entry_id":"abc123"}
```

One JSON object per line (JSONL), pipe-friendly.

**CLI:**

```
aingram watch              # colorized live tail
aingram watch --json       # JSONL output for piping
```

**Edge cases:**

- If the database file does not exist, print an error and exit with nonzero code.
- If the database exists but has no entries, print a waiting message and continue polling.

### Files Changed

- `aingram/watch.py` -- new module
- `aingram/cli.py` -- new `watch` subcommand with `--json` flag

### Testing

- Unit test for entry formatting (color and JSON modes)
- Integration test verifying new entries appear in watch output

---

## 3. Local Visualization

### Problem

AIngram has a knowledge graph, reasoning chains with outcome tracking, entity relationships, and cryptographic chain integrity. None of it is visible to users. The library feels invisible compared to tools with visual output.

### Design

`aingram viz` starts a localhost HTTP server and opens a browser tab with an interactive memory explorer. No Python dependencies beyond stdlib.

**Package structure:**

```
aingram/viz/
  __init__.py
  server.py        -- http.server subclass, JSON API endpoints
  static/
    index.html     -- single-page app, tabs + sidebar layout
    app.js         -- D3.js graph + timeline rendering, fetch calls
    style.css      -- styling, color coding, responsive sidebar
```

**API endpoints:**

| Endpoint | Returns |
|----------|---------|
| `GET /api/entities` | All entities with types, relationship edges, weights |
| `GET /api/chains` | Chain list with entry summaries, types, timestamps |
| `GET /api/entry?id=ID` | Full entry: content, confidence, surprise, hash, verification status, linked entities. Returns 400 if `id` missing, 404 if entry not found (JSON error body). |
| `GET /api/stats` | DB summary: entry count, entity count, chain count |

The server sets `Access-Control-Allow-Origin: *` for localhost use. The SPA must be served by the built-in server, not opened as a `file://` URL.

**Frontend layout:**

- **Tab bar** at top: Graph / Chains
- **Main panel**: active view (force-directed graph or vertical timeline)
- **Right sidebar**: entry inspector, populates on click

**Graph view:** D3 force-directed graph. Nodes colored by entity type, edges weighted by relationship strength. Click a node to see its entries in the sidebar.

**Chain timeline:** Vertical timeline per chain. Entries color-coded by type, matching the watch color scheme (hypothesis=blue, method=gray, result=green, lesson=gold, observation=cyan, decision/meta/other=white). Click an entry to inspect in sidebar.

**Entry inspector (sidebar):** Full content, confidence score, surprise score, entry type, timestamp, cryptographic hash, verification status, linked entities.

**CLI:**

```
aingram viz                # start on localhost:8420, open browser
aingram viz --port 9000    # custom port
aingram viz --no-open      # start server without opening browser
```

### Files Changed

- `aingram/viz/__init__.py` -- package init
- `aingram/viz/server.py` -- HTTP server with API routes
- `aingram/viz/static/index.html` -- single-page app shell
- `aingram/viz/static/d3.min.js` -- bundled D3.js (offline support)
- `aingram/viz/static/app.js` -- D3 rendering + API fetch logic
- `aingram/viz/static/style.css` -- layout and color coding
- `aingram/cli.py` -- new `viz` subcommand with `--port` and `--no-open` flags

### Dependencies

No Python dependencies beyond stdlib. D3.js is bundled as a minified file (`aingram/viz/static/d3.min.js`, ~90KB gzipped) to support offline and air-gapped environments, consistent with AIngram's local-first philosophy. Optional `aingram[viz]` extra in `pyproject.toml` as a marker for discoverability -- adds nothing to core.

### Testing

Unit tests for API endpoints returning correct JSON structure. Manual QA for the frontend.

---

## 4. Framework Integrations

### Problem

AIngram is invisible to users of popular agentic frameworks. Users searching for memory backends in LangChain, CrewAI, etc. won't find AIngram unless it ships adapter packages.

### Design

Thin adapter modules that implement each framework's memory interface by delegating to `MemoryStore`. Installed via pip extras.

**Package structure:**

```
aingram/integrations/
  __init__.py
  langchain.py
  crewai.py
  langgraph.py
  autogen.py
  smolagents.py
```

**Common adapter pattern:**

Each adapter wraps a `MemoryStore` instance and translates between the framework's memory protocol and AIngram's `remember()`/`recall()` API. The adapter:

- Accepts a `db_path` to create its own `MemoryStore`, or accepts an existing instance
- Maps framework-specific types (messages, chat history, task memory) to AIngram entry types
- Maps framework retrieval calls to `recall()` with appropriate filters

**Per-framework contracts:**

| Framework | Interface | Key Methods |
|-----------|-----------|-------------|
| LangChain | `BaseChatMessageHistory` | `add_message()`, `messages` property, `clear()` |
| CrewAI | `Memory` base class | `save()`, `search()`, `reset()` |
| LangGraph | `BaseStore` (langgraph >= 0.2) | `put()`, `get()`, `search()`. Note: `BaseCheckpointSaver` is for execution state, not memory. If `BaseStore` is unavailable, implement as a tool node instead. |
| AutoGen | `MemoryStore` protocol | `add()`, `query()`, `delete()` |
| smolagents | `MemoryStep` pattern | `write_memory()`, `retrieve_memory()` |

**Pip extras in `pyproject.toml`:**

```toml
[project.optional-dependencies]
langchain = ["langchain-core>=0.2"]
crewai = ["crewai>=0.40"]
langgraph = ["langgraph>=0.2"]
autogen = ["autogen-agentchat>=0.4"]
smolagents = ["smolagents>=1.0"]
```

**Build order:** LangChain first (largest user base, best-documented interface), then CrewAI, then the rest.

### Files Changed

- `aingram/integrations/__init__.py` -- package init
- `aingram/integrations/langchain.py` -- LangChain adapter
- `aingram/integrations/crewai.py` -- CrewAI adapter
- `aingram/integrations/langgraph.py` -- LangGraph adapter
- `aingram/integrations/autogen.py` -- AutoGen adapter
- `aingram/integrations/smolagents.py` -- smolagents adapter
- `pyproject.toml` -- new optional dependency groups

### Testing

Each adapter gets a unit test that mocks `MemoryStore` and verifies the framework interface contract. Integration tests with actual frameworks are optional/CI-gated.

---

## 5. int8 Embedding Quantization

### Problem

A 768-dim float32 embedding is 3KB. At 100K entries the vector index is ~300MB. Standard int8 quantization provides ~4x storage reduction with minimal recall impact. This is well-published academic work, distinct from Pro's QJL (1-bit + asymmetric scoring + learned partitions).

### Design

Per-vector min/max scaling to uint8. The int8 table becomes the source of truth; the existing `vec0` float32 table becomes a rebuildable cache so sqlite-vec KNN continues to work natively.

**New table `vec_entries_int8`:**

```sql
CREATE TABLE vec_entries_int8 (
    entry_id TEXT PRIMARY KEY,
    quantized BLOB NOT NULL,   -- uint8 bytes [0, 255]
    scale REAL NOT NULL,        -- (max_val - min_val) / 255
    min_val REAL NOT NULL       -- original minimum value
)
```

**Quantize operation (encode):**

For each vector: compute min/max, then for each float value:
- `scale = (max_val - min_val) / 255`
- `uint8_val = round((float_val - min_val) / scale)`
- **Zero-range edge case:** If `max_val == min_val`, set `scale = 0` and all uint8 values to 0. Dequantize recovers correctly: `0 * 0 + min_val = min_val`.

Store uint8 blob + scale + min_val. One-way destructive operation matching the `compact()` pattern.

Example: a 4-dim vector `[0.1, 0.5, 0.3, 0.9]` with min=0.1, max=0.9 gives scale=0.003137, uint8=[0, 128, 64, 255]. Dequantize: [0.1, 0.501, 0.301, 0.9].

**Method signature:** `def quantize(self, *, confirm: bool = False) -> None` with the same `ValueError` guard as `compact()`.

**Dequantize (rebuild/decode):**

Reconstruct float32 from uint8: `float_val = uint8_val * scale + min_val`. Insert into `vec0` table. Rebuild is automatic after quantize and on any future DB open when the int8 table has data but `vec0` is stale.

**Staleness detection:** A `quantized_version` counter in `db_metadata` increments on each quantize or rebuild. A matching `vec_cache_version` tracks the `vec0` table state. When `quantized_version > vec_cache_version`, the cache is stale and triggers rebuild on open.

**New entries after quantize:**

When quantized mode is active, `remember()` stores both float32 (in `vec0` for immediate KNN) and int8 (in `vec_entries_int8` as source of truth).

**Compatibility with `compact()`:**

Independent operations. Compact reduces dimensions (768 -> 256), quantize reduces precision (float32 -> int8). Both together: 768 * 4 bytes -> 256 * 1 byte = 12x total reduction.

**CLI:**

```
aingram quantize --yes       # one-way conversion, stores int8 source of truth
aingram quantize --rebuild   # rebuild vec0 float32 cache from int8 data
```

### Files Changed

- `aingram/storage/schema.py` -- `vec_entries_int8` table in schema migration
- `aingram/storage/engine.py` -- `quantize_all_embeddings()`, `rebuild_vec_from_int8()`, `store_entry_int8()`, `is_quantized()`
- `aingram/store.py` -- `quantize()` method, updated `remember()` for dual-write
- `aingram/cli.py` -- new `quantize` subcommand

### Testing

- Unit test for quantize/dequantize round-trip accuracy (max error < 0.01 per dimension)
- Integration test verifying search recall quality before vs after quantization
- Storage size comparison test

---

## Build Order

Prioritized by traction per hour of work, matching the source document's recommendation:

1. **FTS5 pre-filter** -- pure performance win, improves benchmark numbers
2. **`aingram watch`** -- demo gold, minimal implementation effort
3. **Local visualization** -- highest traction impact, produces shareable screenshots
4. **Framework integrations** -- opens distribution channels (LangChain first)
5. **int8 quantization** -- solid storage improvement

## What Stays in Pro

- QJL (1-bit quantization + asymmetric scoring + learned partition predictor)
- Hopfield associative memory
- HDBSCAN consolidation
- FSRS spaced repetition
- Hosted dashboard / multi-user visualization
