# AIngram — Developer Reference for AI Coding Agents

## Project Identity

AIngram is a local-first, privacy-first agent memory system built on SQLite and sqlite-vec. All state lives in a single `.db` file — no cloud services, no external APIs at runtime. Core primitives are: typed `MemoryEntry` records, a knowledge graph (entities + relationships), reasoning chains, ONNX vector embeddings (Nomic MiniLM, 768-dim), and optional int8-quantized embeddings. AIngram is **not** a general-purpose vector database, not a cloud memory service, and not an ORM.

## Dev Environment

- Python ≥ 3.11 required
- Install all dependencies: `pip install -e ".[dev]"`
- The ONNX embedding model (Nomic MiniLM) is downloaded to `~/.aingram/models/` on first instantiation of `MemoryStore` if not already cached. Tests that require embeddings inject a mock embedder to avoid this download — don't break that pattern.

## Commands

```
pytest                    # run all tests
pytest tests/test_X.py    # run a single file
pytest -x                 # stop on first failure
ruff check .              # lint
ruff check --fix .        # lint + auto-fix
ruff format .             # format
```

## Module Map

| Path | Responsibility |
|------|----------------|
| `aingram/store.py` | `MemoryStore` — the only public API for reads and writes. All business logic lives here. |
| `aingram/storage/engine.py` | `StorageEngine` — low-level SQLite CRUD and threading lock. No business logic. |
| `aingram/storage/schema.py` | DDL strings, `SCHEMA_VERSION`, `apply_schema()`, migration functions (`_migrate_vN_to_vM`). |
| `aingram/storage/queries.py` | Shared query utilities (reciprocal rank fusion, etc.). |
| `aingram/trust/` | Ed25519 signing, verification, content hashing, RFC-8785 canonicalization. |
| `aingram/trust/session.py` | `SessionManager` — per-session Ed25519 keypair and sequence counter. |
| `aingram/processing/` | `NomicEmbedder` — ONNX inference for vector embeddings. |
| `aingram/models/` | `ModelManager` — ONNX model file download and caching. |
| `aingram/extraction/` | Entity/relationship extraction (local LLM via Ollama, or Anthropic Sonnet). |
| `aingram/consolidation/` | Memory consolidation: decay, merge, contradiction detection, knowledge synthesis. |
| `aingram/graph/` | Knowledge graph traversal for graph-augmented recall. |
| `aingram/integrations/` | Thin adapters: LangChain, CrewAI, LangGraph, AutoGen, smolagents. |
| `aingram/viz/` | Local HTTP visualization server (`aingram viz`). |
| `aingram/watch.py` | `watch_loop()` — live tail of new memory entries. |
| `aingram/cli.py` | Typer CLI entry point — all `aingram` subcommands. |
| `aingram/worker.py` | Background task queue processor (entity extraction jobs). |
| `aingram/config.py` | `AIngramConfig` — layered config (kwargs > env > TOML > defaults). |
| `aingram/mcp_server.py` | MCP server for Claude/Cursor tool integration. |
| `aingram/types.py` | All public dataclasses and enums. |
| `aingram/exceptions.py` | Exception hierarchy. |

## Architecture Invariants

1. **`MemoryStore` is the only public write path.** Never call `engine.store_entry()` or `_conn.execute()` directly from outside the `storage/` package. All business logic — signing, embedding, FTS indexing, dual-write — lives in `MemoryStore.remember()`.

2. **All direct `_conn` access must hold `engine._lock`.** The engine uses `threading.Lock`. Any raw `_conn.execute()` call outside `StorageEngine` methods must be wrapped with `with engine._lock:`.

3. **vec0 virtual tables: DELETE + INSERT only, never UPDATE.** sqlite-vec's `vec0` does not support `UPDATE` on the embedding column. Always delete the row and re-insert.

4. **Schema changes require a migration function.** Bump `SCHEMA_VERSION` in `schema.py` and add a corresponding `_migrate_vN_to_vM(conn)` function. The `apply_schema()` function calls all applicable migrations in order.

5. **CLI commands use the Typer `@app.command()` pattern.** Never use raw `argparse` or `sys.argv`. The database path is passed through `ctx.obj['db']`, which is a resolved `str` (not a `Path`). Do not wrap it in `Path()` — pass it directly to `MemoryStore(ctx.obj['db'])`.

6. **Chunk SQL `IN` clauses to ≤ 900 items.** SQLite's default `SQLITE_LIMIT_VARIABLE_NUMBER` is 999. Use `_SQLITE_VAR_LIMIT = 900` (defined in `engine.py`) when building parameterized `IN (?)` queries.

7. **Dual-write int8 embeddings in `remember()`.** When `engine.is_quantized()` is true, call `engine.store_entry_int8(entry_id, embedding)` after `store_entry()`. The int8 table (`vec_entries_int8`) is the source of truth; `vec_entries` (vec0 float32) is the rebuildable cache.

8. **FTS pre-filter adaptive fallback.** If FTS returns ≥ `fts_prefilter_threshold` candidates, use `search_vectors_filtered()` (Python cosine similarity over the candidate set). If fewer, fall back to full KNN via `search_vectors()`. This is required because sqlite-vec's `MATCH` operator cannot be combined with `IN` filtering.

## Design Rationale

**Trust system.** Every `MemoryEntry` is Ed25519-signed. The `entry_id` is derived from content + parent IDs + public key, forming a DAG. Any post-write mutation of content or chain linkage is detectable during `verify()`. The append-only design follows from this: entries are never mutated; consolidation creates new entries flagged `consolidated=1`.

**FTS pre-filter architecture.** sqlite-vec's `vec0` virtual table cannot combine KNN `MATCH` with `IN` for candidate filtering. The workaround: fetch float32 blobs via a regular `IN` query on `vec_entries`, then compute cosine similarity in Python. This is the `search_vectors_filtered()` path.

**int8 quantization design.** `vec_entries_int8` stores uint8-quantized embeddings as the permanent record (per-vector min/max scaling). `vec_entries` (vec0 float32) is a rebuildable cache. After `quantize()`, new entries written via `remember()` dual-write to both tables. `rebuild_vec_from_int8()` reconstructs the float32 cache at any time.

**Layered configuration.** `AIngramConfig` merges in priority order: constructor kwargs (highest) → env vars → `~/.aingram/config.toml` → dataclass defaults (lowest). Library users override with kwargs; CLI users use env vars; power users configure via TOML.

## Testing Notes

- All tests live in `tests/`; `pytest` is configured in `pyproject.toml`
- Tests create temporary SQLite DBs — no shared persistent state between tests
- Tests that require embeddings inject a mock embedder to avoid model downloads; don't change this pattern
- To test a specific integration adapter, install the relevant extra first: e.g., `pip install -e ".[langchain]"`
- Schema migration tests should assert `get_schema_version()` equals `SCHEMA_VERSION` after `apply_schema()`
