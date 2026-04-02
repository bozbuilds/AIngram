# Design Spec: AGENTS.md + llms.txt
**Date:** 2026-04-01
**Status:** Approved

---

## Goal

Make the AIngram repo immediately useful to two distinct AI agent audiences:

1. **Coding agents** (Claude Code, Codex, Cursor) working *on* the AIngram codebase — they need to understand architecture, invariants, and dev workflow without reading every file.
2. **Consumer agents** (LLMs integrating AIngram as a library) — they need to discover the public API, CLI, and integration adapters quickly from the repo root.

Two files satisfy both needs with zero overlap: `AGENTS.md` (coding agents) and `llms.txt` (consumer agents).

---

## Out of Scope

- Enriching in-code docstrings
- Creating `llms-full.txt` (dual-file llmstxt.org convention)
- Adding structured JSON/YAML blocks to either file
- Changes to `README.md`, `CONTRIBUTING.md`, or any source file

---

## File 1: AGENTS.md

**Location:** repo root (`AGENTS.md`)
**Audience:** Claude Code, Codex, Cursor, and any agent that reads `AGENTS.md` before editing
**Tone:** direct, imperative, no marketing language

### Section 1 — Project Identity

One paragraph. Cover:
- AIngram is a local-first, privacy-first agent memory system built on SQLite + sqlite-vec
- All state lives in a single `.db` file; no cloud, no external services at runtime
- Core primitives: typed `MemoryEntry` records, knowledge graph (entities + relationships), reasoning chains, ONNX vector embeddings (Nomic MiniLM, 768-dim), optional int8-quantized embeddings
- What it is NOT: not a general-purpose vector DB, not a cloud memory service, not an ORM

### Section 2 — Dev Environment

- Python ≥ 3.11 required
- Install: `pip install -e ".[dev]"`
- First embed call downloads Nomic MiniLM ONNX model to `~/.aingram/models/` — tests that trigger embedding should use a mock embedder or the model will be downloaded

### Section 3 — Commands

Exact commands, no prose:

```
pytest                    # run all tests
pytest tests/test_X.py    # single file
pytest -x                 # stop on first failure
ruff check .              # lint
ruff check --fix .        # lint + auto-fix
ruff format .             # format
```

### Section 4 — Module Map

One line per module/subpackage. Specify what each owns and, where ambiguous, what it does NOT own.

| Path | Responsibility |
|------|---------------|
| `aingram/store.py` | `MemoryStore` — the only public API for reads and writes |
| `aingram/storage/engine.py` | `StorageEngine` — low-level SQLite CRUD, threading lock, no business logic |
| `aingram/storage/schema.py` | DDL strings, `SCHEMA_VERSION`, `apply_schema()`, migration functions |
| `aingram/storage/queries.py` | Shared query utilities (reciprocal rank fusion, etc.) |
| `aingram/trust/` | Ed25519 signing, verification, content hashing, RFC-8785 canonicalization |
| `aingram/trust/session.py` | `SessionManager` — per-session keypair and sequence counter |
| `aingram/processing/` | `NomicEmbedder` — ONNX inference for vector embeddings |
| `aingram/models/` | `ModelManager` — ONNX model file download and caching |
| `aingram/extraction/` | Entity/relationship extraction (local LLM via Ollama or Sonnet) |
| `aingram/consolidation/` | Memory consolidation: decay, merge, contradiction detection, knowledge synthesis |
| `aingram/graph/` | Knowledge graph traversal for graph-augmented recall |
| `aingram/integrations/` | Thin adapters for LangChain, CrewAI, LangGraph, AutoGen, smolagents |
| `aingram/viz/` | Local HTTP visualization server (`aingram viz`) |
| `aingram/watch.py` | `watch_loop()` — live tail of new memory entries |
| `aingram/cli.py` | Typer CLI entry point — all `aingram` subcommands |
| `aingram/worker.py` | Background task queue processor (entity extraction jobs) |
| `aingram/config.py` | `AIngramConfig` — layered config (kwargs > env > TOML > defaults) |
| `aingram/mcp_server.py` | MCP server for Claude/Cursor tool integration |
| `aingram/types.py` | All public dataclasses and enums |
| `aingram/exceptions.py` | Exception hierarchy |

### Section 5 — Architecture Invariants

Rules a coding agent must never violate. Present as a numbered list with a brief rationale for each:

1. **`MemoryStore` is the only public write path.** Never call `engine.store_entry()` or `_conn.execute()` directly from outside the `storage/` package — all business logic (signing, embedding, FTS indexing, dual-write) lives in `MemoryStore.remember()`.

2. **All direct `_conn` access must hold `engine._lock`.** The engine uses `threading.Lock`. Any raw `_conn.execute()` call outside `StorageEngine` methods must be wrapped with `with engine._lock:`.

3. **vec0 virtual tables: DELETE + INSERT only, never UPDATE.** sqlite-vec's `vec0` does not support `UPDATE` on the embedding column. Always delete the row and re-insert.

4. **Schema changes require a migration function.** Bump `SCHEMA_VERSION` in `schema.py` and add a corresponding `_migrate_vN_to_vM(conn)` function. The `apply_schema()` function calls all applicable migrations in order.

5. **CLI commands use the Typer `@app.command()` pattern.** Never use raw `argparse` or `sys.argv`. The database path is passed through `ctx.obj['db']`, which is a resolved `str` (not a `Path`). Do not wrap it in `Path()` unnecessarily — pass it directly to `MemoryStore(ctx.obj['db'])`.

6. **Chunk SQL `IN` clauses to ≤ 900 items.** SQLite's default `SQLITE_LIMIT_VARIABLE_NUMBER` is 999. Use `_SQLITE_VAR_LIMIT = 900` (defined in `engine.py`) when building parameterized `IN (?)` queries.

7. **Dual-write int8 embeddings in `remember()`.** When `engine.is_quantized()` is true, call `engine.store_entry_int8(entry_id, embedding)` after `store_entry()`. The int8 table is the source of truth; `vec_entries` is the rebuildable float32 cache.

8. **FTS pre-filter adaptive fallback.** If FTS returns ≥ `fts_prefilter_threshold` candidates, use `search_vectors_filtered()` (Python cosine similarity over the candidate set). If fewer, fall back to full KNN via `search_vectors()`. This is required because sqlite-vec's `MATCH` operator cannot be combined with `IN` filtering.

### Section 6 — Design Rationale

Explain the *why* behind the less-obvious decisions:

**Trust system:** Every `MemoryEntry` is Ed25519-signed. The `entry_id` is derived from content + parent IDs + public key, forming a DAG. This makes any post-write mutation of content or chain linkage detectable during `verify()`. The append-only design follows from this — entries are never mutated, consolidation creates new entries flagged `consolidated=1`.

**FTS pre-filter architecture:** sqlite-vec's `vec0` virtual table does not support combining KNN `MATCH` with `IN` for candidate filtering. The workaround is to fetch float32 blobs directly via a regular `IN` query on `vec_entries`, then compute cosine similarity in Python. This is the `search_vectors_filtered()` path.

**int8 quantization design:** `vec_entries_int8` stores uint8 quantized embeddings as the permanent record. `vec_entries` (vec0 float32) is a rebuildable cache. After a `quantize()` call, new entries written via `remember()` dual-write to both tables. `rebuild_vec_from_int8()` can reconstruct the float32 cache from the int8 source at any time.

**Layered configuration:** `AIngramConfig` merges in priority order: constructor kwargs (highest) → env vars → `~/.aingram/config.toml` → dataclass defaults (lowest). This means library users can always override with kwargs; CLI users use env vars; power users configure via TOML.

### Section 7 — Testing Notes

- All tests live in `tests/`; uses `pytest` with config in `pyproject.toml`
- Tests create temporary SQLite DBs — no shared persistent state
- Tests that require embeddings inject a mock embedder to avoid model downloads
- To test a specific integration adapter, install the relevant extra first: e.g., `pip install -e ".[langchain]"`
- Schema migration tests should assert `get_schema_version()` equals `SCHEMA_VERSION` after `apply_schema()`

---

## File 2: llms.txt

**Location:** repo root (`llms.txt`)
**Audience:** LLMs integrating AIngram as a library — reading the repo to understand the API
**Format:** Markdown. Header block → inline content sections → links block at end
**Tone:** reference documentation, not tutorial prose

### Header Block

```
# AIngram

> Local-first, privacy-first agent memory in a single SQLite file. Apache-2.0.

[3-sentence summary]
- What AIngram is: a Python library that gives AI agents persistent, signed, searchable memory
- What it stores: typed MemoryEntry records with hybrid vector+FTS search, optional knowledge graph, and reasoning chains
- What makes it distinctive: fully local (no cloud), Ed25519-signed entries, ONNX embeddings, MCP-ready
```

### Installation Section (inline)

Show all relevant install variants:
```
pip install aingram                         # core
pip install "aingram[mcp]"                  # + MCP server
pip install "aingram[extraction]"           # + local entity extraction (GLiNER)
pip install "aingram[api]"                  # + Anthropic API (Sonnet extractor)
pip install "aingram[langchain]"            # + LangChain adapter
pip install "aingram[crewai]"               # + CrewAI adapter
pip install "aingram[langgraph]"            # + LangGraph adapter
pip install "aingram[autogen]"              # + AutoGen adapter
pip install "aingram[smolagents]"           # + smolagents adapter
pip install "aingram[all]"                  # all optional extras
```

### Core Python API Section (inline)

**`MemoryStore` constructor** — full signature:
```python
MemoryStore(
    db_path: str,
    *,
    agent_name: str = 'default',
    embedder=None,
    embedding_dim: int | None = None,
    models_dir: Path | str | None = None,
    extractor=None,
    config: AIngramConfig | None = None,
)
```
Note: creates the database and downloads the ONNX model on first instantiation if not already cached in `models_dir`.

**`remember(content, ...)` ** — stores a memory entry, returns `entry_id: str`:
```python
entry_id = mem.remember(
    content,                        # str or dict
    entry_type='observation',       # hypothesis|method|result|lesson|observation|decision|meta
    chain_id=None,                  # attach to a reasoning chain
    tags=None,                      # list[str]
    confidence=None,                # float 0–1
    metadata=None,                  # dict, stored as JSON
    # also accepts: parent_entry_id=None, parent_ids=None (DAG linkage)
)
```

**`recall(query, ...)` ** — hybrid semantic search, returns `list[EntrySearchResult]`:
```python
results = mem.recall(
    query,                  # str — natural language query
    limit=20,               # max results
    verify=True,            # verify Ed25519 signatures
    entry_type=None,        # filter by type
    chain_id=None,          # filter by reasoning chain
    session_id=None,        # filter by session
)
# result.entry: MemoryEntry, result.score: float, result.verified: bool | None
```

**`get_context(query, *, max_tokens=2000)` ** — returns top memories as a single string, token-budget aware.

**Other key methods** (name + one-sentence each):
- `create_chain(title) → str` — create a reasoning chain, returns `chain_id`
- `complete_chain(chain_id, *, outcome)` — mark chain complete with outcome (confirmed/refuted/partial/inconclusive/error)
- `reference(*, source_id, target_id, reference_type)` — create a cross-reference between entries
- `consolidate()` — run decay, merge, contradiction detection, and knowledge synthesis
- `compact(*, confirm, target_dim)` — one-way Matryoshka embedding truncation
- `quantize(*, confirm)` — one-way uint8 quantization (~4x storage reduction)
- `export_json(path)` — export full database to JSON
- `import_json(path, *, merge)` — import from JSON export
- `verify(session_id=None) → VerificationResult` — verify Ed25519 signature chain for a session (defaults to current session; `VerificationResult` has `.valid: bool`, `.session_id: str`, `.errors: list[str]`, `.entries_checked: int`)
- `stats` property — `dict` with `entry_count`, `db_size_bytes`, `embedding_dim`
- `close()` — close the database (also supports context manager)

### CLI Reference Section (inline)

One line per command with key flags:
```
aingram [--db PATH] [--no-telemetry] <command>

setup                             report model cache path and print download instructions
add TEXT                          store a memory entry
search QUERY [--limit N]          semantic search
status                            show DB stats
entities [--limit N]              list known entities
graph NAME                        expand entity relationships
consolidate                       run memory consolidation
compact --target-dim N --yes      truncate embeddings (one-way)
quantize [--yes] [--rebuild]      uint8 quantize or rebuild float32 cache
watch [--json]                    live tail new entries
viz [--port N] [--no-open]        start local visualization server
export PATH [--agent NAME]        export to JSON
import PATH [--merge]             import from JSON
agent create NAME [--role ROLE]   create agent token
agent list                        list agents and roles
agent revoke NAME                 revoke agent token
```

### Framework Integrations Section (inline)

One block per adapter showing the import path and a minimal usage snippet (3–5 lines):

**LangChain:**
```python
from aingram.integrations.langchain import AIngramChatMessageHistory
history = AIngramChatMessageHistory(db_path='memory.db')
# use as chain.memory or pass to ConversationChain
```

**CrewAI:**
```python
from aingram.integrations.crewai import AIngramCrewMemory
memory = AIngramCrewMemory(db_path='memory.db')
memory.save("The user prefers concise answers")
results = memory.search("user preferences")
```

**LangGraph:**
```python
from aingram.integrations.langgraph import AIngramLangGraphStore
store = AIngramLangGraphStore(db_path='memory.db')
store.put(('agent', 'task_001'), 'plan', {'steps': [...]})
item = store.get(('agent', 'task_001'), 'plan')  # returns _StoreItem | None
if item:
    value = item.value  # the original dict
```

**AutoGen:**
```python
from aingram.integrations.autogen import AIngramAutogenMemory
memory = AIngramAutogenMemory(db_path='memory.db')
memory.add("Completed subtask 3 successfully")
context = memory.query("subtask status")  # returns str (newline-joined content)
print(context)
```

**smolagents:**
```python
from aingram.integrations.smolagents import AIngramSmolagentsMemory
memory = AIngramSmolagentsMemory(db_path='memory.db')
memory.write_memory("Tool call failed with timeout")
results = memory.retrieve_memory("tool failures")
```

### Configuration Section (inline, brief)

```
Environment variables (all optional):
  AINGRAM_MODELS_DIR                path to ONNX model cache (default: ~/.aingram/models)
  AINGRAM_EMBEDDING_DIM             embedding dimensions (default: 768)
  AINGRAM_LLM_URL                   Ollama base URL (default: http://localhost:11434)
  AINGRAM_LLM_MODEL                 Ollama model name (default: mistral)
  AINGRAM_EXTRACTOR_MODE            none | local | sonnet (default: none)
  AINGRAM_EXTRACTOR_MODEL           extractor model name (default: aingram-extractor)
  AINGRAM_ONNX_PROVIDER             cpu | cuda | npu (default: auto)
  AINGRAM_TELEMETRY_ENABLED         true | false (default: true)
  AINGRAM_LOG_LEVEL                 DEBUG | INFO | WARNING | ERROR (default: INFO)
  AINGRAM_WORKER_ENABLED            true | false (default: true)
  AINGRAM_CONSOLIDATION_INTERVAL    seconds between auto-consolidation runs (default: unset)
  AINGRAM_FTS_PREFILTER_THRESHOLD   min FTS candidates before filtered vector search (default: 50)

Config file: ~/.aingram/config.toml (TOML, same keys without AINGRAM_ prefix)
```

### Links Section

```
## Optional: more detail

- [README](README.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Examples](examples/)
- [Docs](docs/)
- [MCP server](aingram/mcp_server.py)
- [Schema & migrations](aingram/storage/schema.py)
- [Public types](aingram/types.py)
- [Tests](tests/)
```

---

## Maintenance Notes

- **AGENTS.md** should be updated whenever: a new subpackage is added, a new architectural invariant is established, or the test/lint commands change.
- **llms.txt** should be updated whenever: the `MemoryStore` public API changes, a new integration adapter is added, or new CLI commands are introduced.
- Neither file is auto-generated — they are manually maintained human-readable documents.
