# AGENTS.md + llms.txt Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two root-level files — `AGENTS.md` for AI coding agents and `llms.txt` for AI agents consuming AIngram as a library.

**Architecture:** Two standalone markdown files at repo root. No source changes. `AGENTS.md` covers dev workflow, module map, architectural invariants, and design rationale. `llms.txt` covers the public Python API, CLI reference, framework integration snippets, and configuration — hybrid format (inline content + links).

**Tech Stack:** Plain markdown; no tooling required to generate or validate.

**Spec:** `docs/superpowers/specs/2026-04-01-agents-llms-txt-design.md`

---

## Chunk 1: AGENTS.md

### Task 1: Write AGENTS.md

**Files:**
- Create: `AGENTS.md` (repo root)

- [ ] **Step 1: Write AGENTS.md**

Create the file `AGENTS.md` at the repo root with exactly this content:

````markdown
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
````

- [ ] **Step 2: Verify module map paths exist**

Run from the repo root (`D:/Misc/GitHub/portfolio/AIngram`):

```bash
cd "D:/Misc/GitHub/portfolio/AIngram" && ls \
   aingram/store.py aingram/storage/engine.py aingram/storage/schema.py \
   aingram/storage/queries.py aingram/trust/ aingram/processing/ \
   aingram/models/ aingram/extraction/ aingram/consolidation/ \
   aingram/graph/ aingram/integrations/ aingram/viz/ aingram/watch.py \
   aingram/cli.py aingram/worker.py aingram/config.py aingram/mcp_server.py \
   aingram/types.py aingram/exceptions.py
```

Expected: all listed paths exist. Note: `aingram/pipeline.py`, `aingram/security/`, and `aingram/telemetry.py` also exist in the package but are intentionally omitted from the module map (internal plumbing not relevant to contributors).

- [ ] **Step 3: Verify CLI commands match cli.py**

```bash
cd "D:/Misc/GitHub/portfolio/AIngram" && grep "@app.command" aingram/cli.py
```

Expected: output contains entries for `setup`, `status`, `add`, `search`, `entities`, `graph`, `consolidate`, `compact`, `watch`, `viz`, `quantize`, `export`, and `'import'` (registered as `@app.command('import')`, Python function name is `import_backup`). Confirm every command listed in the Module Map's `cli.py` row and in the `llms.txt` CLI Reference is present in this grep output.

- [ ] **Step 4: Commit AGENTS.md**

```bash
git add AGENTS.md
git commit -m "docs: add AGENTS.md — coding agent reference"
```

---

## Chunk 2: llms.txt

### Task 2: Write llms.txt — header, installation, core Python API

**Files:**
- Create: `llms.txt` (repo root, partial — completed in Task 3)

- [ ] **Step 1: Write the header block and installation section**

Create `llms.txt` with this opening content (Task 3 will append the remaining sections):

````markdown
# AIngram

> Local-first, privacy-first agent memory in a single SQLite file. Apache-2.0.

AIngram is a Python library that gives AI agents persistent, signed, searchable memory — all stored locally in one SQLite file with no cloud dependency. It stores typed `MemoryEntry` records with hybrid vector+FTS5 search, an optional knowledge graph (entities + relationships), and reasoning chains. What makes it distinctive: fully local ONNX embeddings (Nomic MiniLM), Ed25519-signed entries for tamper-evidence, and built-in MCP server support.

## Installation

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

## Core Python API

### MemoryStore

```python
from aingram import MemoryStore

mem = MemoryStore(
    db_path,                          # str — path to SQLite file (created if absent)
    *,
    agent_name='default',             # str — name used to identify this agent session
    embedder=None,                    # override the default NomicEmbedder
    embedding_dim=None,               # int | None — must match existing DB if set
    models_dir=None,                  # Path | str | None — ONNX model cache location
    extractor=None,                   # override entity extractor
    config=None,                      # AIngramConfig | None
)
# Note: creates the database and downloads the ONNX model on first instantiation
# if not already cached in models_dir (default: ~/.aingram/models/).
```

### remember — store a memory entry

```python
entry_id: str = mem.remember(
    content,                          # str or dict
    entry_type='observation',         # hypothesis|method|result|lesson|observation|decision|meta
    chain_id=None,                    # str | None — attach to a reasoning chain
    tags=None,                        # list[str] | None
    confidence=None,                  # float 0–1 | None
    metadata=None,                    # dict | None — stored as JSON
    # also accepts: parent_entry_id=None, parent_ids=None (DAG linkage)
)
```

### recall — hybrid semantic search

```python
results: list[EntrySearchResult] = mem.recall(
    query,                            # str — natural language query
    limit=20,                         # int — max results to return
    verify=True,                      # bool — verify Ed25519 signatures
    entry_type=None,                  # str | None — filter by type
    chain_id=None,                    # str | None — filter by reasoning chain
    session_id=None,                  # str | None — filter by session
    entry_id=None,                    # str | None — fetch a specific entry by ID
)
# Each result: result.entry (MemoryEntry), result.score (float), result.verified (bool | None)
```

### get_context — token-budget-aware string for LLM prompts

```python
context: str = mem.get_context(query, max_tokens=2000)
# Returns top memories as a single newline-separated string, respecting token budget.
```

### Other key methods

```python
mem.create_chain(title: str) -> str
# Create a reasoning chain; returns chain_id.

mem.complete_chain(chain_id, *, outcome)
# Mark chain complete. outcome: confirmed|refuted|partial|inconclusive|error

mem.reference(*, source_id, target_id, reference_type)
# Cross-reference two entries. reference_type: builds_on|contradicts|supports|refines|supersedes

mem.consolidate()
# Run decay + merge + contradiction detection + knowledge synthesis.

mem.compact(*, confirm: bool, target_dim: int)
# One-way Matryoshka embedding truncation. Requires confirm=True.

mem.quantize(*, confirm: bool)
# One-way uint8 quantization (~4x storage reduction). Requires confirm=True.

mem.export_json(path)
# Export full database to JSON.

mem.import_json(path, *, merge: bool = False)
# Import from a JSON export. merge=False requires an empty target DB.

mem.verify(session_id=None) -> VerificationResult
# Verify Ed25519 signature chain. Defaults to current session.
# VerificationResult: .valid (bool), .session_id (str), .errors (list[str]), .entries_checked (int)

mem.stats  # property: dict — entry_count, db_size_bytes, embedding_dim

mem.close()  # close DB — also works as context manager: with MemoryStore(...) as mem:
```
````

- [ ] **Step 2: Verify MemoryStore constructor signature matches store.py**

```bash
cd "D:/Misc/GitHub/portfolio/AIngram" && grep -A 15 "def __init__" aingram/store.py | head -18
```

Confirm that `db_path`, `agent_name`, `embedder`, `embedding_dim`, `models_dir`, `extractor`, `config` all appear in the actual signature.

- [ ] **Step 3: Verify remember() kwargs match store.py**

```bash
cd "D:/Misc/GitHub/portfolio/AIngram" && grep -A 12 "def remember" aingram/store.py | head -15
```

Confirm `content`, `entry_type`, `chain_id`, `parent_entry_id`, `parent_ids`, `tags`, `confidence`, `metadata` are all present in the actual method.

### Task 3: Write llms.txt — CLI, integrations, configuration, links

**Files:**
- Modify: `llms.txt` (append remaining sections)

- [ ] **Step 1: Append CLI reference section to llms.txt**

Append this section to `llms.txt`:

````markdown
## CLI Reference

```
aingram [--db PATH] [--no-telemetry] <command>

setup                             report model cache path and print download instructions
add TEXT                          store a memory entry
search QUERY [--limit N]          semantic search
status                            show DB stats (entry count, size, embedding dim)
entities [--limit N]              list known entities
graph NAME                        expand entity relationships in the knowledge graph
consolidate                       run memory consolidation (decay, merge, synthesis)
compact --target-dim N --yes      truncate embeddings to N dims (one-way, Matryoshka)
quantize [--yes] [--rebuild]      uint8 quantize embeddings or rebuild float32 cache
watch [--json]                    live tail new entries (Ctrl+C to stop)
viz [--port N] [--no-open]        start local visualization server (default port: 8420)
export PATH [--agent NAME]        export database to JSON
import PATH [--merge]             import from JSON export
agent create NAME [--role ROLE]   create agent bearer token (role: reader|contributor|admin)
agent list                        list all agents and their roles
agent revoke NAME                 revoke an agent token
```
````

- [ ] **Step 2: Append framework integrations section to llms.txt**

Append this section:

````markdown
## Framework Integrations

**LangChain** — `pip install "aingram[langchain]"`

```python
from aingram.integrations.langchain import AIngramChatMessageHistory
history = AIngramChatMessageHistory(db_path='memory.db', session_id='my-session')
# Pass to RunnableWithMessageHistory or ConversationChain as the message history backend.
```

**CrewAI** — `pip install "aingram[crewai]"`

```python
from aingram.integrations.crewai import AIngramCrewMemory
memory = AIngramCrewMemory(db_path='memory.db')
memory.save("The user prefers concise answers")
results = memory.search("user preferences")   # returns list[EntrySearchResult]
memory.reset()                                # no-op (AIngram is append-only)
```

**LangGraph** — `pip install "aingram[langgraph]"`

```python
from aingram.integrations.langgraph import AIngramLangGraphStore
store = AIngramLangGraphStore(db_path='memory.db')
store.put(('agent', 'task_001'), 'plan', {'steps': [...]})
item = store.get(('agent', 'task_001'), 'plan')   # returns _StoreItem | None
if item:
    value = item.value                             # the original dict
# Note: search() does not enforce namespace scoping (single-file backend limitation).
```

**AutoGen** — `pip install "aingram[autogen]"`

```python
from aingram.integrations.autogen import AIngramAutogenMemory
memory = AIngramAutogenMemory(db_path='memory.db')
memory.add("Completed subtask 3 successfully")
context = memory.query("subtask status")   # returns str (newline-joined content)
print(context)
```

**smolagents** — `pip install "aingram[smolagents]"`

```python
from aingram.integrations.smolagents import AIngramSmolagentsMemory
memory = AIngramSmolagentsMemory(db_path='memory.db')
memory.write_memory("Tool call failed with timeout")
results = memory.retrieve_memory("tool failures")   # returns list[EntrySearchResult]
```
````

- [ ] **Step 3: Append configuration and links sections to llms.txt**

Append this final block:

````markdown
## Configuration

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
```

Config file: `~/.aingram/config.toml` — same keys as env vars, without the `AINGRAM_` prefix:

```toml
log_level = "DEBUG"
extractor_mode = "local"
llm_url = "http://localhost:11434"
fts_prefilter_threshold = 100
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
````

- [ ] **Step 4: Verify integration class names and import paths**

```bash
cd "D:/Misc/GitHub/portfolio/AIngram" && grep "^class " \
   aingram/integrations/langchain.py \
   aingram/integrations/crewai.py \
   aingram/integrations/langgraph.py \
   aingram/integrations/autogen.py \
   aingram/integrations/smolagents.py
```

Expected output:
```
aingram/integrations/langchain.py:class AIngramChatMessageHistory(BaseChatMessageHistory):
aingram/integrations/crewai.py:class AIngramCrewMemory:
aingram/integrations/langgraph.py:class AIngramLangGraphStore:
aingram/integrations/autogen.py:class AIngramAutogenMemory:
aingram/integrations/smolagents.py:class AIngramSmolagentsMemory:
```

- [ ] **Step 5: Verify env vars match config.py**

```bash
cd "D:/Misc/GitHub/portfolio/AIngram" && grep "AINGRAM_" aingram/config.py
```

Expected output — all 12 `AINGRAM_*` names must appear:
```
    if v := env.get('AINGRAM_MODELS_DIR'):
    if v := env.get('AINGRAM_EMBEDDING_DIM'):
    if v := env.get('AINGRAM_LLM_URL'):
    if v := env.get('AINGRAM_LLM_MODEL'):
    if v := env.get('AINGRAM_LOG_LEVEL'):
    if v := env.get('AINGRAM_WORKER_ENABLED'):
    if v := env.get('AINGRAM_CONSOLIDATION_INTERVAL'):
    if v := env.get('AINGRAM_EXTRACTOR_MODE'):
    if v := env.get('AINGRAM_EXTRACTOR_MODEL'):
    if v := env.get('AINGRAM_ONNX_PROVIDER'):
    if v := env.get('AINGRAM_TELEMETRY_ENABLED'):
    if v := env.get('AINGRAM_FTS_PREFILTER_THRESHOLD'):
```

- [ ] **Step 6: Commit llms.txt**

```bash
git add llms.txt
git commit -m "docs: add llms.txt — consumer agent API reference"
```
