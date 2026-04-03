# Changelog

## 1.1.0

**Search, vectors, and packaging**

- **Schema v9 (migrates from v8):** int8 quantization replaced by **QJL 1-bit** auxiliary vectors in `vec_entries_qjl`, dual-written with float32 `vec_entries` for coarse Hamming search plus float re-ranking.
- **Recall / vector search:** FTS5 **pre-filter** path with configurable `fts_prefilter_threshold`; when the candidate set is large enough, cosine re-ranking runs in Python over FTS hits (workaround for sqlite-vec `MATCH` + `IN` limits). **QJL two-pass** recall for large stores (coarse QJL KNN, then float re-rank).
- **GPU (optional):** `[project.optional-dependencies] gpu` — CUDA 12 pip wheels for ONNX Runtime GPU; README documents `onnxruntime` vs `onnxruntime-gpu` and `AINGRAM_ONNX_PROVIDER`.
- **CLI telemetry (opt-out):** anonymous usage events (command name + version + install id); disable via `--no-telemetry`, `AINGRAM_TELEMETRY_ENABLED=false`, or `telemetry_enabled` in `~/.aingram/config.toml`. See README for details.
- **Docs:** `AGENTS.md` (coding-agent reference), `llms.txt` (consumer API summary); README updates for benchmarks and installation notes.

## 1.0.0

**AIngram Lite** — first public release under Apache-2.0.

- `MemoryStore`: `remember`, `recall`, `get_context`, `reference`, `verify`, `consolidate`, reasoning chains, `compact`, `export_json`, `import_json`.
- Storage: SQLite + sqlite-vec + FTS5; schema version 7; signed entries and session chains.
- Optional extraction (`local` / Sonnet), background task worker for entity linking, knowledge graph traversal.
- Security: MCP-oriented middleware (auth, RBAC, bounds, rate limits) when using `aingram[mcp]`.
- Typer CLI: `setup`, `status`, `add`, `search`, `entities`, `graph`, `consolidate`, `compact`, `export`, `import`, `agent` subcommands.
- Layered config: `AIngramConfig`, environment variables, optional `~/.aingram/config.toml`.
