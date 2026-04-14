# Changelog

## Unreleased

## 1.2.0

- **Capture daemon:** opt-in local HTTP daemon (`aingram[capture]`) that captures prompt/response pairs from seven AI coding tools (Claude Code, Cursor, Gemini CLI, Aider, Copilot, Cline, ChatGPT) and writes them into AIngram memory. Includes per-tool adapters, `@nocapture` opt-out, secret redaction, toggle system, and CLI subcommands (`aingram capture start|stop|status|on|off|install`). Disabled by default; configure via `[capture]` in `config.toml` or `AINGRAM_CAPTURE_ENABLED` env var.
- **Config:** `AIngramConfig` extended with `capture: CaptureConfig | None` field; `_merge_toml_into()` parses `[capture]` TOML sections; `_merge_env_into()` reads `AINGRAM_CAPTURE_ENABLED` and `AINGRAM_CAPTURE_PORT`.
- **Multi-agent example:** `examples/05_multi_agent_shared_memory.py` — self-contained ~100-line demo of three async agents sharing one `MemoryStore` for concurrent hyperparameter exploration with piggyback recall.
- **Example smoke test:** `tests/test_examples.py` added to CI matrix; guards example scripts against bit-rot.
- **README:** new "Multi-agent patterns" and "Capture Daemon" sections; updated install extras to include `[capture]`.
- **Fix:** `aingram.__version__` corrected from `'1.0.0'` to `'1.1.0'` (was not bumped alongside `pyproject.toml` in the 1.1.0 release).

## 1.1.0

### Search, vectors, and packaging

- **Schema v9 (migrates from v8):** int8 quantization replaced by **QJL 1-bit** auxiliary vectors in `vec_entries_qjl`, dual-written with float32 `vec_entries` for coarse Hamming search plus float re-ranking.
- **Recall / vector search:** FTS5 **pre-filter** path with configurable `fts_prefilter_threshold`; when the candidate set is large enough, cosine re-ranking runs in Python over FTS hits (workaround for sqlite-vec `MATCH` + `IN` limits). **QJL two-pass** recall for large stores (coarse QJL KNN, then float re-rank).
- **GPU (optional):** `[project.optional-dependencies] gpu` — CUDA 12 pip wheels for ONNX Runtime GPU; README documents `onnxruntime` vs `onnxruntime-gpu` and `AINGRAM_ONNX_PROVIDER`.
- **CLI telemetry (opt-out):** anonymous usage events (command name + version + install id); disable via `--no-telemetry`, `AINGRAM_TELEMETRY_ENABLED=false`, or `telemetry_enabled` in `~/.aingram/config.toml`. See README for details.
- **Docs:** `AGENTS.md` (coding-agent reference), `llms.txt` (consumer API summary); README updates for benchmarks and installation notes.

## 1.0.0

### AIngram Lite

First public release under Apache-2.0.

- `MemoryStore`: `remember`, `recall`, `get_context`, `reference`, `verify`, `consolidate`, reasoning chains, `compact`, `export_json`, `import_json`.
- Storage: SQLite + sqlite-vec + FTS5; schema version 7; signed entries and session chains.
- Optional extraction (`local` / Sonnet), background task worker for entity linking, knowledge graph traversal.
- Security: MCP-oriented middleware (auth, RBAC, bounds, rate limits) when using `aingram[mcp]`.
- Typer CLI: `setup`, `status`, `add`, `search`, `entities`, `graph`, `consolidate`, `compact`, `export`, `import`, `agent` subcommands.
- Layered config: `AIngramConfig`, environment variables, optional `~/.aingram/config.toml`.
