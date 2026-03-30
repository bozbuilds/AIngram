# AIngram (Lite)

Local-first agent memory in **one SQLite file**: hybrid vector + FTS5 search, optional knowledge graph (entities and relationships), Ed25519-signed entries, and an optional MCP server. **Apache-2.0.** No cloud dependency; embeddings run via ONNX (Nomic) on your machine.

## Install

```bash
pip install aingram
```

Optional extras:

| Extra | Purpose |
|--------|---------|
| `aingram[extraction]` | GLiNER entity extraction (background linking) |
| `aingram[llm]` | HTTP client for Ollama / local LLM (e.g. consolidation) |
| `aingram[mcp]` | MCP server (`FastMCP`) |
| `aingram[api]` | Anthropic API (Sonnet extractor) |
| `aingram[all]` | `mcp`, `extraction`, `llm`, and `cli`-related deps |
| `aingram[gpu]` | CUDA 12 pip wheels (`cuFFT`, cuBLAS, cuDNN, runtime) ONNX Runtime needs on Windows/Linux |

The `aingram` CLI is available once the package is installed (Typer is a core dependency).

**CUDA / GPU (embeddings):** The default package uses the CPU build of ONNX Runtime. Do not install `onnxruntime` and `onnxruntime-gpu` together. Typical GPU setup:

1. `pip uninstall -y onnxruntime onnxruntime-gpu` then `pip install onnxruntime-gpu`
2. `pip install "aingram[gpu]"` — includes **`nvidia-cufft-cu12`** (ORT’s CUDA EP often fails with `cufft64_11.dll` / cuFFT missing if you only installed cuBLAS/cuDNN/runtime)

Set `AINGRAM_ONNX_PROVIDER=cuda` (or `[onnx_provider]` in `config.toml`) if you want to force CUDA instead of auto. See [ONNX Runtime GPU install](https://onnxruntime.ai/docs/install/#install-gpu) for full CUDA/cuDNN notes.

## Quick start (Python)

```python
from aingram import MemoryStore

with MemoryStore('./agent_memory.db') as mem:
    mem.remember('User prefers dark mode and concise answers')
    for r in mem.recall('What does the user prefer?', limit=5):
        print(r.score, r.entry.content)
```

## CLI

```bash
aingram --db ./agent_memory.db status
aingram --db ./agent_memory.db add "User likes Python"
aingram --db ./agent_memory.db search "Python"
aingram --db ./agent_memory.db entities
aingram --db ./agent_memory.db graph "Alice"
aingram --db ./agent_memory.db consolidate
aingram --db ./agent_memory.db compact --yes --target-dim 256
aingram --db ./agent_memory.db export ./backup.json
aingram --db ./agent_memory.db import ./backup.json
```

`compact` is **one-way** (e.g. 768 → 256 embedding width). Use `--yes` to confirm.

## Configuration

Precedence (highest first): constructor kwargs → environment variables → `~/.aingram/config.toml` → defaults.

| Env var | Meaning |
|---------|--------|
| `AINGRAM_MODELS_DIR` | Model cache directory |
| `AINGRAM_EMBEDDING_DIM` | Width for **new** DBs (must match an existing DB after creation) |
| `AINGRAM_LLM_URL` | Ollama base URL |
| `AINGRAM_LLM_MODEL` | Default LLM name |
| `AINGRAM_LOG_LEVEL` | `aingram` logger level |
| `AINGRAM_WORKER_ENABLED` | `true` / `false` |
| `AINGRAM_EXTRACTOR_MODE` | `none`, `local`, or `sonnet` |
| `AINGRAM_EXTRACTOR_MODEL` | Extractor model id |
| `AINGRAM_ONNX_PROVIDER` | `cpu`, `cuda`, `npu`, or omit for auto |
| `AINGRAM_TELEMETRY_ENABLED` | `true` / `false` — anonymous CLI usage telemetry |

Example `~/.aingram/config.toml`:

```toml
embedding_dim = 768
worker_enabled = true
models_dir = "C:/Users/me/.aingram/models"
llm_url = "http://localhost:11434"
llm_model = "mistral"
extractor_mode = "none"
telemetry_enabled = true
```

Use `AIngramConfig` and `load_merged_config()` for the same rules in application code.

## Privacy and anonymous telemetry

**CLI only:** the `aingram` command-line tool may send events. Using the Python API or MCP does not use this channel.

The CLI may send **anonymous usage events** by default: a random install id (`~/.aingram/telemetry_id`), the **name of the top-level command** you ran (e.g. `add`, `status`), and the package version. **No memory text, queries, paths, or secrets** are included. Events are sent over HTTPS to `https://api.aingram.dev/v1/telemetry`.

**Opt out** (any one):

- Add `--no-telemetry` to a single invocation, e.g. `aingram --no-telemetry --db ./agent_memory.db status`.
- Set `telemetry_enabled = false` in `~/.aingram/config.toml`.
- Set `AINGRAM_TELEMETRY_ENABLED=false` in the environment.

This is separate from any future **opt-in** “contribute training examples” feature, which would involve content you deliberately choose to share.

## Consolidation and LLM

Pass an LLM instance into `MemoryStore.consolidate(llm=...)` when you want richer merge/synthesis behavior (optional). Install `aingram[llm]` and use your stack’s Ollama or other client as appropriate.

## Export / import

`MemoryStore.export_json(path)` writes a Lite JSON backup (sessions, chains, entries, graph, vectors). `import_json(path)` targets an **empty** database by default, or `merge=True` to skip entries that already exist. Export and DB `embedding_dim` must match on import.

For programmatic verification of a session chain, use `MemoryStore.verify()`.

## MCP

With `aingram[mcp]` installed, see `aingram.mcp_server.create_server` for tools such as `remember`, `recall`, `reference`, `verify`, and `get_experiment_context`, with optional bearer-token middleware.

## Examples

Runnable scripts using `MemoryStore.remember` / `MemoryStore.recall` live in [`examples/`](examples/). See [`examples/README.md`](examples/README.md).

## Development

```bash
pip install -e ".[dev,all]"
pytest
ruff check aingram/ && ruff format --check aingram/
```

Guidelines and pre-push hygiene (including secret scanning) are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Python **3.11+**.

## Social

Join the official Aingram discord here: https://discord.gg/zSJCFZnXxf

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
