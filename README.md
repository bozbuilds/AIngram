# Aingram (Lite)

**Agent memory that finds the right context. Every time. Locally.**

```
pip install aingram
```

[![PyPI](https://img.shields.io/pypi/v/aingram?style=flat-square&color=0a0e14)](https://pypi.org/project/aingram/)
[![License](https://img.shields.io/badge/license-Apache%202.0-0a0e14?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-0a0e14?style=flat-square)](https://python.org)
[![Discord](https://img.shields.io/badge/discord-join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/zSJCFZnXxf)

---

Most AI memory systems are filing cabinets with a search bar. Store everything, search later, hope the right thing comes back.

Aingram is different. It runs **three retrieval signals simultaneously** — full-text search, semantic vector search, and knowledge graph traversal — and fuses them into a single ranked result using Reciprocal Rank Fusion. Everything lives in **one SQLite file** on your machine. No cloud. No API key. No vendor to trust with your agents' memory.

On LongMemEval — the most rigorous public benchmark for AI memory — Aingram's retrieval pipeline finds the correct context in the top 3 results **for every single query** when the evidence is present. On the real benchmark with full noisy conversation histories, it surfaces the right sessions in the top 10 for **95.5% of queries**. End-to-end answer accuracy (retrieval → gpt-4o-mini) reaches **72.8%** across all question types.

```python
from aingram import MemoryStore

with MemoryStore('./agent_memory.db') as mem:
    mem.remember('The API rate limit is 100 req/min. Exceeding it causes silent drops.')
    mem.remember('Deployment takes ~3 min. Always run migrations before the container swap.')

    results = mem.recall('what do I need to know before deploying?', limit=5)
    for r in results:
        print(r.score, r.entry.content)
```

---

## Numbers

Benchmarked on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., ICLR 2025) — 500 hand-curated questions across multi-session conversation histories averaging 115,000 tokens. All runs on RTX 4060 8GB. No reranking model. No LLM in the retrieval loop.

| Benchmark | Metric | Score |
|-----------|--------|-------|
| LongMemEval (oracle split) | recall_any@3 | **1.000** |
| LongMemEval (oracle split) | ndcg@10 | **0.994** |
| LongMemEval-S (real retrieval) | recall_any@10 | **0.955** |
| LongMemEval-S (real retrieval) | recall_any@3 | **0.902** |
| LongMemEval-S (real retrieval) | ndcg@10 | **0.836** |
| BEIR SciFact | nDCG@10 | **0.703** |

**What these numbers mean:**

- **recall_any@3 = 1.000 (oracle):** When the evidence exists, Aingram puts the right session in the top 3 results for every query across 500 instances. The correct context is always available to your agent.
- **recall_any@10 = 0.955 (real):** On real noisy conversation histories — ~40 sessions of noise per query — the right sessions appear in the top 10 for 95.5% of queries. This sets the ceiling for any downstream LLM accuracy.
- **22ms median retrieval latency** — pure local pipeline, no network round-trip.

### End-to-end answer accuracy (LongMemEval-S)

Full pipeline accuracy: AIngram retrieval → gpt-4o-mini answering 500 questions across multi-session conversation histories. All categories use gpt-4o-mini for clean comparability to published baselines.

| Question Type | Correct | Total | Accuracy |
|---|---|---|---|
| single-session-user | 66 | 70 | **94.3%** |
| knowledge-update | 65 | 78 | **83.3%** |
| multi-session | 102 | 133 | **76.7%** |
| temporal-reasoning | 80 | 133 | **60.2%** |
| single-session-preference | 18 | 30 | **60.0%** |
| single-session-assistant | 33 | 56 | **58.9%** |
| **Overall** | **364** | **500** | **72.8%** |

> ⚠️ **Temporal-reasoning note:** The 60.2% figure uses gpt-4o-mini throughout — this is the clean comparable to published baselines. Running that category with gpt-4o yields **98/133 (73.7%)**, a +13.5pp improvement, and pushes overall accuracy to 76.4% (382/500). That score is not directly comparable to published gpt-4o-mini baselines.

Retrieval speed scales with corpus size. Vector search is the dominant cost at scale — Aingram's QJL two-pass compression keeps it manageable:

| Entries | Full recall | Embedding | Vector search |
|---------|-------------|-----------|---------------|
| 1K | ~16ms | ~8ms | ~3ms |
| 10K | ~47ms | ~9ms | ~34ms |
| 50K | ~222ms | ~11ms | ~160ms |
| 100K | ~347ms | ~11ms | ~320ms |

QJL's two-pass approach (compressed candidates → float32 rerank) breaks even against brute-force at ~30K entries and delivers meaningful speedup above that threshold.

---

## Why not just use a vector database?

Single-signal retrieval misses. Semantic similarity is powerful but breaks on:

- **Exact terminology** — a query about "the 100 req/min rate limit" might not semantically match a memory that says "hard cap: 100/min" without FTS
- **Entity relationships** — "what did Alice decide about auth?" needs graph traversal, not cosine similarity
- **Keyword-first queries** — agents often search with specific technical terms where BM25 outperforms dense retrieval

Aingram runs all three signals and fuses them. The hybrid consistently outperforms any single signal, especially on the kinds of precise, domain-specific queries agents actually make.

---

## How It Works

```
Agent query
    │
    ├──▶ FTS5 (keyword)                        ─┐
    ├──▶ sqlite-vec + QJL two-pass (semantic)  ─┤──▶ RRF fusion ──▶ ranked results
    └──▶ Knowledge graph (entity)              ─┘
```

**FTS5 full-text search** — SQLite's native full-text index. Fast, no embedding required, excellent for exact terminology and technical strings.

**sqlite-vec vector search** — Dense semantic retrieval using [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) running locally via ONNX. 768-dimensional embeddings, CPU or GPU. No external API.

**QJL two-pass vector search** — At larger corpus sizes, vector search dominates retrieval latency. Aingram uses a Quantized Johnson-Lindenstrauss (QJL) two-pass approach: a fast first pass over compressed quantized vectors narrows the candidate pool, then a precise second pass over full float32 vectors reranks the survivors. This trades a small fraction of recall for significantly lower latency at scale — the break-even point is around 30K entries, above which QJL is faster than brute-force float32 search with no meaningful quality loss.

**Knowledge graph traversal** — Entities and relationships extracted from memory entries. Multi-hop queries resolved via CTE. "What did Alice decide about auth?" finds the entity, traverses relationships, returns relevant entries — even if the query didn't match the entry verbatim.

**Reciprocal Rank Fusion** — Results from all three signals are combined and re-ranked. Each signal's rank position, not raw score, contributes to the final order. This makes the fusion robust to scale differences between signals.

---

## Everything in One File

Your entire agent memory — entries, embeddings, entity graph, signing chain — lives in a single SQLite file. No separate vector database to manage. No graph database. No external embedding service. No Docker containers.

```
agent_memory.db     ← that's it
```

Copy it. Back it up with `cp`. Inspect it with any SQLite client. Share it between agents. Export it to JSON. Import it somewhere else.

This is a deliberate design choice. Memory that requires infrastructure to operate is fragile. Memory that's a file is durable.

---

## Cryptographic Integrity

Every memory entry is Ed25519-signed and linked in a tamper-evident hash chain. You can verify that a memory hasn't been modified since it was written — useful when memory is shared between agents or stored across trust boundaries.

```python
result = mem.verify()
# VerificationResult(valid=True, session_id='...', entries_checked=1247, errors=[])
```

---

## Entity Extraction and Knowledge Graph

Install `aingram[extraction]` to enable [GLiNER](https://github.com/urchade/GLiNER)-based entity extraction. Aingram uses the [GLiNER multitask-large](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5) model — a 205M-parameter multitask model that handles person, organization, location, project, and technology extraction in a single pass. Entities and relationships are automatically extracted from memory entries and stored in the knowledge graph as you write.

```python
# After 'User Alice approved the migration to Clerk on Jan 15.'
# is stored, the graph contains:
#   Alice ─[approved]─▶ migration (valid_from: 2026-01-15)
#   migration ─[uses]─▶ Clerk

results = mem.recall('what did Alice decide?')
# Returns entries linked to Alice via graph traversal,
# not just entries that mention "Alice" by text
```

Query the graph directly:

```bash
aingram --db ./agent_memory.db graph "Alice"
aingram --db ./agent_memory.db entities
```

---

## Memory Consolidation

Run `aingram consolidate` (or `mem.consolidate()`) to clean up accumulated memory over time. Consolidation runs four steps:

1. **Decay** — reduces the importance score of memories that haven't been accessed recently (always active, no configuration needed)
2. **Contradiction detection** — finds pairs of memories about the same entity that say conflicting things and marks the older one as superseded
3. **Merge** — clusters near-duplicate memories and synthesizes a single canonical entry (requires Ollama)
4. **Knowledge synthesis** — summarizes chains of related observations into higher-level conclusions (requires Ollama)

**Contradiction detection** is powered by a local DeBERTa-v3 NLI model running via ONNX Runtime — no LLM or network call required at inference time. Enable it in `~/.aingram/config.toml`:

```toml
contradiction_backend = "deberta"   # or "llm" to use Ollama instead
contradiction_threshold = 0.7       # confidence cutoff (0.0–1.0)
```

The DeBERTa model (~740MB) downloads from HuggingFace on first use and caches locally. For contradiction detection to work, entity extraction must have been run on your memories (`aingram[extraction]` required) so entries can be grouped by the entities they mention.

```bash
aingram consolidate        # run all steps, print JSON summary
```

**Capture daemon auto-consolidation:** if the capture daemon is running, it can trigger consolidation automatically every N ingested memories. Set `consolidation_interval_records` in the `[capture]` section of your config (default: 50, set to 0 to disable).

---

## MCP Server

Install `aingram[mcp]` and connect any MCP-compatible agent (Claude, Cursor, Windsurf, Cline) to your memory store.

```bash
aingram --db ./agent_memory.db mcp
```

Tools exposed: `remember`, `recall`, `reference`, `verify`, `get_experiment_context`, and more. Optional bearer-token auth middleware included.

Add to your MCP config:

```json
{
  "mcpServers": {
    "aingram": {
      "command": "aingram",
      "args": ["--db", "/path/to/agent_memory.db", "mcp"]
    }
  }
}
```

---

## Quick Start

```bash
pip install aingram
```

**Python API:**

```python
from aingram import MemoryStore

with MemoryStore('./agent_memory.db') as mem:
    # Store a memory
    mem.remember('Deploy always requires a migration run first.')

    # Recall with hybrid search
    results = mem.recall('deployment checklist', limit=5)
    for r in results:
        print(f'{r.score:.3f}  {r.entry.content}')
```

**CLI:**

```bash
aingram --db ./agent_memory.db status
aingram --db ./agent_memory.db add "API rate limit is 100 req/min"
aingram --db ./agent_memory.db search "rate limiting"
aingram --db ./agent_memory.db entities
aingram --db ./agent_memory.db graph "Alice"
aingram --db ./agent_memory.db export ./backup.json
aingram --db ./agent_memory.db import ./backup.json
```

**GPU embeddings (optional):**

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
pip install "aingram[gpu]"
export AINGRAM_ONNX_PROVIDER=cuda
```

---

## Multi-agent patterns

AIngram supports concurrent multi-agent setups where multiple logical agents share one SQLite memory file. Two shapes work out of the box:

- **Intra-process (simplest):** multiple async tasks share a *single* `MemoryStore` instance. The engine's internal `threading.Lock` serializes writes.
- **Cross-process (or when you want per-agent attribution):** each agent constructs its own `MemoryStore(db_path, agent_name='agent-N')` pointing at the same `.db` file. SQLite's WAL mode handles multi-writer safety.

See **[`examples/05_multi_agent_shared_memory.py`](examples/05_multi_agent_shared_memory.py)** for a self-contained ~100-line demonstration of the intra-process shape: three async agents share one `MemoryStore` to solve a toy hyperparameter search task, with each agent's `recall()` surfacing sibling findings for piggyback exploration.

For a production-grade multi-agent research integration — including three concurrency modes (mock / solo / swarm), subprocess orchestration with CUDA pinning, workspace isolation, and the full Karpathy `autoresearch` loop wired into AIngram's memory layer — see the companion repo **[bozbuilds/aingram-AR](https://github.com/bozbuilds/aingram-AR)**.

---

## Install

```bash
pip install aingram                   # core — CPU embeddings
pip install "aingram[extraction]"     # + GLiNER entity extraction
pip install "aingram[mcp]"            # + MCP server
pip install "aingram[llm]"            # + Ollama/local LLM client
pip install "aingram[api]"            # + Anthropic API extractor
pip install "aingram[gpu]"            # + CUDA ONNX Runtime wheels
pip install "aingram[capture]"        # + capture daemon (starlette, uvicorn, watchdog)
pip install "aingram[all]"            # everything (except capture and gpu)
```

---

## Capture Daemon

Install `aingram[capture]` to enable automatic prompt/response capture from AI coding tools. The daemon runs locally on `localhost:7749` and supports Claude Code, Cursor, Gemini CLI, Aider, Copilot, Cline, and ChatGPT (manual export).

```bash
aingram capture start                 # foreground mode
aingram capture start --daemon        # background mode
aingram capture stop                  # stop the daemon
aingram capture status                # show tool status and queue depth
aingram capture install claude_code   # print hook setup instructions
aingram capture on                    # enable all tools
aingram capture off cursor            # disable a specific tool
```

Captured interactions flow through a filter pipeline (`@nocapture` opt-out, secret redaction) into a separate SQLite queue, then drain into your main memory database via `MemoryStore.remember()`. Configure via `[capture]` in `~/.aingram/config.toml` or `AINGRAM_CAPTURE_ENABLED` / `AINGRAM_CAPTURE_PORT` env vars. Disabled by default.

---

## Configuration

Precedence: constructor kwargs → env vars → `~/.aingram/config.toml` → defaults.

| Env var | Default | Meaning |
|---------|---------|---------|
| `AINGRAM_MODELS_DIR` | `~/.aingram/models` | Model cache directory |
| `AINGRAM_EMBEDDING_DIM` | `768` | Embedding width for new DBs |
| `AINGRAM_LLM_URL` | `http://localhost:11434` | Ollama base URL |
| `AINGRAM_LLM_MODEL` | — | Default LLM model name |
| `AINGRAM_ONNX_PROVIDER` | auto | `cpu`, `cuda`, or `npu` |
| `AINGRAM_EXTRACTOR_MODE` | `none` | `none`, `local`, or `sonnet` |
| `AINGRAM_WORKER_ENABLED` | `true` | Background consolidation worker |
| `AINGRAM_CONTRADICTION_BACKEND` | `none` | `none`, `deberta`, or `llm` |
| `AINGRAM_CONTRADICTION_THRESHOLD` | `0.7` | DeBERTa confidence cutoff (0.0–1.0) |
| `AINGRAM_TELEMETRY_ENABLED` | `true` | Anonymous CLI usage (opt-out below) |

`~/.aingram/config.toml` example (note: flat keys, no section header for core settings):

```toml
embedding_dim = 768
worker_enabled = true
llm_url = "http://localhost:11434"
llm_model = "mistral"
extractor_mode = "local"
contradiction_backend = "deberta"
contradiction_threshold = 0.7
telemetry_enabled = false

[capture]
consolidation_interval_records = 50
```

---

## Privacy and Telemetry

**No memory content ever leaves your machine.** The SQLite database, embeddings, and entity graph stay local.

The CLI may send anonymous usage events by default: a random install ID, the top-level command name (e.g. `search`, `add`), and the package version. No memory text, query content, file paths, or personal data.

**Opt out:**
- `--no-telemetry` on any command
- `telemetry_enabled = false` in `~/.aingram/config.toml`
- `AINGRAM_TELEMETRY_ENABLED=false` environment variable

---

## Comparison

| | **Aingram** | Mem0 | Zep | MemPalace |
|---|---|---|---|---|
| Retrieval signals | FTS5 + vector + graph | Vector only | KG + vector | Vector + heuristics |
| Vector compression | ✅ QJL two-pass | ✗ | ✗ | ✗ |
| Storage | SQLite (one file) | Cloud / self-host | Neo4j / cloud | ChromaDB |
| Cryptographic signing | ✅ Ed25519 + hash chains | ✗ | ✗ | ✗ |
| Knowledge graph | ✅ Built-in | ✗ | ✅ (Neo4j) | ✗ |
| Local-only | ✅ | Optional | Optional | ✅ |
| No API key required | ✅ | ✗ | ✗ | ✅ |
| Python package | ✅ | ✅ | ✅ | ✅ |
| MCP server | ✅ | ✗ | ✗ | ✅ |
| License | Apache 2.0 | Commercial | Commercial | MIT |

---

## Export / Import

```python
# Export everything — entries, graph, vectors
mem.export_json('./backup.json')

# Import into a fresh database
with MemoryStore('./new_memory.db') as fresh:
    fresh.import_json('./backup.json')

# Merge into an existing database (skips duplicates)
with MemoryStore('./existing.db') as existing:
    existing.import_json('./backup.json', merge=True)
```

---

## Benchmarks

Reproduce the retrieval benchmarks from a clone of this repo:

```bash
# Create synthetic benchmark databases
python scripts/seed_bench_db.py

# Run retrieval and embedding timing
python scripts/bench.py
```

The `scripts/bench.py` output gives per-database timing breakdowns for embedding cost, vector search, FTS5, and full hybrid recall at 1K, 10K, 50K, and 100K entries.

---

## Development

```bash
git clone https://github.com/bozbuilds/AIngram
cd AIngram
pip install -e ".[dev,all]"
pytest
ruff check aingram/ && ruff format --check aingram/
```

Python 3.11+. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## What's Next

Aingram Lite is the open-source retrieval and storage foundation. Aingram Pro adds systems built on top of it — GPU-resident neural caching, biological memory consolidation with spaced-repetition scheduling, and multi-agent synchronization primitives, among other additions. [Join the waitlist](https://aingram.dev), or watch this repo for updates.

---

## Community

[Discord](https://discord.gg/zSJCFZnXxf) · [aingram.dev](https://aingram.dev) · [Discussions](https://github.com/bozbuilds/AIngram/discussions)

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
