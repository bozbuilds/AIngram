# Contributing to AIngram

Thanks for helping improve AIngram. This document covers local development and how we verify changes before merge.

## Prerequisites

- Python **3.11+**
- Git

Optional: CUDA-enabled ONNX Runtime if you are testing GPU embeddings locally (see `README.md`).

## Setup

```bash
git clone <your-fork-or-repo-url> aingram
cd aingram
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,all]"
```

## Commands

Run the same checks CI runs:

```bash
ruff check aingram/
ruff format --check aingram/
pytest
```

Format code (if `ruff format --check` fails):

```bash
ruff format aingram/
```

## Before opening a PR

1. Add or update tests for behavior changes.
2. Keep commits focused; avoid unrelated refactors.
3. Run **secret scanning** on your branch before pushing (see below).

### Secret and leak hygiene

Do not commit API keys, tokens, or private model weights. The repository `.gitignore` excludes common weight formats (`.gguf`, `.onnx`, `.safetensors`, etc.).

**Recommended:** install [TruffleHog](https://github.com/trufflesecurity/trufflehog) and run a scan over git history before the first public push, for example:

```bash
trufflehog git file://. --only-verified
```

Also run a quick search for internal codenames or private repo paths you do not intend to publish (see the pre-launch checklist in your maintainers’ docs).

## Examples

The `examples/` directory contains small scripts you can run after `pip install -e ".[dev,all]"` to exercise `MemoryStore.remember` / `MemoryStore.recall`.

## Licensing

By contributing, you agree that your contributions are licensed under the same terms as the project (**Apache-2.0**). See `LICENSE` and `NOTICE`.
