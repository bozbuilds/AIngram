#!/usr/bin/env python3
"""
Real-world example: Using AIngram as Hermes Agent's semantic memory layer.

This shows how to seed and query memory alongside Hermes' built-in fact_store.
Semantic recall finds facts using different words than how they were stored.

Prerequisites:
  pip install aingram

Usage:
  python3 hermes_with_aingram.py
"""

import os
from aingram import MemoryStore


def main():
    db_path = os.path.expanduser("~/.hermes/aingram.db")
    store = MemoryStore(db_path)

    # ── Seed facts ──
    facts = [
        ("Gold trading: Clean bot v1-kronos active, MYTHOS_ENABLED=False.", ["gold", "trading"]),
        ("User: Sim, Singapore yacht broker. Sells Jeanneau SO40 on Xiaohongshu.", ["user", "identity"]),
        ("Hermes: fallback chain deepseek→openrouter→ollama, all verified.", ["hermes", "config"]),
    ]
    for text, tags in facts:
        store.remember(text, tags=tags)

    # ── Semantic recall (different words than stored) ──
    queries = [
        "which trading bot is running?",
        "what does the user do for work?",
        "what happens when deepseek fails?",
    ]

    for q in queries:
        results = store.recall(q, limit=2)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  [{r.score:.4f}] {r.entry.content}")


if __name__ == "__main__":
    main()
