"""Three concurrent research agents share memory via AIngram.

This example demonstrates the multi-agent shared-memory pattern that
AIngram is designed to support. Three logical agents run in one process
(asyncio tasks), share a single MemoryStore, and `recall()` each other's
findings to avoid duplicating exploration.

The toy research task is hyperparameter search for a 2-layer MLP on
synthetic data. Each agent proposes a (hidden_size, activation) combo,
"trains" it via a deterministic mock scoring function, and records the
result. On subsequent iterations, agents recall sibling findings and
bias toward combos that scored well.

No ONNX download (inline stub embedder). No autoresearch dependency.
For the full autoresearch integration, see https://github.com/bozbuilds/aingram-AR
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import struct
import tempfile
from pathlib import Path

from aingram import MemoryStore


class _StubEmbedder:
    """Hash-derived unit-vector embedder. No model download, no network."""

    dim = 768

    def embed(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode('utf-8')).digest()
        vec: list[float] = []
        chunk = seed
        while len(vec) < self.dim:
            chunk = hashlib.sha256(chunk).digest()
            for i in range(0, 32, 4):
                if len(vec) >= self.dim:
                    break
                (x,) = struct.unpack('<f', chunk[i : i + 4])
                if not math.isfinite(x):
                    x = 0.0
                vec.append(x)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


OPTIMAL = {'hidden_size': 64, 'activation': 'relu'}
BASE_LOSS = 1.0


def mock_train(hidden_size: int, activation: str) -> float:
    """Deterministic loss: best when hidden_size=64, activation='relu'."""
    loss = BASE_LOSS
    if hidden_size != OPTIMAL['hidden_size']:
        loss += 0.1 * abs(math.log2(hidden_size / OPTIMAL['hidden_size']))
    if activation != OPTIMAL['activation']:
        loss += 0.15
    return round(loss, 4)


async def run_agent(
    agent_idx: int,
    store: MemoryStore,
    iterations: int,
) -> list[dict]:
    """Run one agent for N iterations: recall → propose → train → remember."""
    candidates = {
        'hidden_size': [16, 32, 64, 128, 256],
        'activation': ['relu', 'gelu', 'tanh'],
    }
    hypothesis = {
        'hidden_size': candidates['hidden_size'][agent_idx % 5],
        'activation': candidates['activation'][agent_idx % 3],
    }
    results = []

    for i in range(iterations):
        recall_hits = list(store.recall('MLP hyperparameter', limit=5))
        sibling_hit = None
        for r in recall_hits:
            entry = getattr(r, 'entry', None)
            if entry is None:
                continue
            content = entry.content
            if 'loss=' in content and 'status=keep' in content:
                sibling_hit = content
                break

        if sibling_hit:
            for line in sibling_hit.splitlines():
                if line.startswith('activation='):
                    sibling_act = line.split('=', 1)[1].strip()
                    if sibling_act in candidates['activation']:
                        hypothesis['activation'] = sibling_act
                    break

        loss = mock_train(hypothesis['hidden_size'], hypothesis['activation'])
        status = 'keep' if loss < BASE_LOSS + 0.1 else 'discard'

        content = (
            f'[mlp_experiment]\n'
            f'agent={agent_idx}\n'
            f'iteration={i}\n'
            f'hidden_size={hypothesis["hidden_size"]}\n'
            f'activation={hypothesis["activation"]}\n'
            f'loss={loss}\n'
            f'status={status}'
        )
        entry_id = store.remember(
            content,
            metadata={
                'agent_idx': agent_idx,
                'iteration': i,
                'hypothesis': dict(hypothesis),
            },
        )
        results.append({'entry_id': entry_id, 'loss': loss, 'status': status})

        # If this config didn't work, explore the next hidden_size. On 'keep'
        # we stay put — this is the config we want sibling agents to piggyback on.
        if status == 'discard':
            idx = candidates['hidden_size'].index(hypothesis['hidden_size'])
            hypothesis['hidden_size'] = candidates['hidden_size'][
                min(idx + 1, len(candidates['hidden_size']) - 1)
            ]

    return results


async def main() -> None:
    """Run 3 agents for 3 iterations each against a single shared MemoryStore."""
    with tempfile.TemporaryDirectory() as tmp_root:
        db_path = Path(tmp_root) / 'shared_memory_demo.db'

        with MemoryStore(str(db_path), embedder=_StubEmbedder()) as store:
            tasks = [run_agent(i, store, iterations=3) for i in range(3)]
            all_results = await asyncio.gather(*tasks)

    print(f'Ran 3 agents × 3 iterations = {sum(len(r) for r in all_results)} total findings')
    for i, results in enumerate(all_results):
        kept = sum(1 for r in results if r['status'] == 'keep')
        best = min(r['loss'] for r in results)
        print(f'  Agent {i}: {len(results)} findings, {kept} kept, best loss = {best}')

    print('\nFor the full autoresearch integration: https://github.com/bozbuilds/aingram-AR')


if __name__ == '__main__':
    asyncio.run(main())
