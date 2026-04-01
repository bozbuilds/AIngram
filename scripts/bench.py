import time
from pathlib import Path

from aingram import MemoryStore
from aingram.config import load_merged_config
from aingram.processing.embedder import NomicEmbedder


def bench_recall(db_path, query, *, embedder: NomicEmbedder, embedding_dim: int, n_trials=20):
    with MemoryStore(
        db_path,
        embedder=embedder,
        embedding_dim=embedding_dim,
    ) as mem:
        mem.recall(query, limit=10, verify=False)
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            mem.recall(query, limit=10, verify=False)
            times.append((time.perf_counter() - start) * 1000)

    ordered = sorted(times)
    mid = len(ordered) // 2
    p95_i = min(int(len(ordered) * 0.95), len(ordered) - 1)
    return {
        "median_ms": ordered[mid],
        "p95_ms": ordered[p95_i],
    }


def bench_embedding_vs_vector_search(
    db_path: str,
    query: str,
    *,
    embedder: NomicEmbedder,
    embedding_dim: int,
    n_trials: int = 20,
) -> None:
    """Split embed vs sqlite-vec search (no signature verify, no full recall pipeline)."""
    with MemoryStore(
        db_path,
        embedder=embedder,
        embedding_dim=embedding_dim,
    ) as mem:
        times_emb: list[float] = []
        embedding: list[float] | None = None
        for _ in range(n_trials):
            start = time.perf_counter()
            embedding = mem._embedder.embed(query)
            times_emb.append((time.perf_counter() - start) * 1000)
        times_emb.sort()
        assert embedding is not None

        times_search: list[float] = []
        for _ in range(n_trials):
            start = time.perf_counter()
            mem._engine.search_vectors(embedding, limit=10)
            times_search.append((time.perf_counter() - start) * 1000)
        times_search.sort()

    mid = n_trials // 2
    print(f"  Embedding only:       {times_emb[mid]:.1f}ms median")
    print(f"  Vector search only:   {times_search[mid]:.1f}ms median")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    bench_dir = root / "benchmarks"
    dbs = sorted(bench_dir.glob("bench_*.db"))
    if not dbs:
        print(f"No benchmark databases in {bench_dir}. Run: python scripts/seed_bench_db.py")
        raise SystemExit(1)

    config = load_merged_config()
    embedder = NomicEmbedder(
        dim=config.embedding_dim,
        models_dir=config.models_dir,
        preferred_provider=config.onnx_provider,
    )
    embedder.embed("warmup")
    print(f"ONNX Runtime providers (embedding): {embedder.active_execution_providers()}")

    query = "learning rate batch training hypothesis"
    for db_path in dbs:
        print(f"\n{db_path.relative_to(root)}")
        stats = bench_recall(
            str(db_path),
            query,
            embedder=embedder,
            embedding_dim=config.embedding_dim,
            n_trials=20,
        )
        print(f"  median_ms: {stats['median_ms']:.2f}")
        print(f"  p95_ms:    {stats['p95_ms']:.2f}")

    breakdown_query = "learning rate instability optimizer"
    print("\nLatency breakdown (embed vs vector search):")
    for db_path in dbs:
        print(f"\n{db_path.relative_to(root)}")
        bench_embedding_vs_vector_search(
            str(db_path),
            breakdown_query,
            embedder=embedder,
            embedding_dim=config.embedding_dim,
            n_trials=20,
        )
