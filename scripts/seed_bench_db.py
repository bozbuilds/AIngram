# scripts/seed_bench_db.py
import random
import time
from pathlib import Path
from aingram import MemoryStore

# Realistic-ish agent reasoning content to seed with
HYPOTHESES = [
    "Reducing learning rate below 1e-5 eliminates loss oscillation",
    "Chunking documents at sentence boundaries improves retrieval recall",
    "Adding warmup steps stabilizes early training",
    "Increasing batch size reduces gradient noise",
    "Layer normalization before attention improves convergence",
    "Dropout at 0.1 prevents overfitting on small datasets",
    "AdamW outperforms Adam on transformer architectures",
    "Gradient clipping at 1.0 prevents exploding gradients",
    "Mixed precision training reduces memory by 40%",
    "Cosine annealing scheduler outperforms step decay",
]

METHODS = [
    "Ran 3 training runs with identical seeds, varied only the target parameter",
    "Compared against baseline with 5-fold cross validation",
    "Ablation study across 10 configurations",
    "Grid search over parameter range with fixed compute budget",
    "A/B test on held-out evaluation set",
]

RESULTS = [
    "Validation loss dropped 18%, no oscillation after epoch 3",
    "Recall@5 improved from 0.61 to 0.82",
    "Training stabilized after 200 warmup steps",
    "Gradient variance reduced by 34%",
    "No measurable improvement over baseline",
    "Mixed results — improved on in-domain, degraded on OOD",
    "12% speedup with no quality regression",
    "Overfitting reduced, but convergence slowed by 2x",
]

LESSONS = [
    "Parameter sensitivity is higher on this dataset than expected",
    "Effect disappears at larger batch sizes — context dependent",
    "Warmup duration matters more than peak learning rate",
    "Result is not reproducible across random seeds",
    "This approach generalizes well to similar architectures",
]

ENTRY_TYPES = ["hypothesis", "method", "result", "observation", "lesson"]

def seed_database(path: str, n_entries: int):
    print(f"Seeding {path} with {n_entries} entries...")
    start = time.perf_counter()

    with MemoryStore(path) as mem:
        chain_id = mem.create_chain(f"benchmark seed run — {n_entries} entries")
        
        for i in range(n_entries):
            # Vary entry types and content to make the database realistic
            entry_type = random.choice(ENTRY_TYPES)
            
            if entry_type == "hypothesis":
                content = random.choice(HYPOTHESES) + f" (variant {i})"
            elif entry_type == "method":
                content = random.choice(METHODS) + f" — run {i}"
            elif entry_type == "result":
                content = random.choice(RESULTS) + f" (experiment {i})"
            elif entry_type == "lesson":
                content = random.choice(LESSONS) + f" (observation {i})"
            else:
                content = f"Agent observation {i}: " + random.choice(HYPOTHESES)
            
            mem.remember(
                content,
                entry_type=entry_type,
                chain_id=chain_id,
                confidence=round(random.uniform(0.5, 0.99), 2),
            )
            
            if i > 0 and i % 1000 == 0:
                elapsed = time.perf_counter() - start
                rate = i / elapsed
                remaining = (n_entries - i) / rate
                print(f"  {i}/{n_entries} entries — {rate:.0f}/sec — ~{remaining:.0f}s remaining")
    
    elapsed = time.perf_counter() - start
    size_mb = Path(path).stat().st_size / 1_000_000
    print(f"Done. {n_entries} entries in {elapsed:.1f}s. DB size: {size_mb:.1f}MB\n")


if __name__ == "__main__":
    sizes = [1_000, 10_000, 50_000, 100_000]
    
    for n in sizes:
        path = f"benchmarks/bench_{n}.db"
        Path("benchmarks").mkdir(exist_ok=True)
        seed_database(path, n)
    
    print("All seed databases created in benchmarks/")
    print("Run: python scripts/bench.py")