"""QJL 1-bit vector compression — projection, encoding, and scoring.

Implements the Quantized Johnson-Lindenstrauss transform for approximate
nearest neighbor search. Vectors are compressed from float32 to 1-bit sign
vectors via random projection, enabling fast Hamming distance search as a
coarse filter before exact float32 re-ranking.

Algorithm overview:
  1. Generate a random projection matrix S (orthogonalized via QR)
  2. For each database vector k: store sign(S @ k) as packed bits
  3. For queries: compute S @ q at full precision (asymmetric)
  4. Coarse search via Hamming distance on bit vectors
  5. Re-rank top candidates using exact float32 cosine distance

The asymmetric design (full-precision query, quantized database) produces
unbiased inner product estimates, unlike symmetric approaches (SimHash).
"""

from __future__ import annotations

import numpy as np

# Hardcoded defaults — no user-facing configuration.
# Seed is stored in db_metadata for projection matrix reproducibility.
SEED = 42
NUM_PROJECTIONS = 768
OVERSAMPLING = 4


def create_projection(dim: int, num_projections: int, seed: int) -> np.ndarray:
    """Generate a deterministic (num_projections, dim) projection matrix.

    The matrix is drawn from N(0,1) and orthogonalized via QR decomposition.
    Orthogonal rows improve estimation accuracy by eliminating redundancy
    between projections. When num_projections > dim, only the first `dim`
    rows can be orthogonalized; remaining rows keep their Gaussian init.

    Args:
        dim: Embedding dimension (e.g. 768 for Nomic).
        num_projections: Number of projection rows = number of output bits.
        seed: RNG seed for deterministic matrix generation.

    Returns:
        float32 array of shape (num_projections, dim).
    """
    rng = np.random.RandomState(seed)
    g = rng.randn(num_projections, dim).astype(np.float32)

    if num_projections <= dim:
        q, _ = np.linalg.qr(g.T)
        return q.T[:num_projections].astype(np.float32)

    q, _ = np.linalg.qr(g[:dim].T)
    result = g.copy()
    result[:dim] = q.T
    return result.astype(np.float32)


def encode(vector: list[float], projection: np.ndarray) -> tuple[bytes, float]:
    """Encode a float32 vector to QJL 1-bit representation.

    Computes sign(S @ vector) and packs the sign bits into bytes.
    A 768-dim vector produces 96 bytes (768 bits / 8).

    The L2 norm is returned alongside the packed bits because the
    asymmetric score formula requires ||k|| to produce unbiased estimates.

    Args:
        vector: Float32 embedding to compress.
        projection: Projection matrix from create_projection().

    Returns:
        (packed_bytes, l2_norm) tuple.
    """
    arr = np.array(vector, dtype=np.float32)
    projected = projection @ arr
    sign_bits = (projected > 0).astype(np.uint8)
    num_proj = projection.shape[0]
    packed_arr = np.packbits(sign_bits)
    byte_len = (num_proj + 7) // 8
    packed = packed_arr[:byte_len].tobytes()
    norm = float(np.linalg.norm(arr))
    return packed, norm


def encode_batch(
    vectors: list[list[float]], projection: np.ndarray
) -> list[tuple[bytes, float]]:
    """Encode a batch of vectors. Returns list of (packed_bytes, l2_norm).

    Vectorized implementation — significantly faster than calling encode()
    in a loop for large batches (e.g. migration backfill).

    Args:
        vectors: List of float32 embeddings.
        projection: Projection matrix from create_projection().

    Returns:
        List of (packed_bytes, l2_norm) tuples, one per input vector.
    """
    if not vectors:
        return []
    arr = np.array(vectors, dtype=np.float32)
    projected = arr @ projection.T
    sign_bits = (projected > 0).astype(np.uint8)
    packed = np.packbits(sign_bits, axis=1)
    norms = np.linalg.norm(arr, axis=1).astype(np.float32)
    num_proj = projection.shape[0]
    byte_len = (num_proj + 7) // 8
    return [
        (packed[i, :byte_len].tobytes(), float(norms[i]))
        for i in range(len(vectors))
    ]


def asymmetric_score(
    query_projected: np.ndarray, packed_bits: bytes, norm: float, num_projections: int
) -> float:
    """QJL asymmetric inner product estimator.

    Computes: (sqrt(pi/2) / m) * ||k|| * <S@q, sign(S@k)>

    This is an unbiased estimator of the true inner product <q, k>.
    The sqrt(pi/2) correction factor accounts for the information lost
    by sign quantization. The asymmetry (full-precision query, quantized
    database) is what makes the estimate unbiased — symmetric quantization
    (SimHash) produces biased estimates.

    Not used in the two-pass search pipeline (which re-ranks with exact
    float32 cosine), but included for completeness and potential future
    QJL-only mode.

    Args:
        query_projected: S @ query (full precision, float32).
        packed_bits: Packed sign bits from encode().
        norm: L2 norm of the database vector (from encode()).
        num_projections: Number of projections (= number of bits).

    Returns:
        Estimated inner product (float).
    """
    bits = np.unpackbits(np.frombuffer(packed_bits, dtype=np.uint8))[:num_projections]
    signs = 2.0 * bits.astype(np.float32) - 1.0
    dot = float(np.dot(query_projected, signs))
    correction = np.sqrt(np.pi / 2) / num_projections
    return float(correction * norm * dot)
