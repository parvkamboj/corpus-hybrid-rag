"""BM25 sparse vector computation for ingestion + retrieval.

Approximates BM25 using TF saturation only — skipping IDF because we'd need
corpus-level stats that don't exist at index time. Seems to work well enough
for the use case, but worth revisiting if retrieval quality is poor on rare terms.
"""

import hashlib
import re
from collections import Counter

# 2^18 = 262144 buckets. collision rate is low enough for our chunk sizes.
# TODO: benchmark whether a smaller dim hurts quality (would save memory)
_SPARSE_DIM = 2**18

_K1 = 1.5  # standard BM25 saturation param, didn't tune this


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\b[a-z0-9]+\b", text.lower()) if len(t) > 1]


def term_to_index(term: str) -> int:
    """SHA-256 hash → bucket index. Stable across restarts."""
    return int(hashlib.sha256(term.encode()).hexdigest()[:8], 16) % _SPARSE_DIM


def compute_document_sparse(text: str) -> tuple[list[int], list[float]]:
    """TF-saturated sparse vector for a chunk. Merges hash collisions by summing."""
    tokens = tokenize(text)
    if not tokens:
        return [], []

    counts = Counter(tokens)
    total = len(tokens)

    merged: dict[int, float] = {}
    for term, count in counts.items():
        tf = count / total
        weight = (_K1 + 1.0) * tf / (_K1 + tf)
        idx = term_to_index(term)
        # hash collisions just get summed — not ideal but rare enough
        merged[idx] = merged.get(idx, 0.0) + weight

    return list(merged.keys()), list(merged.values())


def compute_query_sparse(text: str) -> tuple[list[int], list[float]]:
    """Uniform weight=1.0 per unique query term.

    Combined with document TF via dot product this roughly approximates BM25.
    Good enough for now.
    """
    tokens = tokenize(text)
    if not tokens:
        return [], []

    merged: dict[int, float] = {}
    for term in set(tokens):
        idx = term_to_index(term)
        merged[idx] = merged.get(idx, 0.0) + 1.0

    return list(merged.keys()), list(merged.values())
