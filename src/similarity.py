"""Sentence similarity via MiniLM (cosine), used to compute SDoU.

Paper: "The MiniLM model extracts features for each representation and
calculates sentence similarity using the cosine similarity function."
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np
import torch


_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_minilm():
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(_MINILM_MODEL, device=device)


def encode(sentences: Sequence[str]) -> np.ndarray:
    mdl = _load_minilm()
    return mdl.encode(list(sentences), normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)


def cosine(a: str, b: str) -> float:
    """Cosine similarity between two sentences (MiniLM embeddings)."""
    vecs = encode([a, b])
    return float(np.dot(vecs[0], vecs[1]))


def best_match(reference: str, candidates: Sequence[str],
               top_k: int = 20) -> tuple[str, float, List[tuple[str, float]]]:
    """Filter top-k candidates by similarity to `reference`, return the best.

    Returns (best_candidate, best_score, top_k_list). Used in Experiment B:
    35 variants -> top 20 by MiniLM -> select best.
    """
    if not candidates:
        return reference, 1.0, []
    vecs = encode([reference, *candidates])
    ref_v, cand_v = vecs[0], vecs[1:]
    scores = cand_v @ ref_v
    order = np.argsort(-scores)
    top = [(candidates[i], float(scores[i])) for i in order[:top_k]]
    best_sent, best_score = top[0]
    return best_sent, best_score, top
