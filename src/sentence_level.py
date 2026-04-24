"""Sentence-level DoU (SDoU) quantification and SP optimization.

Pipeline (paper Section III-B, Fig. 3):
  1. Sender paraphrases S with T5_paraphrase  -> U^s
  2. Receiver selects word senses (WDoU level), re-synthesizes with Pegasus -> U^r
  3. SDoU = cosine_MiniLM(U^r, U^s)

SP optimization (Section III-C, Experiment B):
  Sender generates l=35 variants with chatgpt_paraphraser_on_T5_base,
  filters top-20 by MiniLM similarity to S, chooses the best-performing
  version as the new U^s.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .paraphrase import chatgpt_paraphrase, pegasus_paraphrase, t5_paraphrase
from .preprocessing import Token, extract_polysemous_tokens
from .similarity import best_match, cosine
from .word_level import (
    MeaningVector,
    receiver_meaning,
    sender_meaning,
    wdou,
)


@dataclass
class DoUResult:
    sentence: str
    u_s: str                # sender's paraphrase of S
    u_r: str                # receiver's reconstruction
    wdou_level: float       # target WDoU in [0,1]
    wdou: float             # measured WDoU (Eq. 3)
    sdou: float             # measured SDoU (MiniLM cosine)


def _receiver_sentence(tokens: list[Token],
                       receiver: MeaningVector,
                       seed: str) -> str:
    """Build the receiver's input sentence from its chosen synsets.

    Paper: "receiver generates a word semantic sequence W^r and passes it to
    Pegasus to create sentence U^r." We construct W^r by substituting the
    lemma of the chosen synset for each polysemous token, then Pegasus
    rephrases the resulting string.
    """
    from nltk.corpus import wordnet as wn
    tok_map = {t.surface.lower(): t for t in tokens}
    parts = []
    idx = 0
    for word in seed.split():
        clean = "".join(c for c in word.lower() if c.isalpha())
        tok = tok_map.get(clean)
        if tok is not None and idx < len(receiver.synsets):
            try:
                syn = wn.synset(receiver.synsets[idx])
                lemma = syn.lemma_names()[0].replace("_", " ")
            except Exception:
                lemma = word
            parts.append(lemma)
            idx += 1
        else:
            parts.append(word)
    stitched = " ".join(parts)
    out = pegasus_paraphrase(stitched, num_return=1)
    return out[0] if out else stitched


def compute_sdou(sentence: str,
                 wdou_level: float,
                 seed: int = 0,
                 u_s: Optional[str] = None) -> DoUResult:
    """End-to-end pipeline for one sentence at a target WDoU level.

    If u_s is provided, the sender step is skipped (useful when evaluating
    SP optimization variants against the same receiver).
    """
    tokens = extract_polysemous_tokens(sentence)
    s_vec = sender_meaning(tokens)
    rng = random.Random(seed)
    r_vec = receiver_meaning(tokens, wdou_level, s_vec, rng=rng)

    if u_s is None:
        u_s_list = t5_paraphrase(sentence, num_return=1)
        u_s = u_s_list[0] if u_s_list else sentence

    u_r = _receiver_sentence(tokens, r_vec, sentence)

    return DoUResult(
        sentence=sentence,
        u_s=u_s,
        u_r=u_r,
        wdou_level=wdou_level,
        wdou=wdou(s_vec, r_vec, tokens),
        sdou=cosine(u_s, u_r),
    )


def sp_optimize(sentence: str,
                wdou_level: float,
                l: int = 35,
                top_k: int = 20,
                seed: int = 0) -> DoUResult:
    """Sentence Paraphrasing optimization (Experiment B).

    Generate l variants, keep top_k by similarity to the original sentence,
    return the variant whose SDoU against the receiver is the best.
    """
    variants = chatgpt_paraphrase(sentence, num_return=l)
    _, _, top = best_match(sentence, variants, top_k=top_k)

    best: Optional[DoUResult] = None
    for cand, _ in top:
        res = compute_sdou(sentence, wdou_level, seed=seed, u_s=cand)
        if best is None or res.sdou > best.sdou:
            best = res
    # Fallback if generation returned nothing
    return best or compute_sdou(sentence, wdou_level, seed=seed)
