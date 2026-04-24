"""Word-level DoU (WDoU) quantification.

Implements Eq. (3) of the paper:

    sim_w(M_s, M_r) = sum_{i=1..d} ( v_i * u_i * d_i )

where for each polysemous word i in the preprocessed set:
    v_i = 1 if the receiver's synset choice matches the sender's, else 0
    u_i in [0,1] = word importance (Experiment A assumes equal weight)
    d_i = f_i / sum_k f_k = difficulty from number of candidate meanings
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .preprocessing import Token, extract_polysemous_tokens, synset_checksum


@dataclass
class MeaningVector:
    """One side's meaning selection vector M."""
    synsets: List[str]                       # chosen synset for each token
    checksum: str


def sender_meaning(tokens: List[Token]) -> MeaningVector:
    """Sender always picks the 'correct' sense (paper's assumption).

    We use the first synset returned by WordNet as the canonical/true sense.
    """
    chosen = [t.synsets[0] for t in tokens]
    for t, s in zip(tokens, chosen):
        t.chosen_synset = s
    return MeaningVector(synsets=chosen, checksum=synset_checksum(chosen))


def receiver_meaning(tokens: List[Token],
                     wdou_level: float,
                     sender: MeaningVector,
                     rng: Optional[random.Random] = None) -> MeaningVector:
    """Simulate the receiver at a target WDoU level.

    wdou_level in [0,1]: fraction of tokens for which the receiver picks the
    same synset as the sender. For the remainder, the receiver samples a
    wrong synset uniformly from the remaining candidates (paper: 'randomly
    selects an interpretation ... from the available choices').
    """
    rng = rng or random.Random(0)
    n = len(tokens)
    n_correct = round(wdou_level * n)
    correct_idx = set(rng.sample(range(n), n_correct)) if n else set()

    chosen: List[str] = []
    for i, t in enumerate(tokens):
        if i in correct_idx or len(t.synsets) == 1:
            chosen.append(sender.synsets[i])
        else:
            wrong = [s for s in t.synsets if s != sender.synsets[i]]
            chosen.append(rng.choice(wrong) if wrong else sender.synsets[i])
    return MeaningVector(synsets=chosen, checksum=synset_checksum(chosen))


def wdou(sender: MeaningVector,
         receiver: MeaningVector,
         tokens: Sequence[Token],
         importance: Optional[Sequence[float]] = None) -> float:
    """Eq. (3): weighted average factor-based similarity.

    Checksum shortcut (paper): if sender/receiver checksums match, return 1.
    """
    if not tokens:
        return 1.0
    if sender.checksum == receiver.checksum:
        return 1.0

    # Difficulty d_i = f_i / sum_k f_k, so sum_k d_k = 1
    f = [t.f_i for t in tokens]
    total_f = sum(f) or 1
    d = [fi / total_f for fi in f]

    # Importance u_i. Paper: Experiment A uses equal weights per word.
    if importance is None:
        u = [1.0] * len(tokens)
    else:
        u = list(importance)

    # Match indicator v_i
    v = [1.0 if a == b else 0.0
         for a, b in zip(sender.synsets, receiver.synsets)]

    return float(sum(vi * ui * di for vi, ui, di in zip(v, u, d)))


def compute_wdou(sentence: str,
                 wdou_level: float,
                 seed: int = 0) -> dict:
    """End-to-end WDoU pipeline for a single sentence at a target level.

    Returns the sender/receiver synset choices, preprocessed tokens,
    and the Eq. (3) value.
    """
    tokens = extract_polysemous_tokens(sentence)
    s = sender_meaning(tokens)
    r = receiver_meaning(tokens, wdou_level, s, rng=random.Random(seed))
    return {
        "tokens": tokens,
        "sender": s,
        "receiver": r,
        "wdou": wdou(s, r, tokens),
    }
