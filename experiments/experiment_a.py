"""Experiment A: relationship between WDoU and SDoU (Tests 1-3).

For each sentence-length bucket (5 / 15 / 25 words) and each WDoU level
(0%, 50%, 100%) sweep the number of masked words from 1 to n, compute the
mean SDoU across 12 sentences, and produce a figure like Fig. 4-6 in the
paper.

"Mask" here means the receiver is forced to re-select senses for m words
(outside the subset that matches the sender); the remaining (n-m) tokens
use the WDoU-aligned choice. This mirrors the paper's "no. of words
masked" axis.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from data.prepare_data import load_sentences
from src.paraphrase import t5_paraphrase
from src.preprocessing import extract_polysemous_tokens
from src.sentence_level import _receiver_sentence
from src.similarity import cosine
from src.word_level import MeaningVector, receiver_meaning, sender_meaning


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

WDOU_LEVELS = [0.0, 0.5, 1.0]
LENGTH_BUCKETS = [("len5", 5), ("len15", 15), ("len25", 25)]


def _mask_receiver(sender: MeaningVector,
                   tokens,
                   wdou_level: float,
                   n_mask: int,
                   rng: random.Random) -> MeaningVector:
    """Make a receiver meaning vector with exactly n_mask wrong tokens,
    spread according to the WDoU level on the remaining positions."""
    n = len(tokens)
    if n == 0:
        return sender
    n_mask = max(0, min(n, n_mask))
    mask_idx = set(rng.sample(range(n), n_mask))

    remaining = [i for i in range(n) if i not in mask_idx]
    n_correct_remaining = round(wdou_level * len(remaining))
    correct_idx = set(rng.sample(remaining, n_correct_remaining)) if remaining else set()

    chosen = []
    for i, t in enumerate(tokens):
        if i in correct_idx or (i not in mask_idx and len(t.synsets) == 1):
            chosen.append(sender.synsets[i])
        else:
            wrong = [s for s in t.synsets if s != sender.synsets[i]]
            chosen.append(rng.choice(wrong) if wrong else sender.synsets[i])

    from src.preprocessing import synset_checksum
    return MeaningVector(synsets=chosen, checksum=synset_checksum(chosen))


def _mask_axis(n_words_bucket: int) -> List[int]:
    """Odd values 1,3,5,...,n (following the paper's plots)."""
    return list(range(1, n_words_bucket + 1, 2))


def run_test(bucket_name: str, n_words: int, sentences: List[str]) -> Dict:
    print(f"\n=== Experiment A | {bucket_name} ({n_words} words, {len(sentences)} sentences) ===")
    mask_counts = _mask_axis(n_words)
    results = {f"wdou={int(w*100)}%": {m: [] for m in mask_counts} for w in WDOU_LEVELS}

    per_sentence_u_s: Dict[str, str] = {}
    # Pre-compute U^s once per sentence (sender is deterministic in the paper).
    for s in tqdm(sentences, desc="  sender paraphrase"):
        per_sentence_u_s[s] = t5_paraphrase(s, num_return=1)[0]

    for w_level in WDOU_LEVELS:
        key = f"wdou={int(w_level*100)}%"
        for m in mask_counts:
            rng = random.Random(1234)
            scores = []
            for s in sentences:
                tokens = extract_polysemous_tokens(s)
                if len(tokens) == 0:
                    continue
                s_vec = sender_meaning(tokens)
                r_vec = _mask_receiver(s_vec, tokens, w_level, m, rng)
                u_r = _receiver_sentence(tokens, r_vec, s)
                u_s = per_sentence_u_s[s]
                scores.append(cosine(u_s, u_r))
            results[key][m] = float(np.mean(scores)) if scores else 0.0

    return {
        "bucket": bucket_name,
        "n_words": n_words,
        "mask_counts": mask_counts,
        "curves": results,
    }


def plot_test(result: Dict, outfile: Path) -> None:
    plt.figure(figsize=(7, 5))
    for key, series in result["curves"].items():
        xs = list(series.keys())
        ys = [series[x] * 100 for x in xs]
        plt.plot(xs, ys, marker="o", label=key.upper())
    plt.xlabel("No. of Words Masked")
    plt.ylabel("SDoU (%)")
    plt.ylim(0, 100)
    plt.title(f"Sentence with {result['n_words']} words: SDoU vs no. of words masked")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  figure -> {outfile}")


def main() -> None:
    data = load_sentences()
    all_results = []
    for bucket, n in LENGTH_BUCKETS:
        res = run_test(bucket, n, data[bucket])
        all_results.append(res)
        plot_test(res, FIGURES_DIR / f"fig_exp_a_{bucket}.png")

    out = RESULTS_DIR / "experiment_a.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nExperiment A results saved to {out}")


if __name__ == "__main__":
    main()
