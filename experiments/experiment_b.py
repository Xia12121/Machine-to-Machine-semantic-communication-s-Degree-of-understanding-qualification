"""Experiment B: Sentence-Paraphrasing (SP) optimization (Tests 4-5).

For each sentence-length bucket (5 / 15 words) and each WDoU level
(0%, 50%, 100%) compare:
  - Original:  U^s = one T5 paraphrase of S
  - Optimized: U^s = best of top-20/35 chatgpt_paraphraser variants

and plot the SDoU gain, matching Fig. 7 and Fig. 8 of the paper.
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
from src.paraphrase import chatgpt_paraphrase, t5_paraphrase
from src.preprocessing import extract_polysemous_tokens
from src.sentence_level import _receiver_sentence
from src.similarity import best_match, cosine
from src.word_level import receiver_meaning, sender_meaning


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

WDOU_LEVELS = [0.0, 0.5, 1.0]
BUCKETS = [("len5", 5), ("len15", 15)]
N_VARIANTS = 35
TOP_K = 20


def _score_sentence(s: str, w_level: float, u_s: str,
                    rng: random.Random) -> float:
    tokens = extract_polysemous_tokens(s)
    if not tokens:
        return 1.0
    sv = sender_meaning(tokens)
    # "All words were masked" -> receiver gets wdou_level correct, others wrong
    rv = receiver_meaning(tokens, w_level, sv, rng=rng)
    u_r = _receiver_sentence(tokens, rv, s)
    return cosine(u_s, u_r)


def run_bucket(name: str, n: int, sentences: List[str]) -> Dict:
    print(f"\n=== Experiment B | {name} ({n} words, {len(sentences)} sentences) ===")
    baseline = {w: [] for w in WDOU_LEVELS}
    optimized = {w: [] for w in WDOU_LEVELS}

    for s in tqdm(sentences, desc="  per-sentence"):
        # baseline U^s: a single T5 paraphrase
        u_s_base = t5_paraphrase(s, num_return=1)[0]
        # candidate U^s: top-20 of 35 chatgpt_paraphraser variants
        variants = chatgpt_paraphrase(s, num_return=N_VARIANTS)
        _, _, top = best_match(s, variants, top_k=TOP_K)
        top_sents = [t[0] for t in top] or [u_s_base]

        for w in WDOU_LEVELS:
            rng = random.Random(4321)
            baseline[w].append(_score_sentence(s, w, u_s_base, rng))

            # pick the variant that gives the highest SDoU at this WDoU level
            best_score = -1.0
            for cand in top_sents:
                rng2 = random.Random(4321)
                sc = _score_sentence(s, w, cand, rng2)
                if sc > best_score:
                    best_score = sc
            optimized[w].append(best_score)

    summary = {
        "bucket": name,
        "n_words": n,
        "wdou_levels": WDOU_LEVELS,
        "baseline_mean": {w: float(np.mean(v)) for w, v in baseline.items()},
        "optimized_mean": {w: float(np.mean(v)) for w, v in optimized.items()},
    }
    print(f"  baseline : {summary['baseline_mean']}")
    print(f"  optimized: {summary['optimized_mean']}")
    return summary


def plot_bucket(summary: Dict, outfile: Path) -> None:
    xs = [int(w * 100) for w in summary["wdou_levels"]]
    base = [summary["baseline_mean"][w] * 100 for w in summary["wdou_levels"]]
    opt = [summary["optimized_mean"][w] * 100 for w in summary["wdou_levels"]]

    plt.figure(figsize=(7, 5))
    plt.plot(xs, opt, marker="o", label="Optimized")
    plt.plot(xs, base, marker="s", linestyle="--", label="Original")
    plt.xlabel("WDoU (%)")
    plt.ylabel("SDoU (%)")
    plt.ylim(0, 100)
    plt.title(f"SDoU with SP optimization ({summary['n_words']} words)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  figure -> {outfile}")


def main() -> None:
    data = load_sentences()
    all_results = []
    for bucket, n in BUCKETS:
        res = run_bucket(bucket, n, data[bucket])
        all_results.append(res)
        plot_bucket(res, FIGURES_DIR / f"fig_exp_b_{bucket}.png")

    out = RESULTS_DIR / "experiment_b.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nExperiment B results saved to {out}")


if __name__ == "__main__":
    main()
