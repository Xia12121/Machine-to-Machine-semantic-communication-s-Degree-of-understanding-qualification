"""Entry point: run one or all experiments from the ICCT 2024 paper.

Usage:
    python main.py                 # run everything
    python main.py --exp a         # only Experiment A (Tests 1-3)
    python main.py --exp b         # only Experiment B (Tests 4-5)
    python main.py --demo          # quick smoke test on one sentence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _demo() -> None:
    """Minimal smoke test exercising the full pipeline on one sentence."""
    from src.sentence_level import compute_sdou, sp_optimize
    from src.word_level import compute_wdou

    sentence = ("The experienced bank manager carefully reviewed the loan "
                "application and approved a long credit line for the small "
                "local business owner.")
    print(f"Sentence: {sentence}\n")

    for w in (0.0, 0.5, 1.0):
        r = compute_wdou(sentence, wdou_level=w, seed=42)
        print(f"  WDoU target={int(w*100)}%  -> Eq.(3) WDoU = {r['wdou']:.3f}"
              f"  (kept {len(r['tokens'])} polysemous tokens)")

    print("\nRunning SDoU pipeline (WDoU=100%)…")
    res = compute_sdou(sentence, wdou_level=1.0, seed=42)
    print(f"  U_s: {res.u_s}")
    print(f"  U_r: {res.u_r}")
    print(f"  SDoU = {res.sdou:.3f}, WDoU = {res.wdou:.3f}")

    print("\nSP optimization (35 variants -> top 20 -> best)…")
    opt = sp_optimize(sentence, wdou_level=1.0, seed=42)
    print(f"  best U_s: {opt.u_s}")
    print(f"  SDoU(opt) = {opt.sdou:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=["a", "b", "all"], default="all",
                        help="which experiment to run")
    parser.add_argument("--demo", action="store_true",
                        help="quick smoke test on one sentence")
    args = parser.parse_args()

    if args.demo:
        _demo()
        return

    if args.exp in ("a", "all"):
        from experiments.experiment_a import main as run_a
        run_a()
    if args.exp in ("b", "all"):
        from experiments.experiment_b import main as run_b
        run_b()


if __name__ == "__main__":
    main()
