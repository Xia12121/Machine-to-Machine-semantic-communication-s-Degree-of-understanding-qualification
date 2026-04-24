"""Optional dataset preparation.

The paper uses Semeval-2017 (Kaggle: azzouza2018/semevaldatadets) and picks
12 same-length sentences per bucket (5 / 15 / 25 words). Kaggle requires
authentication, so by default we ship a curated fallback in
`data/sentences.json` that reproduces the paper's three length buckets.

If kagglehub is installed AND you have a Kaggle account configured, this
script will try to download the dataset and populate a Semeval-based JSON
alongside the fallback.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
SENTENCES_FILE = ROOT / "sentences.json"
SEMEVAL_FILE = ROOT / "sentences_semeval.json"


def _word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z]+", s))


def _bucketize(sents: List[str]) -> Dict[str, List[str]]:
    buckets = {"len5": [], "len15": [], "len25": []}
    for s in sents:
        n = _word_count(s)
        if n == 5:
            buckets["len5"].append(s)
        elif n == 15:
            buckets["len15"].append(s)
        elif n == 25:
            buckets["len25"].append(s)
    # match the paper: 12 sentences per bucket
    for k in buckets:
        buckets[k] = buckets[k][:12]
    return buckets


def try_download_semeval() -> bool:
    try:
        import kagglehub           # type: ignore
    except Exception:
        print("[prepare_data] kagglehub not installed; skipping Semeval download.",
              file=sys.stderr)
        return False
    try:
        path = kagglehub.dataset_download("azzouza2018/semevaldatadets")
        print(f"[prepare_data] Semeval downloaded to: {path}")
    except Exception as e:
        print(f"[prepare_data] Semeval download failed: {e}", file=sys.stderr)
        return False

    sents: List[str] = []
    for fp in Path(path).rglob("*"):
        if fp.suffix.lower() in {".txt", ".csv", ".tsv"}:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for piece in re.split(r"(?<=[.!?])\s+", text):
                p = piece.strip()
                if 3 <= _word_count(p) <= 30:
                    sents.append(p)

    buckets = _bucketize(sents)
    if all(buckets[k] for k in buckets):
        SEMEVAL_FILE.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
        print(f"[prepare_data] wrote {SEMEVAL_FILE}")
        return True
    print("[prepare_data] couldn't find enough same-length sentences.",
          file=sys.stderr)
    return False


def load_sentences() -> Dict[str, List[str]]:
    """Return the sentence buckets. Prefer Semeval if present, else fallback."""
    if SEMEVAL_FILE.exists():
        data = json.loads(SEMEVAL_FILE.read_text(encoding="utf-8"))
        if all(data.get(k) for k in ("len5", "len15", "len25")):
            return data
    data = json.loads(SENTENCES_FILE.read_text(encoding="utf-8"))
    return {k: data[k] for k in ("len5", "len15", "len25")}


if __name__ == "__main__":
    if not try_download_semeval():
        print("[prepare_data] Using shipped fallback in data/sentences.json")
    data = load_sentences()
    for k, v in data.items():
        print(f"  {k}: {len(v)} sentences")
