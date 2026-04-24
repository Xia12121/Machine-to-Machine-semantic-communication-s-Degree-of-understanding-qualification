# Machine-to-Machine Semantic Communications — Degree of Understanding (DoU) Quantification

> Reference implementation of
> **Xia, L., Cai, J., Hou, R. Y.-T., Jeong, S.-P.**
> *Quantification and Validation for Degree of Understanding in M2M Semantic Communications.*
> IEEE ICCT 2024. arXiv: [2408.00767](https://arxiv.org/abs/2408.00767) ·
> DOI: [10.1109/ICCT62411.2024.10946387](https://doi.org/10.1109/ICCT62411.2024.10946387)

This repository reproduces the paper end-to-end: the two-stage hierarchical
quantification of the **Word-level DoU (WDoU)** and **Sentence-level DoU (SDoU)**,
the feedback-based communication model (Fig. 1), and both experiments
(Fig. 4–8), as a single one-click runnable Python project.

---

## What the paper proposes

Natural-language M2M semantic communication is modelled as a feedback loop
between a *sender* and a *receiver*. The sender transmits a sentence `S`;
the receiver replies with its DoU, measured hierarchically:

1. **Word level (§III-A).** After tokenisation / stopword removal / stemming,
   every polysemous word's WordNet synset ID is encoded into a *semantic
   checksum* appended to the message. The receiver recomputes the checksum;
   if they match, the word level is verified. Otherwise a weighted mismatch
   score is computed (Eq. 3):

   ```
   sim_w(M_s, M_r) = Σ_{i=1..d}  v_i · u_i · d_i
       v_i ∈ {0,1}   — match indicator between sender and receiver choice
       u_i ∈ [0,1]   — importance of word i in the sentence
       d_i = f_i / Σ_k f_k — difficulty (f_i = #senses of word i)
   ```

2. **Sentence level (§III-B).** The sender paraphrases `S` into `U^s` with
   T5; the receiver re-synthesises its understanding `U^r` from its chosen
   word senses via Pegasus; MiniLM sentence embeddings and cosine similarity
   give the SDoU.

3. **Sentence-paraphrasing (SP) optimisation (§III-C).** The sender generates
   `l = 35` paraphrased variants with `chatgpt_paraphraser_on_T5_base`,
   keeps the **top-20** by MiniLM similarity to `S`, and selects the variant
   that yields the highest SDoU. This is the optimisation reported in Fig. 7-8.

The joint objective (Eq. 1–2) becomes:

```
F = min ( f(T) + g(T) ) = min ( (1 − sim_w) + (1 − sim_s) ),   with f, g ∈ [0,1].
```

---

## One-click run

### Windows (cmd or PowerShell)

```cmd
run.bat            :: run both experiments
run.bat demo       :: quick smoke test on a single sentence
run.bat a          :: Experiment A only (Tests 1-3, Fig. 4-6)
run.bat b          :: Experiment B only (Tests 4-5, Fig. 7-8)
```

### Linux / macOS / Git Bash

```bash
bash run.sh        # run both experiments
bash run.sh demo   # quick smoke test
bash run.sh a      # Experiment A only
bash run.sh b      # Experiment B only
```

The launcher will:

1. create a local `.venv/` (idempotent, reused on later calls),
2. install the pinned dependencies from `requirements.txt`,
3. fetch the NLTK resources (WordNet, Punkt, POS tagger, stopwords, OMW),
4. run `main.py` with the requested mode,
5. write **plots into `figures/`** and **JSON dumps into `results/`**.

> **First run downloads ~2 GB of HuggingFace model weights** (T5, Pegasus,
> MiniLM, chatgpt-T5). Subsequent runs are cache-hits. GPU is optional —
> everything runs on CPU, just slower.

---

## Models

All four models from Section IV of the paper are used:

| Role                                     | HuggingFace ID                                 |
| ---------------------------------------- | ---------------------------------------------- |
| Sender paraphraser — produces `U^s`      | `Vamsi/T5_Paraphrase_Paws`                     |
| Receiver paraphraser — produces `U^r`    | `tuner007/pegasus_paraphrase`                  |
| SP variant generator (l = 35 variants)   | `humarin/chatgpt_paraphraser_on_T5_base`       |
| Sentence-similarity (cosine on features) | `sentence-transformers/all-MiniLM-L6-v2`       |

Library pins match the paper's reported stack:
`Python 3.10`, `transformers==4.30.0`, `sentencepiece==0.2.0`.

---

## Data

The paper samples 12 same-length sentences per bucket (5 / 15 / 25 words)
from the **Semeval-2017** dataset. That dataset requires a Kaggle login, so
this repo ships an equivalent curated fallback in
[`data/sentences.json`](data/sentences.json) (public-domain English,
12 sentences per length bucket) so the pipeline runs fully offline.

To use the real Semeval split:

```bash
pip install kagglehub
# configure Kaggle credentials as per kaggle.com/docs/api
python data/prepare_data.py
```

`prepare_data.py` will try to download `azzouza2018/semevaldatadets` and
write `data/sentences_semeval.json`, which is auto-preferred over the
fallback when present.

---

## Repository layout

```
.
├── main.py                        # argparse entry point (exp A / B / demo / all)
├── run.sh  /  run.bat             # one-click launchers (venv + install + run)
├── requirements.txt
│
├── src/
│   ├── preprocessing.py           # tokenize · stopword · stem · WordNet · checksum
│   ├── word_level.py              # Eq. (3) sim_w and the sender/receiver meaning vectors
│   ├── paraphrase.py              # T5 / Pegasus / chatgpt-T5 wrappers (lazy-loaded, cached)
│   ├── similarity.py              # MiniLM cosine + top-k filter
│   └── sentence_level.py          # full SDoU pipeline + sp_optimize
│
├── experiments/
│   ├── experiment_a.py            # Tests 1-3 → figures/fig_exp_a_len{5,15,25}.png
│   └── experiment_b.py            # Tests 4-5 → figures/fig_exp_b_len{5,15}.png
│
├── data/
│   ├── sentences.json             # 12-per-bucket offline fallback
│   └── prepare_data.py            # optional Kaggle/Semeval fetcher
│
├── figures/                       # plots written at run time
└── results/                       # JSON dumps of all per-sentence measurements
```

Paper-to-code map:

| Paper section                                   | Module / function                                                         |
| ----------------------------------------------- | ------------------------------------------------------------------------- |
| §III-A preprocessing + synset checksum          | [`src/preprocessing.py`](src/preprocessing.py) — `extract_polysemous_tokens`, `synset_checksum` |
| §III-A + Eq. (3) WDoU                           | [`src/word_level.py`](src/word_level.py) — `wdou`, `compute_wdou`         |
| §III-B SDoU via T5 → Pegasus → MiniLM           | [`src/sentence_level.py`](src/sentence_level.py) — `compute_sdou`         |
| §III-C SP optimisation (35 → top-20 → best)     | [`src/sentence_level.py`](src/sentence_level.py) — `sp_optimize`          |
| §IV-A Experiment A — Tests 1, 2, 3 / Fig. 4-6   | [`experiments/experiment_a.py`](experiments/experiment_a.py)              |
| §IV-B Experiment B — Tests 4, 5 / Fig. 7-8      | [`experiments/experiment_b.py`](experiments/experiment_b.py)              |

---

## Programmatic API

If you want to call the pipeline from your own code:

```python
from src.word_level     import compute_wdou
from src.sentence_level import compute_sdou, sp_optimize

sentence = "The experienced bank manager approved a long credit line."

# Word-level DoU — Eq. (3)
r = compute_wdou(sentence, wdou_level=1.0, seed=42)
print(r["wdou"])                # 1.0 when receiver aligns with sender

# Sentence-level DoU — full T5 / Pegasus / MiniLM pipeline
res = compute_sdou(sentence, wdou_level=0.5, seed=42)
print(res.u_s, res.u_r, res.sdou, res.wdou)

# SP optimisation (Experiment B)
opt = sp_optimize(sentence, wdou_level=1.0, seed=42)
print(opt.u_s, opt.sdou)
```

---

## Reproducing the figures

| Figure    | Script                                        | Notes                                                                 |
| --------- | --------------------------------------------- | --------------------------------------------------------------------- |
| Fig. 4    | `python main.py --exp a`                      | 5-word sentences × {0,50,100}% WDoU × #masked ∈ {1,3,5}               |
| Fig. 5    | `python main.py --exp a`                      | 15-word sentences × {0,50,100}% WDoU × #masked ∈ {1,3,…,15}           |
| Fig. 6    | `python main.py --exp a`                      | 25-word sentences × {0,50,100}% WDoU × #masked ∈ {1,3,…,25}           |
| Fig. 7    | `python main.py --exp b`                      | 5-word sentences, all words masked, baseline vs SP-optimised          |
| Fig. 8    | `python main.py --exp b`                      | 15-word sentences, all words masked, baseline vs SP-optimised         |

Full run times: ~5 min for a smoke test, ~10 min for Experiment A, ~20–40 min
for Experiment B (it generates 35 paraphrases × 12 sentences × 2 buckets).
GPU cuts this roughly in half.

---

## Tested environment

- **OS**: Windows 10 / 11, Linux
- **Python**: 3.10
- **Key pins**: `torch>=2.0`, `transformers==4.30.0`, `sentencepiece==0.2.0`,
  `sentence-transformers>=2.2.2`, `nltk>=3.8`
- **Hardware in the paper**: Intel Xeon 3.6 GHz + NVIDIA RTX 2080 Ti
  (code is also fine on pure CPU).

---

## Citation

```bibtex
@inproceedings{xia2024dou,
  title     = {Quantification and Validation for Degree of Understanding in
               {M2M} Semantic Communications},
  author    = {Xia, Linhan and Cai, Jiaxin and Hou, Ricky Yuen-Tan and
               Jeong, Seon-Phil},
  booktitle = {Proceedings of the IEEE International Conference on Communication
               Technology (ICCT)},
  year      = {2024},
  doi       = {10.1109/ICCT62411.2024.10946387}
}
```

---

## Acknowledgement

From the paper: supported in part by the Guangdong Provincial Key Laboratory
of Interdisciplinary Research and Application for Data Science, BNU-HKBU
United International College (2022B1212010006), and in part by Guangdong
Higher Education Upgrading Plan (2021–2025) (UIC R0400001-22).

## License

Code released under the MIT License (see `LICENSE`). Third-party model
weights are subject to their own licenses on the HuggingFace Hub.
