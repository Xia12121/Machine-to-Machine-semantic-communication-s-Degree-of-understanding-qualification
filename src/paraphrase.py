"""Paraphrase models used in the paper.

- T5 paraphraser (sender side, Experiment A)        : Vamsi/T5_Paraphrase_Paws
- Pegasus paraphraser (receiver side, Experiment A) : tuner007/pegasus_paraphrase
- ChatGPT-T5 paraphraser (SP optimization, Exp. B)  : humarin/chatgpt_paraphraser_on_T5_base
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import torch


_T5_PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"
_PEGASUS_MODEL = "tuner007/pegasus_paraphrase"
_CHATGPT_T5_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Sender: T5_paraphrase  -> produces U^s from original sentence S
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_t5_paraphrase():
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tok = T5Tokenizer.from_pretrained(_T5_PARAPHRASE_MODEL)
    mdl = T5ForConditionalGeneration.from_pretrained(_T5_PARAPHRASE_MODEL).to(_device())
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def t5_paraphrase(sentence: str, num_return: int = 1,
                  max_length: int = 128) -> List[str]:
    """Sender's paraphrase: one high-quality version close to the true meaning U."""
    tok, mdl = _load_t5_paraphrase()
    text = "paraphrase: " + sentence + " </s>"
    enc = tok.encode_plus(text, padding="longest", return_tensors="pt",
                          truncation=True, max_length=max_length).to(_device())
    out = mdl.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_length=max_length,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return,
    )
    return [tok.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for o in out]


# ---------------------------------------------------------------------------
# Receiver: Pegasus -> reconstructs U^r from the receiver's word sequence W^r
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_pegasus():
    from transformers import PegasusTokenizer, PegasusForConditionalGeneration
    tok = PegasusTokenizer.from_pretrained(_PEGASUS_MODEL)
    mdl = PegasusForConditionalGeneration.from_pretrained(_PEGASUS_MODEL).to(_device())
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def pegasus_paraphrase(sentence: str, num_return: int = 1,
                       max_length: int = 128) -> List[str]:
    """Receiver's paraphrase: recreate a sentence from chosen word meanings."""
    tok, mdl = _load_pegasus()
    enc = tok([sentence], truncation=True, padding="longest",
              max_length=max_length, return_tensors="pt").to(_device())
    out = mdl.generate(
        **enc,
        max_length=max_length,
        num_beams=max(4, num_return),
        num_return_sequences=num_return,
        temperature=1.5,
    )
    return [tok.decode(o, skip_special_tokens=True) for o in out]


# ---------------------------------------------------------------------------
# SP optimization (Experiment B): chatgpt_paraphraser_on_T5_base -> 35 variants
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_chatgpt_t5():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(_CHATGPT_T5_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(_CHATGPT_T5_MODEL).to(_device())
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def chatgpt_paraphrase(sentence: str, num_return: int = 35,
                       max_length: int = 128) -> List[str]:
    """Generate l variants for SP optimization (paper uses l=35)."""
    tok, mdl = _load_chatgpt_t5()
    enc = tok(f"paraphrase: {sentence}", return_tensors="pt",
              padding="longest", truncation=True,
              max_length=max_length).to(_device())
    out = mdl.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_length=max_length,
        temperature=0.8,
        repetition_penalty=1.2,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=num_return,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
    )
    return [tok.decode(o, skip_special_tokens=True) for o in out]
