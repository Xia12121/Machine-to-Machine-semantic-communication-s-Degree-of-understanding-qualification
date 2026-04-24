"""Preprocessing for word-level DoU: tokenize, stopword-remove, stem,
filter single-meaning words, and look up candidate WordNet synsets.

Corresponds to Section III-A of the paper.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import List, Optional

import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


_NLTK_READY = False


def ensure_nltk() -> None:
    """Download NLTK resources once, quietly."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


def _wn_pos(tag: str) -> Optional[str]:
    # WordNet POS constants are resolved lazily so the module can be imported
    # before NLTK data is downloaded.
    pos_map = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}
    return pos_map.get(tag[:1].upper())


@dataclass
class Token:
    """One polysemous word kept for the word-level protocol."""
    surface: str
    stem: str
    pos: Optional[str]
    synsets: List[str] = field(default_factory=list)   # list of synset names
    chosen_synset: Optional[str] = None
    f_i: int = 0                                        # num of available meanings


def extract_polysemous_tokens(sentence: str) -> List[Token]:
    """Tokenize, drop stopwords/punct, stem, keep words with >=2 WordNet senses.

    Returns the ordered list of tokens that participate in the semantic
    checksum and WDoU calculation.
    """
    ensure_nltk()
    sw = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    raw = word_tokenize(sentence)
    tagged = nltk.pos_tag(raw)

    tokens: List[Token] = []
    for word, tag in tagged:
        low = word.lower()
        if low in sw or low in string.punctuation or not low.isalpha():
            continue
        pos = _wn_pos(tag)
        synsets = wn.synsets(low, pos=pos) if pos else wn.synsets(low)
        if len(synsets) < 2:
            # paper: drop words with only one meaning
            continue
        tokens.append(Token(
            surface=word,
            stem=stemmer.stem(low),
            pos=pos,
            synsets=[s.name() for s in synsets],
            f_i=len(synsets),
        ))
    return tokens


def synset_checksum(synset_ids: List[str]) -> str:
    """Deterministic checksum of a sequence of synset IDs (append-to-message)."""
    import hashlib
    payload = "|".join(synset_ids).encode("utf-8")
    return hashlib.md5(payload).hexdigest()
