"""Shared NLP utilities for notebooks in this folder."""

from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

DATASETS_DIR = "../../datasets"

_porter = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_stop_words: set[str] | None = None


def download_nltk_data(extra: list[str] | None = None) -> None:
    """Download NLTK corpora/models used across the NLP notebooks."""
    packages = [
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger_eng",
    ]
    if extra:
        packages.extend(extra)
    for pkg in packages:
        nltk.download(pkg, quiet=True)


def get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def _wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text, stem: bool = False) -> str:
    """Tokenize first, clean per token, drop stopwords, then stem or lemmatize."""
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [re.sub(r"[^a-z]", "", t) for t in tokens]
    sw = get_stop_words()
    tokens = [t for t in tokens if t and t not in sw and len(t) > 1]
    if stem:
        tokens = [_porter.stem(t) for t in tokens]
    else:
        pos_tags = nltk.pos_tag(tokens)
        tokens = [
            _lemmatizer.lemmatize(token, pos=_wordnet_pos(tag))
            for token, tag in pos_tags
        ]
    return " ".join(tokens)
