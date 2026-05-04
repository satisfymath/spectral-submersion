"""Tests for bilingual_validation module."""

import numpy as np
import pandas as pd
import pytest

from spectral_submersion.bilingual_validation import (
    _build_vocab_with_min_freq,
    _df_to_sequences,
    _effective_rank,
    _subsample_sequences,
    build_bilingual_corpus,
    find_cognate_anchors,
)


def _make_bilingual_df(n_lines=100, n_tokens_per_line=10, lang="en", seed=42):
    rng = np.random.RandomState(seed)
    common_words_en = ["the", "a", "is", "in", "to", "and", "of", "for", "it", "on"]
    common_words_fr = ["le", "la", "est", "dans", "de", "et", "un", "pour", "il", "en"]
    shared_words = [
        "restaurant",
        "information",
        "question",
        "table",
        "possible",
        "important",
        "nature",
        "simple",
        "direct",
        "central",
    ]

    if lang == "en":
        base_words = common_words_en + shared_words + [f"en_{i}" for i in range(50)]
    else:
        base_words = common_words_fr + shared_words + [f"fr_{i}" for i in range(50)]

    rows = []
    for line in range(n_lines):
        for pos in range(n_tokens_per_line):
            word = rng.choice(base_words)
            rows.append(
                {
                    "doc_id": f"doc_{lang}",
                    "line_id": line,
                    "position": pos,
                    "token": word,
                    "raw_token": word,
                }
            )
    return pd.DataFrame(rows)


class TestBuildVocabWithMinFreq:
    def test_basic(self):
        tokens = ["a", "b", "a", "c", "a", "b"]
        vocab = _build_vocab_with_min_freq(tokens, min_freq=2)
        assert "a" in vocab
        assert "b" in vocab
        assert "c" not in vocab

    def test_max_vocab(self):
        tokens = [f"w{i}" for i in range(100)] * 10
        vocab = _build_vocab_with_min_freq(tokens, max_vocab=10, min_freq=1)
        assert len(vocab) == 10


class TestFindCognateAnchors:
    def test_identical_strings(self):
        src_vocab = {"hello": 0, "world": 1, "restaurant": 2}
        tgt_vocab = {"bonjour": 0, "monde": 1, "restaurant": 2}
        anchors = find_cognate_anchors(src_vocab, tgt_vocab)
        assert 2 in anchors
        assert anchors[2] == 2
        assert len(anchors) == 1

    def test_with_known_translations(self):
        src_vocab = {"hello": 0, "world": 1}
        tgt_vocab = {"bonjour": 0, "monde": 1}
        translations = {"hello": "bonjour", "world": "monde"}
        anchors = find_cognate_anchors(
            src_vocab, tgt_vocab, known_translations=translations
        )
        assert len(anchors) == 2
        assert anchors[0] == 0
        assert anchors[1] == 1


class TestDfToSequences:
    def test_basic(self):
        df = _make_bilingual_df(n_lines=10, n_tokens_per_line=5, lang="en")
        vocab = _build_vocab_with_min_freq(df["token"].tolist(), min_freq=1)
        seqs = _df_to_sequences(df, vocab)
        assert len(seqs) > 0
        assert all(isinstance(s, list) for s in seqs)
        assert all(len(s) > 0 for s in seqs)


class TestSubsampleSequences:
    def test_basic(self):
        rng = np.random.RandomState(42)
        seqs = [list(range(10))] * 100
        result = _subsample_sequences(seqs, 500, rng)
        total = sum(len(s) for s in result)
        assert total <= 500 + 10
        assert len(result) > 0


class TestBuildBilingualCorpus:
    def test_en_fr_basics(self):
        src_df = _make_bilingual_df(n_lines=50, n_tokens_per_line=10, lang="en")
        tgt_df = _make_bilingual_df(n_lines=50, n_tokens_per_line=10, lang="fr")

        corpus = build_bilingual_corpus(src_df, tgt_df, min_freq=2, max_vocab=50)

        assert corpus["source_vs"] > 0
        assert corpus["target_vs"] > 0
        assert corpus["source_n_tokens"] > 0
        assert corpus["n_cognate_anchors"] >= 5  # shared words should match


class TestEffectiveRank:
    def test_uniform(self):
        sv = np.array([3.0, 3.0, 3.0, 3.0])
        assert _effective_rank(sv) == pytest.approx(4.0, abs=0.01)

    def test_decreasing(self):
        sv = np.array([10.0, 1.0, 0.1, 0.01])
        r = _effective_rank(sv)
        assert 1.0 < r < 4.0
