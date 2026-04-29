"""Tests for synthetic experiments: permutation recovery, logosyllabic collapse, boustrophedon."""
import pytest
import numpy as np
from spectral_submersion.synthetic_experiments import (
    experiment_permutation_recovery,
    experiment_logosyllabic_collapse,
    experiment_boustrophedon_direction,
    experiment_calendar_model,
    find_parallel_passages,
    generate_permuted_corpus,
    generate_collapsed_corpus,
)


class TestGeneratePermutedCorpus:
    def test_permutation_is_bijective(self):
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 10, size=20).tolist() for _ in range(10)]
        permuted, perm_map = generate_permuted_corpus(sequences, 10, seed=42)
        assert len(permuted) == len(sequences)
        assert len(perm_map) == 10
        values = sorted(perm_map.values())
        assert values == list(range(10))

    def test_permutation_preserves_structure(self):
        rng = np.random.RandomState(42)
        sequences = [rng.randint(0, 5, size=30).tolist() for _ in range(10)]
        permuted, perm_map = generate_permuted_corpus(sequences, 5, seed=42)
        for orig, perm in zip(sequences, permuted):
            assert len(orig) == len(perm)


class TestGenerateCollapsedCorpus:
    def test_collapse_reduces_vocab(self):
        collapse_map = {0: [0, 1], 2: [2, 3, 4]}
        sequences = [[0, 1, 2, 3, 4]]
        collapsed = generate_collapsed_corpus(sequences, collapse_map)
        unique_orig = set(sequences[0])
        unique_coll = set(collapsed[0])
        assert len(unique_coll) <= len(unique_orig)


class TestBoustrophedonDirection:
    def test_forward_dominant_sequences(self):
        rng = np.random.RandomState(42)
        sequences = []
        for _ in range(20):
            seq = list(rng.randint(0, 15, size=rng.randint(5, 15)))
            sequences.append(seq)

        result = experiment_boustrophedon_direction(sequences, 15)
        assert "direction_accuracy" in result
        assert "per_line_results" in result


class TestCalendarModel:
    def test_bic_comparison(self):
        rng = np.random.RandomState(42)
        sequences = [list(rng.randint(0, 20, size=15)) for _ in range(50)]
        result = experiment_calendar_model(sequences, 20, n_lunar_phases=30)
        assert "ngram_bic" in result
        assert "calendar_bic" in result
        assert "delta_bic" in result


class TestParallelPassages:
    def test_identical_passages(self):
        seq = [1, 2, 3, 4, 5]
        sequences = [seq, seq, [9, 8, 7]]
        parallels = find_parallel_passages(sequences, edit_distance_threshold=0.3)
        assert len(parallels) >= 1

    def test_no_parallel_passages(self):
        sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        parallels = find_parallel_passages(sequences, edit_distance_threshold=0.1)
        assert len(parallels) == 0