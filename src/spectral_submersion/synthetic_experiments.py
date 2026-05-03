"""Synthetic experiments for validating structural methods under known conditions.

Implements the experimental protocols from Part IV (Sections 17-22):
- Exp 1: Permutation recovery (Procrustes/OT recover known permutation)
- Exp 2: Logosyllabic collapse (many-to-one mappings)
- Exp 3: Unknown segmentation
- Exp 4: Boustrophedon direction recovery
- Exp 5: Parallel passages
- Exp 6: Lunar calendar model comparison

Each experiment produces auditable results with metrics, negative controls,
and explicit claim-level limits.
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from itertools import product as iter_product


def experiment_permutation_recovery(
    source_sequences: list[list[int]],
    target_sequences: list[list[int]],
    source_vocab_size: int,
    target_vocab_size: int,
    n_anchors: int = 20,
    window_size: int = 3,
    k: int = 16,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict:
    """Experiment 1: Recover a known permutation via Procrustes/OT.

    Synthetic test: create artificial writing X = pi(Y) where Y is
    a known corpus. Hide pi and provide m anchors. Evaluate:
    - Acc@k: fraction of source symbols whose true mapping is in top-k
    - MRR: mean reciprocal rank

    If this fails, the method cannot be applied to Rongorongo.

    Args:
        source_sequences: Source corpus token-ID sequences (permuted).
        target_sequences: Target corpus token-ID sequences (original).
        source_vocab_size: Source vocabulary size.
        target_vocab_size: Target vocabulary size.
        n_anchors: Number of anchor pairs to provide.
        window_size: Co-occurrence window.
        k: Embedding dimensionality.
        n_bootstrap: Bootstrap iterations for stability.
        seed: Random seed.

    Returns:
        Dict with Acc@K, MRR, and full metrics.
    """
    from .cooccurrence import cooccurrence_matrix_from_sequences
    from .pmi import ppmi_matrix
    from .spectral import spectral_embedding
    from .alignment import orthogonal_procrustes, pairwise_squared_distances

    rng = np.random.RandomState(seed)

    C_src = cooccurrence_matrix_from_sequences(
        source_sequences, source_vocab_size, window_size=window_size
    )
    C_tgt = cooccurrence_matrix_from_sequences(
        target_sequences, target_vocab_size, window_size=window_size
    )

    M_src = ppmi_matrix(C_src)
    M_tgt = ppmi_matrix(C_tgt)

    E_src, sv_src, _ = spectral_embedding(M_src, k=k)
    E_tgt, sv_tgt, _ = spectral_embedding(M_tgt, k=k)

    n_min = min(E_src.shape[0], E_tgt.shape[0])
    d = min(E_src.shape[1], E_tgt.shape[1])

    freq_ranks_src = np.argsort(-np.linalg.norm(E_src, axis=1))[:n_min]
    freq_ranks_tgt = np.argsort(-np.linalg.norm(E_tgt, axis=1))[:n_min]

    X_anch = E_src[freq_ranks_src[:n_anchors], :d]
    Y_anch = E_tgt[freq_ranks_tgt[:n_anchors], :d]

    Q = orthogonal_procrustes(X_anch, Y_anch)

    E_src_aligned = E_src[freq_ranks_src[:n_min], :d] @ Q
    E_tgt_subset = E_tgt[freq_ranks_tgt[:n_min], :d]

    D = pairwise_squared_distances(E_src_aligned, E_tgt_subset)

    top_k_values = [1, 5, 10]
    acc_at_k = {}
    for topk in top_k_values:
        correct = 0
        for i in range(n_min):
            top_indices = np.argsort(D[i])[:topk]
            if i in top_indices:
                correct += 1
        acc_at_k[topk] = correct / n_min if n_min > 0 else 0.0

    mrr = 0.0
    for i in range(n_min):
        ranking = np.argsort(D[i])
        rank = int(np.where(ranking == i)[0][0]) + 1 if i in ranking else n_min
        mrr += 1.0 / rank
    mrr /= n_min if n_min > 0 else 1

    return {
        "experiment": "permutation_recovery",
        "acc_at_k": acc_at_k,
        "mrr": mrr,
        "n_anchors": n_anchors,
        "n_aligned": n_min,
        "embedding_dim": k,
        "singular_values_source": sv_src.tolist(),
        "singular_values_target": sv_tgt.tolist(),
    }


def experiment_logosyllabic_collapse(
    source_sequences: list[list[int]],
    target_sequences: list[list[int]],
    collapse_map: dict[int, list[int]],
    source_vocab_size: int,
    target_vocab_size: int,
    window_size: int = 3,
    k: int = 16,
    seed: int = 42,
) -> dict:
    """Experiment 2: Logosyllabic collapse (many-to-one mappings).

    Simulates the scenario where multiple target symbols map to
    a single source symbol (common in logosyllabic systems).

    Uses FiberRecall@K instead of Acc@1.

    Args:
        source_sequences: Source corpus (with collapsed symbols).
        target_sequences: Target corpus (original symbols).
        collapse_map: Dict mapping source_symbol -> [list of target symbols].
        source_vocab_size: Source vocabulary size (after collapse).
        target_vocab_size: Target vocabulary size.
        window_size: Co-occurrence window.
        k: Embedding dimensionality.
        seed: Random seed.

    Returns:
        Dict with FiberRecall@K metrics.
    """
    from .cooccurrence import cooccurrence_matrix_from_sequences
    from .pmi import ppmi_matrix
    from .spectral import spectral_embedding
    from .alignment import orthogonal_procrustes, pairwise_squared_distances

    C_src = cooccurrence_matrix_from_sequences(
        source_sequences, source_vocab_size, window_size=window_size
    )
    C_tgt = cooccurrence_matrix_from_sequences(
        target_sequences, target_vocab_size, window_size=window_size
    )

    M_src = ppmi_matrix(C_src)
    M_tgt = ppmi_matrix(C_tgt)

    E_src, _, _ = spectral_embedding(M_src, k=k)
    E_tgt, _, _ = spectral_embedding(M_tgt, k=k)

    n_min = min(E_src.shape[0], E_tgt.shape[0])
    d = min(E_src.shape[1], E_tgt.shape[1])

    freq_ranks_src = np.argsort(-np.linalg.norm(E_src, axis=1))[:n_min]
    freq_ranks_tgt = np.argsort(-np.linalg.norm(E_tgt, axis=1))[:n_min]

    n_anch = min(20, n_min)
    X_anch = E_src[freq_ranks_src[:n_anch], :d]
    Y_anch = E_tgt[freq_ranks_tgt[:n_anch], :d]
    Q = orthogonal_procrustes(X_anch, Y_anch)

    E_src_aligned = E_src[freq_ranks_src[:n_min], :d] @ Q
    E_tgt_subset = E_tgt[freq_ranks_tgt[:n_min], :d]

    D = pairwise_squared_distances(E_src_aligned, E_tgt_subset)

    top_k_values = [1, 3, 5, 10]
    fiber_recall = {}

    inv_map = {}
    for src_sym, tgt_list in collapse_map.items():
        for tgt_sym in tgt_list:
            inv_map[tgt_sym] = inv_map.get(tgt_sym, [])
            if src_sym not in inv_map[tgt_sym]:
                inv_map[tgt_sym].append(src_sym)

    for topk in top_k_values:
        total_recall = 0.0
        n_fibers = 0
        for src_sym, tgt_list in collapse_map.items():
            if src_sym >= n_min or not tgt_list:
                continue
            rank_i = np.where(freq_ranks_src == src_sym)[0]
            if len(rank_i) == 0:
                continue
            rank_i = rank_i[0]
            top_indices = set(np.argsort(D[rank_i])[:topk])
            n_fibers += 1
            fiber_hits = sum(1 for t in tgt_list if t in top_indices)
            total_recall += fiber_hits / len(tgt_list)
        fiber_recall[topk] = total_recall / n_fibers if n_fibers > 0 else 0.0

    return {
        "experiment": "logosyllabic_collapse",
        "fiber_recall_at_k": fiber_recall,
        "n_collapse_groups": len(collapse_map),
        "embedding_dim": k,
    }


def experiment_boustrophedon_direction(
    sequences: list[list[int]],
    vocab_size: int,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict:
    """Experiment 4: Recover boustrophedon direction via likelihood.

    For each line, test whether left-to-right or right-to-left
    yields higher likelihood under a bigram model.

    Args:
        sequences: Per-line token-ID sequences.
        vocab_size: Vocabulary size.
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with direction recovery accuracy and per-line results.
    """
    rng = np.random.RandomState(seed)

    transition_fwd = np.zeros((vocab_size, vocab_size), dtype=float)
    for line in sequences:
        for i in range(len(line) - 1):
            transition_fwd[line[i], line[i + 1]] += 1

    row_sums = transition_fwd.sum(axis=1, keepdims=True)
    transition_fwd = transition_fwd / (row_sums + 1e-10)

    transition_bwd = np.zeros((vocab_size, vocab_size), dtype=float)
    for line in sequences:
        rev = list(reversed(line))
        for i in range(len(rev) - 1):
            transition_bwd[rev[i], rev[i + 1]] += 1

    row_sums_bwd = transition_bwd.sum(axis=1, keepdims=True)
    transition_bwd = transition_bwd / (row_sums_bwd + 1e-10)

    results = []
    total_fwd_likelihood = 0.0
    total_bwd_likelihood = 0.0
    fwd_wins = 0

    for line in sequences:
        if len(line) < 2:
            results.append({
                "direction": "unknown",
                "fwd_ll": float("-inf"),
                "bwd_ll": float("-inf"),
            })
            continue

        fwd_ll = 0.0
        for i in range(len(line) - 1):
            fwd_ll += np.log(transition_fwd[line[i], line[i + 1]] + 1e-15)
        bwd_ll = 0.0
        rev = list(reversed(line))
        for i in range(len(rev) - 1):
            bwd_ll += np.log(transition_bwd[rev[i], rev[i + 1]] + 1e-15)

        direction = "forward" if fwd_ll > bwd_ll else "reverse"
        results.append({
            "direction": direction,
            "fwd_ll": float(fwd_ll),
            "bwd_ll": float(bwd_ll),
            "line_length": len(line),
        })

        total_fwd_likelihood += fwd_ll
        total_bwd_likelihood += bwd_ll
        if fwd_ll > bwd_ll:
            fwd_wins += 1

    n_lines = len([r for r in results if r.get("line_length", 0) >= 2])
    accuracy = fwd_wins / n_lines if n_lines > 0 else 0.0

    return {
        "experiment": "boustrophedon_direction",
        "n_lines": n_lines,
        "forward_wins": fwd_wins,
        "direction_accuracy": accuracy,
        "total_fwd_ll": float(total_fwd_likelihood),
        "total_bwd_ll": float(total_bwd_likelihood),
        "per_line_results": results,
    }


def experiment_calendar_model(
    sequences: list[list[int]],
    vocab_size: int,
    n_lunar_phases: int = 30,
    seed: int = 42,
) -> dict:
    """Experiment 6: Compare calendar vs n-gram models via BIC.

    Tests whether a lunar-phase latent chain explains the sequence
    better than a general n-gram model.

    Args:
        sequences: Per-line token-ID sequences.
        vocab_size: Vocabulary size.
        n_lunar_phases: Number of lunar phases (typically 30).
        seed: Random seed.

    Returns:
        Dict with BIC comparison and model likelihoods.
    """
    rng = np.random.RandomState(seed)

    total_tokens = sum(len(line) for line in sequences)
    all_tokens = [t for line in sequences for t in line]
    transitions = np.zeros((vocab_size, vocab_size), dtype=float)
    for line in sequences:
        for i in range(len(line) - 1):
            transitions[line[i], line[i + 1]] += 1

    transition_probs = transitions / (transitions.sum(axis=1, keepdims=True) + 1e-10)

    transition_params = (vocab_size - 1) * vocab_size

    ngram_ll = 0.0
    n_valid = 0
    for line in sequences:
        for i in range(len(line) - 1):
            p = transition_probs[line[i], line[i + 1]]
            if p > 0:
                ngram_ll += np.log(p)
                n_valid += 1

    ngram_bic = -2 * ngram_ll + transition_params * np.log(total_tokens)

    calendar_params = (vocab_size * n_lunar_phases) + (n_lunar_phases - 1)
    phase_probs = rng.dirichlet(np.ones(vocab_size), size=n_lunar_phases)

    calendar_ll = 0.0
    for line in sequences:
        for i, token in enumerate(line):
            phase = i % n_lunar_phases
            p = phase_probs[phase, token]
            calendar_ll += np.log(p + 1e-15)

    calendar_bic = -2 * calendar_ll + calendar_params * np.log(total_tokens)

    delta_bic = ngram_bic - calendar_bic

    return {
        "experiment": "calendar_model",
        "n_lunar_phases": n_lunar_phases,
        "ngram_ll": float(ngram_ll),
        "ngram_bic": float(ngram_bic),
        "ngram_params": int(transition_params),
        "calendar_ll": float(calendar_ll),
        "calendar_bic": float(calendar_bic),
        "calendar_params": int(calendar_params),
        "delta_bic": float(delta_bic),
        "calendar_preferred": delta_bic > 0,
        "total_tokens": total_tokens,
    }


def find_parallel_passages(
    sequences: list[list[int]],
    edit_distance_threshold: float = 0.3,
    min_length: int = 3,
) -> list[dict]:
    """Experiment 5: Find parallel passages in a corpus.

    Identifies segment pairs with edit similarity above threshold.
    Parallel passages may indicate formulaic structures but
    cannot support translation claims.

    Args:
        sequences: Per-line token-ID sequences.
        edit_distance_threshold: Normalized edit distance threshold.
        min_length: Minimum sequence length to consider.

    Returns:
        List of parallel passage records.
    """
    parallels = []

    def normalized_edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        if m == 0 and n == 0:
            return 0.0
        if m == 0 or n == 0:
            return 1.0
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n] / max(m, n)

    long_seqs = [(i, seq) for i, seq in enumerate(sequences) if len(seq) >= min_length]

    for idx1, (i, seq1) in enumerate(long_seqs):
        for idx2 in range(idx1 + 1, len(long_seqs)):
            j, seq2 = long_seqs[idx2]
            if abs(len(seq1) - len(seq2)) > max(len(seq1), len(seq2)) * 0.5:
                continue
            ned = normalized_edit_distance(seq1, seq2)
            edit_sim = 1.0 - ned
            if edit_sim > (1.0 - edit_distance_threshold):
                parallels.append({
                    "line_idx_1": i,
                    "line_idx_2": j,
                    "edit_similarity": float(edit_sim),
                    "length_1": len(seq1),
                    "length_2": len(seq2),
                    "sequence_1": seq1[:20],
                    "sequence_2": seq2[:20],
                })

    return parallels


def generate_permuted_corpus(
    sequences: list[list[int]],
    vocab_size: int,
    seed: int = 42,
) -> tuple[list[list[int]], dict[int, int]]:
    """Generate a synthetic corpus by applying a random permutation.

    Used in Experiment 1: the ground truth permutation is known,
    so we can evaluate recovery metrics exactly.

    Args:
        sequences: Original token-ID sequences.
        vocab_size: Vocabulary size.
        seed: Random seed.

    Returns:
        Tuple of (permuted_sequences, permutation_map).
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(vocab_size)
    perm_map = {i: int(perm[i]) for i in range(vocab_size)}

    permuted = [[perm[t] for t in line] for line in sequences]
    return permuted, perm_map


def generate_collapsed_corpus(
    sequences: list[list[int]],
    collapse_map: dict[int, list[int]],
) -> list[list[int]]:
    """Generate a corpus where multiple target symbols collapse to one source.

    Used in Experiment 2 (logosyllabic collapse).

    Args:
        sequences: Original tokens.
        collapse_map: Dict mapping source_symbol -> [target symbols].
            Inverse: every target symbol maps to its source group.

    Returns:
        Collapsed sequences.
    """
    inv_map = {}
    for src, tgt_list in collapse_map.items():
        for tgt in tgt_list:
            inv_map[tgt] = src

    collapsed = []
    for line in sequences:
        collapsed_line = [inv_map.get(t, t) for t in line]
        collapsed.append(collapsed_line)
    return collapsed


def experiment_unknown_segmentation(
    sequences: list[list[int]],
    vocab_size: int,
    window_size: int = 3,
    seed: int = 42,
) -> dict:
    """Experiment 3: Segmentation recovery under unknown boundaries.

    Tests whether co-occurrence statistics can recover segment (word)
    boundaries when the script has no explicit delimiters.

    Strategy:
    1. Flatten all sequences into one long token stream (removing boundaries).
    2. Re-segment using three baselines:
       a) Unigram: every token is a segment.
       b) Bigram MI: merge adjacent tokens whose PMI exceeds threshold.
       c) BPE: iteratively merge most frequent pairs (10 merge iterations).
    3. Measure boundary recovery: precision, recall, F1 against true boundaries.

    Returns dict with metrics for each method.
    """
    rng = np.random.RandomState(seed)

    flat = []
    true_boundaries = []
    pos = 0
    for seq in sequences:
        for t in seq:
            flat.append(t)
            pos += 1
        true_boundaries.append(pos - 1)
    true_boundaries_set = set(b for b in true_boundaries if b < len(flat) - 1)
    n_true = len(true_boundaries_set)

    unigram_boundaries = set(range(len(flat) - 1))
    tp_uni = len(true_boundaries_set & unigram_boundaries)
    fp_uni = len(unigram_boundaries) - tp_uni
    fn_uni = n_true - tp_uni
    prec_uni = tp_uni / (tp_uni + fp_uni) if (tp_uni + fp_uni) > 0 else 0
    rec_uni = tp_uni / (tp_uni + fn_uni) if (tp_uni + fn_uni) > 0 else 0
    f1_uni = 2 * prec_uni * rec_uni / (prec_uni + rec_uni) if (prec_uni + rec_uni) > 0 else 0

    bigram_freq = Counter()
    unigram_freq = Counter()
    for t in flat:
        unigram_freq[t] += 1
    for i in range(len(flat) - 1):
        bigram_freq[(flat[i], flat[i + 1])] += 1
    total_bigrams = sum(bigram_freq.values())
    total_unigrams = sum(unigram_freq.values())

    pmi = {}
    for (a, b), count in bigram_freq.items():
        pa = unigram_freq[a] / total_unigrams
        pb = unigram_freq[b] / total_unigrams
        pab = count / total_bigrams
        if pa > 0 and pb > 0 and pab > 0:
            pmi[(a, b)] = np.log(pab / (pa * pb))

    mi_threshold = 0.0
    bigram_boundaries = set()
    for i in range(len(flat) - 1):
        pair = (flat[i], flat[i + 1])
        if pair in pmi and pmi[pair] > mi_threshold:
            pass
        else:
            bigram_boundaries.add(i)
    tp_bi = len(true_boundaries_set & bigram_boundaries)
    fp_bi = len(bigram_boundaries) - tp_bi
    fn_bi = n_true - tp_bi
    prec_bi = tp_bi / (tp_bi + fp_bi) if (tp_bi + fp_bi) > 0 else 0
    rec_bi = tp_bi / (tp_bi + fn_bi) if (tp_bi + fn_bi) > 0 else 0
    f1_bi = 2 * prec_bi * rec_bi / (prec_bi + rec_bi) if (prec_bi + rec_bi) > 0 else 0
    n_bigram_segments = len(bigram_boundaries) + 1

    def run_bpe(tokens, n_merges):
        merges = []
        current = [t for t in tokens]
        for _ in range(n_merges):
            pairs = Counter()
            for i in range(len(current) - 1):
                if current[i] != current[i + 1]:
                    pairs[(current[i], current[i + 1])] += 1
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            merges.append(best)
            new_current = []
            i = 0
            while i < len(current):
                if i < len(current) - 1 and current[i] == best[0] and current[i + 1] == best[1]:
                    new_current.append(f"{best[0]}_{best[1]}")
                    i += 2
                else:
                    new_current.append(current[i])
                    i += 1
            current = new_current
        return current, merges

    bpe_flat, bpe_merges = run_bpe(flat, n_merges=50)
    bpe_boundaries = set()
    flat_idx = 0
    for token in bpe_flat:
        if "_" in str(token):
            parts = str(token).split("_")
            flat_idx += len(parts)
        else:
            flat_idx += 1
        bpe_boundaries.add(flat_idx - 1)
    bpe_boundaries = bpe_boundaries - {len(flat) - 1}
    tp_bpe = len(true_boundaries_set & bpe_boundaries)
    fp_bpe = len(bpe_boundaries - true_boundaries_set)
    fn_bpe = n_true - tp_bpe
    prec_bpe = tp_bpe / (tp_bpe + fp_bpe) if (tp_bpe + fp_bpe) > 0 else 0
    rec_bpe = tp_bpe / (tp_bpe + fn_bpe) if (tp_bpe + fn_bpe) > 0 else 0
    f1_bpe = 2 * prec_bpe * rec_bpe / (prec_bpe + rec_bpe) if (prec_bpe + rec_bpe) > 0 else 0

    return {
        "n_true_boundaries": n_true,
        "total_tokens": len(flat),
        "unigram": {
            "precision": prec_uni,
            "recall": rec_uni,
            "f1": f1_uni,
            "n_predicted_boundaries": len(unigram_boundaries),
        },
        "bigram_mi": {
            "precision": prec_bi,
            "recall": rec_bi,
            "f1": f1_bi,
            "n_predicted_boundaries": n_bigram_segments - 1,
        },
        "bpe_50": {
            "precision": prec_bpe,
            "recall": rec_bpe,
            "f1": f1_bpe,
            "n_predicted_boundaries": len(bpe_boundaries),
            "n_merges": len(bpe_merges),
        },
        "bigram_type_diversity": len(bigram_freq) / max(len(unigram_freq), 1),
        "vocab_coverage": len(unigram_freq) / max(vocab_size, 1),
    }