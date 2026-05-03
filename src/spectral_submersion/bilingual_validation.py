"""Bilingual validation: test the spectral submersion pipeline on known language pairs.

This module validates the entire pipeline (co-occurrence -> PPMI -> SVD ->
Procrustes alignment -> anchor recovery) on language pairs where we know
the ground truth translation. If the method cannot recover English-French
correspondences, it should not be applied to Rongorongo.

Key experiments:
1. Full-data validation: large vocab, many tokens, many anchors
2. Rongorongo-simulating conditions: restricted vocab (V~180, V~941),
   restricted tokens (T~1000, T~5460), few or no anchors
3. Sensitivity to anchor count, vocab size, and token count

Anchors are TRUE translation pairs (frequency-matched cognates).
Evaluation is Acc@K: given a source word, is the correct translation
in the top-K nearest neighbors after Procrustes alignment?
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Sequence

from .cooccurrence import cooccurrence_matrix_from_sequences
from .pmi import ppmi_matrix
from .spectral import spectral_embedding
from .alignment import orthogonal_procrustes, pairwise_squared_distances
from .stability import spectral_stability_bootstrap, cooccurrence_coverage


def _build_vocab_with_min_freq(
    tokens: list[str], max_vocab: int | None = None, min_freq: int = 5
) -> dict[str, int]:
    """Build vocabulary from token list, filtering by frequency."""
    freq = Counter(tokens)
    if min_freq > 1:
        freq = Counter({t: c for t, c in freq.items() if c >= min_freq})
    if max_vocab is not None:
        freq = Counter(dict(freq.most_common(max_vocab)))
    return {tok: i for i, tok in enumerate(sorted(freq.keys()))}


def _df_to_sequences(df, vocab: dict) -> list[list[int]]:
    """Convert DataFrame to list of token-ID sequences grouped by line."""
    sequences = []
    grouped = df.sort_values(["doc_id", "line_id", "position"]).groupby(
        ["doc_id", "line_id"]
    )
    for _, group in grouped:
        toks = group["token"].astype(str).tolist()
        ids = [vocab[t] for t in toks if t in vocab]
        if len(ids) >= 2:
            sequences.append(ids)
    return sequences


def _subsample_sequences(
    sequences: list[list[int]], max_tokens: int, rng: np.random.RandomState
) -> list[list[int]]:
    """Subsample sequences to stay under max_tokens."""
    total = sum(len(s) for s in sequences)
    if total <= max_tokens:
        return sequences
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    result = []
    running = 0
    for i in indices:
        if running + len(sequences[i]) <= max_tokens:
            result.append(sequences[i])
            running += len(sequences[i])
        if running >= max_tokens:
            break
    return result


def find_cognate_anchors(
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    known_translations: dict[str, str] | None = None,
) -> dict[int, int]:
    """Find anchor pairs between source and target vocabularies.

    If known_translations is provided, uses those directly.
    Otherwise, uses identity mapping (cognates: words identical in both
    languages like 'information'/'information', 'restaurant'/'restaurant').

    Args:
        src_vocab: Source token -> id mapping.
        tgt_vocab: Target token -> id mapping.
        known_translations: Optional dict of src_token -> tgt_token.

    Returns:
        Dict of src_id -> tgt_id anchor pairs.
    """
    anchors = {}

    if known_translations is not None:
        for src_tok, tgt_tok in known_translations.items():
            if src_tok in src_vocab and tgt_tok in tgt_vocab:
                anchors[src_vocab[src_tok]] = tgt_vocab[tgt_tok]
        return anchors

    # Default: use identical strings (cognates / loanwords)
    common = set(src_vocab.keys()) & set(tgt_vocab.keys())
    for tok in common:
        anchors[src_vocab[tok]] = tgt_vocab[tok]

    return anchors


def build_bilingual_corpus(
    source_df,
    target_df,
    min_freq: int = 5,
    max_vocab: int | None = None,
    max_tokens: int | None = None,
    seed: int = 42,
) -> dict:
    """Build aligned bilingual corpora from token DataFrames.

    Takes two corpora in the standard (doc_id, line_id, position, token)
    format and builds shared-vocabulary data structures for bilingual validation.

    Args:
        source_df: Source language DataFrame.
        target_df: Target language DataFrame.
        min_freq: Minimum token frequency.
        max_vocab: Maximum vocabulary size.
        max_tokens: Maximum total tokens.
        seed: Random seed.

    Returns:
        Dict with sequences, vocabs, and anchor information.
    """
    rng = np.random.RandomState(seed)

    src_vocab = _build_vocab_with_min_freq(
        source_df["token"].astype(str).tolist(),
        max_vocab=max_vocab, min_freq=min_freq,
    )
    tgt_vocab = _build_vocab_with_min_freq(
        target_df["token"].astype(str).tolist(),
        max_vocab=max_vocab, min_freq=min_freq,
    )

    src_seqs = _df_to_sequences(source_df, src_vocab)
    tgt_seqs = _df_to_sequences(target_df, tgt_vocab)

    if max_tokens is not None:
        src_seqs = _subsample_sequences(src_seqs, max_tokens, rng)
        tgt_seqs = _subsample_sequences(tgt_seqs, max_tokens, rng)

        # Rebuild vocab from actual tokens in subsampled sequences
        src_freq = Counter(t for s in src_seqs for t in s)
        tgt_freq = Counter(t for s in tgt_seqs for t in s)

        # Filter by min_freq in subsampled data
        src_ids_used = {tid for tid, f in src_freq.items() if f >= min_freq}
        tgt_ids_used = {tid for tid, f in tgt_freq.items() if f >= min_freq}

        # Remap sequences to contiguous IDs
        src_id_map = {old: new for new, old in enumerate(sorted(src_ids_used))}
        tgt_id_map = {old: new for new, old in enumerate(sorted(tgt_ids_used))}

        src_seqs = [[src_id_map[t] for t in s if t in src_id_map] for s in src_seqs]
        src_seqs = [s for s in src_seqs if len(s) >= 2]
        tgt_seqs = [[tgt_id_map[t] for t in s if t in tgt_id_map] for s in tgt_seqs]
        tgt_seqs = [s for s in tgt_seqs if len(s) >= 2]

        src_vs = len(src_id_map)
        tgt_vs = len(tgt_id_map)

        # Find cognate anchors in the remapped ID space
        cognate_anchors = find_cognate_anchors(src_vocab, tgt_vocab)

        # Remap anchors to new IDs
        # src_vocab maps token -> old_id, src_id_map maps old_id -> new_id
        inverted_src = {v: k for k, v in src_vocab.items()}
        inverted_tgt = {v: k for k, v in tgt_vocab.items()}

        remapped_anchors = {}
        for old_src_id, old_tgt_id in cognate_anchors.items():
            if old_src_id in src_id_map and old_tgt_id in tgt_id_map:
                remapped_anchors[src_id_map[old_src_id]] = tgt_id_map[old_tgt_id]

        src_vocab_out = {tok: src_id_map[old_id]
                         for tok, old_id in src_vocab.items()
                         if old_id in src_id_map}
        tgt_vocab_out = {tok: tgt_id_map[old_id]
                         for tok, old_id in tgt_vocab.items()
                         if old_id in tgt_id_map}
    else:
        src_vs = len(src_vocab)
        tgt_vs = len(tgt_vocab)
        cognate_anchors = find_cognate_anchors(src_vocab, tgt_vocab)
        remapped_anchors = cognate_anchors
        src_vocab_out = src_vocab
        tgt_vocab_out = tgt_vocab

    src_n_tokens = sum(len(s) for s in src_seqs)
    tgt_n_tokens = sum(len(s) for s in tgt_seqs)

    return {
        "source_sequences": src_seqs,
        "target_sequences": tgt_seqs,
        "source_vocab": src_vocab_out,
        "target_vocab": tgt_vocab_out,
        "source_vs": src_vs,
        "target_vs": tgt_vs,
        "source_n_tokens": src_n_tokens,
        "target_n_tokens": tgt_n_tokens,
        "cognate_anchors": remapped_anchors,
        "n_cognate_anchors": len(remapped_anchors),
    }


def validate_bilingual_pair(
    source_sequences: list[list[int]],
    target_sequences: list[list[int]],
    source_vocab_size: int,
    target_vocab_size: int,
    cognate_anchors: dict[int, int],
    n_anchors: int | None = None,
    anchor_fraction: float | None = None,
    window_size: int = 3,
    k: int = 16,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict:
    """Run full bilingual validation pipeline on a known language pair.

    Builds PPMI embeddings for both languages, aligns via Procrustes with
    cognate anchors, evaluates Acc@K against ground truth.

    Ground truth: cognate anchors themselves. We measure whether, after
    Procrustes alignment, the correct translation of a held-out source word
    is among its top-K nearest neighbors.

    Args:
        source_sequences: Source token-ID sequences.
        target_sequences: Target token-ID sequences.
        source_vocab_size: Source vocabulary size.
        target_vocab_size: Target vocabulary size.
        cognate_anchors: Dict of src_id -> tgt_id (ground-truth translation pairs).
        n_anchors: Number of anchor pairs to use for alignment (from cognate anchors).
        anchor_fraction: If set, use this fraction of cognates as anchors.
        window_size: Co-occurrence window.
        k: Embedding dimensionality.
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with validation metrics.
    """
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

    all_anchors = list(cognate_anchors.items())

    if anchor_fraction is not None:
        n_anch = max(2, int(anchor_fraction * min(source_vocab_size, target_vocab_size)))
    elif n_anchors is not None:
        n_anch = min(n_anchors, len(all_anchors))
    else:
        n_anch = max(2, min(20, len(all_anchors)))

    n_anch = min(n_anch, len(all_anchors))

    rng.shuffle(all_anchors)
    anchors_for_alignment = all_anchors[:n_anch]
    anchors_for_eval = all_anchors[n_anch:]

    if len(anchors_for_eval) < 5:
        anchors_for_eval = all_anchors

    d = min(E_src.shape[1], E_tgt.shape[1])

    src_anch_ids = np.array([int(a[0]) for a in anchors_for_alignment], dtype=np.int64)
    tgt_anch_ids = np.array([int(a[1]) for a in anchors_for_alignment], dtype=np.int64)

    if len(src_anch_ids) >= 2:
        X_anch = E_src[src_anch_ids][:, :d]
        Y_anch = E_tgt[tgt_anch_ids][:, :d]
        Q = orthogonal_procrustes(X_anch, Y_anch)
    else:
        Q = np.eye(d)

    E_src_aligned = E_src[:, :d] @ Q

    eval_src_ids = np.array([int(a[0]) for a in anchors_for_eval], dtype=np.int64)
    eval_tgt_ids = np.array([int(a[1]) for a in anchors_for_eval], dtype=np.int64)

    if len(eval_src_ids) == 0:
        n_min = min(E_src_aligned.shape[0], E_tgt.shape[0])
        src_norms = np.linalg.norm(E_src_aligned, axis=1)
        tgt_norms = np.linalg.norm(E_tgt[:, :d], axis=1)
        src_top = np.argsort(-src_norms)[:n_min]
        tgt_top = np.argsort(-tgt_norms)[:n_min]

        eval_src_ids = src_top
        eval_tgt_ids = tgt_top

    D_eval = pairwise_squared_distances(E_src_aligned[eval_src_ids], E_tgt[eval_tgt_ids][:, :d])

    n_eval = len(eval_src_ids)

    acc_at_k = {}
    for topk in [1, 5, 10, 20]:
        correct = 0
        for i in range(n_eval):
            top_indices = set(np.argsort(D_eval[i])[:topk])
            if i in top_indices:
                correct += 1
        acc_at_k[topk] = correct / n_eval if n_eval > 0 else 0.0

    mrr = 0.0
    for i in range(n_eval):
        ranking = np.argsort(D_eval[i])
        if i in ranking:
            rank = int(np.where(ranking == i)[0][0]) + 1
        else:
            rank = n_eval
        mrr += 1.0 / rank
    mrr /= n_eval if n_eval > 0 else 1

    src_coverage = cooccurrence_coverage(C_src)
    tgt_coverage = cooccurrence_coverage(C_tgt)

    src_n_tokens = sum(len(s) for s in source_sequences)
    src_epc = 2 * window_size * src_n_tokens / (source_vocab_size ** 2) if source_vocab_size > 0 else float("inf")

    src_stab = spectral_stability_bootstrap(
        source_sequences, source_vocab_size, k=k,
        window_size=window_size, n_bootstrap=n_bootstrap, random_state=seed,
    )
    tgt_stab = spectral_stability_bootstrap(
        target_sequences, target_vocab_size, k=k,
        window_size=window_size, n_bootstrap=n_bootstrap, random_state=seed + 1,
    )

    src_reff = _effective_rank(sv_src[:k])
    tgt_reff = _effective_rank(sv_tgt[:k])

    return {
        "experiment": "bilingual_validation",
        "source_vocab_size": source_vocab_size,
        "target_vocab_size": target_vocab_size,
        "source_n_tokens": src_n_tokens,
        "target_n_tokens": sum(len(s) for s in target_sequences),
        "n_alignment_anchors": len(anchors_for_alignment),
        "n_eval_pairs": n_eval,
        "total_cognates": len(cognate_anchors),
        "anchor_fraction_used": len(anchors_for_alignment) / max(min(source_vocab_size, target_vocab_size), 1),
        "embedding_dim": k,
        "window_size": window_size,
        "acc_at_k": acc_at_k,
        "mrr": mrr,
        "src_effective_rank": src_reff,
        "tgt_effective_rank": tgt_reff,
        "src_cooc_coverage": src_coverage,
        "tgt_cooc_coverage": tgt_coverage,
        "src_epc": src_epc,
        "src_spectral_reliability": src_stab["spectral_reliability"],
        "tgt_spectral_reliability": tgt_stab["spectral_reliability"],
        "src_singular_values": sv_src.tolist(),
        "tgt_singular_values": sv_tgt.tolist(),
    }


def _effective_rank(sv: np.ndarray) -> float:
    s = sv[sv > 0]
    if len(s) == 0 or s.sum() == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def restricted_bilingual_experiment(
    source_df,
    target_df,
    conditions: list[dict],
    window_size: int = 3,
    k: int = 16,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Run bilingual validation under multiple restricted conditions.

    Each condition specifies max_vocab, max_tokens, and anchor settings.
    """
    results = []
    for cond in conditions:
        print(f"  Running condition: {cond['name']}...")

        corpus = build_bilingual_corpus(
            source_df, target_df,
            min_freq=5,
            max_vocab=cond.get("max_vocab"),
            max_tokens=cond.get("max_tokens"),
            seed=seed,
        )

        print(f"    V_src={corpus['source_vs']}, V_tgt={corpus['target_vs']}, "
              f"T_src={corpus['source_n_tokens']}, T_tgt={corpus['target_n_tokens']}, "
              f"cognates={corpus['n_cognate_anchors']}")

        if corpus["n_cognate_anchors"] < 5:
            print(f"    SKIP: too few cognate anchors ({corpus['n_cognate_anchors']})")
            continue

        result = validate_bilingual_pair(
            source_sequences=corpus["source_sequences"],
            target_sequences=corpus["target_sequences"],
            source_vocab_size=corpus["source_vs"],
            target_vocab_size=corpus["target_vs"],
            cognate_anchors=corpus["cognate_anchors"],
            n_anchors=cond.get("n_anchors"),
            anchor_fraction=cond.get("anchor_fraction"),
            window_size=window_size,
            k=k,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        result["condition_name"] = cond["name"]
        result["condition_max_vocab"] = cond.get("max_vocab")
        result["condition_max_tokens"] = cond.get("max_tokens")

        print(
            f"    Acc@1={result['acc_at_k'][1]:.4f}, Acc@5={result['acc_at_k'][5]:.4f}, "
            f"Acc@10={result['acc_at_k'][10]:.4f}, MRR={result['mrr']:.4f} | "
            f"Anchors={result['n_alignment_anchors']}, Eval={result['n_eval_pairs']} | "
            f"Rel_src={result['src_spectral_reliability']:.3f}, "
            f"Rel_tgt={result['tgt_spectral_reliability']:.3f}, "
            f"EPC={result['src_epc']:.4f}"
        )
        results.append(result)

    return results