"""Build enhanced embeddings combining spectral + iconographic features.

For the Indus corpus, each sign has a feature vector from the JSON source
(damage, line, uncertainty, plus sign-specific features like branching_factor).
This script extracts those features, normalizes them, and concatenates them
with spectral embeddings to create enhanced sign representations that include
both contextual and visual/iconographic information.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from spectral_submersion.cooccurrence import (
    build_vocab,
    cooccurrence_matrix_from_sequences,
)
from spectral_submersion.pmi import ppmi_matrix
from spectral_submersion.spectral import spectral_embedding
from spectral_submersion.tokenization import get_sequences_by_line, tokens_to_ids


def extract_feature_vectors(
    json_dir: str, token_vocab: dict[str, int]
) -> dict[str, list[float]]:
    """Extract feature vectors for each token from JSON corpus files.

    Returns a dict mapping token -> feature vector (list of floats).
    For tokens not found in JSON, returns a zero vector.
    """
    input_path = Path(json_dir)
    json_files = sorted(input_path.rglob("*.json"))

    # Collect all feature vectors per token
    token_features: dict[str, list[list[int]]] = {}

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for side in data:
            for g in side.get("graphemes", []):
                token = g["id"]
                features = g.get("features", [])
                if token not in token_features:
                    token_features[token] = []
                token_features[token].append(features)

    # Average feature vectors per token (in case of multiple occurrences)
    token_avg_features: dict[str, list[float]] = {}
    for token, feat_list in token_features.items():
        max_len = max(len(f) for f in feat_list)
        # Pad shorter vectors with zeros
        padded = [f + [0] * (max_len - len(f)) for f in feat_list]
        avg = [float(sum(col)) / len(col) for col in zip(*padded)]
        token_avg_features[token] = avg

    # Determine max feature dimension
    max_dim = (
        max(len(v) for v in token_avg_features.values()) if token_avg_features else 0
    )

    # Build final dict aligned with vocab
    result = {}
    for token in token_vocab:
        if token in token_avg_features:
            feat = token_avg_features[token]
            # Pad to max_dim
            feat = feat + [0.0] * (max_dim - len(feat))
            result[token] = feat
        else:
            result[token] = [0.0] * max_dim

    return result


def build_enhanced_embeddings(
    corpus_csv: str,
    json_dir: str,
    output_path: str,
    k: int = 16,
    window: int = 3,
    alpha: float = 0.0,
    feature_weight: float = 1.0,
):
    """Build embeddings = [spectral | normalized_features]."""
    df = pd.read_csv(corpus_csv)
    tokens = df["token"].tolist()
    vocab = build_vocab(tokens)
    sequences = get_sequences_by_line(df)
    seq_ids = [tokens_to_ids(seq, vocab) for seq in sequences]

    # Spectral embeddings
    C = cooccurrence_matrix_from_sequences(seq_ids, len(vocab), window_size=window)
    M = ppmi_matrix(C, epsilon=1e-9)
    E_spec, _, _ = spectral_embedding(M, k=k, alpha=alpha, random_state=42)

    # Feature vectors
    token_features = extract_feature_vectors(json_dir, vocab)
    feat_matrix = np.array([token_features[tok] for tok in vocab])

    # Normalize features
    if feat_matrix.shape[1] > 0 and feat_matrix.max() > 0:
        scaler = StandardScaler()
        feat_norm = scaler.fit_transform(feat_matrix)
    else:
        feat_norm = feat_matrix

    # Concatenate with weight
    E_enhanced = np.hstack([E_spec, feat_norm * feature_weight])

    np.save(output_path, E_enhanced)
    print(f"Spectral shape: {E_spec.shape}")
    print(f"Feature shape: {feat_norm.shape}")
    print(f"Enhanced shape: {E_enhanced.shape}")
    print(f"Saved enhanced embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build enhanced embeddings with iconographic features"
    )
    parser.add_argument(
        "--corpus", default="data/raw/lost_language/corpus_indus_real.csv"
    )
    parser.add_argument("--json-dir", default="/tmp/indus-corpus/corpus")
    parser.add_argument(
        "--output", default="data/processed/embeddings_indus_real_enhanced.npy"
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    args = parser.parse_args()

    # Ensure JSON corpus is available
    if not Path(args.json_dir).exists():
        print(f"ERROR: JSON corpus not found at {args.json_dir}")
        print(
            "Please clone: git clone https://github.com/mayig/indus-valley-script-corpus.git /tmp/indus-corpus"
        )
        return

    build_enhanced_embeddings(
        args.corpus,
        args.json_dir,
        args.output,
        k=args.k,
        window=args.window,
        alpha=args.alpha,
        feature_weight=args.feature_weight,
    )


if __name__ == "__main__":
    main()
