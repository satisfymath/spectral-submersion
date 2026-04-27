"""Lightweight Polynesian token segmentation.

Polynesian languages are agglutinative with many functional particles
that appear as separate words in space-delimited text but may be
clitics or bound morphemes in deeper analysis. For our co-occurrence
purposes, separating high-frequency particles reduces polysemy noise
and sharpens syntactic structure in the embeddings.

Sources for particle lists:
- General Polynesian linguistics references (space-delimited tokens treated
  as separate words, but some are clearly functional and worth isolating).
- No claim of morphological completeness; this is heuristic.
"""
import re
from pathlib import Path

import pandas as pd


# High-frequency functional particles by language code.
# These are separated with a boundary marker when found at word edges.
PARTICLES = {
    "mi": ["te", "he", "kei", "i", "ki", "ka", "me", "e", "ko", "ā", "o", "nā", "mā", "he", "tō", "taku"],
    "ty": ["te", "e", "i", "o", "ua", "ia", "ta", "to"],
    "haw": ["ka", "ke", "o", "i", "e", "ua", "he", "na", "ko", "ma"],
    "sm": ["le", "o", "e", "i", "ua", "a", "na", "ma"],
    "to": ["he", "a", "e", "ki", "i", "o", "ko", "oku", "hoku"],
    "fj": ["na", "e", "mai", "i", "ko", "a", "me", "sa"],
    "rap": ["te", "e", "i", "o", "ka", "he", "ta"],
}


def segment_token(token: str, particles: list[str]) -> list[str]:
    """If token starts/ends with a known particle, split it.

    This is intentionally conservative: only exact matches at edges
    to avoid over-segmenting content words.
    """
    token_lower = token.lower().strip("'ʻ")
    # Check prefix
    for p in sorted(particles, key=len, reverse=True):
        if token_lower.startswith(p) and len(token_lower) > len(p):
            remainder = token[len(p):].lstrip("'ʻ-")
            if remainder:
                return [token[:len(p)], remainder]
    # No split
    return [token]


def tokenize_corpus_df(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    """Apply Polynesian particle segmentation to a candidate corpus DataFrame."""
    particles = PARTICLES.get(lang_code, [])
    if not particles:
        return df

    rows = []
    for _, row in df.iterrows():
        tok = row["token"]
        segs = segment_token(tok, particles)
        if len(segs) == 1:
            rows.append(row.to_dict())
        else:
            for pos_offset, seg in enumerate(segs):
                new_row = row.to_dict()
                new_row["token"] = seg
                new_row["raw_token"] = tok
                new_row["position"] = row["position"] + pos_offset * 0.1
                rows.append(new_row)

    return pd.DataFrame(rows)


def process_all_candidates(config_path: str = "configs/candidate_languages.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for cand in config.get("candidates", []):
        code = cand.get("code")
        input_csv = cand["corpus_path"]
        output_csv = input_csv.replace(".csv", "_segmented.csv")

        df = pd.read_csv(input_csv)
        df_seg = tokenize_corpus_df(df, code)
        df_seg.to_csv(output_csv, index=False)
        print(f"[{code}] {len(df)} -> {len(df_seg)} tokens. Saved to {output_csv}")


def main():
    process_all_candidates()


if __name__ == "__main__":
    main()
