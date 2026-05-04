"""Build clean dataset from raw corpus."""

from pathlib import Path

import pandas as pd

from spectral_submersion.tokenization import read_corpus, normalize_tokens


def main():
    raw_path = Path("data/raw/lost_language/corpus.csv")
    out_path = Path("data/processed/lost_tokens.csv")

    df = read_corpus(str(raw_path))
    df = normalize_tokens(df, lowercase=True, strip=True)
    df = df.sort_values(["doc_id", "line_id", "position"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved clean corpus to {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Unique tokens: {df['token'].nunique()}")
    print(f"Documents: {df['doc_id'].nunique()}")
    print(f"Lines: {df.groupby(['doc_id', 'line_id']).ngroups}")


if __name__ == "__main__":
    main()
