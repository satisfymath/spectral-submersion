"""Convert Tatoeba per-language TSV to project corpus CSV format.

Samples max_sentences sentences per language and tokenizes by whitespace.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^\w'-]", "", token)
    return token if token else "<EMPTY>"


def tatoeba_tsv_to_corpus(
    input_path: str,
    output_path: str,
    max_sentences: int = 50000,
    seed: int = 42,
) -> None:
    df = pd.read_csv(input_path, sep="\t", header=None, names=["sent_id", "lang", "text"], quoting=3)
    df["text"] = df["text"].fillna("").astype(str)
    if max_sentences < len(df):
        df = df.sample(n=max_sentences, random_state=seed).sort_values("sent_id")
    rows = []
    for _, row in df.iterrows():
        sent_id = row["sent_id"]
        text = str(row["text"])
        tokens = text.split()
        for pos, raw_tok in enumerate(tokens, start=1):
            tok = normalize_token(raw_tok)
            if tok != "<EMPTY>":
                rows.append({
                    "doc_id": f"tatoeba_{row['lang']}",
                    "line_id": int(sent_id),
                    "position": pos,
                    "token": tok,
                    "raw_token": raw_tok,
                })
    out_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} tokens ({df.shape[0]} sentences) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Tatoeba TSV to corpus CSV")
    parser.add_argument("--input", required=True, help="Input TSV file from Tatoeba")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--max-sentences", type=int, default=50000, help="Max sentences to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    tatoeba_tsv_to_corpus(args.input, args.output, args.max_sentences, args.seed)


if __name__ == "__main__":
    main()