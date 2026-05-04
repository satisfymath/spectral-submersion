"""Build candidate language corpus CSV from plain text sentences.

Source: OPUS-Tatoeba v2023-04-12 (open, CC-BY 2.0 FR).
Linguistically, these are Polynesian languages used as candidates
for structural comparison. They are NOT Rapa Nui / Rongorongo.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_token(token: str) -> str:
    """Lowercase and strip punctuation for basic token normalization."""
    token = token.lower().strip()
    # Keep alphanumeric, hyphens, apostrophes (common in Polynesian)
    token = re.sub(r"[^a-zāēīōūáéíóúàèìòùâêîôûäëïöüṅṇñ'\-]", "", token)
    return token if token else "<EMPTY>"


def text_to_corpus_csv(input_path: str, output_path: str) -> None:
    """Convert one-sentence-per-line text to project CSV format."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = []
    for sent_id, line in enumerate(lines, start=1):
        tokens = line.split()
        for pos, raw_tok in enumerate(tokens, start=1):
            tok = normalize_token(raw_tok)
            rows.append(
                {
                    "doc_id": f"tatoeba_{input_path.stem}",
                    "line_id": sent_id,
                    "position": pos,
                    "token": tok,
                    "lemma": tok,
                    "pos": "UNKNOWN",
                    "translation": "",
                    "source": f"opus_tatoeba_v2023-04-12_{input_path.stem}",
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"Saved {len(df)} tokens ({df['line_id'].nunique()} sentences) to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build candidate corpus CSV from text")
    parser.add_argument(
        "--input", required=True, help="Input text file (one sentence per line)"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    text_to_corpus_csv(args.input, args.output)


if __name__ == "__main__":
    main()
