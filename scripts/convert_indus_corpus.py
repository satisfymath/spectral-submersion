"""Convert Indus Valley Script Corpus JSON files to project CSV format.

Source: https://github.com/mayig/indus-valley-script-corpus (MIT License)
Digitization of Corpus of Indus Seals and Inscriptions (CISI) by Parpola et al.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def convert_indus_corpus(input_dir: str, output_path: str) -> None:
    """Convert all JSON files in input_dir to a single project CSV."""
    input_path = Path(input_dir)
    json_files = sorted(input_path.rglob("*.json"))

    rows = []
    line_counter = 0
    all_tokens = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for side in data:
            line_counter += 1
            doc_id = side.get("id", f"unknown_{line_counter}")
            description = side.get("description", "")
            graphemes = side.get("graphemes", [])

            for pos, g in enumerate(graphemes, start=1):
                token = g["id"]
                features = g.get("features", [])
                all_tokens.append(token)
                rows.append({
                    "doc_id": doc_id,
                    "line_id": line_counter,
                    "position": pos,
                    "token": token,
                    "features": json.dumps(features),
                    "description": description,
                    "source": "indus_valley_corpus_parpola_cisi_mayig_2025",
                })

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    vocab = Counter(all_tokens)
    print(f"Converted {len(json_files)} JSON files")
    print(f"Total sides (lines): {line_counter}")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Mean tokens per line: {len(all_tokens) / line_counter:.2f}")
    print(f"Most common signs: {vocab.most_common(10)}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Indus Valley corpus JSON to CSV")
    parser.add_argument("--input-dir", default="/tmp/indus-corpus/corpus", help="Directory containing JSON files")
    parser.add_argument("--output", default="data/raw/lost_language/corpus_indus_real.csv", help="Output CSV path")
    args = parser.parse_args()
    convert_indus_corpus(args.input_dir, args.output)


if __name__ == "__main__":
    main()
