"""Parse RR-corpus XML files into project CSV format.

Extracts tablet, side, line, position, glyph_code, and link from
phspaelti/RR-corpus XML transcription of Rongorongo tablets.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def parse_tablet(xml_path: str) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tablet_id = root.get("id", Path(xml_path).stem)
    rows = []
    for side in root.findall("side"):
        side_id = side.get("id", "")
        for line in side.findall("line"):
            line_id = line.get("id", "")
            for idx, glyph in enumerate(line.findall("glyph"), start=1):
                code_el = glyph.find("code")
                link_el = glyph.find("link")
                code = code_el.text if code_el is not None else ""
                link = link_el.text if link_el is not None else ""
                rows.append(
                    {
                        "doc_id": tablet_id,
                        "line_id": line_id,
                        "position": idx,
                        "token": code.strip() if code else "",
                        "raw_token": code.strip() if code else "",
                        "orientation": "normal",
                        "source": "RR-corpus_phspaelti",
                        "link": link.strip() if link else "",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Parse RR-corpus XML to CSV")
    parser.add_argument("--input-dir", default="data/external/rongorongo_rr_corpus")
    parser.add_argument(
        "--output", default="data/raw/lost_language/corpus_rongorongo_real.xml.csv"
    )
    args = parser.parse_args()

    all_rows = []
    xml_dir = Path(args.input_dir)
    for xml_file in sorted(xml_dir.glob("*.xml")):
        print(f"Parsing {xml_file.name} ...")
        rows = parse_tablet(str(xml_file))
        all_rows.extend(rows)
        print(f"  -> {len(rows)} glyphs")

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Stats
    n_tablets = df["doc_id"].nunique()
    n_lines = df["line_id"].nunique()
    n_tokens = len(df)
    vocab = df["token"].nunique()
    print(f"\nTotal: {n_tokens} glyphs")
    print(f"Tablets: {n_tablets}")
    print(f"Lines: {n_lines}")
    print(f"Vocabulary: {vocab}")
    print(f"Saved to {out_path}")

    # Save stats JSON
    import json

    stats = {
        "n_tablets": int(n_tablets),
        "n_lines": int(n_lines),
        "n_tokens": int(n_tokens),
        "vocab_size": int(vocab),
        "type_token_ratio": round(vocab / n_tokens, 6) if n_tokens > 0 else 0,
    }
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
