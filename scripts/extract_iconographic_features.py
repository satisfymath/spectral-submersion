"""Extract iconographic features from RR-corpus SVG glyphs.

Parses SVG <path> elements to compute:
- Path length (approximate stroke count)
- Bounding box aspect ratio (width/height)
- Path point count (complexity proxy)
- Orientation marker (type='b' vs 'f' for boustrophedon)
"""

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

RR_CORPUS_DIR = Path("data/external/rongorongo_rr_corpus")
OUTPUT_PATH = Path("data/processed/rongorongo_iconographic_features.csv")


def parse_svg_features(path_d):
    path_len = len(path_d)
    move_commands = len(re.findall(r"[Mm]", path_d))
    curve_commands = len(re.findall(r"[CcSsQqTtAa]", path_d))
    line_commands = len(re.findall(r"[LlHhVv]", path_d))
    close_commands = len(re.findall(r"[Zz]", path_d))
    all_numbers = re.findall(r"[-+]?\d*\.?\d+", path_d)
    coords = [float(x) for x in all_numbers]
    total_path_complexity = (
        move_commands + curve_commands + line_commands + close_commands
    )
    return {
        "path_length_chars": path_len,
        "move_count": move_commands,
        "curve_count": curve_commands,
        "line_count": line_commands,
        "close_count": close_commands,
        "total_complexity": total_path_complexity,
        "coord_count": len(all_numbers),
    }


def main():
    rows = []
    for xml_file in sorted(RR_CORPUS_DIR.glob("*.xml")):
        tablet = xml_file.stem
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for side in root.iter("side"):
            side_id = side.get("id", "")
            for line_elem in side.iter("line"):
                line_id = line_elem.get("id", "")
                line_num = line_elem.find("line-num")
                line_num_val = int(line_num.text) if line_num is not None else 0

                for glyph in line_elem.iter("glyph"):
                    glyph_id = glyph.get("id", "")
                    code_elem = glyph.find("code")
                    code = code_elem.text if code_elem is not None else ""

                    features = {
                        "tablet": tablet,
                        "side_id": side_id,
                        "line_id": line_id,
                        "glyph_id": glyph_id,
                        "code": code,
                    }

                    for image in glyph.iter("image"):
                        img_type = image.get("type", "")
                        path_elem = image.find("path")
                        x_elem = image.find("x")
                        y_elem = image.find("y")
                        w_elem = image.find("width")
                        h_elem = image.find("height")

                        path_d = path_elem.get("d", "") if path_elem is not None else ""
                        svg_feats = parse_svg_features(path_d)

                        x = float(x_elem.text) if x_elem is not None else 0
                        y = float(y_elem.text) if y_elem is not None else 0
                        w = float(w_elem.text) if w_elem is not None else 0
                        h = float(h_elem.text) if h_elem is not None else 0

                        aspect_ratio = w / h if h > 0 else 0

                        features[f"x_{img_type}"] = x
                        features[f"y_{img_type}"] = y
                        features[f"w_{img_type}"] = w
                        features[f"h_{img_type}"] = h
                        features[f"aspect_ratio_{img_type}"] = aspect_ratio
                        for k, v in svg_feats.items():
                            features[f"{k}_{img_type}"] = v

                    rows.append(features)

    import pandas as pd

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Extracted {len(df)} glyphs with iconographic features")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to {OUTPUT_PATH}")

    stats = (
        df.groupby("code")
        .agg(
            {
                "total_complexity_b": ["mean", "std"],
                "aspect_ratio_b": ["mean", "std"],
                "w_b": "mean",
                "h_b": "mean",
            }
        )
        .reset_index()
    )
    print(f"\nPer-glyph stats: {len(stats)} unique glyphs")


if __name__ == "__main__":
    main()
