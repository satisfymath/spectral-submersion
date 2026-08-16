"""Task 2.5: blind discrimination test - materials generator.

Produces a shuffled, blinded file of N real tablet lines + N v6 outputs
(matched in length distribution), an answer key (kept separate), and an
annotation sheet. Judges label each line real/synthetic; analysis script
reports per-judge AUC, inter-judge Fleiss kappa and bootstrap CIs.
Seed fixed: 42. Execution requires human judges (epigraphers or advanced
amateurs) and is registered as [PENDIENTE: ejecución humana].
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
from translate_to_rongorongo_v6 import RealGlyphLM, beam_translate, load_model  # noqa: E402
from evaluate_rongorongo_v6 import real_stats  # noqa: E402

SEED = 42
N = 20


def main():
    rng = random.Random(SEED)
    lines, _, _ = real_stats("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    # random windows of 6-16 glyphs from real lines (tablet lines are long,
    # ~50 glyphs; windows match the v6 output length distribution)
    long_lines = [seq for _, seq in lines if len(seq) >= 16]
    real_sample = []
    for _ in range(N):
        seq = rng.choice(long_lines)
        w = rng.randint(6, 16)
        start = rng.randint(0, len(seq) - w)
        real_sample.append(seq[start:start + w])

    model, sv, tv = load_model("models/rongorongo_translator_v6")
    lm = RealGlyphLM("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    # v6 outputs for a fresh set of gloss sentences (disjoint from tuning set)
    gloss = [
        "te ariki noho i te hare nui", "e kai te tangata i te ika",
        "ko Tane rere ki te rangi", "te vahine tangi i te po",
        "he manu noho i te rakau", "e inu au i te vai o te ana",
        "ko Rongo korero ki te matua", "te tamaiti oma ki te tai",
        "he ua topa ki te maunga", "e moe te ariki i te ana",
        "te matangi huri i te moana", "ko Hina whiti i te po",
        "he tangata amo i te vaka", "te mokopuna kai i te hua",
        "e haere te taua ki te motu", "ko Makemake tuku i te ora",
        "te ika nui peke i te tai", "he vahine here i te kahu",
        "e tangi te manu i te ao", "ko Tiki tango i te toki",
    ]
    synth = [beam_translate(model, sv, tv, lm, s, beam=5, lm_weight=0.35).split()
             for s in gloss]

    items = [{"line": " ".join(s), "label": "real"} for s in real_sample] + \
            [{"line": " ".join(s), "label": "synthetic"} for s in synth]
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item_id"] = f"L{i:02d}"

    outdir = Path("reports/blind_judge_test")
    outdir.mkdir(parents=True, exist_ok=True)
    # Blinded sheet (no labels)
    pd.DataFrame([{"item_id": it["item_id"], "sequence": it["line"],
                   "judgment (real/synthetic)": "", "confidence (1-5)": ""}
                  for it in items]).to_csv(outdir / "annotation_sheet.csv", index=False)
    # Answer key, separate file
    (outdir / "ANSWER_KEY_DO_NOT_OPEN.json").write_text(
        json.dumps({it["item_id"]: it["label"] for it in items}, indent=2))
    # Protocol
    (outdir / "PROTOCOL.md").write_text(f"""# Blind discrimination test - protocol

- Items: {N} real lines (tablets A-F, length 6-16) + {N} v6 beam+fusion
  outputs from held-out gloss sentences, shuffled (seed {SEED}).
- Judges: k >= 3, epigraphers or advanced amateurs familiar with Barthel
  transcriptions. Each judges all {2*N} items independently.
- Task: label each sequence real / synthetic + confidence 1-5.
- Analysis (script `analyze_blind_judges.py` [CALCULAR al recibir hojas]):
  per-judge ROC-AUC using confidence as score; Fleiss kappa for
  inter-judge agreement; bootstrap 95% CI (B=2000) on pooled AUC.
- Interpretation guard (C1/C2/C3): AUC ~= 0.5 means judges cannot
  discriminate v6 output from real lines at the sequence-statistics
  level. It does NOT mean the output is a valid text - only that this
  discrimination channel carries no signal (claim level C1 about the
  generator, no semantic claim).
- Status: PENDIENTE - requires human judges.
""")
    print(f"Wrote {outdir}/annotation_sheet.csv ({2*N} items), answer key, protocol")


if __name__ == "__main__":
    main()
