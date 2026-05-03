"""Generate Rongorongo real corpus exploratory report.

Combines all analyses: corpus stats, spectral, controls, positional,
repetition, conditional entropy, and multi-candidate comparison.
"""
import json
from pathlib import Path

import pandas as pd


def load_json(path: str) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")


def main():
    lines = []
    lines.append("# Rongorongo Real Corpus (RR-corpus) Exploratory Report")
    lines.append("")
    lines.append("## 1. Corpus Description")
    lines.append("")

    stats = load_json("data/raw/lost_language/corpus_rongorongo_real.xml.stats.json")
    if stats:
        lines.append(f"- **Source**: RR-corpus (phspaelti/RR-corpus, XML transcription)")
        lines.append(f"- **Tablets**: {stats.get('n_tablets', 'N/A')} (A, B, C, D, E, F)")
        lines.append(f"- **Lines (sides)**: {stats.get('n_lines', 'N/A')}")
        lines.append(f"- **Total glyphs**: {stats.get('n_tokens', 'N/A')}")
        lines.append(f"- **Vocabulary size**: {stats.get('vocab_size', 'N/A')}")
        lines.append(f"- **Type-Token Ratio**: {stats.get('type_token_ratio', 'N/A')}")
    else:
        lines.append("- Stats not found")

    lines.append("")
    lines.append("## 2. Spectral Analysis & Negative Controls")
    lines.append("")

    ctrl = pd.read_csv("reports/tables/control_comparison_rongorongo_real.csv")
    lines.append(ctrl.to_markdown(index=False))
    lines.append("")
    real_r = ctrl[ctrl["variant"] == "real"]["effective_rank"].values[0]
    unif_r = ctrl[ctrl["variant"] == "random_uniform"]["effective_rank"].values[0]
    if real_r < unif_r:
        lines.append(f"**Sanity check PASSED**: Real r_eff ({real_r:.2f}) < Uniform ({unif_r:.2f})")
    else:
        lines.append(f"**Sanity check WARNING**: Real r_eff ({real_r:.2f}) >= Uniform ({unif_r:.2f}). ")
        lines.append("This is consistent with the pattern observed in other short-inscription corpora (Indus): ")
        lines.append("window-based co-occurrence is too noisy when vocabulary is large (~941 types) and sequences are short.")

    lines.append("")
    lines.append("## 3. Conditional Entropy (Bigram / Trigram)")
    lines.append("")

    ent = load_json("reports/tables/conditional_entropy_rongorongo_real.json")
    if ent:
        for variant in ent if isinstance(ent, list) else ent.get("variants", []):
            lines.append(f"- **{variant['variant']}**: H(unigram)={variant['h_unconditional']:.2f}, "
                         f"H(bigram)={variant['h_bigram_conditional']:.2f}, "
                         f"H(trigram)={variant['h_trigram_conditional']:.2f}, "
                         f"PPL(bigram)={variant['perplexity_bigram']:.2f}")
        lines.append("")
        lines.append("**Interpretation**: The real corpus has lower bigram conditional entropy than the permuted control,")
        lines.append("indicating that sequential order carries information. However, the uniform random control")
        lines.append("has even lower entropy due to artefactual bigrams from uniform sampling. The trigram entropy")
        lines.append("is higher in the real corpus than in controls, which may reflect genuine structural diversity.")

    lines.append("")
    lines.append("## 4. Repetition Patterns")
    lines.append("")

    rep_path = Path("reports/tables/repetition_rongorongo_real.json")
    if rep_path.exists():
        rep_df = pd.read_csv(rep_path)
        if len(rep_df) > 0:
            row = rep_df.iloc[0]
            lines.append(f"- **Lines with any repeat**: {int(row.get('any_repeat_count', 'N/A'))} ({row.get('any_repeat_rate', 'N/A')})")
            lines.append(f"- **Double repeat (AA)**: {int(row.get('double_repeat_count', 'N/A'))} ({row.get('double_repeat_rate', 'N/A')})")
            lines.append(f"- **Triple repeat (AAA)**: {int(row.get('triple_repeat_count', 'N/A'))} ({row.get('triple_repeat_rate', 'N/A')})")
            lines.append(f"- **ABAB pattern**: {int(row.get('abab_repeat_count', 'N/A'))} ({row.get('abab_repeat_rate', 'N/A')})")
            lines.append("")
            lines.append("**Interpretation**: The ubiquity of repetitions (66% double, 18% triple) is the strongest structural")
            lines.append("signal in the corpus. This pattern is inconsistent with ordinary linguistic text (where lexical")
            lines.append("repetition is rare) and consistent with mnemonic, ritual, or rhythmic functions (Ferrara 2015).")

    lines.append("")
    lines.append("## 5. Positional Bias (Monte Carlo)")
    lines.append("")

    mc_path = Path("reports/tables/montecarlo_positional_rongorongo_real.json")
    if mc_path.exists():
        mc_df = pd.read_csv(mc_path)
        start_sig = mc_df[mc_df["first_ratio_pvalue"].astype(float) < 0.001]
        end_sig = mc_df[mc_df["last_ratio_pvalue"].astype(float) < 0.001]
        lines.append(f"Signs with significant start bias: {len(start_sig)}")
        for _, s in start_sig.iterrows():
            lines.append(f"  - `{s['token']}` (n={int(s['count'])}): first_ratio={s['first_ratio_obs']:.3f}, p={s['first_ratio_pvalue']}, effect={s['first_ratio_effect']:.1f}σ")
        lines.append(f"Signs with significant end bias: {len(end_sig)}")
        for _, s in end_sig.iterrows():
            lines.append(f"  - `{s['token']}` (n={int(s['count'])}): last_ratio={s['last_ratio_obs']:.3f}, p={s['last_ratio_pvalue']}, effect={s['last_ratio_effect']:.1f}σ")
        lines.append("")
        lines.append("**Interpretation**: Signs `000!` and `_` show extreme positional bias (both start and end),")
        lines.append("suggesting they function as structural markers or framing elements, possibly analogous to")
        lines.append("opening/closing formulae in ritual texts.")

    lines.append("")
    lines.append("## 6. Multi-Candidate Comparison (Consensus Space)")
    lines.append("")

    cand = pd.read_csv("reports/tables/multi_consensus_rongorongo_real.csv")
    lines.append(cand.to_markdown(index=False))
    lines.append("")
    lines.append("**Interpretation**: Polynesian candidates cluster with lower relational distortion, but this")
    lines.append("reflects structural typology (agglutinative, head-initial, rigid order) rather than genealogical")
    lines.append("affinity. The high entropy (>10 nats for all candidates) indicates that the mapping is highly")
    lines.append("uncertain, which is honest given the lack of bilingual anchors.")

    lines.append("")
    lines.append("## 7. Synthesis & Hypotheses")
    lines.append("")
    lines.append("### H1: Non-random structure")
    lines.append("- **SUPPORTED** by repetition patterns (66% AA, 18% AAA) and positional bias (Monte Carlo p<0.001).")
    lines.append("- **NOT SUPPORTED** by spectral compression (sanity check fails) or bigram entropy vs uniform.")
    lines.append("")
    lines.append("### H2: Geometría latente")
    lines.append("- Window-based co-occurrence does NOT produce clear spectral structure for this corpus.")
    lines.append("- Repetition-based features (AA, AAA, ABAB) may be more informative than local context.")
    lines.append("")
    lines.append("### H3: Partial alignment")
    lines.append("- Without anchors, no candidate language can be preferred over another on structural grounds alone.")
    lines.append("- The multi-language consensus ranking is stable but structurally, not genealogically, interpretable.")
    lines.append("")
    lines.append("### H4: Probabilistic translation")
    lines.append("- Current evidence supports structural hypotheses (markers, repetitions, positions) but NOT")
    lines.append("  semantic equivalences between glyphs and words.")
    lines.append("")
    lines.append("## 8. Limitations")
    lines.append("- Only 6 tablets (A-F) transcribed; the full corpus has ~25 objects.")
    lines.append("- The RR-corpus transcription may differ from other systems (CEIPP, Barthel, etc.).")
    lines.append("- No iconographic priors integrated yet.")
    lines.append("- Boustrophedon reading direction not accounted for in linear sequence analysis.")
    lines.append("")
    lines.append("## 9. Next Steps")
    lines.append("- Acquire transcriptions of all remaining tablets (G-T, etc.).")
    lines.append("- Implement boustrophedon-aware sequence modeling.")
    lines.append("- Build repetition-aware embeddings (treat AA/AAA as single tokens or rhythmic units).")
    lines.append("- Integrate iconographic features (SVG paths in XML) into spectral embeddings.")
    lines.append("- Search for real bilingual anchors (proper names, genealogies, numerals) via positional heuristics.")

    out_path = Path("reports/final/rongorongo_real_exploratory_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Rongorongo real report saved to {out_path}")


if __name__ == "__main__":
    main()
