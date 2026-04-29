"""Search for bilingual anchors in Rongorongo via positional heuristics.

We know that signs `000!` and `_` have extreme positional bias (6-8 sigma):
  - `_` appears at start AND end of lines (framing marker)
  - `000!` appears mostly at END of lines (closing marker)

Strategy: Use these structural markers as candidate bilingual anchors.
If RR encodes a known language, markers at line boundaries should correspond
to sentence-level delimiters (punctuation, discourse markers, formulaic openings).

For each candidate language, we:
1. Identify their positional markers (tokens with extreme start/end bias)
2. Compute the positional-bias profile: for each token, its start-ratio and end-ratio
3. Correlate the positional profiles between RR and each candidate
4. Also search for repetitive pattern anchors (AA, AAA patterns in RR vs
   repeated function words in candidate languages)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_submersion.tokenization import read_corpus, get_sequences_by_line

RR_PATH = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"

CANDIDATES = {
    "maori":      "data/raw/candidate_languages/maori_tokens.csv",
    "tahitian":    "data/raw/candidate_languages/tahitian_tokens.csv",
    "hawaiian":    "data/raw/candidate_languages/haw_tokens.csv",
    "samoan":      "data/raw/candidate_languages/sm_tokens.csv",
    "tongan":      "data/raw/candidate_languages/to_tokens.csv",
    "fijian":      "data/raw/candidate_languages/fj_tokens.csv",
    "rapa_nui":   "data/raw/candidate_languages/rap_tokens.csv",
    "english":     "data/raw/candidate_languages/english_tokens.csv",
    "spanish":     "data/raw/candidate_languages/spanish_tokens.csv",
}

FAMILIES = {
    "maori": "polynesian", "tahitian": "polynesian", "hawaiian": "polynesian",
    "samoan": "polynesian", "tongan": "polynesian", "fijian": "austronesian",
    "rapa_nui": "polynesian", "english": "germanic", "spanish": "romance",
}


def compute_positional_profile(sequences, min_count=3):
    """Compute positional bias profile for each token.
    
    Returns dict: token -> {count, first_ratio, last_ratio, start_sigma, end_sigma}
    """
    token_starts = Counter()
    token_ends = Counter()
    token_counts = Counter()
    n_lines = len(sequences)

    for seq in sequences:
        if len(seq) == 0:
            continue
        token_counts[seq[0]] += 1
        token_starts[seq[0]] += 1
        token_ends[seq[-1]] += 1
        token_counts[seq[-1]] += 1
        for tok in seq:
            token_counts[tok] += 1

    first_ratio = {}
    last_ratio = {}
    for tok, cnt in token_counts.items():
        if cnt < min_count:
            continue
        first_ratio[tok] = token_starts.get(tok, 0) / cnt
        last_ratio[tok] = token_ends.get(tok, 0) / cnt

    # Monte Carlo: shuffle positions within each line
    rng = np.random.default_rng(42)
    n_mc = 1000
    mc_first = defaultdict(list)
    mc_last = defaultdict(list)

    for _ in range(n_mc):
        shuffled_starts = Counter()
        shuffled_ends = Counter()
        for seq in sequences:
            if len(seq) <= 1:
                continue
            idx = np.arange(len(seq))
            rng.shuffle(idx)
            shuffled_starts[seq[idx[0]]] += 1
            shuffled_ends[seq[idx[-1]]] += 1

        for tok in token_counts:
            if token_counts[tok] < min_count:
                continue
            mc_first[tok].append(shuffled_starts.get(tok, 0) / token_counts[tok])
            mc_last[tok].append(shuffled_ends.get(tok, 0) / token_counts[tok])

    profile = {}
    for tok in sorted(token_counts.keys()):
        if token_counts[tok] < min_count:
            continue
        obs_first = first_ratio.get(tok, 0)
        obs_last = last_ratio.get(tok, 0)
        mc_f = np.array(mc_first.get(tok, [0]))
        mc_l = np.array(mc_last.get(tok, [0]))
        start_sigma = (obs_first - np.mean(mc_f)) / (np.std(mc_f) + 1e-10)
        end_sigma = (obs_last - np.mean(mc_l)) / (np.std(mc_l) + 1e-10)
        profile[tok] = {
            "count": token_counts[tok],
            "first_ratio": round(obs_first, 4),
            "last_ratio": round(obs_last, 4),
            "start_sigma": round(start_sigma, 2),
            "end_sigma": round(end_sigma, 2),
            "first_count": token_starts.get(tok, 0),
            "last_count": token_ends.get(tok, 0),
        }
    return profile


def profile_vector(profile, top_n=50):
    """Convert positional profile to a vector of (start_sigma, end_sigma) for top tokens."""
    sorted_toks = sorted(profile.keys(), key=lambda t: profile[t]["count"], reverse=True)[:top_n]
    vec = []
    for tok in sorted_toks:
        vec.extend([profile[tok]["start_sigma"], profile[tok]["end_sigma"]])
    return np.array(vec)


def main():
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rongorongo profile
    print("Computing Rongorongo positional profile...", flush=True)
    rr_df = read_corpus(RR_PATH)
    rr_seqs = get_sequences_by_line(rr_df)
    rr_profile = compute_positional_profile(rr_seqs, min_count=3)

    # Print top structural markers
    top_start = sorted(rr_profile.items(), key=lambda x: x[1]["start_sigma"], reverse=True)[:10]
    top_end = sorted(rr_profile.items(), key=lambda x: x[1]["end_sigma"], reverse=True)[:10]

    print("\nTop START-biased tokens in Rongorongo:")
    for tok, info in top_start:
        print(f"  {tok:10s}: count={info['count']:4d}, first_ratio={info['first_ratio']:.3f}, "
              f"start_sigma={info['start_sigma']:.2f}")

    print("\nTop END-biased tokens in Rongorongo:")
    for tok, info in top_end:
        print(f"  {tok:10s}: count={info['count']:4d}, last_ratio={info['last_ratio']:.3f}, "
              f"end_sigma={info['end_sigma']:.2f}")

    # 2. Candidate language profiles
    print("\n--- Computing candidate language profiles ---\n", flush=True)
    cand_profiles = {}
    cand_corr = {}
    rr_vec = profile_vector(rr_profile, top_n=30)

    results = []
    for name, path in CANDIDATES.items():
        print(f"Processing {name}...", flush=True)
        try:
            df = pd.read_csv(path)
            if "token" not in df.columns:
                print(f"  Skipping {name}: no 'token' column")
                continue
            df["token"] = df["token"].astype(str).str.lower().str.strip()
            seqs = get_sequences_by_line(df)
            if len(seqs) < 10:
                print(f"  Skipping {name}: only {len(seqs)} lines")
                continue
            profile = compute_positional_profile(seqs, min_count=3)
            cand_profiles[name] = profile

            # Correlation of positional bias profiles
            cand_vec = profile_vector(profile, top_n=30)
            min_len = min(len(rr_vec), len(cand_vec))
            if min_len > 10:
                corr_start = np.corrcoef(rr_vec[:min_len:2], cand_vec[:min_len:2])[0, 1]
                corr_end = np.corrcoef(rr_vec[1:min_len:2], cand_vec[1:min_len:2])[0, 1]
                corr_combined = np.corrcoef(rr_vec[:min_len], cand_vec[:min_len])[0, 1]
            else:
                corr_start = corr_end = corr_combined = 0.0

            cand_corr[name] = {
                "corr_start": float(corr_start),
                "corr_end": float(corr_end),
                "corr_combined": float(corr_combined),
            }

            # Find top structural markers in this language
            top_start_cand = sorted(profile.items(), key=lambda x: x[1]["start_sigma"], reverse=True)[:5]
            top_end_cand = sorted(profile.items(), key=lambda x: x[1]["end_sigma"], reverse=True)[:5]

            family = FAMILIES.get(name, "unknown")
            results.append({
                "language": name,
                "family": family,
                "n_lines": len(seqs),
                "n_types": len(profile),
                "corr_start": round(corr_start, 4),
                "corr_end": round(corr_end, 4),
                "corr_combined": round(corr_combined, 4),
                "top_start_markers": [(t, round(info["start_sigma"], 2)) for t, info in top_start_cand],
                "top_end_markers": [(t, round(info["end_sigma"], 2)) for t, info in top_end_cand],
            })

            top_start_str = ", ".join(f"{t}({s:.1f}σ)" for t, s in results[-1]["top_start_markers"])
            top_end_str = ", ".join(f"{t}({s:.1f}σ)" for t, s in results[-1]["top_end_markers"])
            print(f"  {name}: {len(seqs)} lines, corr_combined={corr_combined:.4f}")
            print(f"    Start markers: {top_start_str}")
            print(f"    End markers: {top_end_str}")

        except Exception as e:
            print(f"  Error with {name}: {e}")
            continue

    # 3. Repetition pattern matching
    print("\n--- Repetition pattern search ---\n", flush=True)
    rr_repeats = Counter()
    for seq in rr_seqs:
        i = 0
        while i < len(seq):
            j = i + 1
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            if j - i >= 2:
                rr_repeats[seq[i]] += 1
            i = j

    print(f"Tokens appearing in AA+ patterns in RR (top 15):")
    for tok, cnt in rr_repeats.most_common(15):
        info = rr_profile.get(tok, {})
        count = info.get("count", "?")
        print(f"  {tok:10s}: {cnt:4d} repeat contexts, total count={count}")

    # 4. Structural marker alignment search
    # RR markers: `_` (start/end), `000!` (end), `040` (start)
    # Hypothesis: if RR encodes a Polynesian language, line markers should map to
    # discourse markers, articles, or sentence boundaries in that language.
    # Search: do candidate languages have similar positional profile correlations?
    print("\n--- Structural marker alignment ---\n", flush=True)
    rr_markers = {"_": "start+end delimiter", "000!": "end marker", "040": "start marker"}
    print(f"RR structural markers: {rr_markers}")

    # For each candidate, find tokens with similar sigma profiles to RR markers
    marker_matches = []
    for name in cand_profiles:
        profile = cand_profiles[name]
        family = FAMILIES.get(name, "unknown")
        # Find tokens in candidate with start_sigma > 3 AND end_sigma > 3 (like `_`)
        both_markers = [(t, info) for t, info in profile.items()
                       if info["start_sigma"] > 3 and info["end_sigma"] > 3]
        # Find tokens with high end_sigma (like `000!`)
        end_markers = [(t, info) for t, info in profile.items()
                      if info["end_sigma"] > 3 and info["start_sigma"] < 1]
        marker_matches.append({
            "language": name,
            "family": family,
            "both_markers": both_markers[:5],
            "end_markers": end_markers[:5],
        })

    print("\nCandidate languages with matching structural markers:")
    for m in marker_matches:
        print(f"\n  {m['language']} ({m['family']}):")
        if m["both_markers"]:
            print(f"    BOTH start+end markers (like RR `_`): ")
            for tok, info in m["both_markers"]:
                print(f"      '{tok}': count={info['count']}, "
                      f"start={info['start_sigma']:.1f}σ, end={info['end_sigma']:.1f}σ")
        else:
            print(f"    No strong start+end markers found")
        if m["end_markers"]:
            print(f"    END-only markers (like RR `000!`): ")
            for tok, info in m["end_markers"][:3]:
                print(f"      '{tok}': count={info['count']}, "
                      f"start={info['start_sigma']:.1f}σ, end={info['end_sigma']:.1f}σ")
        else:
            print(f"    No strong end-only markers found")

    # 5. Summary table
    print("\n" + "=" * 80)
    print("ANCHOR SEARCH SUMMARY")
    print("=" * 80)
    print(f"\n{'Language':15s} {'Family':15s} {'Lines':>6s} {'corr_start':>10s} "
          f"{'corr_end':>10s} {'corr_comb':>10s} {'Both_mrk':>9s} {'End_mrk':>8s}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["corr_combined"], reverse=True):
        mm = next((m for m in marker_matches if m["language"] == r["language"]), None)
        n_both = len(mm["both_markers"]) if mm else 0
        n_end = len(mm["end_markers"]) if mm else 0
        print(f"{r['language']:15s} {r['family']:15s} {r['n_lines']:6d} "
              f"{r['corr_start']:10.4f} {r['corr_end']:10.4f} {r['corr_combined']:10.4f} "
              f"{n_both:9d} {n_end:8d}")

    # Save results
    results_save = []
    for r in results:
        rs = dict(r)
        rs["top_start_markers"] = json.dumps(rs["top_start_markers"])
        rs["top_end_markers"] = json.dumps(rs["top_end_markers"])
        results_save.append(rs)
    df_results = pd.DataFrame(results_save)
    df_results.to_csv(out_dir / "bilingual_anchor_search.csv", index=False)

    # Save RR profile
    with open(out_dir / "rr_positional_profile.json", "w") as f:
        json.dump(rr_profile, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/bilingual_anchor_search.csv")


if __name__ == "__main__":
    main()