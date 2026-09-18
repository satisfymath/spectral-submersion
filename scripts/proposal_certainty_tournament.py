"""Tournament: which Rapanui phrasing of the marriage proposal yields the
Rongorongo glyph sequence we can be MOST certain about?

Certainty here is *not* semantic (Rongorongo is undeciphered). It is:
  1. agreement   — do the three independently trained v6 models emit the same sequence?
  2. stability   — does the sequence survive changes to the decoder (beam, LM weight)?
  3. margin      — how far ahead is top-1 over top-2 in the primary model's beam?
  4. realism     — bits/glyph under the trigram LM of the real tablets, vs real lines
  5. coverage    — fraction of adjacent glyph pairs attested in the real corpus

Usage:
    PYTHONPATH=src python scripts/proposal_certainty_tournament.py
"""
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from translate_to_rongorongo_v6 import RealGlyphLM, load_model  # noqa: E402

REAL = "data/raw/lost_language/corpus_rongorongo_real.xml.csv"
MODELS = {
    "v6": "models/rongorongo_translator_v6",
    "v6_noaug": "models/rongorongo_translator_v6_noaug",
    "v6_noclass": "models/rongorongo_translator_v6_noclass",
}

# Every token must be in the 232-type source vocab (no <unk>).
CANDIDATES = [
    ("ka moe taua", "casémonos (nosotros dos)"),
    ("ka moe taua mo te ora tonu", "casémonos para toda la vida"),
    ("e vahine kou mo au", "tú, esposa para mí"),
    ("e tane au mo kou", "yo, esposo para ti"),
    ("e vahine kou mo au e tane au mo kou", "tú esposa para mí, yo esposo para ti"),
    ("e vahine kou mo au mo te ora tonu", "tú esposa para mí, para toda la vida"),
    ("ko kou taku vahine mo te ora tonu", "tú eres mi esposa para toda la vida"),
    ("makemake au ki a kou mo te vahine", "te quiero / te elijo como esposa"),
    ("makemake au ki a kou e vahine mo au", "te elijo, esposa para mí"),
    ("e aroha au ki a kou ka moe taua", "te amo, casémonos"),
    ("e aroha taua ka moe taua mo te ora tonu", "nos amamos, casémonos para toda la vida"),
    ("e vahine kou mo au e tane au mo kou ka moe taua mo te ora tonu",
     "tú esposa para mí, yo esposo para ti, casémonos para toda la vida"),
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def beam_nbest(model, src_vocab, tgt_vocab, lm, text, beam=5, lm_weight=0.35,
               max_len=40, rep_penalty=1.5):
    """Same scoring as translate_to_rongorongo_v6.beam_translate, but returns
    the finished hypotheses sorted by length-normalised score."""
    tokens = text.strip().lower().split()
    unk = [t for t in tokens if t not in src_vocab.stoi] if hasattr(src_vocab, "stoi") else []
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_emb = model.pos_enc(model.src_emb(src) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb)
    itos = tgt_vocab.itos

    def model_logprobs(prefix_ids):
        ys = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
        tgt_emb = model.pos_enc(model.tgt_emb(ys) * math.sqrt(model.d_model))
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
        out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return torch.log_softmax(model.out_proj(out[:, -1, :])[0], dim=-1)

    beams = [([tgt_vocab.bos_idx], 0.0)]
    done = []
    for _ in range(max_len):
        candidates = []
        for prefix, score in beams:
            if prefix[-1] == tgt_vocab.eos_idx:
                done.append((prefix, score))
                continue
            lp = model_logprobs(prefix)
            topk = torch.topk(lp, min(beam * 2, len(itos)))
            gp = [itos[i] for i in prefix[1:]]
            prev1 = gp[-1] if len(gp) >= 1 else "<s>"
            prev2 = gp[-2] if len(gp) >= 2 else "<s>"
            for logp_m, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                if idx in (tgt_vocab.bos_idx, tgt_vocab.pad_idx):
                    continue
                w = "</s>" if idx == tgt_vocab.eos_idx else itos[idx]
                s = score + (1 - lm_weight) * logp_m + lm_weight * lm.logp(w, prev2, prev1)
                if len(gp) >= 2 and idx != tgt_vocab.eos_idx and gp[-1] == gp[-2] == itos[idx]:
                    s -= rep_penalty
                candidates.append((prefix + [idx], s))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[1] / len(t[0]), reverse=True)
        beams = candidates[:beam]
        if all(p[-1] == tgt_vocab.eos_idx for p, _ in beams):
            done.extend(beams)
            break
    done.extend(b for b in beams if b[0][-1] == tgt_vocab.eos_idx)
    if not done:
        done = beams
    specials = (tgt_vocab.bos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx)
    seen, nbest = set(), []
    for prefix, score in sorted(done, key=lambda t: t[1] / len(t[0]), reverse=True):
        seq = " ".join(itos[i] for i in prefix if i not in specials)
        if seq in seen:
            continue
        seen.add(seq)
        nbest.append((seq, score / len(prefix)))
    return nbest, unk


CANDIDATES_R2 = [
    ("makemake au ki a kou mo te ora tonu", "te elijo para toda la vida"),
    ("makemake au ki a kou e vahine mo te ora tonu", "te elijo, esposa, para toda la vida"),
    ("makemake au ki a kou taku vahine mo te ora tonu", "te elijo, mi esposa, para toda la vida"),
    ("makemake au ki a kou ka moe taua mo te ora tonu", "te elijo, casémonos para toda la vida"),
    ("makemake au ki te vahine mo te ora tonu", "quiero a la esposa para toda la vida"),
    ("e aroha au ki a kou mo te ora tonu", "te amo para toda la vida"),
    ("ko kou taku vahine mo te ora tonu", "tú eres mi esposa para toda la vida"),
    ("ko kou taku vahine e aroha au ki a kou mo te ora tonu", "tú eres mi esposa, te amo para toda la vida"),
    ("ka moe taua mo te ora tonu", "casémonos para toda la vida"),
    ("makemake au ki a kou e aroha au ki a kou mo te ora tonu", "te elijo, te amo, para toda la vida"),
]

# formulas already carved in the poem tablet (tab:poema) that both v6 models reproduce
POEM_FORMULAS = ["041h 670 580", "670 580", "430 022", "002 002", "004 004"]
PERSON_WORDS = {"vahine", "tane", "wahine", "tangata"}


def barthel_base(tok):
    digits = "".join(ch for ch in tok if ch.isdigit())
    return int(digits[:3]) if digits else None


def person_class_ok(text, seq):
    """Every person noun in the text should have at least one anthropomorph (200-399) in the output."""
    n_person = sum(1 for w in text.split() if w in PERSON_WORDS)
    n_anthro = sum(1 for t in seq.split() if (b := barthel_base(t)) is not None and 200 <= b <= 399)
    return 1.0 if n_person == 0 else min(1.0, n_anthro / n_person)


def formula_hits(seq):
    return sum(1 for f in POEM_FORMULAS if f in seq)


def levenshtein_sim(a, b):
    a, b = a.split(), b.split()
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return 1 - d[len(a)][len(b)] / max(len(a), len(b), 1)


def real_line_bits(lm, corpus_csv):
    df = pd.read_csv(corpus_csv)
    bits = []
    for _, line_df in df.groupby(["doc_id", "line_id"], sort=False):
        seq = [t for t in line_df.sort_values("position")["token"].astype(str) if t != "_"]
        if len(seq) >= 4:
            bits.append(lm.bits_per_glyph(seq))
    return bits


def bigram_coverage(lm, seq):
    toks = seq.split()
    pairs = list(zip(toks, toks[1:]))
    if not pairs:
        return 1.0
    return sum(1 for a, b in pairs if lm.bi.get(a, {}).get(b, 0) > 0) / len(pairs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    cands = CANDIDATES if args.round == 1 else CANDIDATES_R2
    lm = RealGlyphLM(REAL)
    real_bits = real_line_bits(lm, REAL)
    mu, sd = statistics.mean(real_bits), statistics.pstdev(real_bits)
    print(f"Real tablet lines: {len(real_bits)}, bits/glyph mean {mu:.2f} ± {sd:.2f} "
          f"(min {min(real_bits):.2f}, max {max(real_bits):.2f})\n")

    models = {}
    for name, path in MODELS.items():
        m, sv, tv = load_model(path)
        models[name] = (m.to(DEVICE), sv, tv)
    print(f"device {DEVICE}\n")
    settings = [(5, 0.35), (10, 0.35), (5, 0.25), (5, 0.45), (10, 0.45)]

    results = []
    for text, gloss in cands:
        row = {"text": text, "gloss": gloss}
        # 1. primary decode per model (beam 5, lm 0.35)
        top1 = {}
        for name, (model, sv, tv) in models.items():
            nbest, unk = beam_nbest(model, sv, tv, lm, text)
            if unk:
                print(f"WARNING {text!r}: unknown tokens {unk}")
            top1[name] = nbest[0][0]
            if name == "v6":
                row["nbest_v6"] = nbest[:3]
                row["margin"] = (nbest[0][1] - nbest[1][1]) if len(nbest) > 1 else float("inf")
        row["seq"] = top1["v6"]
        agree = 1 + (1 if top1["v6_noaug"] == row["seq"] else 0)
        row["agreement"] = agree  # out of 2 comparable models
        row["pair_sim"] = levenshtein_sim(row["seq"], top1["v6_noaug"])
        bag_a, bag_b = Counter(row["seq"].split()), Counter(top1["v6_noaug"].split())
        row["xmodel_support"] = sum((bag_a & bag_b).values()) / max(1, sum(bag_a.values()))
        row["person_ok"] = person_class_ok(text, row["seq"])
        row["formula_hits"] = formula_hits(row["seq"])
        row["top1_by_model"] = top1
        # 2. decoder stability on the primary model
        model, sv, tv = models["v6"]
        variants = [beam_nbest(model, sv, tv, lm, text, beam=b, lm_weight=w)[0][0][0]
                    for b, w in settings]
        row["stability"] = sum(1 for v in variants if v == row["seq"]) / len(variants)
        row["stability_sim"] = statistics.mean(levenshtein_sim(v, row["seq"]) for v in variants)
        # 3. realism + coverage
        row["bits"] = lm.bits_per_glyph(row["seq"].split())
        row["z_real"] = (row["bits"] - mu) / sd
        row["coverage"] = bigram_coverage(lm, row["seq"])
        row["n_glyphs"] = len(row["seq"].split())
        # composite: agreement and stability dominate; realism as tie-breaker
        row["certainty"] = (
            0.20 * (row["agreement"] - 1)          # exact v6 == v6_noaug
            + 0.15 * row["xmodel_support"]         # glyphs shared across the two runs
            + 0.20 * row["stability"]              # survives decoder changes
            + 0.10 * row["stability_sim"]
            + 0.15 * row["person_ok"]              # person nouns land on human figures (Barthel 200-399)
            + 0.10 * min(1.0, row["formula_hits"] / 3)  # reuses formulas carved in the poem tablet
            + 0.05 * row["coverage"]
            + 0.05 * max(0.0, 1 - abs(row["z_real"]) / 2)
        )
        results.append(row)
        print(f"{text}\n  → {row['seq']}\n  agree {agree}/2  xsupport {row['xmodel_support']:.2f}  "
              f"person {row['person_ok']:.1f}  formulas {row['formula_hits']}  "
              f"stability {row['stability']:.2f}  margin {row['margin']:.3f}  "
              f"bits {row['bits']:.2f} (z {row['z_real']:+.2f})  cov {row['coverage']:.2f}  "
              f"→ certainty {row['certainty']:.3f}")
        for name, s in top1.items():
            if s != row["seq"]:
                print(f"     {name}: {s}")
        print()

    results.sort(key=lambda r: r["certainty"], reverse=True)
    print("=" * 78)
    print("RANKING")
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['certainty']:.3f}  agree {r['agreement']}/2  xsup {r['xmodel_support']:.2f}  "
              f"stab {r['stability']:.2f}  person {r['person_ok']:.1f}  form {r['formula_hits']}  "
              f"bits {r['bits']:.2f}  | {r['text']}  → {r['seq']}")
    out = Path(f"reports/proposal_certainty_tournament_r{args.round}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"real_bits_mean": mu, "real_bits_sd": sd, "results": results},
                              indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
