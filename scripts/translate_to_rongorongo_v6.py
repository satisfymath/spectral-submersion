"""Translate to Rongorongo (v6): beam search + shallow fusion with a trigram
LM trained on the REAL Rongorongo corpus (tablets A-F).

score(y) = (1 - lm_weight) * log p_model(y|x) + lm_weight * log p_LM_real(y)

plus a repetition penalty, with length normalization. This pushes outputs
toward sequences that are statistically plausible under the real tablets.

Usage:
    python translate_to_rongorongo_v6.py --text "e aroha au ki a kou"
    python translate_to_rongorongo_v6.py --interactive
"""
import argparse
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

from spectral_submersion.rongorongo_translator import Vocab, TransformerTranslator  # noqa: F401


# ---------------- Real-corpus trigram LM (interpolated, add-k) ----------------
class RealGlyphLM:
    def __init__(self, corpus_csv, k=0.1, lambdas=(0.5, 0.3, 0.2)):
        df = pd.read_csv(corpus_csv)
        self.uni, self.bi, self.tri = Counter(), defaultdict(Counter), defaultdict(Counter)
        for _, line_df in df.groupby(["doc_id", "line_id"], sort=False):
            seq = [t for t in line_df.sort_values("position")["token"].astype(str) if t != "_"]
            seq = ["<s>", "<s>"] + seq + ["</s>"]
            self.uni.update(seq)
            for a, b in zip(seq, seq[1:]):
                self.bi[a][b] += 1
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                self.tri[(a, b)][c] += 1
        self.total = sum(self.uni.values())
        self.V = len(self.uni)
        self.k = k
        self.lambdas = lambdas

    def logp(self, w, prev2, prev1):
        k, V = self.k, self.V
        p_uni = (self.uni.get(w, 0) + k) / (self.total + k * V)
        bctx = self.bi.get(prev1)
        p_bi = ((bctx.get(w, 0) + k) / (sum(bctx.values()) + k * V)) if bctx else p_uni
        tctx = self.tri.get((prev2, prev1))
        p_tri = ((tctx.get(w, 0) + k) / (sum(tctx.values()) + k * V)) if tctx else p_bi
        l3, l2, l1 = self.lambdas
        return math.log(l3 * p_tri + l2 * p_bi + l1 * p_uni)

    def bits_per_glyph(self, seq):
        ctx = ["<s>", "<s>"]
        total = 0.0
        for w in list(seq) + ["</s>"]:
            total += self.logp(w, ctx[-2], ctx[-1])
            ctx.append(w)
        return -total / (len(seq) + 1) / math.log(2)


# ---------------- Model loading ----------------
def load_model(model_dir: str):
    model_dir = Path(model_dir)
    with open(model_dir / "src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(model_dir / "tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    model = TransformerTranslator(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, dropout=0.1,
    )
    ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        ckpt = model_dir / "model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, src_vocab, tgt_vocab


# ---------------- Beam search with shallow fusion ----------------
@torch.no_grad()
def beam_translate(model, src_vocab, tgt_vocab, lm, text, beam=5, lm_weight=0.35,
                   max_len=40, rep_penalty=1.5, device="cpu"):
    tokens = text.strip().lower().split()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_emb = model.pos_enc(model.src_emb(src) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb)

    def model_logprobs(prefix_ids):
        ys = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        tgt_emb = model.pos_enc(model.tgt_emb(ys) * math.sqrt(model.d_model))
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = model.out_proj(out[:, -1, :])
        return torch.log_softmax(logits[0], dim=-1)

    itos = tgt_vocab.itos
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
            glyph_prefix = [itos[i] for i in prefix[1:]]
            prev1 = glyph_prefix[-1] if len(glyph_prefix) >= 1 else "<s>"
            prev2 = glyph_prefix[-2] if len(glyph_prefix) >= 2 else "<s>"
            for logp_m, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                if idx in (tgt_vocab.bos_idx, tgt_vocab.pad_idx):
                    continue
                w = "</s>" if idx == tgt_vocab.eos_idx else itos[idx]
                logp_lm = lm.logp(w, prev2, prev1)
                s = score + (1 - lm_weight) * logp_m + lm_weight * logp_lm
                # penalize a 3rd immediate repetition (2x is a real feature, 3x+ is degenerate)
                if len(glyph_prefix) >= 2 and idx != tgt_vocab.eos_idx \
                        and glyph_prefix[-1] == glyph_prefix[-2] == itos[idx]:
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
    best = max(done, key=lambda t: t[1] / len(t[0]))
    out = [itos[i] for i in best[0] if i not in
           (tgt_vocab.bos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx)]
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser(description="Translate to Rongorongo v6 (real Barthel codes)")
    parser.add_argument("--model-dir", default="models/rongorongo_translator_v6")
    parser.add_argument("--real-corpus", default="data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    parser.add_argument("--text", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--lm-weight", type=float, default=0.35)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir} + real-corpus LM...")
    model, src_vocab, tgt_vocab = load_model(args.model_dir)
    lm = RealGlyphLM(args.real_corpus)
    print(f"Src vocab {len(src_vocab)}, tgt vocab {len(tgt_vocab)} (real Barthel), LM V={lm.V}")

    def run(text):
        out = beam_translate(model, src_vocab, tgt_vocab, lm, text,
                             beam=args.beam, lm_weight=args.lm_weight)
        bits = lm.bits_per_glyph(out.split())
        print(f"Input:  {text}")
        print(f"Output: {out}")
        print(f"Real-LM bits/glyph: {bits:.2f}")

    if args.interactive:
        while True:
            text = input("\nInput: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            run(text)
    elif args.text:
        run(args.text)
    else:
        for sent in ["te tangata haere ki te moana", "ko Hotu Matua e kai ika",
                     "te manu rere i te raa", "e aroha au ki a kou"]:
            run(sent)
            print("-" * 50)


if __name__ == "__main__":
    main()
