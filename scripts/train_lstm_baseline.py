"""Small LSTM seq2seq baseline (Task 2.1c of the v3 protocol).

1-layer encoder/decoder LSTM, d=128, greedy decode. Trained on the same
parallel corpus as v6. Deliberately small: the question it answers is
whether the Transformer earns its keep. Seed fixed: 42.
"""
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectral_submersion.rongorongo_translator import Vocab


class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d=128):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d)
        self.tgt_emb = nn.Embedding(tgt_vocab, d)
        self.enc = nn.LSTM(d, d, batch_first=True)
        self.dec = nn.LSTM(d, d, batch_first=True)
        self.proj = nn.Linear(d, tgt_vocab)

    def forward(self, src, tgt):
        _, state = self.enc(self.src_emb(src))
        out, _ = self.dec(self.tgt_emb(tgt), state)
        return self.proj(out)

    @torch.no_grad()
    def greedy(self, src, bos, eos, max_len=40):
        _, state = self.enc(self.src_emb(src))
        ys = torch.tensor([[bos]], dtype=torch.long)
        out_ids = []
        for _ in range(max_len):
            o, state = self.dec(self.tgt_emb(ys[:, -1:]), state)
            nxt = self.proj(o[:, -1]).argmax(-1)
            if nxt.item() == eos:
                break
            out_ids.append(nxt.item())
            ys = torch.cat([ys, nxt.unsqueeze(0)], dim=1)
        return out_ids


def load_lstm(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "src_vocab.pkl", "rb") as f:
        sv = pickle.load(f)
    with open(model_dir / "tgt_vocab.pkl", "rb") as f:
        tv = pickle.load(f)
    model = LSTMSeq2Seq(len(sv), len(tv))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, sv, tv


def lstm_translate(model, sv, tv, text):
    ids = [sv.bos_idx] + sv.encode(text.lower().split()) + [sv.eos_idx]
    out = model.greedy(torch.tensor([ids]), tv.bos_idx, tv.eos_idx)
    return " ".join(tv.itos[i] for i in out if i < len(tv.itos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/lost_language/parallel_rongorongo_real_v4.csv")
    parser.add_argument("--output-dir", default="models/rongorongo_lstm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.data)
    src_tokens, tgt_tokens = [], []
    for _, row in df.iterrows():
        src_tokens.extend(row["source_text"].split())
        tgt_tokens.extend(row["target_glyphs"].split())
    sv, tv = Vocab(src_tokens), Vocab(tgt_tokens)

    pairs = []
    for _, row in df.iterrows():
        s = [sv.bos_idx] + sv.encode(row["source_text"].split()) + [sv.eos_idx]
        t = [tv.bos_idx] + tv.encode(row["target_glyphs"].split()) + [tv.eos_idx]
        if len(s) <= 50 and len(t) <= 50:
            pairs.append((s, t))

    def collate(batch):
        ms = max(len(s) for s, _ in batch)
        mt = max(len(t) for _, t in batch)
        S = torch.full((len(batch), ms), sv.pad_idx, dtype=torch.long)
        T = torch.full((len(batch), mt), tv.pad_idx, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            S[i, :len(s)] = torch.tensor(s)
            T[i, :len(t)] = torch.tensor(t)
        return S, T

    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = LSTMSeq2Seq(len(sv), len(tv))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=tv.pad_idx)

    for ep in range(1, args.epochs + 1):
        total, nb = 0.0, 0
        for S, T in loader:
            logits = model(S, T[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), T[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        print(f"Epoch {ep}: loss {total/nb:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "src_vocab.pkl", "wb") as f:
        pickle.dump(sv, f)
    with open(out / "tgt_vocab.pkl", "wb") as f:
        pickle.dump(tv, f)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
