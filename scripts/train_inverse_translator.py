"""Inverse translator: Rongorongo glyphs → candidate language.

Trains a separate decoder that maps glyph sequences back to
source language text, enabling bidirectional translation.
"""
import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spectral_submersion.rongorongo_translator import Vocab, TransformerTranslator


class InverseDataset(Dataset):
    """Dataset where source=glyphs and target=language text."""
    def __init__(self, df, glyph_vocab, text_vocab, max_len=50):
        self.pairs = []
        for _, row in df.iterrows():
            glyphs = row["target_glyphs"].strip().split()
            text = row["source_text"].strip().split()
            src_ids = [glyph_vocab.bos_idx] + glyph_vocab.encode(glyphs) + [glyph_vocab.eos_idx]
            tgt_ids = [text_vocab.bos_idx] + text_vocab.encode(text) + [text_vocab.eos_idx]
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.pairs.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, pad_idx):
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    src_pad = torch.full((len(srcs), max_src), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_pad[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return src_pad, tgt_pad


def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    model.train()
    total_loss = 0
    total_tokens = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(src, tgt_input, src_padding_mask=(src == pad_idx),
                       tgt_padding_mask=(tgt_input == pad_idx),
                       memory_key_padding_mask=(src == pad_idx))
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mask = tgt_output != pad_idx
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input, src_padding_mask=(src == pad_idx),
                           tgt_padding_mask=(tgt_input == pad_idx),
                           memory_key_padding_mask=(src == pad_idx))
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            mask = tgt_output != pad_idx
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Train inverse Rongorongo translator")
    parser.add_argument("--data", default="data/raw/lost_language/parallel_rongorongo_massive_v3.csv")
    parser.add_argument("--output-dir", default="models/rongorongo_inverse_translator")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} pairs")

    # Build vocabularies (reversed: glyphs are source, text is target)
    glyph_tokens = []
    text_tokens = []
    for _, row in df.iterrows():
        glyph_tokens.extend(row["target_glyphs"].strip().split())
        text_tokens.extend(row["source_text"].strip().split())

    glyph_vocab = Vocab(glyph_tokens)
    text_vocab = Vocab(text_tokens)
    print(f"Glyph vocab: {len(glyph_vocab)}, Text vocab: {len(text_vocab)}")

    n_train = int(len(df) * 0.95)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]

    train_ds = InverseDataset(train_df, glyph_vocab, text_vocab)
    val_ds = InverseDataset(val_df, glyph_vocab, text_vocab)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    def _collate(batch):
        return collate_fn(batch, glyph_vocab.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=_collate)

    model = TransformerTranslator(
        src_vocab_size=len(glyph_vocab),
        tgt_vocab_size=len(text_vocab),
        d_model=256, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, dropout=0.1,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, glyph_vocab.pad_idx)
        val_loss = evaluate(model, val_loader, criterion, device, glyph_vocab.pad_idx)
        scheduler.step()
        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print("  -> Saved best model")

    # Save vocabularies
    with open(output_dir / "src_vocab.pkl", "wb") as f:
        pickle.dump(glyph_vocab, f)
    with open(output_dir / "tgt_vocab.pkl", "wb") as f:
        pickle.dump(text_vocab, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    torch.save(model.state_dict(), output_dir / "model.pt")

    print(f"\nBest val loss: {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")


if __name__ == "__main__":
    main()
