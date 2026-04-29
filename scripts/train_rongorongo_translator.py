"""Train Transformer seq2seq translator: Polynesian → Rongorongo.

Architecture: compact Transformer (Attention Is All You Need)
- Source: tokenized Polynesian-like sentences
- Target: Rongorongo glyph sequences
- Training on synthetic parallel corpus
- Inference with greedy decoding and beam search
"""
import argparse
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spectral_submersion.rongorongo_translator import Vocab, TransformerTranslator


# ============================================================
# Dataset
# ============================================================

class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int = 50):
        self.pairs = []
        for _, row in df.iterrows():
            src = row["source_text"].strip().split()
            tgt = row["target_glyphs"].strip().split()
            src_ids = [src_vocab.bos_idx] + src_vocab.encode(src) + [src_vocab.eos_idx]
            tgt_ids = [tgt_vocab.bos_idx] + tgt_vocab.encode(tgt) + [tgt_vocab.eos_idx]
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


# ============================================================
# Training
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    model.train()
    total_loss = 0
    total_tokens = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
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
            src = src.to(device)
            tgt = tgt.to(device)
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Rongorongo translator")
    parser.add_argument("--data", default="data/raw/lost_language/parallel_rongorongo_massive.csv")
    parser.add_argument("--output-dir", default="models/rongorongo_translator")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} parallel pairs")

    # Build vocabularies
    src_tokens = []
    tgt_tokens = []
    for _, row in df.iterrows():
        src_tokens.extend(row["source_text"].strip().split())
        tgt_tokens.extend(row["target_glyphs"].strip().split())

    src_vocab = Vocab(src_tokens)
    tgt_vocab = Vocab(tgt_tokens)
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")

    # Split train/val
    n_train = int(len(df) * args.train_split)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]

    train_ds = TranslationDataset(train_df, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_df, src_vocab, tgt_vocab)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx))

    # Model
    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, src_vocab.pad_idx)
        val_loss = evaluate(model, val_loader, criterion, device, src_vocab.pad_idx)
        scheduler.step()
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> Saved best model")

    # Save vocab and config
    with open(output_dir / "src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open(output_dir / "tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
