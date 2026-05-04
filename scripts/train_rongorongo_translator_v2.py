"""Train Transformer seq2seq translator (enhanced v2).

Improvements over v1:
- CosineAnnealingLR for better convergence
- Data augmentation: random token masking, synonym substitution
- More epochs with early stopping patience
- Better initialization
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


class TranslationDataset(Dataset):
    def __init__(
        self, df, src_vocab, tgt_vocab, max_len=50, augment=False, aug_prob=0.1
    ):
        self.pairs = []
        self.augment = augment
        self.aug_prob = aug_prob
        self.src_vocab = src_vocab
        for _, row in df.iterrows():
            src = row["source_text"].strip().split()
            tgt = row["target_glyphs"].strip().split()
            src_ids = [src_vocab.bos_idx] + src_vocab.encode(src) + [src_vocab.eos_idx]
            tgt_ids = [tgt_vocab.bos_idx] + tgt_vocab.encode(tgt) + [tgt_vocab.eos_idx]
            if len(src_ids) <= max_len and len(tgt_ids) <= max_len:
                self.pairs.append((src_ids, tgt_ids, src, tgt))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids, src_tokens, tgt_tokens = self.pairs[idx]
        if self.augment and random.random() < self.aug_prob:
            # Random token masking / substitution in source
            src_ids = self._augment(src_ids)
        return src_ids, tgt_ids

    def _augment(self, src_ids):
        # Randomly replace 10% of non-special tokens with <unk>
        result = []
        for i, tok in enumerate(src_ids):
            if tok not in (
                self.src_vocab.bos_idx,
                self.src_vocab.eos_idx,
                self.src_vocab.pad_idx,
            ):
                if random.random() < 0.1:
                    result.append(self.src_vocab.unk_idx)
                else:
                    result.append(tok)
            else:
                result.append(tok)
        return result


def collate_fn(batch, pad_idx):
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    src_pad = torch.full((len(srcs), max_src), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_pad[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return src_pad, tgt_pad


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
        output = model(
            src,
            tgt_input,
            src_padding_mask=(src == pad_idx),
            tgt_padding_mask=(tgt_input == pad_idx),
            memory_key_padding_mask=(src == pad_idx),
        )
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
            output = model(
                src,
                tgt_input,
                src_padding_mask=(src == pad_idx),
                tgt_padding_mask=(tgt_input == pad_idx),
                memory_key_padding_mask=(src == pad_idx),
            )
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            mask = tgt_output != pad_idx
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Train enhanced Rongorongo translator")
    parser.add_argument(
        "--data", default="data/raw/lost_language/parallel_rongorongo_massive_v3.csv"
    )
    parser.add_argument("--output-dir", default="models/rongorongo_translator_v5")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--train-split", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} parallel pairs")

    src_tokens = []
    tgt_tokens = []
    for _, row in df.iterrows():
        src_tokens.extend(row["source_text"].strip().split())
        tgt_tokens.extend(row["target_glyphs"].strip().split())

    src_vocab = Vocab(src_tokens)
    tgt_vocab = Vocab(tgt_tokens)
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")

    n_train = int(len(df) * args.train_split)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]

    train_ds = TranslationDataset(train_df, src_vocab, tgt_vocab, augment=args.augment)
    val_ds = TranslationDataset(val_df, src_vocab, tgt_vocab, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    def _collate(batch):
        return collate_fn(batch, src_vocab.pad_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=True,
    )

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    best_val_loss = float("inf")
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, src_vocab.pad_idx
        )
        val_loss = evaluate(model, val_loader, criterion, device, src_vocab.pad_idx)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | LR: {current_lr:.2e} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save artifacts
    with open(output_dir / "src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open(output_dir / "tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    # Save final model too
    torch.save(model.state_dict(), output_dir / "model.pt")

    print(f"\nBest val loss: {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.2f})")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
