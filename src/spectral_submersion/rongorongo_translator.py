"""Transformer seq2seq model for Rongorongo translation.

Shared components used by both training and inference scripts.
"""
import math

import torch
import torch.nn as nn


class Vocab:
    def __init__(self, tokens, min_freq=1, special=None):
        if special is None:
            special = ("<pad>", "<unk>", "<bos>", "<eos>")
        self.pad, self.unk, self.bos, self.eos = special
        self.special = special
        from collections import Counter
        counts = Counter(tokens)
        self.itos = list(special) + [t for t, c in counts.items() if c >= min_freq and t not in special]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_idx = self.stoi[self.pad]
        self.unk_idx = self.stoi[self.unk]
        self.bos_idx = self.stoi[self.bos]
        self.eos_idx = self.stoi[self.eos]

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices):
        return [self.itos[i] if 0 <= i < len(self.itos) else self.unk for i in indices]

    def __len__(self):
        return len(self.itos)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=100,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(out)

    def greedy_decode(self, src, src_vocab, tgt_vocab, max_len=50, device="cpu"):
        self.eval()
        with torch.no_grad():
            src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb)
            ys = torch.tensor([[tgt_vocab.bos_idx]], dtype=torch.long, device=device)
            for _ in range(max_len):
                tgt_emb = self.pos_enc(self.tgt_emb(ys) * math.sqrt(self.d_model))
                tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.out_proj(out[:, -1, :])
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == tgt_vocab.eos_idx:
                    break
        return ys[0].cpu().tolist()
