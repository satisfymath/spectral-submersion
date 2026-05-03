"""Analyze learned glyph mappings in the Rongorongo translator.

Shows which source tokens map to which glyph categories,
helping validate that the model learned meaningful structure.
"""
import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

from spectral_submersion.rongorongo_translator import TransformerTranslator


def load_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(model_dir / "tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    config = {
        "d_model": 256, "nhead": 8, "enc_layers": 4,
        "dec_layers": 4, "dim_ff": 512, "dropout": 0.1,
    }
    model = TransformerTranslator(
        len(src_vocab), len(tgt_vocab),
        d_model=config["d_model"], nhead=config["nhead"],
        num_encoder_layers=config["enc_layers"], num_decoder_layers=config["dec_layers"],
        dim_feedforward=config["dim_ff"], dropout=config["dropout"],
    )
    ckpt = model_dir / "model.pt"
    if not ckpt.exists():
        ckpt = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, src_vocab, tgt_vocab


def analyze_mappings(model, src_vocab, tgt_vocab, test_words):
    """Analyze what glyphs the model produces for specific source words."""
    model.eval()
    device = next(model.parameters()).device

    word_to_glyphs = defaultdict(list)

    for word in test_words:
        if word not in src_vocab.stoi:
            continue
        src_ids = [src_vocab.bos_idx, src_vocab.stoi[word], src_vocab.eos_idx]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            src_emb = model.pos_enc(model.src_emb(src) * (model.d_model ** 0.5))
            memory = model.transformer.encoder(src_emb)
            ys = torch.tensor([[tgt_vocab.bos_idx]], dtype=torch.long, device=device)
            for _ in range(20):
                tgt_emb = model.pos_enc(model.tgt_emb(ys) * (model.d_model ** 0.5))
                tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
                out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.out_proj(out[:, -1, :])
                next_token = logits.argmax(dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == tgt_vocab.eos_idx:
                    break

        glyphs = []
        for idx in ys[0].cpu().tolist():
            if idx == tgt_vocab.bos_idx:
                continue
            if idx == tgt_vocab.eos_idx:
                break
            glyphs.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<?>")
        word_to_glyphs[word] = glyphs

    return word_to_glyphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/rongorongo_translator_v5")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading model...")
    model, src_vocab, tgt_vocab = load_model(args.model_dir)
    model = model.to(args.device)

    # Test words by category
    test_sets = {
        "Determinantes": ["te", "he", "ko", "ka", "a", "o", "e", "na", "tau", "nga"],
        "Nombres comunes": ["tangata", "vahine", "tamaiti", "ika", "manu", "rakau", "moana", "hare", "maunga", "raa"],
        "Verbos": ["haere", "noho", "kai", "inu", "moe", "tangi", "kite", "korero", "hula", "makemake"],
        "Nombres propios": ["Hotu", "Matua", "Maui", "Tane", "Rongo", "Tiki", "Hina", "Papa", "Rangi", "Tupa"],
        "Partículas": ["i", "ki", "mai", "atu", "ma", "mo", "pe", "ra", "nei", "ai"],
        "Números": ["tahi", "rua", "toru", "rima", "ono", "tekau", "hongahuru"],
    }

    print("\n" + "=" * 70)
    print("ANÁLISIS DE MAPEOS APRENDIDOS POR EL MODELO")
    print("=" * 70)
    print("(Palabra fuente -> glifos Rongorongo generados)\n")

    for category, words in test_sets.items():
        print(f"\n{category}")
        print("-" * 70)
        mappings = analyze_mappings(model, src_vocab, tgt_vocab, words)
        for word in words:
            if word in mappings:
                glyphs = " ".join(mappings[word])
                # Infer category from glyph prefix
                prefix = mappings[word][0][0] if mappings[word] else "?"
                print(f"  {word:15s} -> {glyphs:25s} (cat={prefix})")
            else:
                print(f"  {word:15s} -> <OOV>")

    # Consistency check
    print("\n" + "=" * 70)
    print("CONSISTENCIA: MISMA PALABRA, DIFERENTES CONTEXTOS")
    print("=" * 70)
    test_phrases = {
        "te": [
            "te tangata haere",
            "te moana nui",
            "te raa mahana",
            "te hare vahu",
        ],
        "tangata": [
            "te tangata haere",
            "he tangata moe",
            "ko tangata kai",
        ],
        "haere": [
            "te tangata haere",
            "he vahine haere",
            "te manu haere",
        ],
    }

    for word, phrases in test_phrases.items():
        print(f"\nPalabra: '{word}'")
        for phrase in phrases:
            tokens = phrase.strip().lower().split()
            src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
            src = torch.tensor([src_ids], dtype=torch.long, device=args.device)
            out_ids = model.greedy_decode(src, src_vocab, tgt_vocab, max_len=50, device=args.device)
            out = []
            for idx in out_ids:
                if idx == tgt_vocab.bos_idx:
                    continue
                if idx == tgt_vocab.eos_idx:
                    break
                out.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<?>")
            print(f"  '{phrase:35s}' -> {' '.join(out)}")


if __name__ == "__main__":
    main()
