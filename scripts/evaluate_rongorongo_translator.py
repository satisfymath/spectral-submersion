"""Evaluate and compare Rongorongo translations across language families.

Tests the translator with inputs from different language families to see
if structural differences in source languages produce different glyph
patterns in the synthetic Rongorongo output.
"""

import argparse
import pickle
from pathlib import Path

import torch

from spectral_submersion.rongorongo_translator import TransformerTranslator


def load_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(model_dir / "tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    config = {
        "d_model": 256,
        "nhead": 8,
        "enc_layers": 4,
        "dec_layers": 4,
        "dim_ff": 512,
        "dropout": 0.1,
    }
    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["enc_layers"],
        num_decoder_layers=config["dec_layers"],
        dim_feedforward=config["dim_ff"],
        dropout=config["dropout"],
    )
    ckpt = model_dir / "model.pt"
    if not ckpt.exists():
        ckpt = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, src_vocab, tgt_vocab


def translate(model, src_vocab, tgt_vocab, text, device="cpu", max_len=50):
    tokens = text.strip().lower().split()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    model = model.to(device)
    out_ids = model.greedy_decode(
        src_tensor, src_vocab, tgt_vocab, max_len=max_len, device=device
    )
    out_tokens = []
    for idx in out_ids:
        if idx == tgt_vocab.bos_idx:
            continue
        if idx == tgt_vocab.eos_idx:
            break
        out_tokens.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<unk>")
    return " ".join(out_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/rongorongo_translator_v3")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("Loading model...")
    model, src_vocab, tgt_vocab = load_model(args.model_dir)

    # Test sentences across families
    tests = {
        "Polynesian (Rapa Nui-like)": [
            "te tangata haere ki te moana",
            "he vahine noho i te hare",
            "ko Hotu Matua e kai ika",
            "te manu rere i te raa",
        ],
        "Polynesian (Hawaiian-like)": [
            "ka haele ke kanaka i ke kai",
            "he wahine noho ma ka hale",
            "o Maui hele i ke mauna",
        ],
        "Austronesian (Fijian-like)": [
            "na tamata lako ki na wasawasa",
            "e dau tiko na yalewa",
        ],
        "Germanic (English-like)": [
            "the man goes to the sea",
            "a woman lives in the house",
            "the bird flies in the sky",
        ],
        "Romance (Spanish-like)": [
            "el hombre va al mar",
            "una mujer vive en la casa",
            "el pajaro vuela en el cielo",
        ],
        "Japonic (Japanese-like)": [
            "hito wa umi e ikimasu",
            "onna wa ie ni sundeimasu",
        ],
    }

    print("\n" + "=" * 70)
    print("CROSS-FAMILY RONGORONGO TRANSLATION COMPARISON")
    print("=" * 70)

    for family, sentences in tests.items():
        print(f"\n{family}")
        print("-" * 70)
        for sent in sentences:
            result = translate(model, src_vocab, tgt_vocab, sent, args.device)
            print(f"  {sent:45s} -> {result}")

    # Structural metrics
    print("\n" + "=" * 70)
    print("STRUCTURAL METRICS")
    print("=" * 70)
    all_results = {}
    for family, sentences in tests.items():
        lengths = []
        uniq_glyphs = []
        for sent in sentences:
            result = translate(model, src_vocab, tgt_vocab, sent, args.device)
            glyphs = result.split()
            lengths.append(len(glyphs))
            uniq_glyphs.append(len(set(glyphs)))
        avg_len = sum(lengths) / len(lengths)
        avg_uniq = sum(uniq_glyphs) / len(uniq_glyphs)
        print(
            f"  {family:30s} | Avg length: {avg_len:5.2f} | Avg diversity: {avg_uniq:5.2f}"
        )
        all_results[family] = {"avg_len": avg_len, "avg_diversity": avg_uniq}


if __name__ == "__main__":
    main()
