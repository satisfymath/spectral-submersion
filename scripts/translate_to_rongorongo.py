"""Translate text to Rongorongo using the trained Transformer model.

Usage:
    python translate_to_rongorongo.py --text "te tangata haere ki te moana"
    python translate_to_rongorongo.py --interactive
"""
import argparse
import pickle
from pathlib import Path

import torch

from spectral_submersion.rongorongo_translator import Vocab, TransformerTranslator


def load_model(model_dir: str):
    model_dir = Path(model_dir)
    with open(model_dir / "src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open(model_dir / "tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)

    # Load config defaults
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


def translate(model, src_vocab, tgt_vocab, text: str, device: str = "cpu", max_len: int = 50) -> str:
    tokens = text.strip().lower().split()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    model = model.to(device)
    out_ids = model.greedy_decode(src_tensor, src_vocab, tgt_vocab, max_len=max_len, device=device)

    # Remove BOS and EOS
    out_tokens = []
    for idx in out_ids:
        if idx == tgt_vocab.bos_idx:
            continue
        if idx == tgt_vocab.eos_idx:
            break
        out_tokens.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<unk>")

    return " ".join(out_tokens)


def interactive_translate(model, src_vocab, tgt_vocab, device: str = "cpu"):
    print("=" * 60)
    print("RONGORONGO TRANSLATOR (interactive)")
    print("Type a sentence in Polynesian-like language.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    while True:
        text = input("\nInput: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = translate(model, src_vocab, tgt_vocab, text, device)
        print(f"Rongorongo: {result}")


def batch_translate(model, src_vocab, tgt_vocab, texts: list[str], device: str = "cpu") -> list[str]:
    results = []
    for text in texts:
        result = translate(model, src_vocab, tgt_vocab, text, device)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Translate to Rongorongo")
    parser.add_argument("--model-dir", default="models/rongorongo_translator")
    parser.add_argument("--text", default=None, help="Single sentence to translate")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model, src_vocab, tgt_vocab = load_model(args.model_dir)
    print(f"Model loaded. Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")

    if args.interactive:
        interactive_translate(model, src_vocab, tgt_vocab, args.device)
    elif args.text:
        result = translate(model, src_vocab, tgt_vocab, args.text, args.device)
        print(f"Input:  {args.text}")
        print(f"Output: {result}")
    else:
        # Demo translations
        demo_sentences = [
            "te tangata haere ki te moana",
            "he vahine noho i te hare",
            "ko Hotu Matua e kai ika",
            "te tamaiti moe i te po",
            "e haere au ki te maunga",
            "te manu rere i te raa",
            "he vai inu mo te tangata",
            "ko Maui tiki ake te ahi",
        ]
        print("\nDemo translations:")
        print("-" * 60)
        for sent in demo_sentences:
            result = translate(model, src_vocab, tgt_vocab, sent, args.device)
            print(f"  {sent:40s} -> {result}")


if __name__ == "__main__":
    main()
