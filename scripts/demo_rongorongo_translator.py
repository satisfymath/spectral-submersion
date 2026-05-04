"""Quick demo of the Rongorongo Translator.

Usage:
    python demo_rongorongo_translator.py
"""

import torch

from spectral_submersion.rongorongo_translator import TransformerTranslator, Vocab
import pickle
from pathlib import Path


def load(model_dir="models/rongorongo_translator_v3"):
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
        len(src_vocab),
        len(tgt_vocab),
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


def translate(model, src_vocab, tgt_vocab, text):
    tokens = text.strip().lower().split()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src = torch.tensor([src_ids], dtype=torch.long)
    out_ids = model.greedy_decode(src, src_vocab, tgt_vocab, device="cpu")
    out = []
    for idx in out_ids:
        if idx == tgt_vocab.bos_idx:
            continue
        if idx == tgt_vocab.eos_idx:
            break
        out.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<?>")
    return " ".join(out)


def main():
    print("=" * 70)
    print("RONGORONGO TRANSLATOR DEMO")
    print("=" * 70)
    print("Loading model...")
    model, src_vocab, tgt_vocab = load()
    print(f"Source vocab: {len(src_vocab)} | Target glyph inventory: {len(tgt_vocab)}")
    print("=" * 70)

    examples = [
        ("Rapa Nui", "te tangata haere ki te moana"),
        ("Rapa Nui", "he vahine noho i te hare"),
        ("Rapa Nui", "ko Hotu Matua e kai ika"),
        ("Rapa Nui", "te manu rere i te raa"),
        ("Hawaiian", "ka haele ke kanaka i ke kai"),
        ("English", "the man goes to the sea"),
        ("Spanish", "el hombre va al mar"),
        ("Japanese", "hito wa umi e ikimasu"),
    ]

    print("\n{:<15s} {:<35s} -> {}".format("Language", "Input", "Rongorongo"))
    print("-" * 70)
    for lang, text in examples:
        result = translate(model, src_vocab, tgt_vocab, text)
        print(f"{lang:<15s} {text:<35s} -> {result}")

    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Type sentences to translate. 'quit' to exit.")
    print("=" * 70)
    while True:
        try:
            inp = input("\n> ").strip()
            if inp.lower() in ("quit", "q", "exit"):
                break
            if inp:
                print(f"  Rongorongo: {translate(model, src_vocab, tgt_vocab, inp)}")
        except (EOFError, KeyboardInterrupt):
            break
    print("\nPoipoia te kakano kia puawai.")


if __name__ == "__main__":
    main()
