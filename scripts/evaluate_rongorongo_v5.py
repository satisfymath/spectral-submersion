"""Comprehensive evaluation of Rongorongo translator v5.

Compares greedy vs beam search and evaluates across language families.
"""

import argparse
import pickle
from pathlib import Path

import torch

from spectral_submersion.rongorongo_translator import TransformerTranslator
from spectral_submersion.beam_search import translate_beam


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


def translate_greedy(model, src_vocab, tgt_vocab, text, device="cpu"):
    tokens = text.strip().lower().split()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    model = model.to(device)
    out_ids = model.greedy_decode(src, src_vocab, tgt_vocab, max_len=50, device=device)
    out = []
    for idx in out_ids:
        if idx == tgt_vocab.bos_idx:
            continue
        if idx == tgt_vocab.eos_idx:
            break
        out.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<?>")
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/rongorongo_translator_v5")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print("Loading model v5...")
    model, src_vocab, tgt_vocab = load_model(args.model_dir)
    print(f"Source vocab: {len(src_vocab)} | Target: {len(tgt_vocab)}")

    tests = {
        "Rapa Nui": [
            "te tangata haere ki te moana",
            "he vahine noho i te hare",
            "ko Hotu Matua e kai ika",
            "te manu rere i te raa",
            "e haere au ki te maunga",
        ],
        "Hawaiian": [
            "ka haele ke kanaka i ke kai",
            "he wahine noho ma ka hale",
            "o Maui hele i ke mauna",
        ],
        "Fijian": [
            "na tamata lako ki na wasawasa",
            "e dau tiko na yalewa",
        ],
        "English": [
            "the man goes to the sea",
            "a woman lives in the house",
            "the bird flies in the sky",
        ],
        "Spanish": [
            "el hombre va al mar",
            "una mujer vive en la casa",
            "el pajaro vuela en el cielo",
        ],
        "Japanese": [
            "hito wa umi e ikimasu",
            "onna wa ie ni sundeimasu",
        ],
    }

    print("\n" + "=" * 80)
    print("GREEDY vs BEAM SEARCH COMPARISON (v5 model)")
    print("=" * 80)

    for family, sentences in tests.items():
        print(f"\n{family}")
        print("-" * 80)
        for sent in sentences:
            greedy = translate_greedy(model, src_vocab, tgt_vocab, sent, args.device)
            beam, score = translate_beam(
                model,
                src_vocab,
                tgt_vocab,
                sent,
                beam_width=5,
                device=args.device,
                length_penalty=0.8,
            )
            g_len = len(greedy.split())
            b_len = len(beam.split())
            g_uniq = len(set(greedy.split()))
            b_uniq = len(set(beam.split()))
            print(f"  Input:    {sent}")
            print(f"  Greedy:   {greedy}  (len={g_len}, uniq={g_uniq})")
            print(
                f"  Beam(5):  {beam}  (len={b_len}, uniq={b_uniq}, score={score:.3f})"
            )
            print()

    # Metrics summary
    print("=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    for family, sentences in tests.items():
        g_lengths, g_uniqs = [], []
        b_lengths, b_uniqs = [], []
        for sent in sentences:
            greedy = translate_greedy(model, src_vocab, tgt_vocab, sent, args.device)
            beam, _ = translate_beam(
                model,
                src_vocab,
                tgt_vocab,
                sent,
                beam_width=5,
                device=args.device,
                length_penalty=0.8,
            )
            g_lengths.append(len(greedy.split()))
            g_uniqs.append(len(set(greedy.split())))
            b_lengths.append(len(beam.split()))
            b_uniqs.append(len(set(beam.split())))

        print(
            f"  {family:15s} | Greedy: len={sum(g_lengths)/len(g_lengths):5.2f} uniq={sum(g_uniqs)/len(g_uniqs):5.2f} | Beam: len={sum(b_lengths)/len(b_lengths):5.2f} uniq={sum(b_uniqs)/len(b_uniqs):5.2f}"
        )


if __name__ == "__main__":
    main()
