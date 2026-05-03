"""Setup vocab and checkpoint for Rongorongo translator v5."""
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from spectral_submersion.rongorongo_translator import Vocab, TransformerTranslator


def build_and_save(data_path, output_dir, model_config):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} pairs")

    src_tokens = []
    tgt_tokens = []
    for _, row in df.iterrows():
        src_tokens.extend(row["source_text"].strip().split())
        tgt_tokens.extend(row["target_glyphs"].strip().split())

    src_vocab = Vocab(src_tokens)
    tgt_vocab = Vocab(tgt_tokens)
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open(out / "tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)
    with open(out / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config["enc_layers"],
        num_decoder_layers=model_config["dec_layers"],
        dim_feedforward=model_config["dim_ff"],
        dropout=model_config["dropout"],
    )
    ckpt = out / "best_model.pt"
    if ckpt.exists():
        print(f"Loading checkpoint from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print("No checkpoint found")
    torch.save(model.state_dict(), out / "model.pt")
    print(f"Saved to {out}")


if __name__ == "__main__":
    config = {
        "d_model": 256,
        "nhead": 8,
        "enc_layers": 4,
        "dec_layers": 4,
        "dim_ff": 512,
        "dropout": 0.1,
    }
    build_and_save(
        "data/raw/lost_language/parallel_rongorongo_massive_v3.csv",
        "models/rongorongo_translator_v5",
        config,
    )
