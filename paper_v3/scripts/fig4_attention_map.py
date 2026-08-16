"""F4: cross-attention map of the v6 Transformer for one gloss verse,
averaged over heads, last decoder layer. Seed 42.
Anti-conclusion: attention shows which source tokens drive which glyph
emissions inside the SYNTHETIC parallel mapping; it is not evidence of
any real rapanui-glyph correspondence.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "scripts")
from translate_to_rongorongo_v6 import RealGlyphLM, beam_translate, load_model  # noqa: E402

SEED = 42
SENT = "makemake au ki a kou i te rau raa"
OUT = Path("paper_v3/figures")


def main():
    torch.manual_seed(SEED)
    model, sv, tv = load_model("models/rongorongo_translator_v6")
    lm = RealGlyphLM("data/raw/lost_language/corpus_rongorongo_real.xml.csv")
    out = beam_translate(model, sv, tv, lm, SENT, beam=5, lm_weight=0.35)
    out_toks = out.split()

    # capture cross-attention of last decoder layer
    captured = {}
    layer = model.transformer.decoder.layers[-1]
    orig_forward = layer.multihead_attn.forward

    def wrapped(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        o, w = orig_forward(*args, **kwargs)
        captured["w"] = w.detach()
        return o, w

    layer.multihead_attn.forward = wrapped

    src_ids = [sv.bos_idx] + sv.encode(SENT.split()) + [sv.eos_idx]
    tgt_ids = [tv.bos_idx] + tv.encode(out_toks)
    with torch.no_grad():
        model(torch.tensor([src_ids]), torch.tensor([tgt_ids]))
    layer.multihead_attn.forward = orig_forward

    A = captured["w"][0].numpy()  # (tgt_len, src_len)
    src_labels = ["<bos>"] + SENT.split() + ["<eos>"]
    tgt_labels = out_toks + ["</s>"] if A.shape[0] == len(out_toks) + 1 else out_toks[:A.shape[0]]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(A[:len(tgt_labels), :], cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(src_labels)), src_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tgt_labels)), tgt_labels, fontsize=8)
    ax.set_xlabel("source (rapanui gloss)")
    ax.set_ylabel("output (Barthel codes)")
    ax.set_title(f"Cross-attention, last decoder layer (head-avg)\n“{SENT}” → {out}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_attention_map.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig4; output:", out)


if __name__ == "__main__":
    main()
