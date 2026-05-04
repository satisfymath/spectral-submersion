"""Beam search decoding for Rongorongo translator.

Extends the base TransformerTranslator with beam search,
which explores multiple translation hypotheses and selects
the one with highest overall probability.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from spectral_submersion.rongorongo_translator import TransformerTranslator


class BeamSearchTranslator(TransformerTranslator):
    """Transformer with beam search decoding."""

    def beam_decode(
        self,
        src: torch.Tensor,
        src_vocab,
        tgt_vocab,
        beam_width: int = 5,
        max_len: int = 50,
        device: str = "cpu",
        length_penalty: float = 1.0,
    ) -> Tuple[List[int], float]:
        """Beam search decoding.

        Returns:
            best_sequence: list of token indices
            best_score: log-probability of the sequence
        """
        self.eval()
        with torch.no_grad():
            src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb)

            # Each beam: (sequence_tensor, score, finished)
            beams = [
                (
                    torch.tensor(
                        [[tgt_vocab.bos_idx]], dtype=torch.long, device=device
                    ),
                    0.0,
                    False,
                )
            ]

            for step in range(max_len):
                new_beams = []
                for seq, score, finished in beams:
                    if finished:
                        new_beams.append((seq, score, True))
                        continue

                    tgt_emb = self.pos_enc(self.tgt_emb(seq) * math.sqrt(self.d_model))
                    tgt_mask = self.transformer.generate_square_subsequent_mask(
                        seq.size(1)
                    ).to(device)
                    out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    log_probs = torch.log_softmax(self.out_proj(out[:, -1, :]), dim=-1)

                    topk_scores, topk_indices = log_probs.topk(beam_width)
                    for k in range(beam_width):
                        next_token = topk_indices[0, k].item()
                        next_score = score + topk_scores[0, k].item()
                        new_seq = torch.cat(
                            [seq, torch.tensor([[next_token]], device=device)], dim=1
                        )
                        is_finished = next_token == tgt_vocab.eos_idx
                        new_beams.append((new_seq, next_score, is_finished))

                # Keep top beam_width beams, apply length penalty
                new_beams.sort(
                    key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True
                )
                beams = new_beams[:beam_width]

                # If all beams finished, stop early
                if all(finished for _, _, finished in beams):
                    break

            # Return best beam
            best_seq, best_score, _ = beams[0]
            return best_seq[0].cpu().tolist(), best_score


def translate_beam(
    model: nn.Module,
    src_vocab,
    tgt_vocab,
    text: str,
    beam_width: int = 5,
    device: str = "cpu",
    max_len: int = 50,
    length_penalty: float = 1.0,
) -> Tuple[str, float]:
    """Translate text using beam search.

    Returns:
        translated_text: space-separated glyph string
        score: log-probability of the translation
    """
    tokens = text.strip().lower().split()
    src_ids = [src_vocab.bos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    model = model.to(device)

    if isinstance(model, BeamSearchTranslator):
        out_ids, score = model.beam_decode(
            src_tensor,
            src_vocab,
            tgt_vocab,
            beam_width=beam_width,
            max_len=max_len,
            device=device,
            length_penalty=length_penalty,
        )
    else:
        out_ids = model.greedy_decode(
            src_tensor, src_vocab, tgt_vocab, max_len=max_len, device=device
        )
        score = 0.0

    out_tokens = []
    for idx in out_ids:
        if idx == tgt_vocab.bos_idx:
            continue
        if idx == tgt_vocab.eos_idx:
            break
        out_tokens.append(tgt_vocab.itos[idx] if idx < len(tgt_vocab.itos) else "<unk>")

    return " ".join(out_tokens), score
