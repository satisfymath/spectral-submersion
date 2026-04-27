"""Decoding / hypothesis generation utilities."""
import numpy as np


def generate_symbol_hypotheses(
    Pi: np.ndarray,
    source_tokens: list[str],
    target_tokens: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Generate ranked candidate hypotheses for each source symbol.

    Args:
        Pi: Probability matrix (n_source x n_target).
        source_tokens: List of source token strings.
        target_tokens: List of target token strings.
        top_k: Number of top candidates to return per symbol.

    Returns:
        List of hypothesis dicts, one per source token.
    """
    hypotheses = []
    for i, src in enumerate(source_tokens):
        probs = Pi[i]
        top_idx = np.argsort(-probs)[:top_k]
        candidates = [
            {"candidate": target_tokens[j], "probability": float(probs[j])}
            for j in top_idx
        ]
        hypotheses.append(
            {
                "source_token": src,
                "entropy": float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()),
                "candidates": candidates,
            }
        )
    return hypotheses
