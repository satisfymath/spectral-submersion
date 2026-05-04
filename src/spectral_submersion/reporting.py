"""Reporting utilities."""

import json
from pathlib import Path

import yaml


def save_hypotheses(
    hypotheses: list[dict],
    path: str | Path,
    format: str = "yaml",
) -> None:
    """Save hypothesis list to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if format == "yaml":
            yaml.dump(hypotheses, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            json.dump(hypotheses, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}")


def generate_markdown_report(
    metadata: dict,
    frequency_stats: dict,
    spectral_stats: dict,
    alignment_stats: dict | None = None,
    output_path: str | Path = "reports/final/first_hypothesis_report.md",
) -> None:
    """Generate a minimal markdown report from computed statistics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# First Hypothesis Report",
        "",
        "## 1. Corpus",
        f"- Number of documents: {metadata.get('num_docs', 'N/A')}",
        f"- Number of lines: {metadata.get('num_lines', 'N/A')}",
        f"- Number of tokens: {metadata.get('num_tokens', 'N/A')}",
        f"- Vocabulary size: {metadata.get('vocab_size', 'N/A')}",
        "",
        "## 2. Frequency Analysis",
        f"- Entropy: {frequency_stats.get('entropy', 'N/A'):.4f}",
        f"- Most frequent signs: {frequency_stats.get('top_tokens', [])[:5]}",
        "",
        "## 3. Spectral Analysis",
        f"- Effective rank: {spectral_stats.get('effective_rank', 'N/A'):.4f}",
        f"- Embedding dimensions used: {spectral_stats.get('embedding_dim', 'N/A')}",
        "",
        "## 4. Limitations",
        "- Analysis is preliminary and subject to corpus size constraints.",
        "- No external anchors or bilingual evidence used.",
        "",
        "## 5. Next Steps",
        "- Expand corpus coverage.",
        "- Introduce candidate language alignment.",
        "- Run negative controls and bootstrap stability tests.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
