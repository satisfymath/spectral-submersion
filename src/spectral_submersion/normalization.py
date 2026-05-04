"""Normalization utilities for corpus preprocessing."""

import re


def normalize_glyph(token: str) -> str:
    """Basic normalization for glyph-like tokens."""
    token = token.strip().lower()
    # Remove non-alphanumeric except hyphens and underscores
    token = re.sub(r"[^a-z0-9\-_]", "", token)
    return token


def remove_empty(tokens: list[str]) -> list[str]:
    """Remove empty or whitespace-only tokens."""
    return [t for t in tokens if t and t.strip()]
