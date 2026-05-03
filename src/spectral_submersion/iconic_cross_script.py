"""Cross-script validation datasets for iconic grounding.

These adapters provide small, auditable validation sets from deciphered or
standardized scripts. They are intentionally conservative: every sign carries a
known referent label and provenance, and experiments skip referents whose real
image evidence is not present locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .iconic_real_data import image_shape_embedding


EGYPTIAN_FONT = "/System/Library/Fonts/Supplemental/NotoSansEgyptianHieroglyphs-Regular.ttf"
CHINESE_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"


@dataclass(frozen=True)
class KnownScriptSign:
    sign_id: str
    script: str
    glyph: str
    referent_id: str
    label: str
    font_path: str
    provenance: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sign_id": self.sign_id,
            "script": self.script,
            "glyph": self.glyph,
            "referent_id": self.referent_id,
            "label": self.label,
            "font_path": self.font_path,
            "provenance": self.provenance,
        }


def default_known_script_signs() -> list[KnownScriptSign]:
    """Return a compact cross-script validation set.

    Egyptian signs use Gardiner/Unicode identifiers and are deciphered
    hieroglyphs. Chinese signs are standard logographs descended from
    pictographic writing; they are used as a conservative supplementary visual
    sanity check, not as oracle-bone paleography.
    """

    return [
        KnownScriptSign(
            "egyptian_G005",
            "egyptian_hieroglyphic",
            "\U00013143",
            "great_frigatebird",
            "falcon/bird",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph G005; Gardiner bird sign.",
        ),
        KnownScriptSign(
            "egyptian_G043",
            "egyptian_hieroglyphic",
            "\U00013171",
            "domestic_chicken",
            "quail chick/bird",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph G043; Gardiner bird sign.",
        ),
        KnownScriptSign(
            "egyptian_K001",
            "egyptian_hieroglyphic",
            "\U0001319b",
            "yellowfin_tuna",
            "fish",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph K001; Gardiner fish sign.",
        ),
        KnownScriptSign(
            "egyptian_N005",
            "egyptian_hieroglyphic",
            "\U000131f3",
            "sun",
            "sun",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph N005; Gardiner sun sign.",
        ),
        KnownScriptSign(
            "egyptian_N011",
            "egyptian_hieroglyphic",
            "\U000131f9",
            "moon",
            "moon",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph N011; Gardiner moon sign.",
        ),
        KnownScriptSign(
            "egyptian_D046",
            "egyptian_hieroglyphic",
            "\U000130a7",
            "hand",
            "hand",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph D046; Gardiner hand sign.",
        ),
        KnownScriptSign(
            "egyptian_D058",
            "egyptian_hieroglyphic",
            "\U000130c0",
            "foot",
            "foot",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph D058; Gardiner foot sign.",
        ),
        KnownScriptSign(
            "egyptian_A001",
            "egyptian_hieroglyphic",
            "\U00013000",
            "head",
            "seated human/person",
            EGYPTIAN_FONT,
            "Unicode Egyptian Hieroglyph A001; Gardiner seated man sign.",
        ),
        KnownScriptSign(
            "chinese_sun",
            "chinese_standard_logograph",
            "日",
            "sun",
            "sun/day",
            CHINESE_FONT,
            "Standard Chinese logograph; supplementary pictographic descendant.",
        ),
        KnownScriptSign(
            "chinese_moon",
            "chinese_standard_logograph",
            "月",
            "moon",
            "moon/month",
            CHINESE_FONT,
            "Standard Chinese logograph; supplementary pictographic descendant.",
        ),
        KnownScriptSign(
            "chinese_hand",
            "chinese_standard_logograph",
            "手",
            "hand",
            "hand",
            CHINESE_FONT,
            "Standard Chinese logograph; supplementary pictographic descendant.",
        ),
        KnownScriptSign(
            "chinese_foot",
            "chinese_standard_logograph",
            "足",
            "foot",
            "foot",
            CHINESE_FONT,
            "Standard Chinese logograph; supplementary pictographic descendant.",
        ),
        KnownScriptSign(
            "chinese_fish",
            "chinese_standard_logograph",
            "魚",
            "yellowfin_tuna",
            "fish",
            CHINESE_FONT,
            "Traditional Chinese logograph; supplementary pictographic descendant.",
        ),
        KnownScriptSign(
            "chinese_bird",
            "chinese_standard_logograph",
            "鳥",
            "great_frigatebird",
            "bird",
            CHINESE_FONT,
            "Traditional Chinese logograph; supplementary pictographic descendant.",
        ),
    ]


def available_known_script_signs(
    referent_ids: set[str],
    signs: list[KnownScriptSign] | None = None,
) -> list[KnownScriptSign]:
    """Filter signs to referents with local real image evidence."""

    signs = signs or default_known_script_signs()
    return [
        sign
        for sign in signs
        if sign.referent_id in referent_ids and Path(sign.font_path).exists()
    ]


def render_known_script_sign(sign: KnownScriptSign, image_size: int = 128):
    """Render a deciphered/standard sign with its documented font."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required for cross-script rendering") from exc

    image = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(sign.font_path, size=int(image_size * 0.72))
    bbox = draw.textbbox((0, 0), sign.glyph, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (image_size - width) / 2 - bbox[0]
    y = (image_size - height) / 2 - bbox[1]
    draw.text((x, y), sign.glyph, font=font, fill=0)
    return image


def build_known_script_embeddings(
    signs: list[KnownScriptSign],
    image_size: int = 128,
    grid_size: int = 32,
) -> dict[str, np.ndarray]:
    """Render signs and encode them with the shared shape descriptor."""

    return {
        sign.sign_id: image_shape_embedding(
            render_known_script_sign(sign, image_size=image_size),
            grid_size=grid_size,
        )
        for sign in signs
    }
