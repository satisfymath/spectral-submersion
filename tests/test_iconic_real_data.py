"""Tests for real-data iconic grounding adapters."""

from pathlib import Path

import numpy as np
import pytest

from spectral_submersion.iconic_grounding import anchor_power_from_counts
from spectral_submersion.iconic_real_data import (
    base_glyph_code,
    build_rongorongo_glyph_embedding_table,
    canonical_glyph_code,
    export_glyph_svg_audit_files,
    image_shape_embedding,
    parse_rongorongo_svg_instances,
    render_svg_path_to_image,
    sample_svg_path,
)


def _write_tiny_rr_xml(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<tablet id="T">
  <side id="Ta">
    <line id="Ta01">
      <glyph id="Ta01-001">
        <code>600!</code>
        <link>.</link>
        <image id="Ta01-001-b" type="b">
          <path id="p1" d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/>
          <x>0</x><y>0</y><width>10</width><height>10</height>
        </image>
      </glyph>
      <glyph id="Ta01-002">
        <code>040a?</code>
        <link>-</link>
        <image id="Ta01-002-b" type="b">
          <path id="p2" d="M 0 0 C 5 10 10 10 15 0"/>
          <x>0</x><y>0</y><width>15</width><height>10</height>
        </image>
      </glyph>
    </line>
  </side>
</tablet>
""",
        encoding="utf-8",
    )


def test_canonical_and_base_codes():
    assert canonical_glyph_code("600!") == "600"
    assert canonical_glyph_code("040a?") == "040a"
    assert base_glyph_code("040a?") == "040"


def test_parse_rongorongo_svg_instances_from_xml(tmp_path):
    _write_tiny_rr_xml(tmp_path / "T.xml")

    instances = parse_rongorongo_svg_instances(tmp_path)

    assert len(instances) == 2
    assert instances[0].raw_code == "600!"
    assert instances[0].base_code == "600"
    assert instances[1].canonical_code == "040a"


def test_sample_svg_path_handles_lines_and_curves():
    subpaths = sample_svg_path("M 0 0 L 10 0 C 12 2 12 8 10 10 Z")

    assert len(subpaths) == 1
    points, closed = subpaths[0]
    assert closed
    assert len(points) > 5


def test_render_svg_path_and_embedding_are_nonempty():
    image = render_svg_path_to_image(
        "M 0 0 L 10 0 L 10 10 L 0 10 Z",
        bbox=(0, 0, 10, 10),
        image_size=64,
    )
    emb = image_shape_embedding(image, grid_size=16)

    assert np.asarray(image).min() == 0
    assert np.linalg.norm(emb) == pytest.approx(1.0)


def test_build_real_glyph_embedding_table_and_export(tmp_path):
    _write_tiny_rr_xml(tmp_path / "T.xml")

    table = build_rongorongo_glyph_embedding_table(
        xml_dir=tmp_path,
        top_n=2,
        max_instances_per_glyph=2,
        image_size=64,
        grid_size=16,
    )
    exported = export_glyph_svg_audit_files(table.instances_by_code, tmp_path / "svg")

    assert set(table.embeddings) == {"600", "040"}
    assert all(
        np.linalg.norm(v) == pytest.approx(1.0) for v in table.embeddings.values()
    )
    assert exported == 2
    assert len(list((tmp_path / "svg").glob("*.svg"))) == 2


def test_anchor_power_from_counts_is_stable_for_large_vocab():
    ap = anchor_power_from_counts(vocab_size=941, anchored_count=20)

    assert 0.0 < ap < 1.0
