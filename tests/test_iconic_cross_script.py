"""Tests for cross-script iconic validation helpers."""

import numpy as np
import pytest

from spectral_submersion.iconic_cross_script import (
    available_known_script_signs,
    build_known_script_embeddings,
    default_known_script_signs,
    render_known_script_sign,
)


def test_default_known_script_signs_have_referents_and_provenance():
    signs = default_known_script_signs()

    assert len(signs) >= 8
    assert all(sign.referent_id for sign in signs)
    assert all(sign.provenance for sign in signs)


def test_available_known_script_signs_filters_by_referent():
    signs = available_known_script_signs({"sun", "moon"})

    assert signs
    assert {sign.referent_id for sign in signs} <= {"sun", "moon"}


def test_render_known_script_sign_and_embedding():
    signs = available_known_script_signs({"sun"})
    if not signs:
        pytest.skip("No local font for known-script signs")

    image = render_known_script_sign(signs[0], image_size=64)
    embeddings = build_known_script_embeddings(signs[:1], image_size=64, grid_size=16)

    assert np.asarray(image).min() < 255
    assert np.linalg.norm(next(iter(embeddings.values()))) == pytest.approx(1.0)
