"""Tests for iconographic grounding: referents, rankings, and C2.5 claims."""

import numpy as np
import pytest

from spectral_submersion.iconic_grounding import (
    C25_CLAIM_LABEL,
    IconicClaimEvidence,
    RapaNuiWorld1500,
    align_glyphs_to_referents,
    anchor_assignment_stability,
    assess_c25_admissibility,
    cosine_similarity_matrix,
    detect_allographs,
    embedding_consensus,
    evaluate_anchor_ranking,
    iconic_recovery_probability_bound,
    predict_iconic_anchors,
    rank_iconic_candidates,
    referent_separation,
    spherical_mean,
    visual_diameter,
)


class TestRapaNuiWorld1500:
    def test_world_has_expected_categories_and_sources(self):
        world = RapaNuiWorld1500()
        categories = set(world.by_category())

        assert {"fauna", "flora", "artifact", "celestial", "human"} <= categories
        assert "great_frigatebird" in world.get_referent_set()
        assert "moon" in world.get_referent_set()
        assert world.validate_min_sources(min_sources=2)["valid"]

    def test_world_by_id(self):
        world = RapaNuiWorld1500()
        referents = world.by_id()

        assert referents["fairy_tern"].rapa_nui_name == "manu tara"
        assert referents["sweet_potato"].rapa_nui_name == "kumara"


class TestSphereUtilities:
    def test_spherical_mean_is_unit_length(self):
        vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.9, -0.1]])
        mean = spherical_mean(vectors)

        assert np.linalg.norm(mean) == pytest.approx(1.0)
        assert mean[0] > 0.95

    def test_embedding_consensus_with_encoder(self):
        table = {
            "view_a": np.array([1.0, 0.0, 0.0]),
            "view_b": np.array([0.9, 0.1, 0.0]),
            "view_c": np.array([0.9, -0.1, 0.0]),
        }

        consensus = embedding_consensus(list(table), encoder=table.__getitem__)

        assert consensus.n_items == 3
        assert consensus.dispersion < 0.1
        assert consensus.embedding[0] > 0.99

    def test_similarity_matrix_normalizes_inputs(self):
        glyphs = {"g1": np.array([2.0, 0.0]), "g2": np.array([0.0, 3.0])}
        refs = {"bird": np.array([1.0, 0.0]), "moon": np.array([0.0, 1.0])}

        sim = cosine_similarity_matrix(glyphs, refs)

        assert sim.shape == (2, 2)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[1, 1] == pytest.approx(1.0)


class TestIconicRanking:
    def test_rank_iconic_candidates_recovers_top_matches(self):
        glyphs = {
            "RR_600": np.array([1.0, 0.0, 0.0]),
            "RR_200": np.array([0.0, 1.0, 0.0]),
        }
        refs = {
            "bird": np.array([0.95, 0.05, 0.0]),
            "moon": np.array([0.05, 0.95, 0.0]),
            "fish": np.array([0.0, 0.0, 1.0]),
        }

        ranked = rank_iconic_candidates(glyphs, refs, top_k=3)

        assert ranked["RR_600"][0].referent_id == "bird"
        assert ranked["RR_200"][0].referent_id == "moon"
        assert ranked["RR_600"][0].deiconization_rate < 0.01

    def test_predict_iconic_anchors_filters_low_scores(self):
        glyphs = {"RR_600": np.array([1.0, 0.0]), "RR_noise": np.array([1.0, 1.0])}
        refs = {"bird": np.array([1.0, 0.0]), "moon": np.array([0.0, 1.0])}

        anchors = predict_iconic_anchors(glyphs, refs, top_k=2, min_iconicity=0.8)

        assert [c.referent_id for c in anchors["RR_600"]] == ["bird"]
        assert anchors["RR_noise"] == []

    def test_evaluate_anchor_ranking(self):
        glyphs = {
            "g_bird": np.array([1.0, 0.0, 0.0]),
            "g_moon": np.array([0.0, 1.0, 0.0]),
        }
        refs = {
            "bird": np.array([1.0, 0.0, 0.0]),
            "moon": np.array([0.0, 1.0, 0.0]),
            "fish": np.array([0.0, 0.0, 1.0]),
        }

        ranked = rank_iconic_candidates(glyphs, refs, top_k=3)
        metrics = evaluate_anchor_ranking(
            ranked, {"g_bird": "bird", "g_moon": "moon"}, k_values=(1, 3)
        )

        assert metrics["accuracy@1"] == pytest.approx(1.0)
        assert metrics["accuracy@3"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)


class TestTheoremHelpers:
    def test_referent_separation_and_bound(self):
        refs = np.array([[1.0, 0.0], [0.0, 1.0]])
        delta_r = referent_separation(refs)
        bound = iconic_recovery_probability_bound(
            coverage_epsilon=0.1,
            lipschitz_constant=0.2,
            deiconization=0.1,
            delta_r=delta_r,
        )

        assert delta_r == pytest.approx(np.sqrt(2.0))
        assert 0.8 < bound < 1.0

    def test_referent_separation_detects_duplicate_referents(self):
        refs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        assert referent_separation(refs) == pytest.approx(0.0)

    def test_visual_diameter(self):
        diameter = visual_diameter(np.array([[1.0, 0.0], [0.0, 1.0]]))

        assert diameter == pytest.approx(np.sqrt(2.0))


class TestC25Admissibility:
    def test_c25_admitted_when_all_criteria_pass(self):
        evidence = IconicClaimEvidence(
            iota_max=0.75,
            anchor_power=0.2,
            bootstrap_stability=0.8,
            cross_script_acc_at_5=0.7,
            negative_control_gap=3.5,
            in_world_reconstruction=True,
            bibliographic_sources=2,
        )

        decision = assess_c25_admissibility(evidence)

        assert decision.admissible
        assert decision.max_claim_label == C25_CLAIM_LABEL
        assert decision.failed_criteria == ()

    def test_c25_blocks_and_reports_failed_criteria(self):
        evidence = IconicClaimEvidence(
            iota_max=0.55,
            anchor_power=0.2,
            bootstrap_stability=0.65,
            cross_script_acc_at_5=0.7,
            negative_control_gap=2.0,
            in_world_reconstruction=True,
            bibliographic_sources=1,
        )

        decision = assess_c25_admissibility(evidence)

        assert not decision.admissible
        assert "iconicity" in decision.failed_criteria
        assert "negative_control_gap" in decision.failed_criteria
        assert "bibliographic_sources" in decision.failed_criteria


class TestAllographsAndAlignment:
    def test_detect_allographs_groups_similar_variants(self):
        glyphs = {
            "A": np.array([1.0, 0.0]),
            "A_variant": np.array([0.99, 0.05]),
            "B": np.array([0.0, 1.0]),
        }

        clusters = detect_allographs(glyphs, threshold=0.95)

        assert clusters["A"] == clusters["A_variant"]
        assert clusters["A"] != clusters["B"]

    def test_anchor_assignment_stability(self):
        stability = anchor_assignment_stability(
            [
                {"g1": "bird", "g2": "moon"},
                {"g1": "bird", "g2": "moon"},
                {"g1": "bird", "g2": "fish"},
            ]
        )

        assert stability == pytest.approx(2.0 / 3.0)

    def test_align_glyphs_to_referents_requires_explicit_anchors(self):
        glyphs = {"g1": np.array([1.0, 0.0]), "g2": np.array([0.0, 1.0])}
        refs = {"r1": np.array([0.0, 1.0]), "r2": np.array([-1.0, 0.0])}

        result = align_glyphs_to_referents(
            glyphs, refs, anchors=[("g1", "r1"), ("g2", "r2")]
        )
        ranked = rank_iconic_candidates(result.aligned_glyph_embeddings, refs, top_k=1)

        assert ranked["g1"][0].referent_id == "r1"
        assert ranked["g2"][0].referent_id == "r2"

    def test_unanchored_alignment_is_rejected(self):
        glyphs = {"g1": np.array([1.0, 0.0])}
        refs = {"r1": np.array([1.0, 0.0])}

        with pytest.raises(ValueError):
            align_glyphs_to_referents(glyphs, refs, anchors=[])
