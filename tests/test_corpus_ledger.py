"""Tests for corpus ledger: data structures and I/O."""

from pathlib import Path
from spectral_submersion.corpus_ledger import (
    ArtifactRecord,
    CorpusLedger,
    GlyphInstanceRecord,
    build_glyph_instance_id,
    glyph_instances_from_sequences,
)


class TestGlyphInstanceId:
    def test_format(self):
        gid = build_glyph_instance_id("A", "r", 3, 17)
        assert gid == "RR_A_r_03_017"


class TestGlyphInstanceRecord:
    def test_to_dict_removes_empty(self):
        r = GlyphInstanceRecord(
            glyph_instance_id="RR_A_r_03_017",
            artifact_id="A",
            side="r",
            line=3,
            position_in_line=17,
            global_position=231,
        )
        d = r.to_dict()
        assert d["glyph_instance_id"] == "RR_A_r_03_017"
        assert "alternative_codes" not in d

    def test_to_dict_keeps_data(self):
        r = GlyphInstanceRecord(
            glyph_instance_id="RR_A_r_03_017",
            artifact_id="A",
            side="r",
            line=3,
            position_in_line=17,
            global_position=231,
            damage_score=0.3,
            reading_uncertainty=0.4,
        )
        d = r.to_dict()
        assert d["damage_score"] == 0.3
        assert d["reading_uncertainty"] == 0.4


class TestCorpusLedger:
    def test_add_and_save(self, tmp_path):
        ledger = CorpusLedger(tmp_path / "ledger")
        artifact = ArtifactRecord(
            artifact_id="A",
            artifact_name="Tahua",
            artifact_type="tablet",
            source_refs=["barthel1958"],
        )
        ledger.add_artifact(artifact)
        glyph = GlyphInstanceRecord(
            glyph_instance_id="RR_A_r_01_001",
            artifact_id="A",
            side="r",
            line=1,
            position_in_line=1,
            global_position=1,
            barthel_code="200",
            damage_score=0.1,
            reading_uncertainty=0.2,
        )
        ledger.add_glyph_instance(glyph)
        ledger.save()

        assert (tmp_path / "ledger" / "artifacts.jsonl").exists()
        assert (tmp_path / "ledger" / "glyph_instances.jsonl").exists()

    def test_summary(self):
        ledger = CorpusLedger(Path("/tmp/test_ledger"))
        ledger.add_artifact(ArtifactRecord("A", "Tahua", "tablet"))
        summary = ledger.summary()
        assert summary["n_artifacts"] == 1
        assert summary["n_glyph_instances"] == 0


class TestGlyphInstancesFromSequences:
    def test_basic_conversion(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        sequences = [[0, 1, 2], [2, 1, 0]]
        records = glyph_instances_from_sequences(sequences, vocab, artifact_id="TEST")
        assert len(records) == 6
        assert records[0].artifact_id == "TEST"
        assert records[0].global_position == 0
        assert records[-1].global_position == 5
