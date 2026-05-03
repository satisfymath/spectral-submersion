"""Corpus ledger with uncertainty and paleographic metadata.

Implements the data structures from Sections 14-16 of the guide:
- glyph_instances.jsonl
- artifacts.jsonl
- uncertain_readings.jsonl
- parallel_passages.jsonl
- hypothesis ledger entries

Each glyph instance carries uncertainty, direction, damage, and provenance.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_name: str
    artifact_type: str
    source_refs: list[str] = field(default_factory=list)
    museum_id: str = ""
    dimensions_mm: list[float] | None = None
    material: str = ""
    dating: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GlyphInstanceRecord:
    glyph_instance_id: str
    artifact_id: str
    side: str
    line: int
    position_in_line: int
    global_position: int
    direction: str = "unknown"
    barthel_code: str = ""
    alternative_codes: list[dict] = field(default_factory=list)
    bbox_2d: list[float] | None = None
    surface_coordinates_3d: list[float] | None = None
    damage_score: float = 0.0
    reading_uncertainty: float = 0.0
    is_ligature: bool = False
    possible_components: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None and v != "" and v != []}
        return d


@dataclass
class UncertainReadingRecord:
    glyph_instance_id: str
    reading_system: str
    code: str
    confidence: float
    paleographic_notes: str = ""
    image_ref: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v}
        return d


@dataclass
class ParallelPassageRecord:
    passage_id: str
    artifact_id: str
    side: str
    line_start: int
    line_end: int
    glyph_start: int
    glyph_end: int
    glyph_sequence: list[str]
    parallel_passage_id: str = ""
    edit_similarity: float = 0.0
    parallel_type: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DirectionMetadataRecord:
    artifact_id: str
    side: str
    line: int
    direction: str
    confidence: float = 1.0
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CorpusLedger:
    """Full corpus ledger following the data structure from Section 14.

    Manages artifacts, glyph instances, uncertain readings,
    parallel passages, and direction metadata as JSONL files.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.artifacts: list[ArtifactRecord] = []
        self.glyph_instances: list[GlyphInstanceRecord] = []
        self.uncertain_readings: list[UncertainReadingRecord] = []
        self.parallel_passages: list[ParallelPassageRecord] = []
        self.direction_metadata: list[DirectionMetadataRecord] = []

    def add_artifact(self, record: ArtifactRecord) -> None:
        self.artifacts.append(record)

    def add_glyph_instance(self, record: GlyphInstanceRecord) -> None:
        self.glyph_instances.append(record)

    def add_uncertain_reading(self, record: UncertainReadingRecord) -> None:
        self.uncertain_readings.append(record)

    def add_parallel_passage(self, record: ParallelPassageRecord) -> None:
        self.parallel_passages.append(record)

    def add_direction_metadata(self, record: DirectionMetadataRecord) -> None:
        self.direction_metadata.append(record)

    def save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

        for name, records in [
            ("artifacts", self.artifacts),
            ("glyph_instances", self.glyph_instances),
            ("uncertain_readings", self.uncertain_readings),
            ("parallel_passages", self.parallel_passages),
            ("direction_metadata", self.direction_metadata),
        ]:
            path = self.base_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> None:
        self.artifacts.clear()
        self.glyph_instances.clear()
        self.uncertain_readings.clear()
        self.parallel_passages.clear()
        self.direction_metadata.clear()

        for name, record_class, target_list in [
            ("artifacts", ArtifactRecord, self.artifacts),
            ("glyph_instances", GlyphInstanceRecord, self.glyph_instances),
            ("uncertain_readings", UncertainReadingRecord, self.uncertain_readings),
            ("parallel_passages", ParallelPassageRecord, self.parallel_passages),
            ("direction_metadata", DirectionMetadataRecord, self.direction_metadata),
        ]:
            path = self.base_dir / f"{name}.jsonl"
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        try:
                            target_list.append(record_class(**d))
                        except TypeError:
                            target_list.append(d)

    def summary(self) -> dict:
        return {
            "n_artifacts": len(self.artifacts),
            "n_glyph_instances": len(self.glyph_instances),
            "n_uncertain_readings": len(self.uncertain_readings),
            "n_parallel_passages": len(self.parallel_passages),
            "n_direction_metadata": len(self.direction_metadata),
        }

    @staticmethod
    def config_hash(config: dict) -> str:
        s = json.dumps(config, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]


def build_glyph_instance_id(
    artifact_id: str, side: str, line: int, position: int
) -> str:
    return f"RR_{artifact_id}_{side}_{line:02d}_{position:03d}"


def glyph_instances_from_sequences(
    sequences: list[list[int]],
    vocab: dict[str, int],
    artifact_id: str = "SYNTH",
    side: str = "r",
    direction: str = "unknown",
    barthel_prefix: str = "",
    damage_score: float = 0.0,
    reading_uncertainty: float = 0.0,
) -> list[GlyphInstanceRecord]:
    """Convert plain sequences to glyph instance records.

    Args:
        sequences: List of per-line token-ID sequences.
        vocab: Token-to-index mapping.
        artifact_id: Artifact identifier.
        side: Side (r/l).
        direction: Reading direction.
        barthel_prefix: Prefix for Barthel codes.
        damage_score: Default damage score.
        reading_uncertainty: Default uncertainty.

    Returns:
        List of GlyphInstanceRecord.
    """
    inv_vocab = {v: k for k, v in vocab.items()}
    records = []
    global_pos = 0

    for line_idx, line_tokens in enumerate(sequences, start=1):
        for pos_idx, token_id in enumerate(line_tokens, start=1):
            token_str = inv_vocab.get(token_id, f"UNK_{token_id}")
            record = GlyphInstanceRecord(
                glyph_instance_id=build_glyph_instance_id(
                    artifact_id, side, line_idx, pos_idx
                ),
                artifact_id=artifact_id,
                side=side,
                line=line_idx,
                position_in_line=pos_idx,
                global_position=global_pos,
                direction=direction,
                barthel_code=f"{barthel_prefix}{token_str}" if barthel_prefix else token_str,
                damage_score=damage_score,
                reading_uncertainty=reading_uncertainty,
                source_refs=["synthetic"],
            )
            records.append(record)
            global_pos += 1

    return records