"""Iconic grounding for visual anchor generation.

This module implements the first executable slice of
``iconic_grounding_guide.md``:

- Rapa Nui 1500 referent reconstruction (Definition 3.1, Section 8).
- Unit-sphere visual embedding utilities (Definitions 3.2-3.4).
- Spherical Frechet means for multi-view glyph/referent consensus.
- Glyph-to-referent iconicity ranking and cross-script metrics.
- Claim C2.5 admissibility checks from Section 13.

The code intentionally accepts precomputed embeddings or lightweight encoder
callables. Heavy vision models such as DINOv2/SigLIP can be plugged in later
without making the core theorem/audit layer depend on GPU packages.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


EPS = 1e-12
C25_CLAIM_LABEL = "C2.5_ICONOGRAPHIC"


@dataclass(frozen=True)
class HistoricalReferent:
    """A historically admissible visual referent.

    Args:
        referent_id: Stable machine-readable identifier.
        label: Human-readable label.
        category: Coarse group: fauna, flora, artifact, celestial, human.
        subcategory: Finer group within the category.
        scientific_name: Optional Latin binomial or conventional name.
        rapa_nui_name: Optional Rapa Nui name when documented.
        expected_iconicity: Qualitative prior from the guide.
        sources: Bibliographic/source notes supporting presence in the world.
        notes: Additional provenance or cultural context.
    """

    referent_id: str
    label: str
    category: str
    subcategory: str
    scientific_name: str | None = None
    rapa_nui_name: str | None = None
    expected_iconicity: str | None = None
    sources: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def source_count(self) -> int:
        return len({s for s in self.sources if s})

    def as_dict(self) -> dict[str, Any]:
        return {
            "referent_id": self.referent_id,
            "label": self.label,
            "category": self.category,
            "subcategory": self.subcategory,
            "scientific_name": self.scientific_name,
            "rapa_nui_name": self.rapa_nui_name,
            "expected_iconicity": self.expected_iconicity,
            "sources": list(self.sources),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class WorldReconstruction:
    """A verifiable historical world ``W_{tau, G}``.

    This is the concrete carrier for Definition 3.1 in the guide.
    """

    world_id: str
    region: str
    period: str
    referents: tuple[HistoricalReferent, ...]
    notes: str = ""

    def get_referent_set(self) -> list[str]:
        """Return the referent IDs forming ``R_{tau, G}``."""

        return [r.referent_id for r in self.referents]

    def by_id(self) -> dict[str, HistoricalReferent]:
        return {r.referent_id: r for r in self.referents}

    def by_category(self) -> dict[str, list[HistoricalReferent]]:
        grouped: dict[str, list[HistoricalReferent]] = defaultdict(list)
        for referent in self.referents:
            grouped[referent.category].append(referent)
        return dict(grouped)

    def validate_min_sources(self, min_sources: int = 2) -> dict[str, Any]:
        """Check bibliographic coverage for the reconstruction.

        Section 8.7 requires at least two independent source notes per
        referent before it can support C2.5 claims.
        """

        missing = [
            r.referent_id for r in self.referents if r.source_count < min_sources
        ]
        return {
            "valid": len(missing) == 0,
            "min_sources": min_sources,
            "n_referents": len(self.referents),
            "missing": missing,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "region": self.region,
            "period": self.period,
            "notes": self.notes,
            "referents": [r.as_dict() for r in self.referents],
        }


def _ref(
    referent_id: str,
    label: str,
    category: str,
    subcategory: str,
    sources: tuple[str, ...],
    scientific_name: str | None = None,
    rapa_nui_name: str | None = None,
    expected_iconicity: str | None = None,
    notes: str = "",
) -> HistoricalReferent:
    return HistoricalReferent(
        referent_id=referent_id,
        label=label,
        category=category,
        subcategory=subcategory,
        scientific_name=scientific_name,
        rapa_nui_name=rapa_nui_name,
        expected_iconicity=expected_iconicity,
        sources=sources,
        notes=notes,
    )


class RapaNuiWorld1500(WorldReconstruction):
    """Operational reconstruction of ``R_{1500, Rapa Nui}``.

    The inventory follows Section 8 of ``iconic_grounding_guide.md``. Source
    strings are provenance hooks, not full bibliographic records; the point is
    to keep every candidate referent auditable until a dedicated datasheet is
    added.
    """

    def __init__(self) -> None:
        fauna_sources = (
            "Steadman et al. 1994 Anakena faunal remains",
            "Hunt and Lipo 2018 Rapa Nui ecology",
        )
        marine_sources = (
            "Metraux 1940 Rapa Nui ethnography",
            "Hunt and Lipo 2018 Rapa Nui ecology",
        )
        flora_sources = (
            "Flenley and King 1984 Rapa Nui pollen evidence",
            "Horrocks and Wozniak 2008 Rapa Nui microfossils",
        )
        artifact_sources = (
            "Metraux 1940 Rapa Nui material culture",
            "Heyerdahl and Ferdon 1961 archaeological reports",
        )
        celestial_sources = (
            "Esen-Baur 1990 Rapa Nui astronomy",
            "Edwards and Edwards 2013 Rapa Nui archaeoastronomy",
        )
        human_sources = (
            "Metraux 1940 Rapa Nui iconography",
            "Lee 1992 Rapa Nui rock art corpus",
        )

        referents = (
            _ref(
                "great_frigatebird",
                "great frigatebird",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Fregata minor",
                rapa_nui_name="taha",
                expected_iconicity="high",
                notes="Associated with Makemake in the guide.",
            ),
            _ref(
                "masked_booby",
                "masked booby",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Sula dactylatra",
                expected_iconicity="medium",
            ),
            _ref(
                "red_tailed_tropicbird",
                "red-tailed tropicbird",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Phaethon rubricauda",
                rapa_nui_name="tavake",
                expected_iconicity="high",
            ),
            _ref(
                "great_winged_petrel",
                "great-winged petrel",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Pterodroma macroptera",
                expected_iconicity="medium",
            ),
            _ref(
                "sooty_tern",
                "sooty tern",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Sterna fuscata",
                expected_iconicity="medium",
            ),
            _ref(
                "fairy_tern",
                "fairy tern",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Gygis alba",
                rapa_nui_name="manu tara",
                expected_iconicity="high",
                notes="Central to the Tangata Manu cult in the guide.",
            ),
            _ref(
                "brown_noddy",
                "brown noddy",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Anous stolidus",
                expected_iconicity="medium",
            ),
            _ref(
                "wedge_tailed_shearwater",
                "wedge-tailed shearwater",
                "fauna",
                "marine_bird",
                fauna_sources,
                scientific_name="Puffinus pacificus",
                expected_iconicity="medium",
            ),
            _ref(
                "domestic_chicken",
                "domestic chicken",
                "fauna",
                "land_bird",
                fauna_sources,
                scientific_name="Gallus gallus",
                expected_iconicity="high",
            ),
            _ref(
                "extinct_parrot",
                "extinct parrot",
                "fauna",
                "land_bird",
                fauna_sources,
                scientific_name="Cyanoramphus sp.",
                expected_iconicity="medium",
            ),
            _ref(
                "rail",
                "rail",
                "fauna",
                "land_bird",
                fauna_sources,
                scientific_name="Porzana sp.",
                expected_iconicity="medium",
            ),
            _ref(
                "heron",
                "heron",
                "fauna",
                "land_bird",
                fauna_sources,
                scientific_name="Ardea sp.",
                expected_iconicity="medium",
            ),
            _ref(
                "tiger_shark",
                "tiger shark",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Galeocerdo cuvier",
                expected_iconicity="high",
            ),
            _ref(
                "yellowfin_tuna",
                "yellowfin tuna",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Thunnus albacares",
                expected_iconicity="high",
            ),
            _ref(
                "bigeye_tuna",
                "bigeye tuna",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Thunnus obesus",
                expected_iconicity="high",
            ),
            _ref(
                "moray_eel",
                "moray eel",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Gymnothorax sp.",
                expected_iconicity="high",
            ),
            _ref(
                "green_sea_turtle",
                "green sea turtle",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Chelonia mydas",
                expected_iconicity="high",
            ),
            _ref(
                "hawksbill_turtle",
                "hawksbill turtle",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Eretmochelys imbricata",
                expected_iconicity="high",
            ),
            _ref(
                "octopus",
                "octopus",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Octopus spp.",
                expected_iconicity="high",
            ),
            _ref(
                "spiny_lobster_pascuensis",
                "Rapa Nui spiny lobster",
                "fauna",
                "marine_life",
                marine_sources,
                scientific_name="Panulirus pascuensis",
                expected_iconicity="medium",
            ),
            _ref(
                "parrotfish",
                "parrotfish",
                "fauna",
                "marine_life",
                marine_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "snapper",
                "snapper",
                "fauna",
                "marine_life",
                marine_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "polynesian_rat",
                "Polynesian rat",
                "fauna",
                "mammal",
                fauna_sources,
                scientific_name="Rattus exulans",
                rapa_nui_name="kiore",
                expected_iconicity="medium",
            ),
            _ref(
                "easter_island_palm",
                "Easter Island palm",
                "flora",
                "palm",
                flora_sources,
                scientific_name="Paschalococos disperta",
                expected_iconicity="high",
            ),
            _ref(
                "toromiro",
                "toromiro",
                "flora",
                "tree",
                flora_sources,
                scientific_name="Sophora toromiro",
                expected_iconicity="medium",
            ),
            _ref(
                "hau_hau",
                "hau hau",
                "flora",
                "tree",
                flora_sources,
                scientific_name="Triumfetta semitriloba",
                expected_iconicity="medium",
            ),
            _ref(
                "paper_mulberry",
                "paper mulberry",
                "flora",
                "crop",
                flora_sources,
                scientific_name="Broussonetia papyrifera",
                expected_iconicity="medium",
            ),
            _ref(
                "taro",
                "taro",
                "flora",
                "crop",
                flora_sources,
                scientific_name="Colocasia esculenta",
                expected_iconicity="medium",
            ),
            _ref(
                "sweet_potato",
                "sweet potato",
                "flora",
                "crop",
                flora_sources,
                scientific_name="Ipomoea batatas",
                rapa_nui_name="kumara",
                expected_iconicity="medium",
            ),
            _ref(
                "banana",
                "banana",
                "flora",
                "crop",
                flora_sources,
                scientific_name="Musa sp.",
                expected_iconicity="medium",
            ),
            _ref(
                "sugarcane",
                "sugarcane",
                "flora",
                "crop",
                flora_sources,
                scientific_name="Saccharum officinarum",
                expected_iconicity="medium",
            ),
            _ref(
                "yam",
                "yam",
                "flora",
                "crop",
                flora_sources,
                scientific_name="Dioscorea sp.",
                expected_iconicity="medium",
            ),
            _ref(
                "coprosma",
                "coprosma",
                "flora",
                "shrub",
                flora_sources,
                scientific_name="Coprosma spp.",
                expected_iconicity="low",
            ),
            _ref(
                "moai",
                "moai",
                "artifact",
                "monument",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "reimiro",
                "reimiro",
                "artifact",
                "ornament",
                artifact_sources,
                expected_iconicity="high",
                notes="Crescent-shaped pectoral.",
            ),
            _ref(
                "ao",
                "ao ceremonial paddle",
                "artifact",
                "ceremonial_object",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "hami",
                "hami ceremonial belt",
                "artifact",
                "ceremonial_object",
                artifact_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "mataa",
                "mataa obsidian point",
                "artifact",
                "tool",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "toki",
                "toki adze",
                "artifact",
                "tool",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "hare_paenga",
                "hare paenga boat house",
                "artifact",
                "architecture",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "tahonga",
                "tahonga pendant",
                "artifact",
                "ornament",
                artifact_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "paoa",
                "paoa club",
                "artifact",
                "weapon",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "rapa",
                "rapa dance paddle",
                "artifact",
                "ceremonial_object",
                artifact_sources,
                expected_iconicity="high",
            ),
            _ref(
                "sun",
                "sun",
                "celestial",
                "body",
                celestial_sources,
                rapa_nui_name="raa",
                expected_iconicity="high",
            ),
            _ref(
                "moon",
                "moon",
                "celestial",
                "body",
                celestial_sources,
                rapa_nui_name="mahina",
                expected_iconicity="high",
            ),
            _ref(
                "pleiades",
                "Pleiades",
                "celestial",
                "asterism",
                celestial_sources,
                rapa_nui_name="matariki",
                expected_iconicity="medium",
            ),
            _ref(
                "venus",
                "Venus",
                "celestial",
                "planet",
                celestial_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "southern_cross",
                "Southern Cross",
                "celestial",
                "asterism",
                celestial_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "hand",
                "hand",
                "human",
                "body_part",
                human_sources,
                rapa_nui_name="rima",
                expected_iconicity="high",
            ),
            _ref(
                "foot",
                "foot",
                "human",
                "body_part",
                human_sources,
                rapa_nui_name="vae",
                expected_iconicity="high",
            ),
            _ref(
                "head",
                "head",
                "human",
                "body_part",
                human_sources,
                rapa_nui_name="puoko",
                expected_iconicity="high",
            ),
            _ref(
                "seated_posture",
                "seated posture",
                "human",
                "posture",
                human_sources,
                expected_iconicity="high",
            ),
            _ref(
                "standing_posture",
                "standing posture",
                "human",
                "posture",
                human_sources,
                expected_iconicity="medium",
            ),
            _ref(
                "arms_raised",
                "arms raised posture",
                "human",
                "posture",
                human_sources,
                expected_iconicity="high",
            ),
            _ref(
                "fertility_genitals",
                "fertility/genital sign",
                "human",
                "body_part",
                human_sources,
                expected_iconicity="medium",
            ),
        )

        super().__init__(
            world_id="rapa_nui_1500",
            region="Rapa Nui",
            period="circa 1500 CE",
            referents=referents,
            notes="Operational referent set from iconic_grounding_guide.md Section 8.",
        )


def _as_2d(vectors: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 1D or 2D embedding array")
    return arr


def l2_normalize(
    vectors: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    axis: int = -1,
    eps: float = EPS,
) -> np.ndarray:
    """L2-normalize vectors and reject zero vectors."""

    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot normalize zero-length embedding vectors")
    return arr / norms


def spherical_mean(
    vectors: np.ndarray | Sequence[Sequence[float]],
    weights: Sequence[float] | np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute a Frechet mean on the unit sphere.

    This follows the fixed-point tangent-space algorithm in Section 9.3.
    """

    X = l2_normalize(_as_2d(vectors), axis=1)
    n = X.shape[0]
    if n == 0:
        raise ValueError("At least one vector is required")
    if n == 1:
        return X[0].copy()

    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must have one entry per vector")
        if np.any(w < 0) or float(w.sum()) <= EPS:
            raise ValueError("weights must be non-negative with positive sum")
        w = w / w.sum()

    mean = np.sum(X * w[:, None], axis=0)
    if np.linalg.norm(mean) <= EPS:
        mean = X[0].copy()
    else:
        mean = l2_normalize(mean)

    for _ in range(max_iter):
        tangents = []
        for v in X:
            cos_theta = float(np.clip(np.dot(mean, v), -1.0, 1.0))
            theta = float(np.arccos(cos_theta))
            if theta < 1e-10:
                tangents.append(np.zeros_like(v))
            else:
                tangents.append(theta * (v - cos_theta * mean) / np.sin(theta))

        tangent_mean = np.sum(np.asarray(tangents) * w[:, None], axis=0)
        norm_t = float(np.linalg.norm(tangent_mean))
        if norm_t < tol:
            break

        new_mean = (
            np.cos(norm_t) * mean + np.sin(norm_t) * tangent_mean / norm_t
        )
        new_mean = l2_normalize(new_mean)
        if np.linalg.norm(new_mean - mean) < tol:
            mean = new_mean
            break
        mean = new_mean

    return mean


def weighted_spherical_mean(
    vectors: np.ndarray | Sequence[Sequence[float]],
    weights: Sequence[float] | np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Convenience wrapper for weighted multi-view consensus."""

    return spherical_mean(vectors, weights=weights, max_iter=max_iter, tol=tol)


def embedding_dispersion(vectors: np.ndarray, center: np.ndarray | None = None) -> float:
    """Mean geodesic spread around a consensus embedding."""

    X = l2_normalize(_as_2d(vectors), axis=1)
    c = spherical_mean(X) if center is None else l2_normalize(center)
    dots = np.clip(X @ c, -1.0, 1.0)
    return float(np.mean(np.arccos(dots)))


@dataclass(frozen=True)
class EmbeddingConsensus:
    embedding: np.ndarray
    dispersion: float
    n_items: int


def embedding_consensus(
    items: Sequence[Any],
    encoder: Callable[[Any], np.ndarray],
    weights: Sequence[float] | np.ndarray | None = None,
) -> EmbeddingConsensus:
    """Encode many views/images and return a spherical consensus."""

    if len(items) == 0:
        raise ValueError("At least one item is required for consensus")
    embeddings = np.vstack([np.asarray(encoder(item), dtype=float) for item in items])
    center = spherical_mean(embeddings, weights=weights)
    return EmbeddingConsensus(
        embedding=center,
        dispersion=embedding_dispersion(embeddings, center=center),
        n_items=len(items),
    )


def glyph_to_embedding(
    glyph_views: Sequence[Any],
    encoder: Callable[[Any], np.ndarray],
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute a consensus glyph embedding from rendered/augmented views."""

    return embedding_consensus(glyph_views, encoder, weights=weights).embedding


def referent_to_embedding(
    referent_images: Sequence[Any],
    encoder: Callable[[Any], np.ndarray],
    weights: Sequence[float] | np.ndarray | None = None,
) -> EmbeddingConsensus:
    """Compute a referent consensus embedding and dispersion estimate."""

    return embedding_consensus(referent_images, encoder, weights=weights)


def _embedding_table(
    embeddings: Mapping[str, np.ndarray] | np.ndarray,
    ids: Sequence[str] | None,
    prefix: str,
) -> tuple[list[str], np.ndarray]:
    if isinstance(embeddings, Mapping):
        labels = list(embeddings.keys())
        if len(labels) == 0:
            raise ValueError(f"{prefix} embeddings cannot be empty")
        matrix = np.vstack([np.asarray(embeddings[label], dtype=float) for label in labels])
    else:
        matrix = _as_2d(embeddings)
        labels = list(ids) if ids is not None else [
            f"{prefix}_{i}" for i in range(matrix.shape[0])
        ]

    if len(labels) != matrix.shape[0]:
        raise ValueError(f"{prefix} ids must match embedding rows")
    return labels, l2_normalize(matrix, axis=1)


def cosine_similarity_matrix(
    glyph_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    referent_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    glyph_ids: Sequence[str] | None = None,
    referent_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute ``iota(x, r) = <Phi(x), Psi(r)>`` for all pairs."""

    _, G = _embedding_table(glyph_embeddings, glyph_ids, "glyph")
    _, R = _embedding_table(referent_embeddings, referent_ids, "referent")
    if G.shape[1] != R.shape[1]:
        raise ValueError("Glyph and referent embeddings must share a dimension")
    return G @ R.T


def geodesic_distance_matrix(
    glyph_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    referent_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    glyph_ids: Sequence[str] | None = None,
    referent_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Compute spherical geodesic distances for all glyph/referent pairs."""

    sim = cosine_similarity_matrix(
        glyph_embeddings, referent_embeddings, glyph_ids, referent_ids
    )
    return np.arccos(np.clip(sim, -1.0, 1.0))


def deiconization_rate(iconicity: float) -> float:
    """Definition 3.4: ``delta(x) = 1 - iota*(x)``."""

    return float(1.0 - iconicity)


@dataclass(frozen=True)
class IconicAnchorCandidate:
    glyph_id: str
    referent_id: str
    score: float
    rank: int
    geodesic_distance: float
    deiconization_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "glyph_id": self.glyph_id,
            "referent_id": self.referent_id,
            "score": self.score,
            "rank": self.rank,
            "geodesic_distance": self.geodesic_distance,
            "deiconization_rate": self.deiconization_rate,
        }


def rank_iconic_candidates(
    glyph_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    referent_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    glyph_ids: Sequence[str] | None = None,
    referent_ids: Sequence[str] | None = None,
    top_k: int = 5,
) -> dict[str, list[IconicAnchorCandidate]]:
    """Rank candidate referents for each glyph by iconicity."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    g_ids, G = _embedding_table(glyph_embeddings, glyph_ids, "glyph")
    r_ids, R = _embedding_table(referent_embeddings, referent_ids, "referent")
    if G.shape[1] != R.shape[1]:
        raise ValueError("Glyph and referent embeddings must share a dimension")

    sim = G @ R.T
    dist = np.arccos(np.clip(sim, -1.0, 1.0))
    k = min(top_k, len(r_ids))

    ranked: dict[str, list[IconicAnchorCandidate]] = {}
    for i, glyph_id in enumerate(g_ids):
        order = np.argsort(-sim[i])[:k]
        ranked[glyph_id] = [
            IconicAnchorCandidate(
                glyph_id=glyph_id,
                referent_id=r_ids[j],
                score=float(sim[i, j]),
                rank=rank,
                geodesic_distance=float(dist[i, j]),
                deiconization_rate=deiconization_rate(float(sim[i, j])),
            )
            for rank, j in enumerate(order, start=1)
        ]
    return ranked


def predict_iconic_anchors(
    glyph_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    referent_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    glyph_ids: Sequence[str] | None = None,
    referent_ids: Sequence[str] | None = None,
    top_k: int = 5,
    min_iconicity: float = 0.6,
) -> dict[str, list[IconicAnchorCandidate]]:
    """Return top-k candidates that clear the iconicity threshold."""

    ranked = rank_iconic_candidates(
        glyph_embeddings,
        referent_embeddings,
        glyph_ids=glyph_ids,
        referent_ids=referent_ids,
        top_k=top_k,
    )
    return {
        glyph_id: [c for c in candidates if c.score >= min_iconicity]
        for glyph_id, candidates in ranked.items()
    }


def evaluate_anchor_ranking(
    predictions: Mapping[str, Sequence[IconicAnchorCandidate]],
    gold: Mapping[str, str],
    k_values: Sequence[int] = (1, 5),
) -> dict[str, float]:
    """Evaluate blind cross-script recovery with Acc@K and MRR."""

    if len(gold) == 0:
        raise ValueError("gold mapping cannot be empty")

    ranks = []
    for glyph_id, true_ref in gold.items():
        candidates = predictions.get(glyph_id, [])
        rank = None
        for candidate in candidates:
            if candidate.referent_id == true_ref:
                rank = candidate.rank
                break
        ranks.append(rank)

    metrics: dict[str, float] = {"n": float(len(ranks))}
    for k in k_values:
        metrics[f"accuracy@{k}"] = float(
            np.mean([rank is not None and rank <= k for rank in ranks])
        )
    reciprocal = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    metrics["mrr"] = float(np.mean(reciprocal))
    return metrics


def visual_diameter(embeddings: Mapping[str, np.ndarray] | np.ndarray) -> float:
    """Maximum Euclidean distance among normalized embedding vectors."""

    _, X = _embedding_table(embeddings, None, "embedding")
    if X.shape[0] < 2:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=2)))


def referent_separation(referent_embeddings: Mapping[str, np.ndarray] | np.ndarray) -> float:
    """Compute ``Delta_r`` from Theorem 3.5."""

    _, R = _embedding_table(referent_embeddings, None, "referent")
    if R.shape[0] < 2:
        return float("inf")
    diffs = R[:, None, :] - R[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def delta_star(delta0: float, lipschitz_constant: float, glyph_diameter: float) -> float:
    """Threshold ``delta_*`` from Theorem 3.5."""

    return float(delta0 + lipschitz_constant * glyph_diameter)


def iconic_recovery_probability_bound(
    coverage_epsilon: float,
    lipschitz_constant: float,
    deiconization: float,
    delta_r: float,
) -> float:
    """Lower bound from Theorem 3.5, clipped to ``[0, 1]``."""

    if delta_r <= EPS:
        return 0.0
    if np.isinf(delta_r):
        raw = 1.0 - coverage_epsilon
    else:
        raw = 1.0 - coverage_epsilon - (lipschitz_constant * deiconization / delta_r)
    return float(np.clip(raw, 0.0, 1.0))


def anchor_power_from_counts(vocab_size: int, anchored_count: int) -> float:
    """Compute AnchorPower for ``Sym(n)`` with ``m`` fixed anchors.

    This log-gamma implementation avoids constructing huge factorial integers
    for real vocabularies.
    """

    if vocab_size <= 1:
        return 1.0
    if anchored_count < 0:
        raise ValueError("anchored_count must be non-negative")
    if anchored_count > vocab_size:
        raise ValueError("anchored_count cannot exceed vocab_size")

    log_aut = math.lgamma(vocab_size + 1)
    log_anchored = math.lgamma(vocab_size - anchored_count + 1)
    if log_aut <= EPS:
        return 1.0
    return float(np.clip(1.0 - (log_anchored / log_aut), 0.0, 1.0))


@dataclass(frozen=True)
class IconicClaimEvidence:
    """Evidence vector for Section 13 claim C2.5."""

    iota_max: float
    anchor_power: float
    bootstrap_stability: float
    cross_script_acc_at_5: float
    negative_control_gap: float
    in_world_reconstruction: bool
    bibliographic_sources: int


@dataclass(frozen=True)
class IconicClaimDecision:
    requested_label: str
    admissible: bool
    max_claim_label: str
    failed_criteria: tuple[str, ...]
    criteria: dict[str, bool]


def assess_c25_admissibility(
    evidence: IconicClaimEvidence,
    min_iconicity: float = 0.6,
    min_anchor_power: float = 0.15,
    min_bootstrap_stability: float = 0.7,
    min_cross_script_acc_at_5: float = 0.6,
    min_negative_control_gap: float = 3.0,
    min_bibliographic_sources: int = 2,
) -> IconicClaimDecision:
    """Apply Definition 13.1 for the new C2.5 iconographic claim."""

    criteria = {
        "iconicity": evidence.iota_max >= min_iconicity,
        "anchor_power": evidence.anchor_power >= min_anchor_power,
        "bootstrap_stability": (
            evidence.bootstrap_stability >= min_bootstrap_stability
        ),
        "cross_script_validation": (
            evidence.cross_script_acc_at_5 >= min_cross_script_acc_at_5
        ),
        "negative_control_gap": (
            evidence.negative_control_gap >= min_negative_control_gap
        ),
        "world_coverage": evidence.in_world_reconstruction,
        "bibliographic_sources": (
            evidence.bibliographic_sources >= min_bibliographic_sources
        ),
    }
    failed = tuple(name for name, ok in criteria.items() if not ok)
    admissible = len(failed) == 0
    return IconicClaimDecision(
        requested_label=C25_CLAIM_LABEL,
        admissible=admissible,
        max_claim_label=C25_CLAIM_LABEL if admissible else "C2_FUNCTIONAL_OR_LOWER",
        failed_criteria=failed,
        criteria=criteria,
    )


def anchor_assignment_stability(assignments: Sequence[Mapping[str, str]]) -> float:
    """Bootstrap stability for top-1 iconographic assignments.

    Returns the mean pairwise agreement on glyphs shared by bootstrap samples.
    """

    if len(assignments) < 2:
        return float("nan")

    agreements = []
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            common = set(assignments[i]).intersection(assignments[j])
            if not common:
                continue
            agreements.append(
                np.mean([
                    assignments[i][glyph_id] == assignments[j][glyph_id]
                    for glyph_id in common
                ])
            )

    if not agreements:
        return float("nan")
    return float(np.mean(agreements))


def detect_allographs(
    glyph_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    glyph_ids: Sequence[str] | None = None,
    threshold: float = 0.85,
) -> dict[str, int]:
    """Cluster visually similar glyph variants into allograph classes."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    labels, X = _embedding_table(glyph_embeddings, glyph_ids, "glyph")
    if len(labels) == 1:
        return {labels[0]: 1}

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sim = np.clip(X @ X.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return {label: int(cluster) for label, cluster in zip(labels, clusters)}


@dataclass(frozen=True)
class ProcrustesAlignmentResult:
    aligned_glyph_embeddings: dict[str, np.ndarray]
    rotation: np.ndarray
    anchor_condition: float


def align_glyphs_to_referents(
    glyph_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    referent_embeddings: Mapping[str, np.ndarray] | np.ndarray,
    anchors: Sequence[tuple[str, str]],
    glyph_ids: Sequence[str] | None = None,
    referent_ids: Sequence[str] | None = None,
) -> ProcrustesAlignmentResult:
    """Align glyph embeddings to referents using explicit anchors.

    The unanchored case is deliberately not implemented here: the previous
    no-identifiability theorem says we should not silently infer an absolute
    semantic frame without external anchors.
    """

    if len(anchors) == 0:
        raise ValueError("At least one explicit anchor is required")

    from .alignment import orthogonal_procrustes
    from .identifiability import anchor_condition_number

    g_ids, G = _embedding_table(glyph_embeddings, glyph_ids, "glyph")
    r_ids, R = _embedding_table(referent_embeddings, referent_ids, "referent")
    if G.shape[1] != R.shape[1]:
        raise ValueError("Glyph and referent embeddings must share a dimension")

    g_index = {glyph_id: i for i, glyph_id in enumerate(g_ids)}
    r_index = {referent_id: i for i, referent_id in enumerate(r_ids)}
    try:
        X_anchor = np.vstack([G[g_index[glyph_id]] for glyph_id, _ in anchors])
        Y_anchor = np.vstack([R[r_index[referent_id]] for _, referent_id in anchors])
    except KeyError as exc:
        raise KeyError(f"Unknown anchor id: {exc}") from exc

    Q = orthogonal_procrustes(X_anchor, Y_anchor)
    aligned = l2_normalize(G @ Q, axis=1)
    return ProcrustesAlignmentResult(
        aligned_glyph_embeddings={
            glyph_id: aligned[i] for i, glyph_id in enumerate(g_ids)
        },
        rotation=Q,
        anchor_condition=anchor_condition_number(X_anchor, Y_anchor),
    )
