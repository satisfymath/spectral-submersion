"""Real-data adapters for iconic grounding.

The core :mod:`spectral_submersion.iconic_grounding` module is deliberately
model-agnostic. This module connects it to real Rongorongo SVG paths from the
RR-corpus XML files and to real referent image folders on disk.
"""
from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .iconic_grounding import embedding_dispersion, l2_normalize, spherical_mean


_PATH_TOKEN_RE = re.compile(
    r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)
_COMMANDS = set("MmZzLlHhVvCcSsQqTtAa")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RongorongoGlyphSvgInstance:
    """One visual glyph instance extracted from RR-corpus XML."""

    tablet: str
    side_id: str
    line_id: str
    position: int
    glyph_id: str
    raw_code: str
    canonical_code: str
    base_code: str
    link: str
    image_type: str
    path_d: str
    x: float
    y: float
    width: float
    height: float
    source_xml: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tablet": self.tablet,
            "side_id": self.side_id,
            "line_id": self.line_id,
            "position": self.position,
            "glyph_id": self.glyph_id,
            "raw_code": self.raw_code,
            "canonical_code": self.canonical_code,
            "base_code": self.base_code,
            "link": self.link,
            "image_type": self.image_type,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "source_xml": self.source_xml,
        }


@dataclass(frozen=True)
class GlyphEmbeddingMetadata:
    glyph_code: str
    group_by: str
    n_instances_total: int
    n_instances_used: int
    dispersion: float
    raw_codes: tuple[str, ...]
    example_glyph_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "glyph_code": self.glyph_code,
            "group_by": self.group_by,
            "n_instances_total": self.n_instances_total,
            "n_instances_used": self.n_instances_used,
            "dispersion": self.dispersion,
            "raw_codes": "|".join(self.raw_codes),
            "example_glyph_ids": "|".join(self.example_glyph_ids),
        }


@dataclass(frozen=True)
class RealGlyphEmbeddingTable:
    embeddings: dict[str, np.ndarray]
    metadata: dict[str, GlyphEmbeddingMetadata]
    instances_by_code: dict[str, tuple[RongorongoGlyphSvgInstance, ...]]


@dataclass(frozen=True)
class ReferentImageMetadata:
    referent_id: str
    n_images: int
    dispersion: float
    image_paths: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "referent_id": self.referent_id,
            "n_images": self.n_images,
            "dispersion": self.dispersion,
            "image_paths": "|".join(self.image_paths),
        }


@dataclass(frozen=True)
class ReferentImageEmbeddingTable:
    embeddings: dict[str, np.ndarray]
    metadata: dict[str, ReferentImageMetadata]


def canonical_glyph_code(code: str) -> str:
    """Remove uncertainty punctuation while preserving Barthel variants."""

    cleaned = (code or "").strip()
    cleaned = re.sub(r"[!?*]+$", "", cleaned)
    return cleaned


def base_glyph_code(code: str) -> str:
    """Return the three-digit Barthel base code when present."""

    cleaned = canonical_glyph_code(code)
    match = re.search(r"\d{3}", cleaned)
    return match.group(0) if match else cleaned


def _text_or_empty(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def _float_child(element: ET.Element, name: str) -> float:
    text = _text_or_empty(element.find(name))
    return float(text) if text else 0.0


def parse_rongorongo_svg_instances(
    xml_dir: str | Path = "data/external/rongorongo_rr_corpus",
    image_type: str | None = "b",
) -> list[RongorongoGlyphSvgInstance]:
    """Parse real SVG glyph paths from the RR-corpus XML files.

    Args:
        xml_dir: Directory containing tablet XML files.
        image_type: ``"b"`` or ``"f"`` for the corpus' two path variants, or
            ``None`` to include every available image path.
    """

    xml_root = Path(xml_dir)
    instances: list[RongorongoGlyphSvgInstance] = []

    for xml_path in sorted(xml_root.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tablet = root.get("id", xml_path.stem)
        for side in root.findall("side"):
            side_id = side.get("id", "")
            for line in side.findall("line"):
                line_id = line.get("id", "")
                for position, glyph in enumerate(line.findall("glyph"), start=1):
                    glyph_id = glyph.get("id", f"{line_id}-{position:03d}")
                    raw_code = _text_or_empty(glyph.find("code"))
                    link = _text_or_empty(glyph.find("link"))
                    for image in glyph.findall("image"):
                        img_type = image.get("type", "")
                        if image_type is not None and img_type != image_type:
                            continue
                        path_el = image.find("path")
                        path_d = "" if path_el is None else path_el.get("d", "")
                        if not path_d:
                            continue
                        instances.append(
                            RongorongoGlyphSvgInstance(
                                tablet=tablet,
                                side_id=side_id,
                                line_id=line_id,
                                position=position,
                                glyph_id=glyph_id,
                                raw_code=raw_code,
                                canonical_code=canonical_glyph_code(raw_code),
                                base_code=base_glyph_code(raw_code),
                                link=link,
                                image_type=img_type,
                                path_d=path_d,
                                x=_float_child(image, "x"),
                                y=_float_child(image, "y"),
                                width=_float_child(image, "width"),
                                height=_float_child(image, "height"),
                                source_xml=str(xml_path),
                            )
                        )
    return instances


def _tokenize_path(path_d: str) -> list[str]:
    return _PATH_TOKEN_RE.findall(path_d)


def _is_command(token: str) -> bool:
    return token in _COMMANDS


def _cubic(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float
) -> np.ndarray:
    return (
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t**2) * p2
        + (t**3) * p3
    )


def _quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t**2) * p2


def sample_svg_path(
    path_d: str,
    samples_per_curve: int = 12,
) -> list[tuple[list[tuple[float, float]], bool]]:
    """Sample common SVG path commands into polylines.

    Arcs are conservatively approximated by a straight segment to their
    endpoint. The RR-corpus paths are dominated by cubic curves, which are
    sampled explicitly.
    """

    tokens = _tokenize_path(path_d)
    i = 0
    command: str | None = None
    current = np.array([0.0, 0.0])
    start = np.array([0.0, 0.0])
    last_control: np.ndarray | None = None
    last_command: str | None = None
    current_points: list[tuple[float, float]] = []
    subpaths: list[tuple[list[tuple[float, float]], bool]] = []

    def flush(closed: bool = False) -> None:
        nonlocal current_points
        if current_points:
            subpaths.append((current_points, closed))
            current_points = []

    def has_numbers(n: int) -> bool:
        return i + n <= len(tokens) and all(not _is_command(t) for t in tokens[i : i + n])

    def read_numbers(n: int) -> list[float]:
        nonlocal i
        values = [float(t) for t in tokens[i : i + n]]
        i += n
        return values

    def add_point(point: np.ndarray) -> None:
        current_points.append((float(point[0]), float(point[1])))

    while i < len(tokens):
        if _is_command(tokens[i]):
            command = tokens[i]
            i += 1
        if command is None:
            break

        absolute = command.isupper()
        cmd = command.upper()

        if cmd == "M":
            if not has_numbers(2):
                continue
            flush(False)
            x, y = read_numbers(2)
            point = np.array([x, y])
            if not absolute:
                point = current + point
            current = point
            start = point.copy()
            add_point(current)
            last_control = None
            last_command = command
            command = "L" if absolute else "l"
            continue

        if cmd == "Z":
            if current_points and current_points[-1] != (float(start[0]), float(start[1])):
                add_point(start)
            current = start.copy()
            flush(True)
            last_control = None
            last_command = command
            command = None
            continue

        if cmd == "L":
            while has_numbers(2):
                x, y = read_numbers(2)
                point = np.array([x, y])
                if not absolute:
                    point = current + point
                current = point
                add_point(current)
            last_control = None
            last_command = command
            continue

        if cmd == "H":
            while has_numbers(1):
                (x,) = read_numbers(1)
                current = np.array([x if absolute else current[0] + x, current[1]])
                add_point(current)
            last_control = None
            last_command = command
            continue

        if cmd == "V":
            while has_numbers(1):
                (y,) = read_numbers(1)
                current = np.array([current[0], y if absolute else current[1] + y])
                add_point(current)
            last_control = None
            last_command = command
            continue

        if cmd == "C":
            while has_numbers(6):
                x1, y1, x2, y2, x3, y3 = read_numbers(6)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                if not absolute:
                    p1, p2, p3 = current + p1, current + p2, current + p3
                p0 = current.copy()
                for step in range(1, samples_per_curve + 1):
                    add_point(_cubic(p0, p1, p2, p3, step / samples_per_curve))
                current = p3
                last_control = p2
            last_command = command
            continue

        if cmd == "S":
            while has_numbers(4):
                x2, y2, x3, y3 = read_numbers(4)
                if last_control is not None and last_command and last_command.upper() in {"C", "S"}:
                    p1 = current + (current - last_control)
                else:
                    p1 = current.copy()
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                if not absolute:
                    p2, p3 = current + p2, current + p3
                p0 = current.copy()
                for step in range(1, samples_per_curve + 1):
                    add_point(_cubic(p0, p1, p2, p3, step / samples_per_curve))
                current = p3
                last_control = p2
            last_command = command
            continue

        if cmd == "Q":
            while has_numbers(4):
                x1, y1, x2, y2 = read_numbers(4)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                if not absolute:
                    p1, p2 = current + p1, current + p2
                p0 = current.copy()
                for step in range(1, samples_per_curve + 1):
                    add_point(_quadratic(p0, p1, p2, step / samples_per_curve))
                current = p2
                last_control = p1
            last_command = command
            continue

        if cmd == "T":
            while has_numbers(2):
                x2, y2 = read_numbers(2)
                if last_control is not None and last_command and last_command.upper() in {"Q", "T"}:
                    p1 = current + (current - last_control)
                else:
                    p1 = current.copy()
                p2 = np.array([x2, y2])
                if not absolute:
                    p2 = current + p2
                p0 = current.copy()
                for step in range(1, samples_per_curve + 1):
                    add_point(_quadratic(p0, p1, p2, step / samples_per_curve))
                current = p2
                last_control = p1
            last_command = command
            continue

        if cmd == "A":
            while has_numbers(7):
                _rx, _ry, _xrot, _large, _sweep, x, y = read_numbers(7)
                point = np.array([x, y])
                if not absolute:
                    point = current + point
                current = point
                add_point(current)
            last_control = None
            last_command = command
            continue

        break

    flush(False)
    return subpaths


def render_svg_path_to_image(
    path_d: str,
    bbox: tuple[float, float, float, float] | None = None,
    image_size: int = 128,
    padding: int = 8,
    samples_per_curve: int = 12,
):
    """Render a sampled SVG path to a Pillow grayscale image."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Pillow is required for real iconic image rendering") from exc

    subpaths = sample_svg_path(path_d, samples_per_curve=samples_per_curve)
    points = [point for subpath, _ in subpaths for point in subpath]
    image = Image.new("L", (image_size, image_size), 255)
    if not points:
        return image

    if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        width = max(xmax - xmin, 1.0)
        height = max(ymax - ymin, 1.0)

    canvas = max(image_size - 2 * padding, 1)
    scale = canvas / max(width, height, 1e-9)
    x_offset = padding + 0.5 * (canvas - width * scale)
    y_offset = padding + 0.5 * (canvas - height * scale)

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return ((x - xmin) * scale + x_offset, (y - ymin) * scale + y_offset)

    draw = ImageDraw.Draw(image)
    line_width = max(1, image_size // 64)
    for subpath, closed in subpaths:
        if len(subpath) < 2:
            continue
        transformed = [transform(p) for p in subpath]
        if closed and len(transformed) >= 3:
            draw.polygon(transformed, fill=0)
        draw.line(transformed, fill=0, width=line_width, joint="curve")
    return image


def glyph_instance_to_image(
    instance: RongorongoGlyphSvgInstance,
    image_size: int = 128,
    padding: int = 8,
):
    return render_svg_path_to_image(
        instance.path_d,
        bbox=instance.bbox,
        image_size=image_size,
        padding=padding,
    )


def image_shape_embedding(
    image,
    grid_size: int = 32,
    threshold: float = 0.12,
) -> np.ndarray:
    """Extract a deterministic shape embedding from a real glyph/referent image."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Pillow is required for real iconic image features") from exc

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    gray = image.convert("L").resize((grid_size, grid_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=float) / 255.0

    dark_ink = 1.0 - arr
    gy, gx = np.gradient(arr)
    edges = np.sqrt(gx**2 + gy**2)
    if float(edges.max()) > 1e-12:
        edges = edges / float(edges.max())

    # Glyph SVGs are rendered as dark ink on a light background. For natural
    # referent photos, edges add a rough silhouette cue without pretending this
    # lightweight descriptor is a semantic vision model.
    ink = np.maximum(dark_ink, 0.75 * edges)
    ink = np.where(ink >= threshold, ink, 0.0)
    if float(ink.sum()) <= 1e-9:
        ink = np.maximum(dark_ink, edges)

    yy, xx = np.indices(ink.shape)
    mass = float(ink.sum()) + 1e-12
    x_norm = xx / max(grid_size - 1, 1)
    y_norm = yy / max(grid_size - 1, 1)
    cx = float((ink * x_norm).sum() / mass)
    cy = float((ink * y_norm).sum() / mass)
    sx = float(np.sqrt((ink * (x_norm - cx) ** 2).sum() / mass))
    sy = float(np.sqrt((ink * (y_norm - cy) ** 2).sum() / mass))
    cov = float((ink * (x_norm - cx) * (y_norm - cy)).sum() / mass)

    rows = np.where(ink.sum(axis=1) > 1e-6)[0]
    cols = np.where(ink.sum(axis=0) > 1e-6)[0]
    if len(rows) and len(cols):
        bbox_w = (cols.max() - cols.min() + 1) / grid_size
        bbox_h = (rows.max() - rows.min() + 1) / grid_size
    else:
        bbox_w = bbox_h = 0.0

    projections = np.concatenate([ink.sum(axis=0), ink.sum(axis=1)])
    projections = projections / (np.linalg.norm(projections) + 1e-12)
    coarse = np.asarray(
        gray.resize((12, 12), Image.Resampling.LANCZOS), dtype=float
    )
    coarse_ink = 1.0 - coarse / 255.0
    geom = np.array(
        [
            ink.mean(),
            cx,
            cy,
            sx,
            sy,
            cov,
            bbox_w,
            bbox_h,
            bbox_w / (bbox_h + 1e-12),
        ],
        dtype=float,
    )
    vector = np.concatenate([coarse_ink.ravel(), projections, geom])
    return l2_normalize(vector)


def glyph_instance_embedding(
    instance: RongorongoGlyphSvgInstance,
    image_size: int = 128,
    grid_size: int = 32,
) -> np.ndarray:
    image = glyph_instance_to_image(instance, image_size=image_size)
    return image_shape_embedding(image, grid_size=grid_size)


def _select_instances(
    instances: Sequence[RongorongoGlyphSvgInstance],
    max_instances: int | None,
) -> list[RongorongoGlyphSvgInstance]:
    ordered = sorted(instances, key=lambda g: (g.tablet, g.line_id, g.position, g.glyph_id))
    if max_instances is None or len(ordered) <= max_instances:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, max_instances).round().astype(int)
    return [ordered[i] for i in idx]


def build_rongorongo_glyph_embedding_table(
    xml_dir: str | Path = "data/external/rongorongo_rr_corpus",
    image_type: str | None = "b",
    group_by: str = "base_code",
    min_instances: int = 1,
    top_n: int | None = None,
    max_instances_per_glyph: int | None = 25,
    image_size: int = 128,
    grid_size: int = 32,
) -> RealGlyphEmbeddingTable:
    """Build visual consensus embeddings from real RR-corpus SVG paths."""

    if group_by not in {"base_code", "canonical_code", "raw_code"}:
        raise ValueError("group_by must be base_code, canonical_code, or raw_code")

    instances = parse_rongorongo_svg_instances(xml_dir=xml_dir, image_type=image_type)
    grouped: dict[str, list[RongorongoGlyphSvgInstance]] = defaultdict(list)
    for instance in instances:
        key = getattr(instance, group_by)
        if key:
            grouped[key].append(instance)

    items = [
        (code, group)
        for code, group in grouped.items()
        if len(group) >= min_instances
    ]
    items.sort(key=lambda item: (-len(item[1]), item[0]))
    if top_n is not None:
        items = items[:top_n]

    embeddings: dict[str, np.ndarray] = {}
    metadata: dict[str, GlyphEmbeddingMetadata] = {}
    used_instances: dict[str, tuple[RongorongoGlyphSvgInstance, ...]] = {}

    for code, group in items:
        selected = _select_instances(group, max_instances_per_glyph)
        instance_embeddings = np.vstack(
            [
                glyph_instance_embedding(
                    instance,
                    image_size=image_size,
                    grid_size=grid_size,
                )
                for instance in selected
            ]
        )
        center = spherical_mean(instance_embeddings)
        embeddings[code] = center
        metadata[code] = GlyphEmbeddingMetadata(
            glyph_code=code,
            group_by=group_by,
            n_instances_total=len(group),
            n_instances_used=len(selected),
            dispersion=embedding_dispersion(instance_embeddings, center=center),
            raw_codes=tuple(sorted({g.raw_code for g in group})),
            example_glyph_ids=tuple(g.glyph_id for g in selected[:5]),
        )
        used_instances[code] = tuple(selected)

    return RealGlyphEmbeddingTable(
        embeddings=embeddings,
        metadata=metadata,
        instances_by_code=used_instances,
    )


def save_glyph_embedding_table(
    table: RealGlyphEmbeddingTable,
    embeddings_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Persist real glyph embeddings and their audit metadata."""

    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    glyph_ids = np.array(list(table.embeddings.keys()))
    matrix = np.vstack([table.embeddings[glyph_id] for glyph_id in glyph_ids])
    np.savez_compressed(embeddings_path, glyph_ids=glyph_ids, embeddings=matrix)

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "glyph_code",
                "group_by",
                "n_instances_total",
                "n_instances_used",
                "dispersion",
                "raw_codes",
                "example_glyph_ids",
            ],
        )
        writer.writeheader()
        for glyph_id in glyph_ids:
            writer.writerow(table.metadata[str(glyph_id)].as_dict())


def export_glyph_svg_audit_files(
    instances_by_code: Mapping[str, Sequence[RongorongoGlyphSvgInstance]],
    output_dir: str | Path,
    max_per_code: int = 3,
) -> int:
    """Write standalone SVG audit files for the real glyph instances used."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for code, instances in instances_by_code.items():
        for instance in list(instances)[:max_per_code]:
            width = max(instance.width, 1.0)
            height = max(instance.height, 1.0)
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="{instance.x} {instance.y} {width} {height}">\n'
                f'  <path d="{instance.path_d}" fill="black"/>\n'
                "</svg>\n"
            )
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance.glyph_id)
            path = output / f"{code}__{safe_id}__{instance.image_type}.svg"
            path.write_text(svg, encoding="utf-8")
            count += 1
    return count


def iter_referent_image_paths(
    image_root: str | Path,
    referent_ids: Iterable[str] | None = None,
) -> dict[str, list[Path]]:
    """Find real referent images stored as ``image_root/<referent_id>/*``."""

    root = Path(image_root)
    wanted = set(referent_ids) if referent_ids is not None else None
    grouped: dict[str, list[Path]] = {}
    if not root.exists():
        return grouped
    for referent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted is not None and referent_dir.name not in wanted:
            continue
        images = [
            p
            for p in sorted(referent_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        ]
        if images:
            grouped[referent_dir.name] = images
    return grouped


def load_referent_image_embedding_table(
    image_root: str | Path,
    referent_ids: Iterable[str] | None = None,
    min_images: int = 1,
    max_images_per_referent: int | None = 10,
    grid_size: int = 32,
) -> ReferentImageEmbeddingTable:
    """Build referent embeddings from real images on disk."""

    grouped = iter_referent_image_paths(image_root, referent_ids=referent_ids)
    embeddings: dict[str, np.ndarray] = {}
    metadata: dict[str, ReferentImageMetadata] = {}

    for referent_id, paths in grouped.items():
        if len(paths) < min_images:
            continue
        selected = paths[:max_images_per_referent] if max_images_per_referent else paths
        image_embeddings = np.vstack(
            [image_shape_embedding(path, grid_size=grid_size) for path in selected]
        )
        center = spherical_mean(image_embeddings)
        embeddings[referent_id] = center
        metadata[referent_id] = ReferentImageMetadata(
            referent_id=referent_id,
            n_images=len(selected),
            dispersion=embedding_dispersion(image_embeddings, center=center),
            image_paths=tuple(str(path) for path in selected),
        )

    return ReferentImageEmbeddingTable(embeddings=embeddings, metadata=metadata)


def save_referent_embedding_table(
    table: ReferentImageEmbeddingTable,
    embeddings_path: str | Path,
    metadata_path: str | Path,
) -> None:
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    referent_ids = np.array(list(table.embeddings.keys()))
    matrix = np.vstack([table.embeddings[ref_id] for ref_id in referent_ids])
    np.savez_compressed(embeddings_path, referent_ids=referent_ids, embeddings=matrix)

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["referent_id", "n_images", "dispersion", "image_paths"],
        )
        writer.writeheader()
        for ref_id in referent_ids:
            writer.writerow(table.metadata[str(ref_id)].as_dict())
