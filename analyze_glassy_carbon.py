#!/usr/bin/env python3
"""Structural analysis for glassy-carbon LAMMPS snapshots.

This script is designed for the simulation outputs in ``my_runs/*`` and focuses
on analyses that are routinely useful for disordered carbon:

- density and box metrics
- cutoff-based bond graph and coordination statistics
- bond-length and bond-angle distributions
- radial distribution function g(r)
- local planarity and graphitic-normal alignment for 3-coordinated atoms
- bounded shortest-path ring proxy on the bond graph

Outputs are written as JSON, CSV, and simple SVG plots without requiring
matplotlib.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CARBON_MOLAR_MASS_G_PER_MOL = 12.011
AVOGADRO = 6.02214076e23
ANGSTROM3_TO_CM3 = 1.0e-24


class Structure:
    def __init__(
        self,
        positions: np.ndarray,
        atom_ids: np.ndarray,
        atom_types: np.ndarray,
        box_lo: np.ndarray,
        box_hi: np.ndarray,
        source: str,
        timestep: Optional[int] = None,
    ) -> None:
        self.positions = positions
        self.atom_ids = atom_ids
        self.atom_types = atom_types
        self.box_lo = box_lo
        self.box_hi = box_hi
        self.source = source
        self.timestep = timestep

    @property
    def lengths(self) -> np.ndarray:
        return self.box_hi - self.box_lo

    @property
    def volume(self) -> float:
        lengths = self.lengths
        return float(lengths[0] * lengths[1] * lengths[2])

    @property
    def atom_count(self) -> int:
        return int(self.positions.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="LAMMPS data file or lammpstrj dump")
    parser.add_argument(
        "--format",
        choices=("auto", "data", "lammpstrj"),
        default="auto",
        help="input format; default: infer from extension/content",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for analysis outputs; default: sibling '<stem>_analysis'",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.85,
        help="bond cutoff in Angstrom for carbon-carbon neighbors",
    )
    parser.add_argument(
        "--rdf-rmax",
        type=float,
        default=10.0,
        help="maximum radius for g(r) in Angstrom",
    )
    parser.add_argument(
        "--rdf-dr",
        type=float,
        default=0.05,
        help="bin width for g(r) in Angstrom",
    )
    parser.add_argument(
        "--angle-bin-deg",
        type=float,
        default=2.0,
        help="bin width for bond-angle histogram in degrees",
    )
    parser.add_argument(
        "--ring-max",
        type=int,
        default=8,
        help="maximum cycle size for the bounded ring proxy",
    )
    return parser.parse_args()


def infer_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".lammpstrj", ".dump", ".traj"}:
        return "lammpstrj"
    if suffix in {".data", ".dat", ".lmp"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("ITEM: TIMESTEP"):
                    return "lammpstrj"
                if "atoms" in line.lower():
                    return "data"
    raise ValueError(f"could not infer format for {path}")


def wrap_positions(positions: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray) -> np.ndarray:
    lengths = box_hi - box_lo
    return box_lo + np.mod(positions - box_lo, lengths)


def minimum_image_vectors(vectors: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return vectors - lengths * np.round(vectors / lengths)


def distance_pbc(a: np.ndarray, b: np.ndarray, lengths: np.ndarray) -> float:
    delta = minimum_image_vectors(b - a, lengths)
    return float(np.linalg.norm(delta))


def read_lammps_data(path: Path) -> Structure:
    atom_count = None
    box_lo = np.zeros(3, dtype=float)
    box_hi = np.zeros(3, dtype=float)
    atoms_start = None

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith("atoms") and atom_count is None:
            atom_count = int(stripped.split()[0])
        elif stripped.endswith("xlo xhi"):
            parts = stripped.split()
            box_lo[0], box_hi[0] = float(parts[0]), float(parts[1])
        elif stripped.endswith("ylo yhi"):
            parts = stripped.split()
            box_lo[1], box_hi[1] = float(parts[0]), float(parts[1])
        elif stripped.endswith("zlo zhi"):
            parts = stripped.split()
            box_lo[2], box_hi[2] = float(parts[0]), float(parts[1])
        elif stripped.startswith("Atoms"):
            atoms_start = index + 2
            break

    if atom_count is None or atoms_start is None:
        raise ValueError(f"failed to parse LAMMPS data file {path}")

    atom_ids = np.zeros(atom_count, dtype=int)
    atom_types = np.zeros(atom_count, dtype=int)
    positions = np.zeros((atom_count, 3), dtype=float)

    read_count = 0
    for line in lines[atoms_start:]:
        stripped = line.strip()
        if not stripped:
            if read_count >= atom_count:
                break
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        atom_ids[read_count] = int(parts[0])
        atom_types[read_count] = int(parts[1])
        positions[read_count] = [float(parts[2]), float(parts[3]), float(parts[4])]
        read_count += 1
        if read_count == atom_count:
            break

    if read_count != atom_count:
        raise ValueError(f"expected {atom_count} atoms but read {read_count} from {path}")

    positions = wrap_positions(positions, box_lo, box_hi)
    order = np.argsort(atom_ids)
    return Structure(
        positions=positions[order],
        atom_ids=atom_ids[order],
        atom_types=atom_types[order],
        box_lo=box_lo,
        box_hi=box_hi,
        source=str(path),
    )


def read_last_lammpstrj(path: Path) -> Structure:
    last_structure = None
    for structure in iter_lammpstrj_frames(path):
        last_structure = structure

    if last_structure is None:
        raise ValueError(f"failed to read any frames from {path}")

    return last_structure


def iter_lammpstrj_frames(path: Path) -> Iterable[Structure]:
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())

            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"malformed dump near timestep {timestep} in {path}")
            atom_count = int(handle.readline().strip())

            box_lo = np.zeros(3, dtype=float)
            box_hi = np.zeros(3, dtype=float)
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"malformed box header near timestep {timestep} in {path}")
            for axis in range(3):
                lo, hi, *_ = handle.readline().split()
                box_lo[axis] = float(lo)
                box_hi[axis] = float(hi)

            atoms_header = handle.readline().strip().split()
            if atoms_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError(f"malformed atom header near timestep {timestep} in {path}")
            columns = atoms_header[2:]
            column_index = {name: idx for idx, name in enumerate(columns)}
            required = ("id", "type", "x", "y", "z")
            if any(name not in column_index for name in required):
                raise ValueError(f"missing required atom columns in {path}: {required}")

            atom_ids = np.zeros(atom_count, dtype=int)
            atom_types = np.zeros(atom_count, dtype=int)
            positions = np.zeros((atom_count, 3), dtype=float)
            for i in range(atom_count):
                parts = handle.readline().split()
                atom_ids[i] = int(parts[column_index["id"]])
                atom_types[i] = int(parts[column_index["type"]])
                positions[i] = [
                    float(parts[column_index["x"]]),
                    float(parts[column_index["y"]]),
                    float(parts[column_index["z"]]),
                ]

            positions = wrap_positions(positions, box_lo, box_hi)
            order = np.argsort(atom_ids)
            yield Structure(
                positions=positions[order],
                atom_ids=atom_ids[order],
                atom_types=atom_types[order],
                box_lo=box_lo.copy(),
                box_hi=box_hi.copy(),
                source=str(path),
                timestep=timestep,
            )


def build_bond_graph(
    positions: np.ndarray, lengths: np.ndarray, cutoff: float
) -> Tuple[List[List[int]], List[Tuple[int, int, float]]]:
    atom_count = positions.shape[0]
    adjacency = [[] for _ in range(atom_count)]
    bonds = []  # type: List[Tuple[int, int, float]]
    cutoff_sq = cutoff * cutoff

    for i in range(atom_count - 1):
        deltas = positions[i + 1 :] - positions[i]
        deltas = minimum_image_vectors(deltas, lengths)
        dist_sq = np.einsum("ij,ij->i", deltas, deltas)
        neighbor_offsets = np.where(dist_sq <= cutoff_sq)[0]
        for offset in neighbor_offsets:
            j = i + 1 + int(offset)
            distance = float(math.sqrt(dist_sq[offset]))
            adjacency[i].append(j)
            adjacency[j].append(i)
            bonds.append((i, j, distance))

    return adjacency, bonds


def coordination_counts(adjacency: List[List[int]]) -> np.ndarray:
    return np.array([len(neighbors) for neighbors in adjacency], dtype=int)


def coordination_histogram(coordination: np.ndarray) -> List[Dict[str, int]]:
    unique, counts = np.unique(coordination, return_counts=True)
    return [
        {"coordination": int(coord), "count": int(count)}
        for coord, count in zip(unique, counts)
    ]


def calculate_density_g_cm3(atom_count: int, volume_ang3: float) -> float:
    mass_g = atom_count * CARBON_MOLAR_MASS_G_PER_MOL / AVOGADRO
    volume_cm3 = volume_ang3 * ANGSTROM3_TO_CM3
    return mass_g / volume_cm3


def compute_rdf(
    positions: np.ndarray, lengths: np.ndarray, r_max: float, dr: float
) -> Tuple[np.ndarray, np.ndarray]:
    atom_count = positions.shape[0]
    rho = atom_count / float(np.prod(lengths))
    edges = np.arange(0.0, r_max + dr, dr)
    counts = np.zeros(edges.size - 1, dtype=float)

    for i in range(atom_count - 1):
        deltas = positions[i + 1 :] - positions[i]
        deltas = minimum_image_vectors(deltas, lengths)
        distances = np.linalg.norm(deltas, axis=1)
        valid = distances < r_max
        hist, _ = np.histogram(distances[valid], bins=edges)
        counts += hist

    r_centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    normalization = 0.5 * atom_count * rho * shell_volumes
    g_r = np.divide(counts, normalization, out=np.zeros_like(counts), where=normalization > 0)
    return r_centers, g_r


def compute_bond_angles(
    positions: np.ndarray, adjacency: List[List[int]], lengths: np.ndarray
) -> np.ndarray:
    angles = []  # type: List[float]
    for center, neighbors in enumerate(adjacency):
        if len(neighbors) < 2:
            continue
        vectors = positions[np.array(neighbors)] - positions[center]
        vectors = minimum_image_vectors(vectors, lengths)
        norms = np.linalg.norm(vectors, axis=1)
        for i in range(len(neighbors) - 1):
            for j in range(i + 1, len(neighbors)):
                denom = norms[i] * norms[j]
                if denom == 0.0:
                    continue
                cosine = float(np.dot(vectors[i], vectors[j]) / denom)
                cosine = max(-1.0, min(1.0, cosine))
                angles.append(math.degrees(math.acos(cosine)))
    return np.array(angles, dtype=float)


def compute_threefold_planarity(
    positions: np.ndarray,
    adjacency: List[List[int]],
    lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.full((positions.shape[0], 3), np.nan, dtype=float)
    planarity = np.full(positions.shape[0], np.nan, dtype=float)
    pyramidalization = np.full(positions.shape[0], np.nan, dtype=float)

    for center, neighbors in enumerate(adjacency):
        if len(neighbors) != 3:
            continue
        vectors = positions[np.array(neighbors)] - positions[center]
        vectors = minimum_image_vectors(vectors, lengths)
        cross_sum = (
            np.cross(vectors[0], vectors[1])
            + np.cross(vectors[1], vectors[2])
            + np.cross(vectors[2], vectors[0])
        )
        norm = np.linalg.norm(cross_sum)
        if norm == 0.0:
            continue
        normal = cross_sum / norm
        signed_distances = vectors @ normal
        planarity[center] = float(np.sqrt(np.mean(signed_distances**2)))
        pyramidalization[center] = float(np.mean(np.abs(signed_distances) / np.linalg.norm(vectors, axis=1)))
        normals[center] = normal

    return normals, planarity, pyramidalization


def compute_normal_alignment(
    adjacency: List[List[int]],
    normals: np.ndarray,
) -> np.ndarray:
    alignments = []  # type: List[float]
    for i, neighbors in enumerate(adjacency):
        if np.isnan(normals[i, 0]):
            continue
        for j in neighbors:
            if j <= i or np.isnan(normals[j, 0]):
                continue
            alignments.append(float(abs(np.dot(normals[i], normals[j]))))
    return np.array(alignments, dtype=float)


def shortest_path_excluding_edge(
    adjacency: List[List[int]],
    start: int,
    goal: int,
    max_depth: int,
) -> Optional[int]:
    frontier = [(start, 0)]
    visited = {start}
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for neighbor in adjacency[node]:
            if (node == start and neighbor == goal) or (node == goal and neighbor == start):
                continue
            if neighbor == goal:
                return depth + 1
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append((neighbor, depth + 1))
    return None


def compute_ring_proxy(adjacency: List[List[int]], max_ring: int) -> List[Dict[str, int]]:
    histogram = {}  # type: Dict[int, int]
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if j <= i:
                continue
            path_len = shortest_path_excluding_edge(adjacency, i, j, max_ring - 1)
            if path_len is None:
                continue
            ring_size = path_len + 1
            if 3 <= ring_size <= max_ring:
                histogram[ring_size] = histogram.get(ring_size, 0) + 1
    return [
        {"ring_size": ring_size, "bond_count": histogram[ring_size]}
        for ring_size in sorted(histogram)
    ]


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def histogram_rows(values: np.ndarray, bin_edges: np.ndarray) -> List[Dict[str, float]]:
    counts, _ = np.histogram(values, bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = bin_edges[1:] - bin_edges[:-1]
    density = np.divide(counts, counts.sum() * widths, out=np.zeros_like(centers), where=counts.sum() > 0)
    return [
        {
            "bin_left": float(bin_edges[i]),
            "bin_right": float(bin_edges[i + 1]),
            "bin_center": float(centers[i]),
            "count": int(counts[i]),
            "density": float(density[i]),
        }
        for i in range(len(counts))
    ]


def svg_line_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 900, 540
    left, right, top, bottom = 80, 30, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = 0.0, float(np.max(y) * 1.05 if np.max(y) > 0 else 1.0)

    def sx(value: float) -> float:
        if x_max == x_min:
            return left + plot_width / 2
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    points = " ".join(f"{sx(xv):.2f},{sy(yv):.2f}" for xv, yv in zip(x, y))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="20" font-family="Helvetica">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black"/>',
        f'<polyline fill="none" stroke="#005f73" stroke-width="2" points="{points}"/>',
        f'<text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-size="16" font-family="Helvetica">{x_label}</text>',
        f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="16" font-family="Helvetica" transform="rotate(-90 22 {height/2:.1f})">{y_label}</text>',
        f'<text x="{left}" y="{height-42}" font-size="12" font-family="Helvetica">{x_min:.2f}</text>',
        f'<text x="{left+plot_width}" y="{height-42}" text-anchor="end" font-size="12" font-family="Helvetica">{x_max:.2f}</text>',
        f'<text x="{left-10}" y="{top+plot_height}" text-anchor="end" font-size="12" font-family="Helvetica">{y_min:.2f}</text>',
        f'<text x="{left-10}" y="{top+12}" text-anchor="end" font-size="12" font-family="Helvetica">{y_max:.2f}</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_bar_plot(
    path: Path,
    labels: List[str],
    values: List[float],
    title: str,
    y_label: str,
) -> None:
    width, height = 900, 540
    left, right, top, bottom = 80, 30, 40, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(values) * 1.1 if values else 1.0
    bar_width = plot_width / max(len(values), 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="20" font-family="Helvetica">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black"/>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * bar_width + 0.15 * bar_width
        rect_width = 0.7 * bar_width
        rect_height = 0.0 if y_max == 0 else plot_height * value / y_max
        y = top + plot_height - rect_height
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{rect_width:.2f}" height="{rect_height:.2f}" fill="#0a9396"/>')
        lines.append(f'<text x="{x + rect_width/2:.2f}" y="{top+plot_height+20}" text-anchor="middle" font-size="12" font-family="Helvetica">{label}</text>')
        lines.append(f'<text x="{x + rect_width/2:.2f}" y="{max(y-6, top+12):.2f}" text-anchor="middle" font-size="11" font-family="Helvetica">{value:.0f}</text>')

    lines.extend(
        [
            f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="16" font-family="Helvetica" transform="rotate(-90 22 {height/2:.1f})">{y_label}</text>',
            f'<text x="{left-10}" y="{top+plot_height}" text-anchor="end" font-size="12" font-family="Helvetica">0</text>',
            f'<text x="{left-10}" y="{top+12}" text-anchor="end" font-size="12" font-family="Helvetica">{y_max:.0f}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(
    structure: Structure,
    cutoff: float,
    rdf_rmax: float,
    rdf_dr: float,
    angle_bin_deg: float,
    ring_max: int,
) -> Dict[str, object]:
    lengths = structure.lengths
    adjacency, bonds = build_bond_graph(structure.positions, lengths, cutoff)
    coordination = coordination_counts(adjacency)
    bond_lengths = np.array([distance for _, _, distance in bonds], dtype=float)
    bond_angles = compute_bond_angles(structure.positions, adjacency, lengths)
    rdf_r, rdf_g = compute_rdf(structure.positions, lengths, rdf_rmax, rdf_dr)
    normals, planarity, pyramidalization = compute_threefold_planarity(structure.positions, adjacency, lengths)
    alignment = compute_normal_alignment(adjacency, normals)
    ring_proxy = compute_ring_proxy(adjacency, ring_max)

    threefold_mask = coordination == 3
    results = {
        "source": structure.source,
        "timestep": structure.timestep,
        "atom_count": structure.atom_count,
        "box_lo_angstrom": structure.box_lo.tolist(),
        "box_hi_angstrom": structure.box_hi.tolist(),
        "box_lengths_angstrom": lengths.tolist(),
        "volume_angstrom3": structure.volume,
        "density_g_cm3": calculate_density_g_cm3(structure.atom_count, structure.volume),
        "bond_cutoff_angstrom": cutoff,
        "bond_count": len(bonds),
        "mean_coordination": float(np.mean(coordination)),
        "coordination_histogram": coordination_histogram(coordination),
        "sp2_like_fraction": float(np.mean(coordination == 3)),
        "sp3_like_fraction": float(np.mean(coordination == 4)),
        "undercoordinated_fraction": float(np.mean(coordination <= 2)),
        "overcoordinated_fraction": float(np.mean(coordination >= 5)),
        "bond_length_mean_angstrom": float(np.mean(bond_lengths)) if bond_lengths.size else None,
        "bond_length_std_angstrom": float(np.std(bond_lengths)) if bond_lengths.size else None,
        "bond_angle_mean_deg": float(np.mean(bond_angles)) if bond_angles.size else None,
        "bond_angle_std_deg": float(np.std(bond_angles)) if bond_angles.size else None,
        "threefold_atom_count": int(np.count_nonzero(threefold_mask)),
        "threefold_planarity_rms_mean_angstrom": float(np.nanmean(planarity)) if np.any(threefold_mask) else None,
        "threefold_pyramidalization_mean": float(np.nanmean(pyramidalization)) if np.any(threefold_mask) else None,
        "threefold_normal_alignment_mean_abs_cos": float(np.mean(alignment)) if alignment.size else None,
        "ring_proxy_bond_histogram": ring_proxy,
    }
    results["_coordination"] = coordination
    results["_bond_lengths"] = bond_lengths
    results["_bond_angles"] = bond_angles
    results["_rdf_r"] = rdf_r
    results["_rdf_g"] = rdf_g
    results["_planarity"] = planarity
    results["_pyramidalization"] = pyramidalization
    results["_alignment"] = alignment
    return results


def write_outputs(output_dir: Path, results: Dict[str, object], angle_bin_deg: float, cutoff: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {key: value for key, value in results.items() if not key.startswith("_")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_csv(
        output_dir / "coordination_histogram.csv",
        ["coordination", "count"],
        summary["coordination_histogram"],  # type: ignore[arg-type]
    )

    bond_lengths = results["_bond_lengths"]
    if isinstance(bond_lengths, np.ndarray) and bond_lengths.size:
        bins = np.arange(0.0, max(float(np.max(bond_lengths)) + 0.02, cutoff + 0.02), 0.02)
        rows = histogram_rows(bond_lengths, bins)
        write_csv(
            output_dir / "bond_length_distribution.csv",
            ["bin_left", "bin_right", "bin_center", "count", "density"],
            rows,
        )
        svg_line_plot(
            output_dir / "bond_length_distribution.svg",
            np.array([row["bin_center"] for row in rows]),
            np.array([row["density"] for row in rows]),
            "Bond-Length Distribution",
            "Bond length (A)",
            "Probability density",
        )

    bond_angles = results["_bond_angles"]
    if isinstance(bond_angles, np.ndarray) and bond_angles.size:
        bins = np.arange(0.0, 180.0 + angle_bin_deg, angle_bin_deg)
        rows = histogram_rows(bond_angles, bins)
        write_csv(
            output_dir / "bond_angle_distribution.csv",
            ["bin_left", "bin_right", "bin_center", "count", "density"],
            rows,
        )
        svg_line_plot(
            output_dir / "bond_angle_distribution.svg",
            np.array([row["bin_center"] for row in rows]),
            np.array([row["density"] for row in rows]),
            "Bond-Angle Distribution",
            "Bond angle (deg)",
            "Probability density",
        )

    rdf_r = results["_rdf_r"]
    rdf_g = results["_rdf_g"]
    if isinstance(rdf_r, np.ndarray) and isinstance(rdf_g, np.ndarray):
        rdf_rows = [
            {"r_angstrom": float(r), "g_r": float(g)}
            for r, g in zip(rdf_r, rdf_g)
        ]
        write_csv(output_dir / "rdf.csv", ["r_angstrom", "g_r"], rdf_rows)
        svg_line_plot(output_dir / "rdf.svg", rdf_r, rdf_g, "Radial Distribution Function", "r (A)", "g(r)")

    planarity = results["_planarity"]
    if isinstance(planarity, np.ndarray) and np.any(~np.isnan(planarity)):
        valid = planarity[~np.isnan(planarity)]
        bins = np.arange(0.0, max(float(np.max(valid)) + 0.01, 0.25), 0.01)
        rows = histogram_rows(valid, bins)
        write_csv(
            output_dir / "threefold_planarity_distribution.csv",
            ["bin_left", "bin_right", "bin_center", "count", "density"],
            rows,
        )
        svg_line_plot(
            output_dir / "threefold_planarity_distribution.svg",
            np.array([row["bin_center"] for row in rows]),
            np.array([row["density"] for row in rows]),
            "Threefold Planarity",
            "RMS distance from local plane (A)",
            "Probability density",
        )

    alignment = results["_alignment"]
    if isinstance(alignment, np.ndarray) and alignment.size:
        bins = np.arange(0.0, 1.0 + 0.025, 0.025)
        rows = histogram_rows(alignment, bins)
        write_csv(
            output_dir / "threefold_normal_alignment.csv",
            ["bin_left", "bin_right", "bin_center", "count", "density"],
            rows,
        )
        svg_line_plot(
            output_dir / "threefold_normal_alignment.svg",
            np.array([row["bin_center"] for row in rows]),
            np.array([row["density"] for row in rows]),
            "Threefold Normal Alignment",
            "|n_i . n_j|",
            "Probability density",
        )

    ring_proxy = summary["ring_proxy_bond_histogram"]
    if ring_proxy:
        write_csv(output_dir / "ring_proxy_bond_histogram.csv", ["ring_size", "bond_count"], ring_proxy)  # type: ignore[arg-type]
        svg_bar_plot(
            output_dir / "ring_proxy_bond_histogram.svg",
            [str(row["ring_size"]) for row in ring_proxy],  # type: ignore[index]
            [float(row["bond_count"]) for row in ring_proxy],  # type: ignore[index]
            "Bounded Ring Proxy",
            "Bond count",
        )

    coord_hist = summary["coordination_histogram"]
    svg_bar_plot(
        output_dir / "coordination_histogram.svg",
        [str(row["coordination"]) for row in coord_hist],  # type: ignore[index]
        [float(row["count"]) for row in coord_hist],  # type: ignore[index]
        "Coordination Histogram",
        "Atom count",
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    file_format = infer_format(input_path, args.format)

    if file_format == "data":
        structure = read_lammps_data(input_path)
    elif file_format == "lammpstrj":
        structure = read_last_lammpstrj(input_path)
    else:
        raise AssertionError(f"unsupported format {file_format}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_path.with_name(f"{input_path.stem}_analysis")
    )

    results = analyze(
        structure=structure,
        cutoff=args.cutoff,
        rdf_rmax=args.rdf_rmax,
        rdf_dr=args.rdf_dr,
        angle_bin_deg=args.angle_bin_deg,
        ring_max=args.ring_max,
    )
    write_outputs(output_dir, results, args.angle_bin_deg, args.cutoff)

    summary = {key: value for key, value in results.items() if not key.startswith("_")}
    print(f"Wrote analysis to {output_dir}")
    print(f"Atoms: {summary['atom_count']}")
    print(f"Density (g/cm^3): {summary['density_g_cm3']:.4f}")
    print(f"Mean coordination: {summary['mean_coordination']:.4f}")
    print(f"sp2-like fraction: {summary['sp2_like_fraction']:.4f}")
    print(f"sp3-like fraction: {summary['sp3_like_fraction']:.4f}")
    if summary["bond_length_mean_angstrom"] is not None:
        print(f"Mean bond length (A): {summary['bond_length_mean_angstrom']:.4f}")
    if summary["bond_angle_mean_deg"] is not None:
        print(f"Mean bond angle (deg): {summary['bond_angle_mean_deg']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
