#!/usr/bin/env python3
"""Generate randomly oriented graphene flakes in a simulation box.

The atom count is computed from the requested box dimensions and mass density.
Atoms are grouped into small graphene-like flakes whose centers are placed
uniformly at random in the box. Flakes are rotated randomly in 3D and inserted
with rejection sampling so the final configuration satisfies a minimum
interatomic separation.
"""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


AVOGADRO = 6.02214076e23
CARBON_MOLAR_MASS = 12.011  # g/mol
ANGSTROM3_TO_CM3 = 1.0e-24
GRAPHENE_BOND_LENGTH = 1.42  # A
GRAPHENE_AREA_PER_ATOM = 3.0 * math.sqrt(3.0) * GRAPHENE_BOND_LENGTH**2 / 4.0
DEFAULT_FLAKE_ATOM_COUNT = 24
DEFAULT_FLAKE_AREA = DEFAULT_FLAKE_ATOM_COUNT * GRAPHENE_AREA_PER_ATOM
MIN_INTERATOMIC_DISTANCE = 1.2  # A
MAX_PLACEMENT_ATTEMPTS = 1000


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def infer_format(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".extxyz":
        return "extxyz"
    if suffix in {".data", ".lmp", ".lammps"}:
        return "lammps"
    raise ValueError(
        "could not infer output format from file extension; "
        "use --format extxyz or --format lammps"
    )


def atom_count_from_density(
    density_g_cm3: float,
    box_lengths: Tuple[float, float, float],
    rounding: str,
) -> int:
    volume_cm3 = box_lengths[0] * box_lengths[1] * box_lengths[2] * ANGSTROM3_TO_CM3
    total_mass_g = density_g_cm3 * volume_cm3
    exact_atom_count = total_mass_g * AVOGADRO / CARBON_MOLAR_MASS

    if rounding == "floor":
        atom_count = math.floor(exact_atom_count)
    elif rounding == "ceil":
        atom_count = math.ceil(exact_atom_count)
    else:
        atom_count = round(exact_atom_count)

    return max(1, atom_count)


def achieved_density_g_cm3(atom_count: int, box_lengths: Tuple[float, float, float]) -> float:
    volume_cm3 = box_lengths[0] * box_lengths[1] * box_lengths[2] * ANGSTROM3_TO_CM3
    total_mass_g = atom_count * CARBON_MOLAR_MASS / AVOGADRO
    return total_mass_g / volume_cm3


def generate_graphene_flake(
    atom_count: int,
    bond_length: float = GRAPHENE_BOND_LENGTH,
) -> List[Tuple[float, float, float]]:
    if atom_count <= 0:
        raise ValueError("atom_count must be positive")
    if atom_count == 1:
        return [(0.0, 0.0, 0.0)]

    cell_count = math.ceil(atom_count / 2.0)
    nx = math.ceil(math.sqrt(cell_count))
    ny = math.ceil(cell_count / nx)

    a1 = (math.sqrt(3.0) * bond_length, 0.0)
    a2 = (math.sqrt(3.0) * bond_length / 2.0, 1.5 * bond_length)
    basis = (
        (0.0, 0.0),
        (math.sqrt(3.0) * bond_length / 2.0, bond_length / 2.0),
    )

    positions: List[Tuple[float, float, float]] = []
    for i in range(nx):
        for j in range(ny):
            origin_x = i * a1[0] + j * a2[0]
            origin_y = i * a1[1] + j * a2[1]
            for basis_x, basis_y in basis:
                positions.append((origin_x + basis_x, origin_y + basis_y, 0.0))

    positions.sort(key=lambda point: (point[0] ** 2 + point[1] ** 2, point[0], point[1]))
    positions = positions[:atom_count]
    center_x = sum(x for x, _, _ in positions) / len(positions)
    center_y = sum(y for _, y, _ in positions) / len(positions)
    return [(x - center_x, y - center_y, z) for x, y, z in positions]


def flake_atom_count_from_area(flake_area: float) -> int:
    return max(1, round(flake_area / GRAPHENE_AREA_PER_ATOM))


def choose_flake_sizes(atom_count: int, target_flake_atoms: int) -> List[int]:
    counts: List[int] = []
    remaining = atom_count
    while remaining > 0:
        flake_size = min(target_flake_atoms, remaining)
        counts.append(flake_size)
        remaining -= flake_size
    return counts


def random_rotation_matrix(rng: random.Random) -> Tuple[Tuple[float, float, float], ...]:
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()

    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)

    return (
        (
            1.0 - 2.0 * (q3 * q3 + q4 * q4),
            2.0 * (q2 * q3 - q1 * q4),
            2.0 * (q2 * q4 + q1 * q3),
        ),
        (
            2.0 * (q2 * q3 + q1 * q4),
            1.0 - 2.0 * (q2 * q2 + q4 * q4),
            2.0 * (q3 * q4 - q1 * q2),
        ),
        (
            2.0 * (q2 * q4 - q1 * q3),
            2.0 * (q3 * q4 + q1 * q2),
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
        ),
    )


def rotate_point(
    point: Tuple[float, float, float],
    rotation: Tuple[Tuple[float, float, float], ...],
) -> Tuple[float, float, float]:
    x, y, z = point
    return (
        rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z,
        rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z,
        rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z,
    )


def minimum_image_delta(delta: float, box_length: float) -> float:
    return delta - box_length * round(delta / box_length)


def minimum_image_distance_sq(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
    box_lengths: Tuple[float, float, float],
) -> float:
    dx = minimum_image_delta(point_a[0] - point_b[0], box_lengths[0])
    dy = minimum_image_delta(point_a[1] - point_b[1], box_lengths[1])
    dz = minimum_image_delta(point_a[2] - point_b[2], box_lengths[2])
    return dx * dx + dy * dy + dz * dz


class SpatialGrid:
    def __init__(self, box_lengths: Tuple[float, float, float], cutoff: float) -> None:
        self.box_lengths = box_lengths
        self.cutoff = cutoff
        self.cell_size = cutoff
        self.shape = tuple(
            max(1, int(math.floor(length / self.cell_size))) for length in box_lengths
        )
        self.cells: Dict[Tuple[int, int, int], List[Tuple[float, float, float]]] = defaultdict(list)

    def _cell_index(self, point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        ix = min(int(point[0] / self.cell_size), self.shape[0] - 1)
        iy = min(int(point[1] / self.cell_size), self.shape[1] - 1)
        iz = min(int(point[2] / self.cell_size), self.shape[2] - 1)
        return (ix, iy, iz)

    def _neighbor_indices(self, index: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        ix, iy, iz = index
        neighbors: List[Tuple[int, int, int]] = []
        for dx in (-1, 0, 1):
            nx = (ix + dx) % self.shape[0]
            for dy in (-1, 0, 1):
                ny = (iy + dy) % self.shape[1]
                for dz in (-1, 0, 1):
                    nz = (iz + dz) % self.shape[2]
                    neighbors.append((nx, ny, nz))
        return neighbors

    def has_overlap(
        self,
        candidate: Sequence[Tuple[float, float, float]],
        min_distance_sq: float,
    ) -> bool:
        for point in candidate:
            for neighbor_index in self._neighbor_indices(self._cell_index(point)):
                for other in self.cells.get(neighbor_index, ()):
                    if minimum_image_distance_sq(point, other, self.box_lengths) < min_distance_sq:
                        return True
        return False

    def add_points(self, points: Sequence[Tuple[float, float, float]]) -> None:
        for point in points:
            self.cells[self._cell_index(point)].append(point)


def try_place_flake(
    flake: Sequence[Tuple[float, float, float]],
    rotation: Tuple[Tuple[float, float, float], ...],
    box_lengths: Tuple[float, float, float],
    rng: random.Random,
) -> Optional[List[Tuple[float, float, float]]]:
    lx, ly, lz = box_lengths
    rotated = [rotate_point(point, rotation) for point in flake]

    min_x = min(x for x, _, _ in rotated)
    max_x = max(x for x, _, _ in rotated)
    min_y = min(y for _, y, _ in rotated)
    max_y = max(y for _, y, _ in rotated)
    min_z = min(z for _, _, z in rotated)
    max_z = max(z for _, _, z in rotated)

    if max_x - min_x > lx or max_y - min_y > ly or max_z - min_z > lz:
        return None

    center = (
        rng.uniform(-min_x, lx - max_x),
        rng.uniform(-min_y, ly - max_y),
        rng.uniform(-min_z, lz - max_z),
    )
    cx, cy, cz = center
    return [(cx + x, cy + y, cz + z) for x, y, z in rotated]


def graphene_flake_positions(
    atom_count: int,
    box_lengths: Tuple[float, float, float],
    flake_area: float,
    seed: Optional[int],
) -> List[Tuple[float, float, float]]:
    rng = random.Random(seed)
    target_flake_atoms = flake_atom_count_from_area(flake_area)
    flake_cache: Dict[int, List[Tuple[float, float, float]]] = {}
    positions: List[Tuple[float, float, float]] = []
    min_distance_sq = MIN_INTERATOMIC_DISTANCE**2
    spatial_grid = SpatialGrid(box_lengths, MIN_INTERATOMIC_DISTANCE)

    for flake_size in choose_flake_sizes(atom_count, target_flake_atoms):
        flake = flake_cache.setdefault(flake_size, generate_graphene_flake(flake_size))
        candidate = None
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            rotation = random_rotation_matrix(rng)
            candidate = try_place_flake(flake, rotation, box_lengths, rng)
            if candidate is None:
                continue
            if not spatial_grid.has_overlap(candidate, min_distance_sq):
                positions.extend(candidate)
                spatial_grid.add_points(candidate)
                break
        else:
            raise RuntimeError(
                "could not place all flakes without overlap; "
                "try a larger box, lower density, or smaller --flake-area"
            )

    return positions


def write_extxyz(
    output_path: Path,
    positions: List[Tuple[float, float, float]],
    box_lengths: Tuple[float, float, float],
) -> None:
    lx, ly, lz = box_lengths
    with output_path.open("w", encoding="ascii") as handle:
        handle.write(f"{len(positions)}\n")
        handle.write(
            f'Lattice="{lx:.8f} 0 0 0 {ly:.8f} 0 0 0 {lz:.8f}" '
            'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        )
        for x, y, z in positions:
            handle.write(f"C {x:.8f} {y:.8f} {z:.8f}\n")


def write_lammps_data(
    output_path: Path,
    positions: List[Tuple[float, float, float]],
    box_lengths: Tuple[float, float, float],
) -> None:
    lx, ly, lz = box_lengths
    with output_path.open("w", encoding="ascii") as handle:
        handle.write("LAMMPS data file for randomly placed graphene-flake carbon atoms\n\n")
        handle.write(f"{len(positions)} atoms\n")
        handle.write("1 atom types\n\n")
        handle.write(f"0.0 {lx:.8f} xlo xhi\n")
        handle.write(f"0.0 {ly:.8f} ylo yhi\n")
        handle.write(f"0.0 {lz:.8f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        handle.write(f"1 {CARBON_MOLAR_MASS:.6f}\n\n")
        handle.write("Atoms # atomic\n\n")
        for atom_id, (x, y, z) in enumerate(positions, start=1):
            handle.write(f"{atom_id} 1 {x:.8f} {y:.8f} {z:.8f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Place small randomly oriented graphene flakes in a simulation box "
            "for a requested mass density."
        )
    )
    parser.add_argument(
        "--box",
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        type=positive_float,
        required=True,
        help="simulation box lengths in angstrom",
    )
    parser.add_argument(
        "--density",
        type=positive_float,
        required=True,
        help="target mass density in g/cm^3",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output path; extension .extxyz infers extxyz and .data/.lmp/.lammps infers LAMMPS",
    )
    parser.add_argument(
        "--format",
        choices=("extxyz", "lammps"),
        help="explicitly select output format instead of inferring from the file extension",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducible coordinates",
    )
    parser.add_argument(
        "--flake-area",
        type=positive_float,
        default=DEFAULT_FLAKE_AREA,
        help=(
            "target area of each graphene flake in square angstrom "
            f"(default: {DEFAULT_FLAKE_AREA:.4f})"
        ),
    )
    parser.add_argument(
        "--rounding",
        choices=("round", "floor", "ceil"),
        default="round",
        help="how to convert the exact atom count from density to an integer (default: round)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    box_lengths = tuple(args.box)
    output_format = args.format or infer_format(args.output)
    atom_count = atom_count_from_density(args.density, box_lengths, args.rounding)
    positions = graphene_flake_positions(atom_count, box_lengths, args.flake_area, args.seed)
    achieved_density = achieved_density_g_cm3(atom_count, box_lengths)

    if output_format == "extxyz":
        write_extxyz(args.output, positions, box_lengths)
    else:
        write_lammps_data(args.output, positions, box_lengths)

    print(f"target_density_g_cm3: {args.density:.8f}")
    print(f"achieved_density_g_cm3: {achieved_density:.8f}")
    print(f"box_A: {box_lengths[0]:.8f} {box_lengths[1]:.8f} {box_lengths[2]:.8f}")
    print(f"num_atoms: {atom_count}")
    print(f"flake_area_A2: {args.flake_area:.8f}")
    print(f"rounding_mode: {args.rounding}")
    print(f"output_format: {output_format}")
    print(f"output_path: {args.output}")
    if args.seed is not None:
        print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
