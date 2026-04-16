#!/usr/bin/env python3
"""Time-series structural analysis for glassy-carbon LAMMPS trajectories."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analyze_glassy_carbon import (
    analyze,
    iter_lammpstrj_frames,
    svg_line_plot,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="LAMMPS lammpstrj file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for outputs; default: sibling '<stem>_timeseries'",
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
        help="maximum radius for per-frame g(r) calculation",
    )
    parser.add_argument(
        "--rdf-dr",
        type=float,
        default=0.05,
        help="bin width for per-frame g(r) calculation",
    )
    parser.add_argument(
        "--angle-bin-deg",
        type=float,
        default=2.0,
        help="bin width for per-frame bond-angle histogram",
    )
    parser.add_argument(
        "--ring-max",
        type=int,
        default=8,
        help="maximum cycle size for the bounded ring proxy",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="analyze every Nth frame",
    )
    parser.add_argument(
        "--coordination-log",
        default=None,
        help="optional fix coordlog file to merge into the output table",
    )
    return parser.parse_args()


def read_coordination_log(path: Path) -> Dict[int, Dict[str, float]]:
    by_timestep = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            timestep = int(parts[0])
            by_timestep[timestep] = {
                "coordlog_n2": float(parts[1]),
                "coordlog_n3": float(parts[2]),
                "coordlog_n4": float(parts[3]),
            }
    return by_timestep


def series_svg(output_path: Path, rows: List[Dict[str, float]], key: str, title: str, y_label: str) -> None:
    if not rows:
        return
    x = np.array([row["timestep"] for row in rows], dtype=float)
    y = np.array([row[key] for row in rows], dtype=float)
    svg_line_plot(output_path, x, y, title, "Timestep", y_label)


def main() -> int:
    args = parse_args()
    trajectory = Path(args.trajectory).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else trajectory.with_name("%s_timeseries" % trajectory.stem)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coordlog = None  # type: Optional[Dict[int, Dict[str, float]]]
    if args.coordination_log:
        coordlog = read_coordination_log(Path(args.coordination_log).resolve())

    rows = []  # type: List[Dict[str, float]]
    frame_count = 0
    analyzed_count = 0
    for structure in iter_lammpstrj_frames(trajectory):
        frame_count += 1
        if (frame_count - 1) % args.every != 0:
            continue

        results = analyze(
            structure=structure,
            cutoff=args.cutoff,
            rdf_rmax=args.rdf_rmax,
            rdf_dr=args.rdf_dr,
            angle_bin_deg=args.angle_bin_deg,
            ring_max=args.ring_max,
        )
        analyzed_count += 1

        row = {
            "frame_index": float(frame_count - 1),
            "timestep": float(structure.timestep),
            "density_g_cm3": float(results["density_g_cm3"]),
            "mean_coordination": float(results["mean_coordination"]),
            "sp2_like_fraction": float(results["sp2_like_fraction"]),
            "sp3_like_fraction": float(results["sp3_like_fraction"]),
            "undercoordinated_fraction": float(results["undercoordinated_fraction"]),
            "bond_length_mean_angstrom": float(results["bond_length_mean_angstrom"]),
            "bond_length_std_angstrom": float(results["bond_length_std_angstrom"]),
            "bond_angle_mean_deg": float(results["bond_angle_mean_deg"]),
            "bond_angle_std_deg": float(results["bond_angle_std_deg"]),
            "threefold_planarity_rms_mean_angstrom": float(results["threefold_planarity_rms_mean_angstrom"])
            if results["threefold_planarity_rms_mean_angstrom"] is not None
            else np.nan,
            "threefold_pyramidalization_mean": float(results["threefold_pyramidalization_mean"])
            if results["threefold_pyramidalization_mean"] is not None
            else np.nan,
            "threefold_normal_alignment_mean_abs_cos": float(results["threefold_normal_alignment_mean_abs_cos"])
            if results["threefold_normal_alignment_mean_abs_cos"] is not None
            else np.nan,
        }

        coord_hist = results["coordination_histogram"]
        coord_counts = {}
        for item in coord_hist:
            coord_counts[int(item["coordination"])] = float(item["count"])
        row["coord_2_count"] = coord_counts.get(2, 0.0)
        row["coord_3_count"] = coord_counts.get(3, 0.0)
        row["coord_4_count"] = coord_counts.get(4, 0.0)

        ring_hist = results["ring_proxy_bond_histogram"]
        ring_counts = {}
        for item in ring_hist:
            ring_counts[int(item["ring_size"])] = float(item["bond_count"])
        for ring_size in range(3, args.ring_max + 1):
            row["ring_%d_bond_count" % ring_size] = ring_counts.get(ring_size, 0.0)

        if coordlog is not None and structure.timestep in coordlog:
            row.update(coordlog[structure.timestep])

        rows.append(row)

    if not rows:
        raise ValueError("no frames were analyzed")

    fieldnames = list(rows[0].keys())
    write_csv(output_dir / "trajectory_summary.csv", fieldnames, rows)

    metadata = {
        "trajectory": str(trajectory),
        "frame_count": frame_count,
        "analyzed_frame_count": analyzed_count,
        "sampling_stride": args.every,
        "bond_cutoff_angstrom": args.cutoff,
        "ring_max": args.ring_max,
    }
    if args.coordination_log:
        metadata["coordination_log"] = str(Path(args.coordination_log).resolve())
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    series_svg(output_dir / "density_vs_timestep.svg", rows, "density_g_cm3", "Density Evolution", "Density (g/cm^3)")
    series_svg(output_dir / "mean_coordination_vs_timestep.svg", rows, "mean_coordination", "Mean Coordination Evolution", "Mean coordination")
    series_svg(output_dir / "sp2_fraction_vs_timestep.svg", rows, "sp2_like_fraction", "sp2-Like Fraction Evolution", "Fraction")
    series_svg(output_dir / "sp3_fraction_vs_timestep.svg", rows, "sp3_like_fraction", "sp3-Like Fraction Evolution", "Fraction")
    series_svg(
        output_dir / "bond_length_mean_vs_timestep.svg",
        rows,
        "bond_length_mean_angstrom",
        "Mean Bond Length Evolution",
        "Bond length (A)",
    )
    series_svg(
        output_dir / "bond_angle_mean_vs_timestep.svg",
        rows,
        "bond_angle_mean_deg",
        "Mean Bond Angle Evolution",
        "Bond angle (deg)",
    )
    series_svg(
        output_dir / "planarity_vs_timestep.svg",
        rows,
        "threefold_planarity_rms_mean_angstrom",
        "Threefold Planarity Evolution",
        "RMS distance from plane (A)",
    )
    series_svg(
        output_dir / "normal_alignment_vs_timestep.svg",
        rows,
        "threefold_normal_alignment_mean_abs_cos",
        "Threefold Normal Alignment Evolution",
        "|n_i . n_j|",
    )

    if "coordlog_n3" in rows[0]:
        series_svg(output_dir / "coordlog_n2_vs_timestep.svg", rows, "coordlog_n2", "Coordination Log n2", "Count")
        series_svg(output_dir / "coordlog_n3_vs_timestep.svg", rows, "coordlog_n3", "Coordination Log n3", "Count")
        series_svg(output_dir / "coordlog_n4_vs_timestep.svg", rows, "coordlog_n4", "Coordination Log n4", "Count")

    first = rows[0]
    last = rows[-1]
    print("Wrote trajectory analysis to %s" % output_dir)
    print("Frames read: %d" % frame_count)
    print("Frames analyzed: %d" % analyzed_count)
    print("sp2-like fraction: %.4f -> %.4f" % (first["sp2_like_fraction"], last["sp2_like_fraction"]))
    print("sp3-like fraction: %.4f -> %.4f" % (first["sp3_like_fraction"], last["sp3_like_fraction"]))
    print("mean coordination: %.4f -> %.4f" % (first["mean_coordination"], last["mean_coordination"]))
    print("density (g/cm^3): %.4f -> %.4f" % (first["density_g_cm3"], last["density_g_cm3"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
