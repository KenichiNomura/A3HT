#!/usr/bin/env python3
"""Build a tabular ML dataset from glassy-carbon run outputs.

This script aggregates per-run structural analysis outputs into one feature
table suitable for downstream regression models such as XGBoost.

Features come from:
- analysis/anneal/summary.json
- analysis/nemd/summary.json
- analysis/anneal_timeseries/trajectory_summary.csv
- histogram-style CSV outputs in analysis/anneal and analysis/nemd
- final NEMD thermal-conductivity outputs in data/gc_rebo2_hotcold.dat

Optionally, the script can generate missing analysis artifacts by invoking the
existing analysis scripts in this directory.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYZE_SNAPSHOT = ROOT / "analyze_glassy_carbon.py"
ANALYZE_TRAJECTORY = ROOT / "analyze_glassy_carbon_trajectory.py"

SUMMARY_KEYS = [
    "atom_count",
    "volume_angstrom3",
    "density_g_cm3",
    "bond_count",
    "mean_coordination",
    "sp2_like_fraction",
    "sp3_like_fraction",
    "undercoordinated_fraction",
    "overcoordinated_fraction",
    "bond_length_mean_angstrom",
    "bond_length_std_angstrom",
    "bond_angle_mean_deg",
    "bond_angle_std_deg",
    "threefold_atom_count",
    "threefold_planarity_rms_mean_angstrom",
    "threefold_pyramidalization_mean",
    "threefold_normal_alignment_mean_abs_cos",
]

TIMESERIES_KEYS = [
    "density_g_cm3",
    "mean_coordination",
    "sp2_like_fraction",
    "sp3_like_fraction",
    "undercoordinated_fraction",
    "bond_length_mean_angstrom",
    "bond_length_std_angstrom",
    "bond_angle_mean_deg",
    "bond_angle_std_deg",
    "threefold_planarity_rms_mean_angstrom",
    "threefold_pyramidalization_mean",
    "threefold_normal_alignment_mean_abs_cos",
    "coord_2_count",
    "coord_3_count",
    "coord_4_count",
    "coordlog_n2",
    "coordlog_n3",
    "coordlog_n4",
]

HISTOGRAM_FILES = [
    "bond_angle_distribution.csv",
    "bond_length_distribution.csv",
    "coordination_histogram.csv",
    "rdf.csv",
    "ring_proxy_bond_histogram.csv",
    "threefold_normal_alignment.csv",
    "threefold_planarity_distribution.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=str(ROOT / "my_runs"),
        help="root directory containing per-run subdirectories",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "ml_features.csv"),
        help="output feature table CSV path",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT / "ml_features_summary.json"),
        help="output summary/manifest JSON path",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help="optional specific run ids to include",
    )
    parser.add_argument(
        "--generate-missing-analysis",
        action="store_true",
        help="generate missing analysis outputs before feature extraction",
    )
    parser.add_argument(
        "--overwrite-analysis",
        action="store_true",
        help="when generating, overwrite existing analysis outputs",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="minimum number of anneal trajectory frames required",
    )
    return parser.parse_args()


def _float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = float("nan")
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def list_run_dirs(runs_root: Path, run_ids: Optional[Sequence[str]]) -> List[Path]:
    if run_ids:
        return [runs_root / run_id for run_id in run_ids]
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def generate_analysis(run_dir: Path, overwrite: bool) -> None:
    analysis_dir = run_dir / "analysis"
    data_dir = run_dir / "data"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    anneal_snapshot = data_dir / "anneal_gc_rebo2.data"
    anneal_traj = data_dir / "anneal_gc_rebo2.lammpstrj"
    coordlog = data_dir / "anneal_gc_rebo2_coordination.dat"
    nemd_traj = data_dir / "gc_rebo2_nemd.lammpstrj"

    jobs: List[Tuple[List[str], Path]] = []
    if anneal_snapshot.exists():
        jobs.append(
            (
                [
                    sys.executable,
                    str(ANALYZE_SNAPSHOT),
                    str(anneal_snapshot),
                    "--output-dir",
                    str(analysis_dir / "anneal"),
                ],
                analysis_dir / "anneal" / "summary.json",
            )
        )
    if nemd_traj.exists():
        jobs.append(
            (
                [
                    sys.executable,
                    str(ANALYZE_SNAPSHOT),
                    str(nemd_traj),
                    "--output-dir",
                    str(analysis_dir / "nemd"),
                ],
                analysis_dir / "nemd" / "summary.json",
            )
        )
    if anneal_traj.exists():
        cmd = [
            sys.executable,
            str(ANALYZE_TRAJECTORY),
            str(anneal_traj),
            "--output-dir",
            str(analysis_dir / "anneal_timeseries"),
        ]
        if coordlog.exists():
            cmd.extend(["--coordination-log", str(coordlog)])
        jobs.append((cmd, analysis_dir / "anneal_timeseries" / "trajectory_summary.csv"))

    for cmd, sentinel in jobs:
        if sentinel.exists() and not overwrite:
            continue
        subprocess.run(cmd, check=True)


def extract_summary_features(summary: Mapping[str, object], prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for key in SUMMARY_KEYS:
        features[f"{prefix}_{key}"] = _float(summary.get(key))

    coord_hist = {int(item["coordination"]): float(item["count"]) for item in summary.get("coordination_histogram", [])}
    atom_count = max(1.0, features.get(f"{prefix}_atom_count", float("nan")))
    for coord in (2, 3, 4):
        count = coord_hist.get(coord, 0.0)
        features[f"{prefix}_coord_{coord}_count"] = count
        features[f"{prefix}_coord_{coord}_fraction"] = count / atom_count

    ring_hist = {int(item["ring_size"]): float(item["bond_count"]) for item in summary.get("ring_proxy_bond_histogram", [])}
    total_ring_bonds = sum(ring_hist.values()) or 1.0
    for ring_size in range(3, 9):
        count = ring_hist.get(ring_size, 0.0)
        features[f"{prefix}_ring_{ring_size}_bond_count"] = count
        features[f"{prefix}_ring_{ring_size}_bond_fraction"] = count / total_ring_bonds
    return features


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_centered, y - y.mean()) / denom)


def add_series_stats(features: Dict[str, float], rows: Sequence[Mapping[str, float]], key: str, prefix: str) -> None:
    values = np.array([float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))], dtype=float)
    if values.size == 0:
        return
    idx = np.arange(values.size, dtype=float)
    features[f"{prefix}_{key}_first"] = float(values[0])
    features[f"{prefix}_{key}_last"] = float(values[-1])
    features[f"{prefix}_{key}_delta"] = float(values[-1] - values[0])
    features[f"{prefix}_{key}_mean"] = float(values.mean())
    features[f"{prefix}_{key}_std"] = float(values.std(ddof=0))
    features[f"{prefix}_{key}_min"] = float(values.min())
    features[f"{prefix}_{key}_max"] = float(values.max())
    features[f"{prefix}_{key}_range"] = float(values.max() - values.min())
    features[f"{prefix}_{key}_q25"] = float(np.quantile(values, 0.25))
    features[f"{prefix}_{key}_q50"] = float(np.quantile(values, 0.50))
    features[f"{prefix}_{key}_q75"] = float(np.quantile(values, 0.75))
    features[f"{prefix}_{key}_slope_per_frame"] = linear_slope(idx, values)


def extract_timeseries_features(rows: Sequence[Mapping[str, float]], prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {f"{prefix}_frame_count": float(len(rows))}
    if not rows:
        return features
    for key in TIMESERIES_KEYS:
        add_series_stats(features, rows, key, prefix)
    timesteps = np.array([float(row["timestep"]) for row in rows], dtype=float)
    features[f"{prefix}_timestep_start"] = float(timesteps[0])
    features[f"{prefix}_timestep_end"] = float(timesteps[-1])
    features[f"{prefix}_timestep_span"] = float(timesteps[-1] - timesteps[0])
    return features


def weighted_stats(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    if "bin_center" not in rows[0]:
        return {}
    bin_center = np.array([float(row["bin_center"]) for row in rows], dtype=float)
    weight_key = "count" if "count" in rows[0] else "density"
    weights = np.array([float(row[weight_key]) for row in rows], dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        return {"mass": 0.0}
    probs = weights / total
    mean = float(np.dot(probs, bin_center))
    variance = float(np.dot(probs, (bin_center - mean) ** 2))
    cdf = np.cumsum(probs)

    def quantile(q: float) -> float:
        index = int(np.searchsorted(cdf, q, side="left"))
        index = max(0, min(index, len(bin_center) - 1))
        return float(bin_center[index])

    positive_probs = probs[probs > 0]
    entropy = float(-np.sum(positive_probs * np.log(positive_probs)))

    return {
        "mass": total,
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min_center": float(bin_center[np.argmax(weights > 0)]) if np.any(weights > 0) else float("nan"),
        "max_center": float(bin_center[len(bin_center) - 1 - np.argmax(weights[::-1] > 0)]) if np.any(weights > 0) else float("nan"),
        "peak_center": float(bin_center[int(np.argmax(weights))]),
        "peak_value": float(np.max(weights)),
        "entropy": entropy,
        "q10": quantile(0.10),
        "q25": quantile(0.25),
        "q50": quantile(0.50),
        "q75": quantile(0.75),
        "q90": quantile(0.90),
    }


def rdf_features(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    r = np.array([float(row["r_angstrom"]) for row in rows], dtype=float)
    g_r = np.array([float(row["g_r"]) for row in rows], dtype=float)
    peak_idx = int(np.argmax(g_r))
    cumulative = np.cumsum(np.maximum(g_r, 0.0))
    total = float(cumulative[-1]) if cumulative.size else 0.0

    def crossing(threshold: float) -> float:
        mask = g_r >= threshold
        return float(r[np.argmax(mask)]) if np.any(mask) else float("nan")

    return {
        "peak_r": float(r[peak_idx]),
        "peak_g": float(g_r[peak_idx]),
        "mean_g": float(g_r.mean()),
        "std_g": float(g_r.std(ddof=0)),
        "integral_like": total,
        "first_r_gte_1": crossing(1.0),
        "first_r_gte_2": crossing(2.0),
        "last_g": float(g_r[-1]),
    }


def extract_histogram_features(rows: Sequence[Mapping[str, float]], prefix: str, stem: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if not rows:
        return features
    if stem == "rdf":
        for key, value in rdf_features(rows).items():
            features[f"{prefix}_{stem}_{key}"] = value
        return features
    for key, value in weighted_stats(rows).items():
        features[f"{prefix}_{stem}_{key}"] = value
    return features


def parse_hotcold_target(path: Path) -> Dict[str, float]:
    last_values = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            last_values = [float(value) for value in parts[:6]]
    if last_values is None:
        raise ValueError(f"no data rows found in {path}")
    timestep, hot_avg, cold_avg, d_t, j_z, kappa = last_values
    return {
        "target_final_thermal_conductivity": kappa,
        "target_final_delta_t": d_t,
        "target_final_heat_flux_jz": j_z,
        "target_final_hot_temp": hot_avg,
        "target_final_cold_temp": cold_avg,
        "target_timestep": timestep,
    }


def add_delta_features(features: Dict[str, float], left_prefix: str, right_prefix: str, keys: Sequence[str]) -> None:
    for key in keys:
        left = features.get(f"{left_prefix}_{key}")
        right = features.get(f"{right_prefix}_{key}")
        if left is None or right is None:
            continue
        if math.isnan(left) or math.isnan(right):
            continue
        features[f"delta_{right_prefix}_minus_{left_prefix}_{key}"] = right - left


def collect_run_features(run_dir: Path, min_frames: int) -> Dict[str, float]:
    analysis_dir = run_dir / "analysis"
    data_dir = run_dir / "data"

    anneal_summary_path = analysis_dir / "anneal" / "summary.json"
    nemd_summary_path = analysis_dir / "nemd" / "summary.json"
    trajectory_path = analysis_dir / "anneal_timeseries" / "trajectory_summary.csv"
    hotcold_path = data_dir / "gc_rebo2_hotcold.dat"

    if not anneal_summary_path.exists():
        raise FileNotFoundError(f"missing {anneal_summary_path}")
    if not nemd_summary_path.exists():
        raise FileNotFoundError(f"missing {nemd_summary_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"missing {trajectory_path}")
    if not hotcold_path.exists():
        raise FileNotFoundError(f"missing {hotcold_path}")

    features: Dict[str, float] = {
        "run_id": run_dir.name,
        "run_numeric_id": float(run_dir.name) if run_dir.name.isdigit() else float("nan"),
    }

    anneal_summary = read_json(anneal_summary_path)
    nemd_summary = read_json(nemd_summary_path)
    trajectory_rows = read_csv_rows(trajectory_path)
    if len(trajectory_rows) < min_frames:
        raise ValueError(f"only {len(trajectory_rows)} frames in {trajectory_path}")

    features.update(extract_summary_features(anneal_summary, "anneal"))
    features.update(extract_summary_features(nemd_summary, "nemd"))
    features.update(extract_timeseries_features(trajectory_rows, "anneal_ts"))
    features.update(parse_hotcold_target(hotcold_path))

    add_delta_features(features, "anneal", "nemd", SUMMARY_KEYS)

    for phase in ("anneal", "nemd"):
        phase_dir = analysis_dir / phase
        for filename in HISTOGRAM_FILES:
            path = phase_dir / filename
            if not path.exists():
                continue
            rows = read_csv_rows(path)
            stem = path.stem
            features.update(extract_histogram_features(rows, phase, stem))

    return features


def write_feature_csv(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}, key=lambda x: (x != "run_id", x))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    summary_json = Path(args.summary_json).resolve()

    run_dirs = list_run_dirs(runs_root, args.run_ids)
    rows: List[Dict[str, float]] = []
    skipped: List[Dict[str, str]] = []

    for run_dir in run_dirs:
        if not run_dir.exists():
            skipped.append({"run_id": run_dir.name, "reason": "run directory missing"})
            continue
        try:
            if args.generate_missing_analysis:
                generate_analysis(run_dir, overwrite=args.overwrite_analysis)
            rows.append(collect_run_features(run_dir, min_frames=args.min_frames))
        except Exception as exc:  # pragma: no cover - explicit logging path
            skipped.append({"run_id": run_dir.name, "reason": str(exc)})

    if not rows:
        raise SystemExit("no usable runs found; no feature table written")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_feature_csv(output_csv, rows)

    summary = {
        "runs_root": str(runs_root),
        "row_count": len(rows),
        "feature_count": len(rows[0]),
        "target_column": "target_final_thermal_conductivity",
        "output_csv": str(output_csv),
        "included_run_ids": [row["run_id"] for row in rows],
        "skipped": skipped,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote feature table: {output_csv}")
    print(f"Included runs: {len(rows)}")
    print(f"Skipped runs: {len(skipped)}")
    print(f"Summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
