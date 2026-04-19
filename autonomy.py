#!/usr/bin/env python3
"""Helpers for autonomous cohort control in A3HT."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "my_runs"

TARGET_KAPPA_W_MK = 6.0
TARGET_RELATIVE_UNCERTAINTY_PCT = 10.0
MIN_COHORT_SUCCESS_SEEDS = 5


def read_text(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_last_kappa(hotcold_file: Path) -> Optional[float]:
    text = read_text(hotcold_file)
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 6:
        return None
    try:
        return float(parts[5])
    except ValueError:
        return None


def canonicalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {key: parameters[key] for key in sorted(parameters)}


def cohort_id_from_parameters(parameters: Dict[str, Any]) -> str:
    payload = json.dumps(canonicalize_parameters(parameters), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_plan(run_dir: Path) -> Optional[Dict[str, Any]]:
    plan_json = run_dir / "simulation_plan.json"
    if not plan_json.is_file():
        return None
    try:
        payload = json.loads(plan_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    params = payload.get("recommended_parameters")
    if not isinstance(params, dict):
        return None
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["_meta"] = meta
    meta.setdefault("cohort_id", cohort_id_from_parameters(params))
    return payload


def collect_run_records(runs_root: Path) -> List[Dict[str, Any]]:
    records = []  # type: List[Dict[str, Any]]
    if not runs_root.exists():
        return records

    for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda path: int(path.name) if path.name.isdigit() else -1):
        try:
            seed = int(run_dir.name)
        except ValueError:
            continue

        plan = load_plan(run_dir)
        status = read_text(run_dir / "run_status.txt")
        if plan is None and status is None:
            continue

        normalized_status = status or "PLANNED"
        record = {
            "seed": seed,
            "run_dir": str(run_dir),
            "status": normalized_status,
            "plan": plan,
            "kappa_w_mk": read_last_kappa(run_dir / "data" / "gc_edip_hotcold.cont.dat"),
        }
        if plan is not None:
            params = plan.get("recommended_parameters", {})
            record["cohort_id"] = plan["_meta"].get("cohort_id") or cohort_id_from_parameters(params)
            record["parameters"] = params
        records.append(record)
    return records


def _sample_stddev(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(variance)


def summarize_loop_state(
    records: List[Dict[str, Any]],
    target_kappa_w_mk: float = TARGET_KAPPA_W_MK,
    target_relative_uncertainty_pct: float = TARGET_RELATIVE_UNCERTAINTY_PCT,
    min_cohort_success_seeds: int = MIN_COHORT_SUCCESS_SEEDS,
) -> Dict[str, Any]:
    cohorts = {}  # type: Dict[str, Dict[str, Any]]
    for record in records:
        cohort_id = record.get("cohort_id")
        if not cohort_id:
            continue
        cohort = cohorts.setdefault(
            cohort_id,
            {
                "cohort_id": cohort_id,
                "parameters": record.get("parameters"),
                "first_seed": record["seed"],
                "latest_seed": record["seed"],
                "planned_count": 0,
                "pending_count": 0,
                "success_count": 0,
                "failed_count": 0,
                "kappa_values": [],
            },
        )
        cohort["parameters"] = cohort.get("parameters") or record.get("parameters")
        cohort["first_seed"] = min(cohort["first_seed"], record["seed"])
        cohort["latest_seed"] = max(cohort["latest_seed"], record["seed"])
        cohort["planned_count"] += 1

        status = record["status"]
        if status == "SUCCESS":
            cohort["success_count"] += 1
        elif status == "FAILED":
            cohort["failed_count"] += 1
        else:
            cohort["pending_count"] += 1

        kappa = record.get("kappa_w_mk")
        if isinstance(kappa, (int, float)):
            cohort["kappa_values"].append(float(kappa))

    cohort_list = []  # type: List[Dict[str, Any]]
    for cohort in cohorts.values():
        kappa_values = list(cohort["kappa_values"])
        evaluable_successes = len(kappa_values)
        mean_kappa = sum(kappa_values) / float(evaluable_successes) if evaluable_successes else None
        stddev_kappa = _sample_stddev(kappa_values)
        stderr_kappa = None
        if stddev_kappa is not None and evaluable_successes > 0:
            stderr_kappa = stddev_kappa / math.sqrt(float(evaluable_successes))
        relative_uncertainty_pct = None
        if stderr_kappa is not None and mean_kappa not in (None, 0.0):
            relative_uncertainty_pct = abs(stderr_kappa / mean_kappa) * 100.0

        stop_met = bool(
            evaluable_successes >= min_cohort_success_seeds
            and mean_kappa is not None
            and relative_uncertainty_pct is not None
            and mean_kappa >= target_kappa_w_mk
            and relative_uncertainty_pct < target_relative_uncertainty_pct
        )

        cohort_summary = {
            "cohort_id": cohort["cohort_id"],
            "parameters": cohort["parameters"],
            "first_seed": cohort["first_seed"],
            "latest_seed": cohort["latest_seed"],
            "planned_count": cohort["planned_count"],
            "pending_count": cohort["pending_count"],
            "success_count": cohort["success_count"],
            "failed_count": cohort["failed_count"],
            "evaluable_success_count": evaluable_successes,
            "mean_kappa_w_mk": mean_kappa,
            "sample_stddev_kappa_w_mk": stddev_kappa,
            "standard_error_kappa_w_mk": stderr_kappa,
            "relative_uncertainty_pct": relative_uncertainty_pct,
            "stop_met": stop_met,
        }
        cohort_list.append(cohort_summary)

    cohort_list.sort(key=lambda item: (item["latest_seed"], item["first_seed"]))

    stop_cohort = None
    for cohort in cohort_list:
        if cohort["stop_met"]:
            stop_cohort = cohort

    if stop_cohort is not None:
        return {
            "target_kappa_w_mk": target_kappa_w_mk,
            "target_relative_uncertainty_pct": target_relative_uncertainty_pct,
            "min_cohort_success_seeds": min_cohort_success_seeds,
            "stop_condition_met": True,
            "action": "stop",
            "reason": "A cohort satisfied the target conductivity and uncertainty thresholds.",
            "active_cohort": stop_cohort,
            "cohorts": cohort_list,
        }

    if not cohort_list:
        return {
            "target_kappa_w_mk": target_kappa_w_mk,
            "target_relative_uncertainty_pct": target_relative_uncertainty_pct,
            "min_cohort_success_seeds": min_cohort_success_seeds,
            "stop_condition_met": False,
            "action": "plan_new_cohort",
            "reason": "No existing cohort plans were found.",
            "active_cohort": None,
            "cohorts": cohort_list,
        }

    active_cohort = cohort_list[-1]
    evaluable = active_cohort["evaluable_success_count"]
    pending = active_cohort["pending_count"]
    if evaluable >= min_cohort_success_seeds:
        action = "plan_new_cohort"
        reason = "The latest cohort has enough completed seeds for evaluation but did not meet the stop condition."
    elif evaluable + pending >= min_cohort_success_seeds:
        action = "wait_active_cohort"
        reason = "The latest cohort already has enough planned/running seeds to reach the minimum cohort size."
    else:
        action = "reuse_active_cohort"
        reason = "The latest cohort needs more repeated seeds with identical physical parameters."

    return {
        "target_kappa_w_mk": target_kappa_w_mk,
        "target_relative_uncertainty_pct": target_relative_uncertainty_pct,
        "min_cohort_success_seeds": min_cohort_success_seeds,
        "stop_condition_met": False,
        "action": action,
        "reason": reason,
        "active_cohort": active_cohort,
        "cohorts": cohort_list,
    }
