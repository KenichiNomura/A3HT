#!/usr/bin/env python3
"""Cohort state machine and run-record helpers for the A3HT autonomous loop.

A *cohort* is a group of simulation runs that share identical physical
parameters (box, density, flake geometry, MD schedule).  Different seeds
within a cohort produce independent structures so their thermal conductivities
can be averaged to estimate uncertainty.

The stop condition is met when any cohort has:
  - mean kappa  >= target_kappa_w_mk  (target thermal conductivity)
  - stderr/mean <  target_relative_uncertainty_pct / 100
  - at least    >= min_cohort_success_seeds  evaluable runs

All thresholds are read from config.toml [goals].
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


from config import load as _load_config

ROOT = Path(__file__).resolve().parent.parent  # repo root
RUNS_ROOT = Path(os.environ.get("A3HT_RUNS_ROOT", str(ROOT / "my_runs")))

# --- goal constants (from config.toml [goals]) ---
_goals = _load_config()["goals"]
TARGET_KAPPA_W_MK               = _goals["target_kappa_w_mk"]
TARGET_RELATIVE_UNCERTAINTY_PCT = _goals["target_relative_uncertainty_pct"]
MIN_COHORT_SUCCESS_SEEDS        = _goals["min_cohort_success_seeds"]
MAX_SIMULTANEOUS_COHORTS        = int(
    os.environ.get("A3HT_MAX_SIMULTANEOUS_COHORTS", _goals["max_simultaneous_cohorts"])
)


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------

def read_text(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_last_kappa(hotcold_file: Path) -> Optional[float]:
    """Return the final thermal conductivity (W/m-K) from a gc_rebo2_hotcold.dat file.

    The file has comment lines starting with '#' and data rows where column 6
    (0-indexed: 5) is the running kappa estimate.  We take the last data row.
    """
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


# ---------------------------------------------------------------------------
# Cohort identification
# ---------------------------------------------------------------------------

def cohort_id_from_parameters(parameters: Dict[str, Any]) -> str:
    """Stable 12-char hex digest of the sorted parameter dict.

    Two runs with identical recommended_parameters get the same cohort_id
    regardless of the order keys appear in the JSON.
    """
    payload = json.dumps({k: parameters[k] for k in sorted(parameters)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-run plan loading
# ---------------------------------------------------------------------------

def load_plan(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load simulation_plan.json from a run directory, patching in cohort_id if absent."""
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


# ---------------------------------------------------------------------------
# Run-record cache (terminal states never change, so they are cached to disk)
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = {"SUCCESS", "FAILED"}


def _load_records_cache(cache_file: Path) -> Dict[int, Dict[str, Any]]:
    if not cache_file.is_file():
        return {}
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _save_records_cache(cache_file: Path, cache: Dict[int, Dict[str, Any]]) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): v for k, v in cache.items()}, separators=(",", ":")))
        tmp.rename(cache_file)
    except OSError:
        pass


def collect_run_records(runs_root: Path, cache_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan runs_root and return one record dict per seed directory.

    Terminal runs (SUCCESS / FAILED) are served from a JSON cache on disk so
    repeated calls during one queue-fill cycle do not re-parse completed
    directories.  Only RUNNING / PLANNED runs are re-read every time.
    """
    if cache_file is None:
        state_dir = Path(os.environ.get("A3HT_STATE_DIR", str(runs_root.parent / ".queue_state")))
        cache_file = state_dir / "run_records_cache.json"

    # Terminal states (SUCCESS, FAILED) never change — load once and cache forever.
    cache = _load_records_cache(cache_file)

    records = []  # type: List[Dict[str, Any]]
    updated_cache = {}  # type: Dict[int, Dict[str, Any]]

    if not runs_root.exists():
        return records

    for entry in os.scandir(runs_root):
        if not entry.is_dir():
            continue
        try:
            seed = int(entry.name)
        except ValueError:
            continue

        # Serve terminal-state records straight from cache.
        if seed in cache and cache[seed].get("status") in _TERMINAL_STATUSES:
            record = cache[seed]
            records.append(record)
            updated_cache[seed] = record
            continue

        # Re-read for non-terminal or uncached runs.
        run_dir = Path(entry.path)
        plan = load_plan(run_dir)
        status = read_text(run_dir / "run_status.txt")
        if plan is None and status is None:
            continue

        normalized_status = status or "PLANNED"
        record = {
            "seed": seed,
            "run_dir": entry.path,
            "status": normalized_status,
            "plan": plan,
            "kappa_w_mk": read_last_kappa(run_dir / "data" / "gc_rebo2_hotcold.dat") if normalized_status == "SUCCESS" else None,
        }
        if plan is not None:
            params = plan.get("recommended_parameters", {})
            record["cohort_id"] = plan["_meta"].get("cohort_id") or cohort_id_from_parameters(params)
            record["parameters"] = params
        records.append(record)

        # Add terminal runs to cache (strip the large plan dict to keep cache compact).
        if normalized_status in _TERMINAL_STATUSES:
            updated_cache[seed] = {
                "seed": seed,
                "run_dir": entry.path,
                "status": normalized_status,
                "cohort_id": record.get("cohort_id"),
                "parameters": record.get("parameters"),
                "kappa_w_mk": record.get("kappa_w_mk"),
            }

    _save_records_cache(cache_file, updated_cache)
    return records


# ---------------------------------------------------------------------------
# Cohort statistics and loop-action decision
# ---------------------------------------------------------------------------

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
    max_simultaneous_cohorts: int = MAX_SIMULTANEOUS_COHORTS,
) -> Dict[str, Any]:
    """Aggregate run records by cohort and return the next loop action.

    Possible actions
    ----------------
    stop                 A cohort met the kappa + uncertainty target.
    plan_new_cohort      There is room to open a fresh parameter set.
    reuse_active_cohort  All cohort slots are full; add seeds to the neediest one.
    wait_active_cohorts  All slots are full and each has enough running jobs; do nothing.
    """
    # --- group records into per-cohort accumulators ---
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
        elif status == "RUNNING":
            cohort["pending_count"] += 1

        kappa = record.get("kappa_w_mk")
        if isinstance(kappa, (int, float)):
            cohort["kappa_values"].append(float(kappa))

    # --- compute statistics for each cohort and check stop condition ---
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

    # Sort by most-recently-used cohort so the newest work appears last.
    cohort_list.sort(key=lambda item: (item["latest_seed"], item["first_seed"]))

    # --- decide action ---

    # If any cohort already met the target, report stop immediately.
    stop_cohort = None
    for cohort in cohort_list:
        if cohort["stop_met"]:
            stop_cohort = cohort

    if stop_cohort is not None:
        return {
            "target_kappa_w_mk": target_kappa_w_mk,
            "target_relative_uncertainty_pct": target_relative_uncertainty_pct,
            "min_cohort_success_seeds": min_cohort_success_seeds,
            "max_simultaneous_cohorts": max_simultaneous_cohorts,
            "stop_condition_met": True,
            "action": "stop",
            "reason": "A cohort satisfied the target conductivity and uncertainty thresholds.",
            "selected_cohort": stop_cohort,
            "active_cohorts": [],
            "cohorts": cohort_list,
        }

    if not cohort_list:
        return {
            "target_kappa_w_mk": target_kappa_w_mk,
            "target_relative_uncertainty_pct": target_relative_uncertainty_pct,
            "min_cohort_success_seeds": min_cohort_success_seeds,
            "max_simultaneous_cohorts": max_simultaneous_cohorts,
            "stop_condition_met": False,
            "action": "plan_new_cohort",
            "reason": "No existing cohort plans were found.",
            "selected_cohort": None,
            "active_cohorts": [],
            "cohorts": cohort_list,
        }

    # Active cohorts still need more evaluable seeds to reach min_cohort_success_seeds.
    active_cohorts = [
        cohort for cohort in cohort_list if cohort["evaluable_success_count"] < min_cohort_success_seeds
    ]
    # Reusable = active cohorts that cannot reach the minimum even with all their pending jobs.
    reusable_cohorts = [
        cohort
        for cohort in active_cohorts
        if cohort["evaluable_success_count"] + cohort["pending_count"] < min_cohort_success_seeds
    ]
    # Prefer the cohort with the fewest total planned seeds (least investment so far).
    reusable_cohorts.sort(
        key=lambda cohort: (
            cohort["planned_count"],
            cohort["evaluable_success_count"] + cohort["pending_count"],
            cohort["latest_seed"],
        )
    )

    selected_cohort = reusable_cohorts[0] if reusable_cohorts else None
    if len(active_cohorts) < max_simultaneous_cohorts:
        action = "plan_new_cohort"
        reason = "There is room to open another cohort while existing cohorts continue in parallel."
    elif selected_cohort is not None:
        action = "reuse_active_cohort"
        reason = "All cohort slots are occupied, so reuse the open cohort that most needs additional repeated seeds."
    else:
        action = "wait_active_cohorts"
        reason = "The maximum number of simultaneous cohorts is already open and each has enough running jobs to potentially reach the minimum cohort size."

    return {
        "target_kappa_w_mk": target_kappa_w_mk,
        "target_relative_uncertainty_pct": target_relative_uncertainty_pct,
        "min_cohort_success_seeds": min_cohort_success_seeds,
        "max_simultaneous_cohorts": max_simultaneous_cohorts,
        "stop_condition_met": False,
        "action": action,
        "reason": reason,
        "selected_cohort": selected_cohort,
        "active_cohorts": active_cohorts,
        "cohorts": cohort_list,
    }
