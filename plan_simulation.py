#!/usr/bin/env python3
"""Generate per-run simulation plans using ALCF inference endpoint, with random fallback."""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from autonomy import (
    MIN_COHORT_SUCCESS_SEEDS,
    TARGET_KAPPA_W_MK,
    TARGET_RELATIVE_UNCERTAINTY_PCT,
    cohort_id_from_parameters,
    collect_run_records,
    summarize_loop_state,
)

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "my_runs"
SCHEMA_PATH = ROOT / "simulation_plan_schema.json"
AUTH_SCRIPT = ROOT / "inference_auth_token.py"
ALCF_ENDPOINT = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
ALCF_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
ALCF_PYTHON = "/home/knomura/lammps/.venv/bin/python3"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="run seed to plan for")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run directory where planning artifacts should be written",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="root directory containing historical runs",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=10,
        help="maximum number of recent successful runs to summarize for the planner",
    )
    parser.add_argument(
        "--disable-planner",
        action="store_true",
        help="skip ALCF planner; fails unless a cohort reuse is available (no random fallback)",
    )
    parser.add_argument(
        "--alcf-model",
        default=None,
        help=f"ALCF model name; default: A3HT_ALCF_MODEL env var, then {ALCF_DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--alcf-endpoint",
        default=None,
        help=f"ALCF vLLM endpoint URL; default: {ALCF_ENDPOINT}",
    )
    parser.add_argument(
        "--alcf-auth-script",
        default=None,
        help=f"path to inference_auth_token.py; default: {AUTH_SCRIPT}",
    )
    return parser.parse_args()



def collect_history(runs_root: Path, max_history: int) -> Dict[str, Any]:
    all_records = collect_run_records(runs_root)
    recent_successes = []
    for record in reversed(all_records):
        if record.get("status") != "SUCCESS":
            continue
        entry = {"seed": record["seed"], "cohort_id": record.get("cohort_id")}
        if isinstance(record.get("kappa_w_mk"), (int, float)):
            entry["final_kappa_w_mk"] = record["kappa_w_mk"]
        params = record.get("parameters") or {}
        if params:
            entry["parameters"] = {
                key: params[key]
                for key in ("flake_area_a2", "box_x_a", "box_y_a", "box_z_a", "density_g_cm3")
                if key in params
            }
        recent_successes.append(entry)
        if len(recent_successes) >= max_history:
            break
    recent_successes.reverse()
    return {
        "recent_successes": recent_successes,
        "loop_state": summarize_loop_state(all_records),
    }


def planner_prompt(seed: int, history: Dict[str, Any]) -> str:
    return (
        "You are planning the next MD run for this repository.\n\n"
        f"Target goal: reach {TARGET_KAPPA_W_MK:.1f} W/m-K thermal conductivity with relative "
        f"uncertainty below {TARGET_RELATIVE_UNCERTAINTY_PCT:.1f}%.\n"
        f"Each same-parameter cohort must collect at least {MIN_COHORT_SUCCESS_SEEDS} evaluable seeds.\n"
        f"This plan is for run seed {seed}. The same physical parameter set may be reused across "
        "different random seeds to estimate uncertainty.\n\n"
        "Hard constraints:\n"
        "- flake_area_a2 must remain within 25-100 A^2\n"
        "- box_x_a must remain within 20-50 A\n"
        "- box_y_a must remain within 20-50 A\n"
        "- box_z_a must remain within 40-100 A\n"
        "- nemd_eflux_ev_ps must remain within 1-3 eV/ps\n\n"
        "Return one JSON object matching the provided schema.\n"
        "Keep recommended_parameters concrete and numerically explicit.\n"
        "If history is sparse or inconclusive, prefer conservative defaults and use repeated seeds "
        "for uncertainty estimation.\n\n"
        "Recent run summary:\n"
        f"{json.dumps(history, indent=2, sort_keys=True)}\n"
    )



def get_alcf_token(auth_script: Path) -> str:
    python = ALCF_PYTHON if Path(ALCF_PYTHON).is_file() else sys.executable
    result = subprocess.run(
        [python, str(auth_script), "get_access_token"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"inference_auth_token.py failed: {result.stderr.strip()}")
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("inference_auth_token.py returned an empty token")
    return token


def run_alcf_llm(
    seed: int,
    history: Dict[str, Any],
    model: str,
    endpoint: str,
    auth_script: Path,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package not installed; run: pip install openai") from exc

    token = get_alcf_token(auth_script)
    client = OpenAI(api_key=token, base_url=endpoint)

    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    system_msg = (
        "You are a materials simulation planner. "
        "Respond with a single valid JSON object that matches the schema below exactly. "
        "Do not include any text outside the JSON object.\n\n"
        f"Schema:\n{schema_text}"
    )
    user_msg = planner_prompt(seed, history)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ALCF LLM returned invalid JSON: {exc}\nRaw response: {raw[:300]}") from exc


def random_plan(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)

    def rnd(lo, hi, step):
        steps = int((hi - lo) / step)
        return round(lo + rng.randint(0, steps) * step, 10)

    density  = rnd(1.5, 2.0, 0.05)
    box_x    = rnd(20.0, 50.0, 5.0)
    box_y    = rnd(20.0, 50.0, 5.0)
    box_z    = rnd(40.0, 100.0, 10.0)
    eflux    = rnd(1.0, 3.0, 0.5)

    return {
        "reasoning_summary": (
            f"Random fallback plan (ALCF unavailable): "
            f"density={density} g/cm3, box={box_x}x{box_y}x{box_z} A, eflux={eflux} eV/ps."
        ),
        "uncertainty_strategy": (
            "Random parameter exploration to maintain throughput while the primary planner is unavailable."
        ),
        "recommended_parameters": {
            "density_g_cm3":          density,
            "flake_area_a2":          25.0,
            "box_x_a":                box_x,
            "box_y_a":                box_y,
            "box_z_a":                box_z,
            "anneal_timestep_ps":     0.0002,
            "anneal_10ps_steps":      50000,
            "anneal_50ps_steps":      250000,
            "thermalize_temperature_k": 300.0,
            "thermalize_timestep_ps": 0.0001,
            "thermalize_nvt_steps":   300000,
            "thermalize_npt_steps":   300000,
            "thermalize_nve_steps":   300000,
            "nemd_timestep_ps":       0.0001,
            "nemd_slab_width_a":      5.0,
            "nemd_freeze_width_a":    5.0,
            "nemd_bin_size_a":        5.0,
            "nemd_eflux_ev_ps":       eflux,
            "nemd_steps":             2000000,
        },
    }


def validate_positive_number(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def validate_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan.get("reasoning_summary"), str) or not plan["reasoning_summary"].strip():
        raise ValueError("reasoning_summary must be a non-empty string")
    if not isinstance(plan.get("uncertainty_strategy"), str) or not plan["uncertainty_strategy"].strip():
        raise ValueError("uncertainty_strategy must be a non-empty string")

    params = plan.get("recommended_parameters")
    if not isinstance(params, dict):
        raise ValueError("recommended_parameters must be an object")

    validated = {
        "density_g_cm3": validate_positive_number("density_g_cm3", params.get("density_g_cm3")),
        "flake_area_a2": validate_positive_number("flake_area_a2", params.get("flake_area_a2")),
        "box_x_a": validate_positive_number("box_x_a", params.get("box_x_a")),
        "box_y_a": validate_positive_number("box_y_a", params.get("box_y_a")),
        "box_z_a": validate_positive_number("box_z_a", params.get("box_z_a")),
        "anneal_timestep_ps": validate_positive_number("anneal_timestep_ps", params.get("anneal_timestep_ps")),
        "anneal_10ps_steps": validate_positive_int("anneal_10ps_steps", params.get("anneal_10ps_steps")),
        "anneal_50ps_steps": validate_positive_int("anneal_50ps_steps", params.get("anneal_50ps_steps")),
        "thermalize_temperature_k": validate_positive_number(
            "thermalize_temperature_k", params.get("thermalize_temperature_k")
        ),
        "thermalize_timestep_ps": validate_positive_number(
            "thermalize_timestep_ps", params.get("thermalize_timestep_ps")
        ),
        "thermalize_nvt_steps": validate_positive_int(
            "thermalize_nvt_steps", params.get("thermalize_nvt_steps")
        ),
        "thermalize_npt_steps": validate_positive_int(
            "thermalize_npt_steps", params.get("thermalize_npt_steps")
        ),
        "thermalize_nve_steps": validate_positive_int(
            "thermalize_nve_steps", params.get("thermalize_nve_steps")
        ),
        "nemd_timestep_ps": validate_positive_number("nemd_timestep_ps", params.get("nemd_timestep_ps")),
        "nemd_slab_width_a": validate_positive_number("nemd_slab_width_a", params.get("nemd_slab_width_a")),
        "nemd_freeze_width_a": validate_positive_number("nemd_freeze_width_a", params.get("nemd_freeze_width_a")),
        "nemd_bin_size_a": validate_positive_number("nemd_bin_size_a", params.get("nemd_bin_size_a")),
        "nemd_eflux_ev_ps": validate_positive_number("nemd_eflux_ev_ps", params.get("nemd_eflux_ev_ps")),
        "nemd_steps": validate_positive_int("nemd_steps", params.get("nemd_steps")),
    }

    if not 25.0 <= validated["flake_area_a2"] <= 100.0:
        raise ValueError("flake_area_a2 violates hard constraints")
    if not 20.0 <= validated["box_x_a"] <= 50.0:
        raise ValueError("box_x_a violates hard constraints")
    if not 20.0 <= validated["box_y_a"] <= 50.0:
        raise ValueError("box_y_a violates hard constraints")
    if not 40.0 <= validated["box_z_a"] <= 100.0:
        raise ValueError("box_z_a violates hard constraints")
    if not 1.0 <= validated["nemd_eflux_ev_ps"] <= 3.0:
        raise ValueError("nemd_eflux_ev_ps violates hard constraints")
    if 2.0 * (validated["nemd_freeze_width_a"] + validated["nemd_slab_width_a"]) >= validated["box_z_a"]:
        raise ValueError("box_z_a is too short for the requested freeze and slab widths")

    return {
        "reasoning_summary": plan["reasoning_summary"].strip(),
        "uncertainty_strategy": plan["uncertainty_strategy"].strip(),
        "recommended_parameters": validated,
    }


def build_reuse_plan(seed: int, active_cohort: Dict[str, Any]) -> Dict[str, Any]:
    parameters = dict(active_cohort["parameters"])
    needed = max(MIN_COHORT_SUCCESS_SEEDS - int(active_cohort.get("evaluable_success_count") or 0), 0)
    return {
        "reasoning_summary": (
            "Reuse the active cohort parameters to build out the minimum repeated-seed set "
            "needed for uncertainty estimation."
        ),
        "uncertainty_strategy": (
            "Keep the physical parameters fixed for this cohort and vary only the random seed "
            "until at least {} evaluable seeds are available.".format(MIN_COHORT_SUCCESS_SEEDS)
        ),
        "recommended_parameters": parameters,
        "_meta": {
            "planner_source": "cohort_reuse",
            "goal_target_kappa_w_mk": TARGET_KAPPA_W_MK,
            "goal_max_relative_uncertainty_pct": TARGET_RELATIVE_UNCERTAINTY_PCT,
            "cohort_id": active_cohort["cohort_id"],
            "cohort_seed_target": MIN_COHORT_SUCCESS_SEEDS,
            "cohort_repeat_seed": seed,
            "cohort_remaining_needed_evaluable": needed,
        },
    }


def plan_to_env(seed: int, plan: Dict[str, Any]) -> Dict[str, str]:
    params = plan["recommended_parameters"]
    source = plan["_meta"]["planner_source"]
    status = "ok" if source in ("alcf_llm", "cohort_reuse") else "degraded"
    env: Dict[str, str] = {
        "A3HT_PLAN_SOURCE": source,
        "A3HT_PLANNER_STATUS": status,
    }
    if plan["_meta"].get("planner_error"):
        env["A3HT_PLANNER_ERROR"] = str(plan["_meta"]["planner_error"])
    env.update({
        "A3HT_COHORT_ID": plan["_meta"]["cohort_id"],
        "A3HT_COHORT_SEED_TARGET": str(plan["_meta"]["cohort_seed_target"]),
        "A3HT_GOAL_TARGET_KAPPA_W_MK": f"{plan['_meta']['goal_target_kappa_w_mk']:.6f}",
        "A3HT_GOAL_MAX_REL_UNCERT_PCT": f"{plan['_meta']['goal_max_relative_uncertainty_pct']:.6f}",
        "A3HT_REASONING_SUMMARY": plan["reasoning_summary"].replace("\n", " "),
        "A3HT_UNCERTAINTY_STRATEGY": plan["uncertainty_strategy"].replace("\n", " "),
        "A3HT_RUN_SEED": str(seed),
        "A3HT_STRUCTURE_BOX_X_A": f"{params['box_x_a']:.6f}",
        "A3HT_STRUCTURE_BOX_Y_A": f"{params['box_y_a']:.6f}",
        "A3HT_STRUCTURE_BOX_Z_A": f"{params['box_z_a']:.6f}",
        "A3HT_STRUCTURE_DENSITY_G_CM3": f"{params['density_g_cm3']:.6f}",
        "A3HT_FLAKE_AREA_A2": f"{params['flake_area_a2']:.6f}",
        "A3HT_ANNEAL_TIMESTEP_PS": f"{params['anneal_timestep_ps']:.6f}",
        "A3HT_ANNEAL_10PS_STEPS": str(params["anneal_10ps_steps"]),
        "A3HT_ANNEAL_50PS_STEPS": str(params["anneal_50ps_steps"]),
        "A3HT_THERMALIZE_TEMPERATURE_K": f"{params['thermalize_temperature_k']:.6f}",
        "A3HT_THERMALIZE_TIMESTEP_PS": f"{params['thermalize_timestep_ps']:.6f}",
        "A3HT_THERMALIZE_NVT_STEPS": str(params["thermalize_nvt_steps"]),
        "A3HT_THERMALIZE_NPT_STEPS": str(params["thermalize_npt_steps"]),
        "A3HT_THERMALIZE_NVE_STEPS": str(params["thermalize_nve_steps"]),
        "A3HT_NEMD_TIMESTEP_PS": f"{params['nemd_timestep_ps']:.6f}",
        "A3HT_NEMD_SLAB_WIDTH_A": f"{params['nemd_slab_width_a']:.6f}",
        "A3HT_NEMD_FREEZE_WIDTH_A": f"{params['nemd_freeze_width_a']:.6f}",
        "A3HT_NEMD_BIN_SIZE_A": f"{params['nemd_bin_size_a']:.6f}",
        "A3HT_NEMD_EFLUX_EV_PS": f"{params['nemd_eflux_ev_ps']:.6f}",
        "A3HT_NEMD_STEPS": str(params["nemd_steps"]),
        "A3HT_ANNEAL_VELOCITY_SEED": str(seed * 1000 + 101),
        "A3HT_THERMALIZE_VELOCITY_SEED": str(seed * 1000 + 202),
    })
    return env


def shell_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


def write_env_file(path: Path, values: Dict[str, str]) -> None:
    lines = ['{}="{}"'.format(key, shell_escape(value)) for key, value in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lammps_include(path: Path, values: Dict[str, str]) -> None:
    text = "\n".join(
        [
            f"variable anneal_tstart_k equal 1500.0",
            f"variable anneal_t1_k equal 2500.0",
            f"variable anneal_t2_k equal 3000.0",
            f"variable anneal_t3_k equal 3500.0",
            f"variable anneal_t4_k equal 4000.0",
            f"variable anneal_t5_k equal 4000.0",
            f"variable anneal_tdamp_ps equal 0.1",
            f"variable anneal_pdamp_ps equal 1.0",
            f"variable anneal_coord_cutoff_a equal 1.85",
            f"variable anneal_timestep_ps equal {values['A3HT_ANNEAL_TIMESTEP_PS']}",
            f"variable anneal_10ps_steps equal {values['A3HT_ANNEAL_10PS_STEPS']}",
            f"variable anneal_50ps_steps equal {values['A3HT_ANNEAL_50PS_STEPS']}",
            f"variable anneal_velocity_seed equal {values['A3HT_ANNEAL_VELOCITY_SEED']}",
            f"variable thermalize_temperature_k equal {values['A3HT_THERMALIZE_TEMPERATURE_K']}",
            f"variable thermalize_timestep_ps equal {values['A3HT_THERMALIZE_TIMESTEP_PS']}",
            f"variable thermalize_velocity_seed equal {values['A3HT_THERMALIZE_VELOCITY_SEED']}",
            f"variable thermalize_slab_width_a equal 5.0",
            f"variable thermalize_bin_size_a equal 5.0",
            f"variable thermalize_eflux_ev_ps equal 1.0",
            f"variable thermalize_nvt_steps equal {values['A3HT_THERMALIZE_NVT_STEPS']}",
            f"variable thermalize_npt_steps equal {values['A3HT_THERMALIZE_NPT_STEPS']}",
            f"variable thermalize_nve_steps equal {values['A3HT_THERMALIZE_NVE_STEPS']}",
            f"variable nemd_timestep_ps equal {values['A3HT_NEMD_TIMESTEP_PS']}",
            f"variable nemd_slab_width_a equal {values['A3HT_NEMD_SLAB_WIDTH_A']}",
            f"variable nemd_freeze_width_a equal {values['A3HT_NEMD_FREEZE_WIDTH_A']}",
            f"variable nemd_bin_size_a equal {values['A3HT_NEMD_BIN_SIZE_A']}",
            f"variable nemd_eflux_ev_ps equal {values['A3HT_NEMD_EFLUX_EV_PS']}",
            f"variable nemd_steps equal {values['A3HT_NEMD_STEPS']}",
        ]
    )
    path.write_text(text + "\n", encoding="ascii")


def sanitize_note(message: str) -> str:
    return re.sub(r"\s+", " ", message).strip()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    history = collect_history(args.runs_root.resolve(), args.max_history)
    loop_state = history["loop_state"]

    if loop_state.get("action") == "reuse_active_cohort" and loop_state.get("selected_cohort"):
        plan = build_reuse_plan(args.seed, loop_state["selected_cohort"])
    elif args.disable_planner:
        print("error: planner disabled via --disable-planner and no reuse cohort available", file=sys.stderr)
        return 1
    else:
        alcf_model    = args.alcf_model or os.environ.get("A3HT_ALCF_MODEL") or ALCF_DEFAULT_MODEL
        alcf_endpoint = args.alcf_endpoint or ALCF_ENDPOINT
        alcf_auth     = Path(args.alcf_auth_script) if args.alcf_auth_script else AUTH_SCRIPT

        # --- primary: ALCF inference endpoint ---
        candidate = None
        planner_source = None
        try:
            candidate = run_alcf_llm(args.seed, history, alcf_model, alcf_endpoint, alcf_auth)
            planner_source = "alcf_llm"
        except Exception as alcf_exc:
            print(f"warning: ALCF planner failed ({alcf_exc}); using random fallback", file=sys.stderr)

        # --- fallback: random parameter exploration ---
        if candidate is None:
            candidate = random_plan(args.seed)
            planner_source = "random_fallback"

        try:
            validated = validate_plan(candidate)
        except Exception as val_exc:
            print(
                f"error: plan from {planner_source} failed validation: {val_exc}",
                file=sys.stderr,
            )
            return 1

        cohort_id = cohort_id_from_parameters(validated["recommended_parameters"])
        plan = {
            **validated,
            "_meta": {
                "planner_source": planner_source,
                "goal_target_kappa_w_mk": TARGET_KAPPA_W_MK,
                "goal_max_relative_uncertainty_pct": TARGET_RELATIVE_UNCERTAINTY_PCT,
                "cohort_id": cohort_id,
                "cohort_seed_target": MIN_COHORT_SUCCESS_SEEDS,
            },
        }

    plan["_meta"].setdefault("cohort_id", cohort_id_from_parameters(plan["recommended_parameters"]))
    plan["_meta"].setdefault("cohort_seed_target", MIN_COHORT_SUCCESS_SEEDS)

    env_values = plan_to_env(args.seed, plan)

    (run_dir / "simulation_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_env_file(run_dir / "simulation_plan.env", env_values)
    write_lammps_include(run_dir / "simulation_plan.lmp", env_values)

    print(json.dumps({"run_dir": str(run_dir), "planner_source": plan["_meta"]["planner_source"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
