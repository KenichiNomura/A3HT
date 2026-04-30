#!/usr/bin/env python3
"""Generate per-run simulation plans using ALCF inference endpoint, with random fallback."""

import argparse
import json
import os
import random
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
from config import get as _cfg, load as _load_config, shell_escape

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "simulation_plan_schema.json"
AUTH_SCRIPT = ROOT / "inference_auth_token.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path,
                        default=Path(os.environ.get("A3HT_RUNS_ROOT", str(ROOT.parent / "my_runs"))))
    parser.add_argument("--max-history", type=int, default=_cfg("alcf.max_history", 10))
    parser.add_argument("--disable-planner", action="store_true")
    parser.add_argument("--alcf-model", default=None)
    parser.add_argument("--alcf-endpoint", default=None)
    parser.add_argument("--alcf-auth-script", default=None)
    return parser.parse_args()


def collect_history(runs_root: Path, max_history: int) -> Dict[str, Any]:
    all_records = collect_run_records(runs_root)
    recent_successes: List[Dict[str, Any]] = []
    for record in reversed(all_records):
        if record.get("status") != "SUCCESS":
            continue
        entry: Dict[str, Any] = {"seed": record["seed"], "cohort_id": record.get("cohort_id")}
        if isinstance(record.get("kappa_w_mk"), (int, float)):
            entry["final_kappa_w_mk"] = record["kappa_w_mk"]
        params = record.get("parameters") or {}
        if params:
            entry["parameters"] = {
                k: params[k] for k in ("flake_area_a2", "box_x_a", "box_y_a", "box_z_a", "density_g_cm3")
                if k in params
            }
        recent_successes.append(entry)
        if len(recent_successes) >= max_history:
            break
    recent_successes.reverse()
    return {"recent_successes": recent_successes, "loop_state": summarize_loop_state(all_records)}


def planner_prompt(seed: int, history: Dict[str, Any]) -> str:
    c = _load_config()["constraints"]
    return (
        "You are planning the next MD run for this repository.\n\n"
        f"Target goal: reach {TARGET_KAPPA_W_MK:.1f} W/m-K thermal conductivity with relative "
        f"uncertainty below {TARGET_RELATIVE_UNCERTAINTY_PCT:.1f}%.\n"
        f"Each same-parameter cohort must collect at least {MIN_COHORT_SUCCESS_SEEDS} evaluable seeds.\n"
        f"This plan is for run seed {seed}.\n\n"
        "Hard constraints:\n"
        f"- flake_area_a2: {c['flake_area_a2'][0]}-{c['flake_area_a2'][1]} A^2\n"
        f"- box_x_a: {c['box_x_a'][0]}-{c['box_x_a'][1]} A\n"
        f"- box_y_a: {c['box_y_a'][0]}-{c['box_y_a'][1]} A\n"
        f"- box_z_a: {c['box_z_a'][0]}-{c['box_z_a'][1]} A\n"
        f"- nemd_eflux_ev_ps: {c['nemd_eflux_ev_ps'][0]}-{c['nemd_eflux_ev_ps'][1]} eV/ps\n\n"
        "Return one JSON object matching the provided schema.\n"
        "Keep recommended_parameters concrete and numerically explicit.\n"
        "If history is sparse or inconclusive, prefer conservative defaults and use repeated seeds "
        "for uncertainty estimation.\n\n"
        f"Recent run summary:\n{json.dumps(history, indent=2, sort_keys=True)}\n"
    )


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

    python = _cfg("paths.python", sys.executable)
    if not Path(python).is_file():
        python = sys.executable
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

    client = OpenAI(api_key=token, base_url=endpoint)
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    system_msg = (
        "You are a materials simulation planner. "
        "Respond with a single valid JSON object that matches the schema below exactly. "
        "Do not include any text outside the JSON object.\n\n"
        f"Schema:\n{schema_text}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": planner_prompt(seed, history)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ALCF LLM returned invalid JSON: {exc}\nRaw: {raw[:300]}") from exc


def random_plan(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    c = _load_config()["constraints"]

    def rnd(lo, hi, step):
        steps = int((hi - lo) / step)
        return round(lo + rng.randint(0, steps) * step, 10)

    density = rnd(*c["density_g_cm3"], 0.05)
    box_x   = rnd(*c["box_x_a"], 5.0)
    box_y   = rnd(*c["box_y_a"], 5.0)
    box_z   = rnd(*c["box_z_a"], 10.0)
    eflux   = rnd(*c["nemd_eflux_ev_ps"], 0.5)

    return {
        "reasoning_summary": (
            f"Random fallback plan (ALCF unavailable): "
            f"density={density} g/cm3, box={box_x}x{box_y}x{box_z} A, eflux={eflux} eV/ps."
        ),
        "uncertainty_strategy": (
            "Random parameter exploration to maintain throughput while the primary planner is unavailable."
        ),
        "recommended_parameters": {
            "density_g_cm3":            density,
            "flake_area_a2":            25.0,
            "box_x_a":                  box_x,
            "box_y_a":                  box_y,
            "box_z_a":                  box_z,
            "anneal_timestep_ps":       0.0002,
            "anneal_10ps_steps":        50000,
            "anneal_50ps_steps":        250000,
            "thermalize_temperature_k": 300.0,
            "thermalize_timestep_ps":   0.0001,
            "thermalize_nvt_steps":     300000,
            "thermalize_npt_steps":     300000,
            "thermalize_nve_steps":     300000,
            "nemd_timestep_ps":         0.0001,
            "nemd_slab_width_a":        5.0,
            "nemd_freeze_width_a":      5.0,
            "nemd_bin_size_a":          5.0,
            "nemd_eflux_ev_ps":         eflux,
            "nemd_steps":               2000000,
        },
    }


def _validate_pos(name: str, value: Any, as_int: bool = False):
    if as_int:
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    else:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value) if as_int else float(value)


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan.get("reasoning_summary"), str) or not plan["reasoning_summary"].strip():
        raise ValueError("reasoning_summary must be a non-empty string")
    if not isinstance(plan.get("uncertainty_strategy"), str) or not plan["uncertainty_strategy"].strip():
        raise ValueError("uncertainty_strategy must be a non-empty string")

    params = plan.get("recommended_parameters")
    if not isinstance(params, dict):
        raise ValueError("recommended_parameters must be an object")

    v = {
        "density_g_cm3":            _validate_pos("density_g_cm3", params.get("density_g_cm3")),
        "flake_area_a2":            _validate_pos("flake_area_a2", params.get("flake_area_a2")),
        "box_x_a":                  _validate_pos("box_x_a", params.get("box_x_a")),
        "box_y_a":                  _validate_pos("box_y_a", params.get("box_y_a")),
        "box_z_a":                  _validate_pos("box_z_a", params.get("box_z_a")),
        "anneal_timestep_ps":       _validate_pos("anneal_timestep_ps", params.get("anneal_timestep_ps")),
        "anneal_10ps_steps":        _validate_pos("anneal_10ps_steps", params.get("anneal_10ps_steps"), True),
        "anneal_50ps_steps":        _validate_pos("anneal_50ps_steps", params.get("anneal_50ps_steps"), True),
        "thermalize_temperature_k": _validate_pos("thermalize_temperature_k", params.get("thermalize_temperature_k")),
        "thermalize_timestep_ps":   _validate_pos("thermalize_timestep_ps", params.get("thermalize_timestep_ps")),
        "thermalize_nvt_steps":     _validate_pos("thermalize_nvt_steps", params.get("thermalize_nvt_steps"), True),
        "thermalize_npt_steps":     _validate_pos("thermalize_npt_steps", params.get("thermalize_npt_steps"), True),
        "thermalize_nve_steps":     _validate_pos("thermalize_nve_steps", params.get("thermalize_nve_steps"), True),
        "nemd_timestep_ps":         _validate_pos("nemd_timestep_ps", params.get("nemd_timestep_ps")),
        "nemd_slab_width_a":        _validate_pos("nemd_slab_width_a", params.get("nemd_slab_width_a")),
        "nemd_freeze_width_a":      _validate_pos("nemd_freeze_width_a", params.get("nemd_freeze_width_a")),
        "nemd_bin_size_a":          _validate_pos("nemd_bin_size_a", params.get("nemd_bin_size_a")),
        "nemd_eflux_ev_ps":         _validate_pos("nemd_eflux_ev_ps", params.get("nemd_eflux_ev_ps")),
        "nemd_steps":               _validate_pos("nemd_steps", params.get("nemd_steps"), True),
    }

    c = _load_config()["constraints"]
    checks = [
        ("flake_area_a2", "flake_area_a2"),
        ("box_x_a", "box_x_a"),
        ("box_y_a", "box_y_a"),
        ("box_z_a", "box_z_a"),
        ("nemd_eflux_ev_ps", "nemd_eflux_ev_ps"),
    ]
    for param_key, constraint_key in checks:
        lo, hi = c[constraint_key]
        if not lo <= v[param_key] <= hi:
            raise ValueError(f"{param_key} violates hard constraints [{lo}, {hi}]")
    if 2.0 * (v["nemd_freeze_width_a"] + v["nemd_slab_width_a"]) >= v["box_z_a"]:
        raise ValueError("box_z_a is too short for the requested freeze and slab widths")

    return {
        "reasoning_summary": plan["reasoning_summary"].strip(),
        "uncertainty_strategy": plan["uncertainty_strategy"].strip(),
        "recommended_parameters": v,
    }


def build_reuse_plan(seed: int, active_cohort: Dict[str, Any]) -> Dict[str, Any]:
    needed = max(MIN_COHORT_SUCCESS_SEEDS - int(active_cohort.get("evaluable_success_count") or 0), 0)
    return {
        "reasoning_summary": (
            "Reuse the active cohort parameters to build out the minimum repeated-seed set "
            "needed for uncertainty estimation."
        ),
        "uncertainty_strategy": (
            f"Keep the physical parameters fixed for this cohort and vary only the random seed "
            f"until at least {MIN_COHORT_SUCCESS_SEEDS} evaluable seeds are available."
        ),
        "recommended_parameters": dict(active_cohort["parameters"]),
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
    env: Dict[str, str] = {
        "A3HT_PLAN_SOURCE": source,
        "A3HT_PLANNER_STATUS": "ok" if source in ("alcf_llm", "cohort_reuse") else "degraded",
    }
    if plan["_meta"].get("planner_error"):
        env["A3HT_PLANNER_ERROR"] = str(plan["_meta"]["planner_error"])
    env.update({
        "A3HT_COHORT_ID":                   plan["_meta"]["cohort_id"],
        "A3HT_COHORT_SEED_TARGET":          str(plan["_meta"]["cohort_seed_target"]),
        "A3HT_GOAL_TARGET_KAPPA_W_MK":      f"{plan['_meta']['goal_target_kappa_w_mk']:.6f}",
        "A3HT_GOAL_MAX_REL_UNCERT_PCT":     f"{plan['_meta']['goal_max_relative_uncertainty_pct']:.6f}",
        "A3HT_REASONING_SUMMARY":           plan["reasoning_summary"].replace("\n", " "),
        "A3HT_UNCERTAINTY_STRATEGY":        plan["uncertainty_strategy"].replace("\n", " "),
        "A3HT_RUN_SEED":                    str(seed),
        "A3HT_STRUCTURE_BOX_X_A":           f"{params['box_x_a']:.6f}",
        "A3HT_STRUCTURE_BOX_Y_A":           f"{params['box_y_a']:.6f}",
        "A3HT_STRUCTURE_BOX_Z_A":           f"{params['box_z_a']:.6f}",
        "A3HT_STRUCTURE_DENSITY_G_CM3":     f"{params['density_g_cm3']:.6f}",
        "A3HT_FLAKE_AREA_A2":               f"{params['flake_area_a2']:.6f}",
        "A3HT_ANNEAL_TIMESTEP_PS":          f"{params['anneal_timestep_ps']:.6f}",
        "A3HT_ANNEAL_10PS_STEPS":           str(params["anneal_10ps_steps"]),
        "A3HT_ANNEAL_50PS_STEPS":           str(params["anneal_50ps_steps"]),
        "A3HT_THERMALIZE_TEMPERATURE_K":    f"{params['thermalize_temperature_k']:.6f}",
        "A3HT_THERMALIZE_TIMESTEP_PS":      f"{params['thermalize_timestep_ps']:.6f}",
        "A3HT_THERMALIZE_NVT_STEPS":        str(params["thermalize_nvt_steps"]),
        "A3HT_THERMALIZE_NPT_STEPS":        str(params["thermalize_npt_steps"]),
        "A3HT_THERMALIZE_NVE_STEPS":        str(params["thermalize_nve_steps"]),
        "A3HT_NEMD_TIMESTEP_PS":            f"{params['nemd_timestep_ps']:.6f}",
        "A3HT_NEMD_SLAB_WIDTH_A":           f"{params['nemd_slab_width_a']:.6f}",
        "A3HT_NEMD_FREEZE_WIDTH_A":         f"{params['nemd_freeze_width_a']:.6f}",
        "A3HT_NEMD_BIN_SIZE_A":             f"{params['nemd_bin_size_a']:.6f}",
        "A3HT_NEMD_EFLUX_EV_PS":           f"{params['nemd_eflux_ev_ps']:.6f}",
        "A3HT_NEMD_STEPS":                  str(params["nemd_steps"]),
        "A3HT_ANNEAL_VELOCITY_SEED":        str(seed * 1000 + 101),
        "A3HT_THERMALIZE_VELOCITY_SEED":    str(seed * 1000 + 202),
    })
    return env


def write_env_file(path: Path, values: Dict[str, str]) -> None:
    lines = ['{}="{}"'.format(k, shell_escape(v)) for k, v in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lammps_include(path: Path, values: Dict[str, str]) -> None:
    a = _load_config()["anneal"]
    lines = [
        f"variable anneal_tstart_k equal {a['tstart_k']}",
        f"variable anneal_t1_k equal {a['t1_k']}",
        f"variable anneal_t2_k equal {a['t2_k']}",
        f"variable anneal_t3_k equal {a['t3_k']}",
        f"variable anneal_t4_k equal {a['t4_k']}",
        f"variable anneal_t5_k equal {a['t4_k']}",
        f"variable anneal_tdamp_ps equal {a['tdamp_ps']}",
        f"variable anneal_pdamp_ps equal {a['pdamp_ps']}",
        f"variable anneal_coord_cutoff_a equal {a['coord_cutoff_a']}",
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
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    history = collect_history(args.runs_root.resolve(), args.max_history)
    loop_state = history["loop_state"]

    if loop_state.get("action") == "reuse_active_cohort" and loop_state.get("selected_cohort"):
        plan = build_reuse_plan(args.seed, loop_state["selected_cohort"])
    elif args.disable_planner:
        print("error: planner disabled and no reuse cohort available", file=sys.stderr)
        return 1
    else:
        alcf_model    = args.alcf_model or os.environ.get("A3HT_ALCF_MODEL") or _cfg("alcf.model")
        alcf_endpoint = args.alcf_endpoint or _cfg("alcf.endpoint")
        alcf_auth     = Path(args.alcf_auth_script) if args.alcf_auth_script else AUTH_SCRIPT

        candidate = None
        planner_source = None
        try:
            candidate = run_alcf_llm(args.seed, history, alcf_model, alcf_endpoint, alcf_auth)
            planner_source = "alcf_llm"
        except Exception as exc:
            print(f"warning: ALCF planner failed ({exc}); using random fallback", file=sys.stderr)

        if candidate is None:
            candidate = random_plan(args.seed)
            planner_source = "random_fallback"

        try:
            validated = validate_plan(candidate)
        except Exception as exc:
            print(f"error: plan from {planner_source} failed validation: {exc}", file=sys.stderr)
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
    (run_dir / "simulation_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_env_file(run_dir / "simulation_plan.env", env_values)
    write_lammps_include(run_dir / "simulation_plan.lmp", env_values)

    print(json.dumps({"run_dir": str(run_dir), "planner_source": plan["_meta"]["planner_source"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
