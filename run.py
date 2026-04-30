#!/usr/bin/env python3
#PBS -A FoundMLIP
#PBS -l filesystems=eagle
#PBS -N a3ht
#PBS -q workq
#PBS -l select=4:ncpus=256
#PBS -l walltime=12:00:00
#PBS -j oe

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _resolve_rootdir() -> Path:
    # A3HT_ROOT_DIR is set explicitly by fill_queue; PBS_O_WORKDIR is the PBS fallback
    # when the job is submitted from the repo root; __file__ handles interactive runs.
    if os.environ.get("A3HT_ROOT_DIR"):
        return Path(os.environ["A3HT_ROOT_DIR"]).resolve()
    if os.environ.get("PBS_O_WORKDIR"):
        return Path(os.environ["PBS_O_WORKDIR"]).resolve()
    return Path(__file__).resolve().parent


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Module-level state updated throughout execution; used by fail() and finish_run.
stage = "startup"
status_file = None   # Path set after run_dir is known
failure_file = None  # Path set after run_dir is known


def fail(message: str) -> None:
    if failure_file is not None:
        failure_file.write_text(
            f"timestamp={timestamp_utc()}\nstage={stage}\nmessage={message}\n"
        )
    if status_file is not None:
        status_file.write_text("FAILED\n")
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def detect_ntasks() -> int:
    # PBS_NODEFILE lists one hostname per slot, so unique nodes × PPN gives total MPI ranks.
    # PBS_NP is a single total-rank count and is used as a fallback when PPN is unavailable.
    default_ppn = 128
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if nodefile and Path(nodefile).is_file():
        unique_nodes = len(set(Path(nodefile).read_text().splitlines()))
        ppn_str = os.environ.get("PBS_NUM_PPN", "")
        if re.match(r"^[1-9][0-9]*$", ppn_str):
            return unique_nodes * int(ppn_str)
        return unique_nodes * default_ppn
    np_str = os.environ.get("PBS_NP", "")
    if re.match(r"^[1-9][0-9]*$", np_str):
        return int(np_str)
    return 128


def load_plan_env(plan_env_path: Path) -> dict:
    """Parse KEY="value" lines into os.environ and return as a dict."""
    env = {}
    for line in plan_env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"')
        env[k] = v
        os.environ[k] = v
    return env


def run_lammps(log_name, input_file, mpi_launcher, ntasks, lmp_bin, procx, procy, procz):
    cmd = mpi_launcher + [
        str(ntasks), lmp_bin,
        "-var", "procx", str(procx),
        "-var", "procy", str(procy),
        "-var", "procz", str(procz),
        "-log", log_name,
        "-in", str(input_file),
    ]
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    global stage, status_file, failure_file

    rootdir = _resolve_rootdir()
    os.chdir(rootdir)
    sys.path.insert(0, str(rootdir / "src"))
    import config as _config  # noqa: E402 (src/config.py)

    os.environ["LD_LIBRARY_PATH"] = _config._runtime_lib_path()
    lammps_dir = _config._resolve(_config.load().get("paths", {}).get("lammps_dir", ""))
    os.environ["PATH"] = lammps_dir + os.pathsep + os.environ.get("PATH", "")

    # MPI launcher
    mpi_launcher = None
    for name in ("mpiexec", "mpirun"):
        if shutil.which(name):
            mpi_launcher = [name, "-n"]
            break
    if mpi_launcher is None:
        print("error: neither mpiexec nor mpirun is available in PATH", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        usage="%(prog)s [--seed N] [--ntasks N] [--processors auto|Px,Py,Pz]"
    )
    parser.add_argument("--seed",       type=str, default=None)
    parser.add_argument("--ntasks",     type=int, default=detect_ntasks())
    parser.add_argument("--processors", type=str, default="auto")
    args = parser.parse_args()

    seed_str = args.seed or os.environ.get("A3HT_SEED", "123")
    if not re.match(r"^[0-9]+$", seed_str):
        print("error: --seed must be a non-negative integer", file=sys.stderr)
        return 1
    seed = int(seed_str)
    ntasks = args.ntasks

    structure_base_angle_deg    = os.environ.get("A3HT_STRUCTURE_BASE_ANGLE_DEG",    "90.0")
    structure_angle_disturb_deg = os.environ.get("A3HT_STRUCTURE_ANGLE_DISTURB_DEG", "20.0")
    structure_tilt_max_deg      = os.environ.get("A3HT_STRUCTURE_TILT_MAX_DEG",      "90.0")

    cfg = _config.load()
    campaign_name = cfg.get("campaign", {}).get("name", "default")
    runs_root_default = rootdir / "campaigns" / campaign_name / "my_runs"
    runs_root = Path(os.environ.get("A3HT_RUNS_ROOT", str(runs_root_default)))

    run_dir = runs_root / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    status_file  = run_dir / "run_status.txt"
    failure_file = run_dir / "run_failure.txt"
    plan_env     = run_dir / "simulation_plan.env"
    plan_lmp     = run_dir / "simulation_plan.lmp"
    plan_json    = run_dir / "simulation_plan.json"

    status_file.write_text("RUNNING\n")
    try:
        failure_file.unlink()
    except FileNotFoundError:
        pass

    if not re.match(r"^[1-9][0-9]*$", str(ntasks)):
        fail("--ntasks must be a positive integer")

    if args.processors == "auto":
        procx = procy = procz = "*"  # LAMMPS wildcard: let it choose the decomposition
    else:
        m = re.fullmatch(r"([1-9][0-9]*),([1-9][0-9]*),([1-9][0-9]*)", args.processors)
        if not m:
            fail("--processors must be 'auto' or 'Px,Py,Pz'")
        procx, procy, procz = m.group(1), m.group(2), m.group(3)
        if int(procx) * int(procy) * int(procz) != ntasks:
            fail(f"--processors must satisfy Px*Py*Pz = ntasks ({ntasks})")

    lmp_bin = os.environ.get("LAMMPS_BIN", str(Path(lammps_dir) / "lmp"))
    planner_script = Path(
        os.environ.get("A3HT_PLANNER_SCRIPT", str(rootdir / "src" / "plan_simulation.py"))
    )
    print(lmp_bin)

    success = False
    try:
        stage = "environment_check"
        if not os.access(lmp_bin, os.X_OK):
            fail(f"LAMMPS executable not found: {lmp_bin}")
        if not (rootdir / "nemd" / "CH.rebo").is_file():
            fail(f"REBO2 parameter file not found: {rootdir / 'nemd' / 'CH.rebo'}")

        os.chdir(run_dir)

        stage = "simulation_planning"
        # Re-run the planner if any plan file is missing; fill_queue normally pre-generates them.
        if not plan_env.exists() or not plan_lmp.exists() or not plan_json.exists():
            if not planner_script.is_file():
                fail(f"planner script not found: {planner_script}")
            result = subprocess.run(
                [
                    sys.executable, str(planner_script),
                    "--seed", str(seed),
                    "--run-dir", str(run_dir),
                    "--runs-root", str(runs_root),
                ],
            )
            if result.returncode != 0:
                fail(f"simulation planner failed for seed {seed}")

        if not plan_env.exists():
            fail(f"simulation plan env file not found: {plan_env}")
        if not plan_lmp.exists():
            fail(f"simulation plan LAMMPS include not found: {plan_lmp}")

        plan_vars = load_plan_env(plan_env)

        if plan_vars.get("A3HT_PLANNER_STATUS", "ok") != "ok":
            warn_msg = (
                f"Planner degraded: source={plan_vars.get('A3HT_PLAN_SOURCE', 'unknown')}\n"
                f"{plan_vars.get('A3HT_PLANNER_ERROR', '')}"
            )
            Path("planner_warning.txt").write_text(warn_msg)
            print(
                f"warning: Planner degraded: source={plan_vars.get('A3HT_PLAN_SOURCE', 'unknown')}",
                file=sys.stderr,
            )
        print(
            f"Plan source: {plan_vars.get('A3HT_PLAN_SOURCE', '')}  "
            f"cohort: {plan_vars.get('A3HT_COHORT_ID', '')}  "
            f"target_kappa: {plan_vars.get('A3HT_GOAL_TARGET_KAPPA_W_MK', '')} W/m-K"
        )
        print(
            f"Orientation: base={structure_base_angle_deg} "
            f"disturb={structure_angle_disturb_deg} tilt={structure_tilt_max_deg}"
        )

        stage = "structure_generation"
        subprocess.run(
            [
                sys.executable,
                str(rootdir / "src" / "generate_random_carbon.py"),
                "--box",
                plan_vars["A3HT_STRUCTURE_BOX_X_A"],
                plan_vars["A3HT_STRUCTURE_BOX_Y_A"],
                plan_vars["A3HT_STRUCTURE_BOX_Z_A"],
                "--density",      plan_vars["A3HT_STRUCTURE_DENSITY_G_CM3"],
                "--seed",         str(seed),
                "--output",       "random_carbon.extxyz",
                "--flake-area",   plan_vars["A3HT_FLAKE_AREA_A2"],
                "--base-angle-deg",    structure_base_angle_deg,
                "--angle-disturb-deg", structure_angle_disturb_deg,
                "--tilt-max-deg",      structure_tilt_max_deg,
                "--format", "lammps",
            ],
            check=True,
        )
        Path("random_carbon.extxyz").rename("random_carbon.dat")
        shutil.copy2(str(rootdir / "nemd" / "CH.rebo"), "CH.rebo")
        print(f"Copied {rootdir / 'nemd' / 'CH.rebo'} -> CH.rebo")

        stage = "anneal"
        run_lammps(
            "anneal.log", rootdir / "nemd" / "anneal.in",
            mpi_launcher, ntasks, lmp_bin, procx, procy, procz,
        )
        shutil.copy2("data/anneal_gc_rebo2.restart", "gc_rebo2.restart")
        print("Copied data/anneal_gc_rebo2.restart -> gc_rebo2.restart")

        stage = "thermalize"
        run_lammps(
            "thermalize.log", rootdir / "nemd" / "thermalize.in",
            mpi_launcher, ntasks, lmp_bin, procx, procy, procz,
        )
        shutil.copy2("data/gc_rebo2_thermalize.restart", "gc_rebo2.restart")
        print("Copied data/gc_rebo2_thermalize.restart -> gc_rebo2.restart")

        stage = "nemd"
        run_lammps(
            "nemd.log", rootdir / "nemd" / "nemd.in",
            mpi_launcher, ntasks, lmp_bin, procx, procy, procz,
        )

        success = True
        return 0

    finally:
        # Always write a terminal status so autonomy.py can detect completion or failure
        # even when an exception bypasses the normal return path.
        if status_file is not None:
            if success:
                try:
                    failure_file.unlink()
                except FileNotFoundError:
                    pass
                status_file.write_text("SUCCESS\n")
            else:
                if not failure_file.exists():
                    failure_file.write_text(
                        f"timestamp={timestamp_utc()}\nstage={stage}\n"
                        f"message=run exited with nonzero status\n"
                    )
                status_file.write_text("FAILED\n")


if __name__ == "__main__":
    sys.exit(main())
