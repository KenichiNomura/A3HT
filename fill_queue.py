#!/usr/bin/env python3
"""Fill the PBS job queue up to the configured target number of active jobs."""

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
import config  # noqa: E402 (src/config.py)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(log_file: Path, message: str) -> None:
    with log_file.open("a") as fh:
        fh.write(f"{timestamp_utc()} {message}\n")


def require_cmd(name: str, log_file: Path) -> str:
    path = shutil.which(name)
    if path:
        return path
    # PBS tools are often absent from PATH when invoked by cron; probe known locations.
    for candidate in (
        f"/opt/pbs/bin/{name}",
        f"/usr/local/pbs/bin/{name}",
        f"/usr/pbs/bin/{name}",
    ):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    log(log_file, f"missing required command: {name}")
    sys.exit(1)


def peek_next_seed(counter_file: Path, initial_seed: int) -> int:
    if not counter_file.exists():
        counter_file.write_text(f"{initial_seed}\n")
    return int(counter_file.read_text().strip())


def advance_next_seed(counter_file: Path, seed: int) -> None:
    counter_file.write_text(f"{seed + 1}\n")


def _active_lines(path: Path):
    """Yield first token of each non-blank, non-comment line."""
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped.split()[0]


def peek_retry_seed(retry_file: Path):
    if not retry_file.exists():
        return None
    for token in _active_lines(retry_file):
        return int(token)
    return None


def consume_retry_seed(retry_file: Path, seed: int) -> None:
    if not retry_file.exists():
        return
    lines = retry_file.read_text().splitlines(keepends=True)
    consumed = False
    out = []
    for raw in lines:
        stripped = raw.strip()
        if (
            not consumed
            and stripped
            and not stripped.startswith("#")
            and stripped.split()[0] == str(seed)
        ):
            consumed = True
            continue
        out.append(raw)
    tmp = retry_file.with_suffix(".txt.tmp")
    tmp.write_text("".join(out))
    tmp.replace(retry_file)  # atomic on POSIX same-filesystem


def count_active_jobs(qselect_cmd, qstat_cmd: str, user: str, job_name: str) -> int:
    if qselect_cmd:
        result = subprocess.run(
            [qselect_cmd, "-u", user, "-N", job_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return len([l for l in result.stdout.splitlines() if l.strip()])

    result = subprocess.run(
        [qstat_cmd, "-u", user],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    count = 0
    for line in result.stdout.splitlines():
        if line.startswith("Job") or line.startswith("---") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[1] == job_name and parts[2] == user:
            count += 1
    return count


def call_loop_status(python3_cmd: str, script: Path, runs_root: Path) -> dict:
    result = subprocess.run(
        [python3_cmd, str(script), "--runs-root", str(runs_root), "--format", "env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    env = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"')
    return env


def _load_settings() -> dict:
    cfg = config.load()
    name = cfg.get("campaign", {}).get("name", "default")
    return {
        "runs_root": ROOT_DIR / "campaigns" / name / "my_runs",
        "state_dir": ROOT_DIR / "campaigns" / name / ".queue_state",
        "initial_seed": int(cfg.get("campaign", {}).get("initial_seed", 1000)),
        "target_jobs": int(cfg.get("queue", {}).get("target_jobs", 10)),
        "job_name": cfg.get("queue", {}).get("job_name", "a3ht"),
        "base_angle_deg": cfg.get("structure", {}).get("base_angle_deg", 90.0),
        "angle_disturb_deg": cfg.get("structure", {}).get("angle_disturb_deg", 30.0),
        "tilt_max_deg": cfg.get("structure", {}).get("tilt_max_deg", 30.0),
        "alcf_model": cfg.get("alcf", {}).get("model", ""),
        "python3": cfg.get("paths", {}).get("python", "python3"),
    }


def _run(
    settings: dict,
    log_file: Path,
    counter_file: Path,
    retry_file: Path,
    job_script: Path,
    planner_script: Path,
    loop_status_script: Path,
) -> int:
    os.environ["PATH"] = (
        "/opt/pbs/bin:/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")
    )

    qsub_cmd = require_cmd("qsub", log_file)
    qstat_cmd = require_cmd("qstat", log_file)
    qselect_cmd = shutil.which("qselect")  # preferred over qstat; None if unavailable
    python3_cmd = settings["python3"]
    if not os.access(python3_cmd, os.X_OK):
        python3_cmd = require_cmd("python3", log_file)

    for fpath in (job_script, planner_script, loop_status_script):
        if not fpath.is_file():
            log(log_file, f"file not found: {fpath}")
            return 1

    user = os.environ.get("USER", "")
    job_name = settings["job_name"]
    target_jobs = settings["target_jobs"]
    runs_root = settings["runs_root"]

    try:
        active_jobs = count_active_jobs(qselect_cmd, qstat_cmd, user, job_name)
    except Exception as exc:
        log(log_file, f"failed to query active jobs: {exc}")
        return 1

    if active_jobs >= target_jobs:
        log(log_file, f"active={active_jobs} target={target_jobs} submitted=0")
        return 0

    try:
        loop_env = call_loop_status(python3_cmd, loop_status_script, runs_root)
    except subprocess.CalledProcessError as exc:
        log(log_file, f"loop_status failed: {exc.stderr.strip() if exc.stderr else exc}")
        return 1

    if loop_env.get("A3HT_LOOP_STOP_CONDITION_MET") == "1":
        log(
            log_file,
            f"stop_condition_met=1 action={loop_env.get('A3HT_LOOP_ACTION', '')} submitted=0",
        )
        return 0
    if loop_env.get("A3HT_LOOP_ACTION") == "wait_active_cohorts":
        log(
            log_file,
            f"action={loop_env.get('A3HT_LOOP_ACTION', '')} "
            f"active_cohort_count={loop_env.get('A3HT_ACTIVE_COHORT_COUNT', '')} submitted=0",
        )
        return 0

    qsub_var_parts = [
        f"A3HT_ROOT_DIR={ROOT_DIR}",
        f"A3HT_RUNS_ROOT={runs_root}",
        f"A3HT_STATE_DIR={settings['state_dir']}",
        f"A3HT_STRUCTURE_BASE_ANGLE_DEG={settings['base_angle_deg']}",
        f"A3HT_STRUCTURE_ANGLE_DISTURB_DEG={settings['angle_disturb_deg']}",
        f"A3HT_STRUCTURE_TILT_MAX_DEG={settings['tilt_max_deg']}",
    ]
    if settings["alcf_model"]:
        qsub_var_parts.append(f"A3HT_ALCF_MODEL={settings['alcf_model']}")
    qsub_vars = ",".join(qsub_var_parts)

    jobs_to_submit = target_jobs - active_jobs
    submitted = 0

    while submitted < jobs_to_submit:
        # Drain the retry queue before advancing to new seeds so failed runs get reprocessed first.
        seed = peek_retry_seed(retry_file)
        if seed is not None:
            seed_source = "retry_queue"
        else:
            seed = peek_next_seed(counter_file, settings["initial_seed"])
            seed_source = "next_seed"

        run_dir = runs_root / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)

        planner_proc = subprocess.run(
            [
                python3_cmd,
                str(planner_script),
                "--seed", str(seed),
                "--run-dir", str(run_dir),
                "--runs-root", str(runs_root),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if planner_proc.returncode != 0:
            log(log_file, f"planning failed seed={seed}")
            return 1
        planner_result = planner_proc.stdout.strip()

        qsub_proc = subprocess.run(
            [
                qsub_cmd,
                "-N", job_name,
                "-v", f"{qsub_vars},A3HT_SEED={seed}",
                str(job_script),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if qsub_proc.returncode != 0:
            log(log_file, f"qsub failed seed={seed}")
            return 1
        job_id = qsub_proc.stdout.strip()

        # Advance the appropriate pointer only after qsub succeeds to avoid skipping seeds.
        if seed_source == "retry_queue":
            consume_retry_seed(retry_file, seed)
        else:
            advance_next_seed(counter_file, seed)

        submitted += 1
        log(
            log_file,
            f"planner={planner_result} seed={seed} source={seed_source} "
            f"action={loop_env.get('A3HT_LOOP_ACTION', '')} "
            f"cohort={loop_env.get('A3HT_SELECTED_COHORT_ID', '')}",
        )
        log(log_file, f"submitted job_id={job_id} seed={seed}")

    log(log_file, f"active={active_jobs} target={target_jobs} submitted={submitted}")
    return 0


def main() -> int:
    settings = _load_settings()
    state_dir = settings["state_dir"]
    state_dir.mkdir(parents=True, exist_ok=True)

    log_file = state_dir / "fill_queue.log"
    lock_dir = state_dir / "lock"
    counter_file = state_dir / "next_seed"
    retry_file = state_dir / "resubmit_seeds.txt"

    job_script = Path(os.environ.get("A3HT_JOB_SCRIPT", str(ROOT_DIR / "run.py")))
    planner_script = Path(
        os.environ.get("A3HT_PLANNER_SCRIPT", str(ROOT_DIR / "src" / "plan_simulation.py"))
    )
    loop_status_script = Path(
        os.environ.get("A3HT_LOOP_STATUS_SCRIPT", str(ROOT_DIR / "src" / "loop_status.py"))
    )

    hostname = (
        subprocess.run(["hostname"], stdout=subprocess.PIPE, text=True).stdout.strip()
        or "unknown"
    )
    print(f"Running on host: {hostname}")
    log(log_file, f"running_on_host={hostname}")

    # mkdir is atomic on POSIX; used as a lock to prevent concurrent fill_queue runs.
    try:
        lock_dir.mkdir()
    except FileExistsError:
        log(log_file, "another queue-fill run is still active")
        return 0

    try:
        return _run(
            settings,
            log_file,
            counter_file,
            retry_file,
            job_script,
            planner_script,
            loop_status_script,
        )
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
