#!/usr/bin/env python3
"""Queue failed A3HT runs for resubmission and purge their partial outputs.

This script is conservative by default:
- queue runs whose ``run_status.txt`` says ``FAILED``
- leave ``RUNNING`` runs untouched

It writes the retry queue consumed by ``cron_queue.sh`` to:
  .queue_state/resubmit_seeds.txt
and stores a manifest of the queued seeds plus the failure reasons.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = ROOT / "my_runs"
DEFAULT_STATE_DIR = ROOT / ".queue_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="directory containing per-seed run directories",
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_STATE_DIR),
        help="directory that stores queue-state files",
    )
    parser.add_argument(
        "--retry-file",
        default=None,
        help="optional explicit retry queue path; default: <state-dir>/resubmit_seeds.txt",
    )
    parser.add_argument(
        "--manifest-json",
        default=None,
        help="optional explicit manifest path; default: <state-dir>/resubmit_manifest.json",
    )
    parser.add_argument(
        "--purge-run-dirs",
        action="store_true",
        help="remove queued run directories before writing the retry queue",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="also queue runs whose status file still says RUNNING",
    )
    return parser.parse_args()


def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def parse_failure(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not path.exists():
        return None, None

    stage = None
    message = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("stage="):
            stage = line.split("=", 1)[1]
        elif line.startswith("message="):
            message = line.split("=", 1)[1]
    return stage, message


def collect_candidates(runs_root: Path, include_running: bool) -> List[Dict[str, object]]:
    candidates = []  # type: List[Dict[str, object]]
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        try:
            seed = int(run_dir.name)
        except ValueError:
            continue

        status = read_text(run_dir / "run_status.txt")
        if status is None:
            continue
        if status == "SUCCESS":
            continue
        if status == "RUNNING" and not include_running:
            continue

        stage, reason = parse_failure(run_dir / "run_failure.txt")
        candidates.append(
            {
                "seed": seed,
                "status": status,
                "reason": reason,
                "stage": stage,
                "run_dir": str(run_dir),
            }
        )
    return candidates


def purge_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)


def write_retry_file(path: Path, seeds: List[int]) -> None:
    lines = [f"{seed}\n" for seed in seeds]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    state_dir = Path(args.state_dir)
    retry_file = Path(args.retry_file) if args.retry_file else state_dir / "resubmit_seeds.txt"
    manifest_json = Path(args.manifest_json) if args.manifest_json else state_dir / "resubmit_manifest.json"
    state_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(runs_root, include_running=args.include_running)
    seeds = [int(candidate["seed"]) for candidate in candidates]

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_root": str(runs_root),
        "retry_file": str(retry_file),
        "purged_run_dirs": bool(args.purge_run_dirs),
        "include_running": bool(args.include_running),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.purge_run_dirs:
        for candidate in candidates:
            purge_run_dir(Path(str(candidate["run_dir"])))

    write_retry_file(retry_file, seeds)

    print(f"Queued {len(seeds)} seeds for resubmission")
    print(f"Retry file: {retry_file}")
    print(f"Manifest: {manifest_json}")
    if args.purge_run_dirs:
        print("Purged queued run directories before resubmission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
