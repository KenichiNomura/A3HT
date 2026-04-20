#!/usr/bin/env python3
"""Report A3HT autonomous loop status for shell integration."""

import argparse
import json
from pathlib import Path

from autonomy import collect_run_records, summarize_loop_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=None, help="runs root to inspect")
    parser.add_argument(
        "--format",
        choices=("json", "env"),
        default="json",
        help="output format for the computed loop status",
    )
    return parser.parse_args()


def shell_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


def main():
    args = parse_args()
    runs_root = args.runs_root.resolve() if args.runs_root else None
    records = collect_run_records(runs_root) if runs_root else collect_run_records(Path(__file__).resolve().parent / "my_runs")
    summary = summarize_loop_state(records)

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    selected = summary.get("selected_cohort") or {}
    active_cohorts = summary.get("active_cohorts") or []
    lines = [
        'A3HT_LOOP_ACTION="{}"'.format(shell_escape(summary.get("action", ""))),
        'A3HT_LOOP_REASON="{}"'.format(shell_escape(summary.get("reason", ""))),
        'A3HT_LOOP_STOP_CONDITION_MET="{}"'.format("1" if summary.get("stop_condition_met") else "0"),
        'A3HT_LOOP_TARGET_KAPPA_W_MK="{}"'.format(shell_escape(summary.get("target_kappa_w_mk", ""))),
        'A3HT_LOOP_TARGET_RELATIVE_UNCERTAINTY_PCT="{}"'.format(
            shell_escape(summary.get("target_relative_uncertainty_pct", ""))
        ),
        'A3HT_LOOP_MIN_COHORT_SUCCESS_SEEDS="{}"'.format(shell_escape(summary.get("min_cohort_success_seeds", ""))),
        'A3HT_ACTIVE_COHORT_COUNT="{}"'.format(shell_escape(len(active_cohorts))),
        'A3HT_SELECTED_COHORT_ID="{}"'.format(shell_escape(selected.get("cohort_id", ""))),
        'A3HT_SELECTED_COHORT_SUCCESS_COUNT="{}"'.format(shell_escape(selected.get("success_count", ""))),
        'A3HT_SELECTED_COHORT_PENDING_COUNT="{}"'.format(shell_escape(selected.get("pending_count", ""))),
        'A3HT_SELECTED_COHORT_EVALUABLE_SUCCESS_COUNT="{}"'.format(
            shell_escape(selected.get("evaluable_success_count", ""))
        ),
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
