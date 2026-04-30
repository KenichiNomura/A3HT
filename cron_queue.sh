#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load all campaign/queue parameters from config.toml
eval "$("${ROOT_DIR}/config.py" --shell-env)"
# Sets: A3HT_RUNS_ROOT, A3HT_STATE_DIR, A3HT_STRUCTURE_BASE_ANGLE_DEG,
#       A3HT_STRUCTURE_ANGLE_DISTURB_DEG, A3HT_STRUCTURE_TILT_MAX_DEG,
#       A3HT_ALCF_MODEL, A3HT_TARGET_JOBS, A3HT_JOB_NAME,
#       A3HT_INITIAL_SEED, LAMMPS_DIR, A3HT_PYTHON3

JOB_SCRIPT="${A3HT_JOB_SCRIPT:-${ROOT_DIR}/run.sh}"
PLANNER_SCRIPT="${A3HT_PLANNER_SCRIPT:-${ROOT_DIR}/plan_simulation.py}"
LOOP_STATUS_SCRIPT="${A3HT_LOOP_STATUS_SCRIPT:-${ROOT_DIR}/loop_status.py}"
STATE_DIR="${A3HT_STATE_DIR}"
LOCK_DIR="${STATE_DIR}/lock"
COUNTER_FILE="${STATE_DIR}/next_seed"
RETRY_FILE="${STATE_DIR}/resubmit_seeds.txt"
LOG_FILE="${STATE_DIR}/fill_queue.log"
PATH="/opt/pbs/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

mkdir -p "${STATE_DIR}"

hostname_value="$(hostname 2>/dev/null || uname -n)"
printf 'Running on host: %s\n' "${hostname_value}"
printf '%s running_on_host=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${hostname_value}" >> "${LOG_FILE}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    printf '%s another queue-fill run is still active\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "${LOG_FILE}"
    exit 0
fi
trap 'rmdir "${LOCK_DIR}"' EXIT INT TERM

require_cmd() {
    cmd_name="$1"
    if command -v "${cmd_name}" >/dev/null 2>&1; then
        command -v "${cmd_name}"; return 0
    fi
    for candidate in "/opt/pbs/bin/${cmd_name}" "/usr/local/pbs/bin/${cmd_name}" "/usr/pbs/bin/${cmd_name}"; do
        if [ -x "${candidate}" ]; then printf '%s\n' "${candidate}"; return 0; fi
    done
    printf '%s missing required command: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${cmd_name}" >> "${LOG_FILE}"
    exit 1
}

peek_next_seed() {
    if [ ! -f "${COUNTER_FILE}" ]; then printf '%s\n' "${A3HT_INITIAL_SEED:-1000}" > "${COUNTER_FILE}"; fi
    cat "${COUNTER_FILE}"
}

advance_next_seed() { printf '%s\n' "$(($1 + 1))" > "${COUNTER_FILE}"; }

peek_retry_seed() {
    [ -f "${RETRY_FILE}" ] || return 1
    awk 'NF && !/^[[:space:]]*#/ {print $1; exit}' "${RETRY_FILE}"
}

consume_retry_seed() {
    [ -f "${RETRY_FILE}" ] || return 0
    awk -v seed="$1" 'BEGIN{r=0} NF && !/^[[:space:]]*#/ && !r && $1==seed {r=1;next} {print}' \
        "${RETRY_FILE}" > "${RETRY_FILE}.tmp"
    mv "${RETRY_FILE}.tmp" "${RETRY_FILE}"
}

count_active_jobs() {
    if [ -n "${QSELECT_CMD:-}" ]; then
        out="$("${QSELECT_CMD}" -u "${USER}" -N "${A3HT_JOB_NAME}")" || return 1
        [ -z "${out}" ] && printf '0\n' || printf '%s\n' "${out}" | wc -l | awk '{print $1}'
        return
    fi
    "$QSTAT_CMD" -u "${USER}" | awk -v n="${A3HT_JOB_NAME}" -v u="${USER}" \
        '!/^Job/ && !/^---/ && NF>=5 && $2==n && $3==u {c++} END{print c+0}'
}

QSUB_CMD="$(require_cmd qsub)"
QSTAT_CMD="$(require_cmd qstat)"
QSELECT_CMD="$(command -v qselect 2>/dev/null || true)"
PYTHON3_CMD="${A3HT_PYTHON3:-python3}"
if [ ! -x "${PYTHON3_CMD}" ]; then PYTHON3_CMD="$(require_cmd python3)"; fi

for f in "${JOB_SCRIPT}" "${PLANNER_SCRIPT}" "${LOOP_STATUS_SCRIPT}"; do
    if [ ! -f "${f}" ]; then
        printf '%s file not found: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${f}" >> "${LOG_FILE}"
        exit 1
    fi
done

active_jobs="$(count_active_jobs)" || {
    printf '%s failed to query active jobs\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "${LOG_FILE}"
    exit 1
}

case "${active_jobs}" in ''|*[!0-9]*)
    printf '%s bad active job count: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${active_jobs}" >> "${LOG_FILE}"
    exit 1
esac

if [ "${active_jobs}" -ge "${A3HT_TARGET_JOBS}" ]; then
    printf '%s active=%s target=%s submitted=0\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${active_jobs}" "${A3HT_TARGET_JOBS}" >> "${LOG_FILE}"
    exit 0
fi

loop_env="$("${PYTHON3_CMD}" "${LOOP_STATUS_SCRIPT}" --runs-root "${A3HT_RUNS_ROOT}" --format env)"
eval "${loop_env}"

if [ "${A3HT_LOOP_STOP_CONDITION_MET}" = "1" ]; then
    printf '%s stop_condition_met=1 action=%s submitted=0\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${A3HT_LOOP_ACTION}" >> "${LOG_FILE}"
    exit 0
fi
if [ "${A3HT_LOOP_ACTION}" = "wait_active_cohorts" ]; then
    printf '%s action=%s active_cohort_count=%s submitted=0\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${A3HT_LOOP_ACTION}" "${A3HT_ACTIVE_COHORT_COUNT}" >> "${LOG_FILE}"
    exit 0
fi

jobs_to_submit=$((A3HT_TARGET_JOBS - active_jobs))
submitted=0
qsub_vars="A3HT_ROOT_DIR=${ROOT_DIR},A3HT_RUNS_ROOT=${A3HT_RUNS_ROOT},A3HT_STATE_DIR=${STATE_DIR}"
qsub_vars="${qsub_vars},A3HT_STRUCTURE_BASE_ANGLE_DEG=${A3HT_STRUCTURE_BASE_ANGLE_DEG}"
qsub_vars="${qsub_vars},A3HT_STRUCTURE_ANGLE_DISTURB_DEG=${A3HT_STRUCTURE_ANGLE_DISTURB_DEG}"
qsub_vars="${qsub_vars},A3HT_STRUCTURE_TILT_MAX_DEG=${A3HT_STRUCTURE_TILT_MAX_DEG}"
[ -n "${A3HT_ALCF_MODEL}" ] && qsub_vars="${qsub_vars},A3HT_ALCF_MODEL=${A3HT_ALCF_MODEL}"

while [ "${submitted}" -lt "${jobs_to_submit}" ]; do
    seed_source="next_seed"
    seed="$(peek_retry_seed || true)"
    if [ -n "${seed}" ]; then
        seed_source="retry_queue"
    else
        seed="$(peek_next_seed)"
    fi
    run_dir="${A3HT_RUNS_ROOT}/${seed}"
    mkdir -p "${run_dir}"
    if ! planner_result="$("${PYTHON3_CMD}" "${PLANNER_SCRIPT}" --seed "${seed}" --run-dir "${run_dir}" --runs-root "${A3HT_RUNS_ROOT}")"; then
        printf '%s planning failed seed=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${seed}" >> "${LOG_FILE}"
        exit 1
    fi
    if ! job_id="$("${QSUB_CMD}" -N "${A3HT_JOB_NAME}" -v "${qsub_vars},A3HT_SEED=${seed}" "${JOB_SCRIPT}")"; then
        printf '%s qsub failed seed=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${seed}" >> "${LOG_FILE}"
        exit 1
    fi
    if [ "${seed_source}" = "retry_queue" ]; then consume_retry_seed "${seed}"; else advance_next_seed "${seed}"; fi
    submitted=$((submitted + 1))
    printf '%s planner=%s seed=%s source=%s action=%s cohort=%s\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${planner_result}" "${seed}" "${seed_source}" \
        "${A3HT_LOOP_ACTION}" "${A3HT_SELECTED_COHORT_ID}" >> "${LOG_FILE}"
    printf '%s submitted job_id=%s seed=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${job_id}" "${seed}" >> "${LOG_FILE}"
done

printf '%s active=%s target=%s submitted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${active_jobs}" "${A3HT_TARGET_JOBS}" "${submitted}" >> "${LOG_FILE}"
