#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/lus/eagle/projects/uMLIP-PET-FT/knomura/a3ht"
JOB_SCRIPT="${A3HT_JOB_SCRIPT:-${ROOT_DIR}/run.sh}"
PLANNER_SCRIPT="${A3HT_PLANNER_SCRIPT:-${ROOT_DIR}/plan_simulation.py}"
LOOP_STATUS_SCRIPT="${A3HT_LOOP_STATUS_SCRIPT:-${ROOT_DIR}/loop_status.py}"
JOB_NAME="${A3HT_JOB_NAME:-a3ht}"
TARGET_JOBS="${A3HT_TARGET_JOBS:-10}"
CODEX_BIN_VALUE="${A3HT_CODEX_BIN:-}"
STATE_DIR="${A3HT_STATE_DIR:-${ROOT_DIR}/.queue_state}"
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

cleanup() {
    rmdir "${LOCK_DIR}"
}

trap cleanup EXIT INT TERM

find_command() {
    cmd_name="$1"

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        command -v "${cmd_name}"
        return 0
    fi

    for candidate in \
        "/opt/pbs/bin/${cmd_name}" \
        "/usr/local/pbs/bin/${cmd_name}" \
        "/usr/pbs/bin/${cmd_name}"
    do
        if [ -x "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

require_command() {
    resolved_path="$(find_command "$1" || true)"
    if [ -z "${resolved_path}" ]; then
        printf '%s missing required command: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" >> "${LOG_FILE}"
        exit 1
    fi

    printf '%s\n' "${resolved_path}"
}

peek_next_seed() {
    if [ ! -f "${COUNTER_FILE}" ]; then
        printf '1000\n' > "${COUNTER_FILE}"
    fi

    cat "${COUNTER_FILE}"
}

advance_next_seed() {
    seed="$1"
    next_value=$((seed + 1))
    printf '%s\n' "${next_value}" > "${COUNTER_FILE}"
}

peek_retry_seed() {
    if [ ! -f "${RETRY_FILE}" ]; then
        return 1
    fi

    awk '
        NF == 0 {next}
        $0 ~ /^[[:space:]]*#/ {next}
        {print $1; exit}
    ' "${RETRY_FILE}"
}

consume_retry_seed() {
    seed="$1"

    if [ ! -f "${RETRY_FILE}" ]; then
        return 0
    fi

    tmp_file="${RETRY_FILE}.tmp"
    awk -v seed="${seed}" '
        BEGIN {removed = 0}
        NF == 0 {next}
        $0 ~ /^[[:space:]]*#/ {next}
        !removed && $1 == seed {removed = 1; next}
        {print}
    ' "${RETRY_FILE}" > "${tmp_file}"
    mv "${tmp_file}" "${RETRY_FILE}"
}

count_active_jobs() {
    if [ -n "${QSELECT_CMD:-}" ]; then
        qselect_output="$("${QSELECT_CMD}" -u "${USER}" -N "${JOB_NAME}")" || return 1
        if [ -z "${qselect_output}" ]; then
            printf '0\n'
        else
            printf '%s\n' "${qselect_output}" | wc -l | awk '{print $1}'
        fi
        return
    fi

    qstat_output="$("${QSTAT_CMD}" -u "${USER}")" || return 1
    printf '%s\n' "${qstat_output}" | awk -v user="${USER}" -v name="${JOB_NAME}" '
        $0 ~ /^Job/ {next}
        $0 ~ /^---/ {next}
        NF >= 5 && $2 == name && $3 == user {count++}
        END {print count + 0}
    '
}

QSUB_CMD="$(require_command qsub)"
QSTAT_CMD="$(require_command qstat)"
QSELECT_CMD="$(find_command qselect || true)"
PYTHON3_CMD="$(require_command python3)"

if [ ! -f "${JOB_SCRIPT}" ]; then
    printf '%s job script not found: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${JOB_SCRIPT}" >> "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${PLANNER_SCRIPT}" ]; then
    printf '%s planner script not found: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${PLANNER_SCRIPT}" >> "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${LOOP_STATUS_SCRIPT}" ]; then
    printf '%s loop status script not found: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${LOOP_STATUS_SCRIPT}" >> "${LOG_FILE}"
    exit 1
fi

if ! active_jobs="$(count_active_jobs)"; then
    printf '%s failed to query active jobs via scheduler\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "${LOG_FILE}"
    exit 1
fi

case "${active_jobs}" in
    ''|*[!0-9]*)
        printf '%s failed to determine active job count: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${active_jobs}" >> "${LOG_FILE}"
        exit 1
        ;;
esac

case "${TARGET_JOBS}" in
    ''|*[!0-9]*)
        printf '%s invalid target job count: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${TARGET_JOBS}" >> "${LOG_FILE}"
        exit 1
        ;;
esac

if [ "${active_jobs}" -ge "${TARGET_JOBS}" ]; then
    printf '%s active=%s target=%s submitted=0\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${active_jobs}" "${TARGET_JOBS}" >> "${LOG_FILE}"
    exit 0
fi

jobs_to_submit=$((TARGET_JOBS - active_jobs))
submitted=0

while [ "${submitted}" -lt "${jobs_to_submit}" ]; do
    loop_env="$("${PYTHON3_CMD}" "${LOOP_STATUS_SCRIPT}" --runs-root "${ROOT_DIR}/my_runs" --format env)"
    eval "${loop_env}"
    if [ "${A3HT_LOOP_STOP_CONDITION_MET}" = "1" ]; then
        printf '%s stop_condition_met=1 action=%s reason=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${A3HT_LOOP_ACTION}" "${A3HT_LOOP_REASON}" >> "${LOG_FILE}"
        break
    fi
    if [ "${A3HT_LOOP_ACTION}" = "wait_active_cohorts" ]; then
        printf '%s action=%s active_cohort_count=%s reason=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${A3HT_LOOP_ACTION}" "${A3HT_ACTIVE_COHORT_COUNT}" "${A3HT_LOOP_REASON}" >> "${LOG_FILE}"
        break
    fi
    seed_source="next_seed"
    seed="$(peek_retry_seed || true)"
    if [ -n "${seed}" ]; then
        seed_source="retry_queue"
    else
        seed="$(peek_next_seed)"
    fi
    run_dir="${ROOT_DIR}/my_runs/${seed}"
    mkdir -p "${run_dir}"
    if ! planner_result="$("${PYTHON3_CMD}" "${PLANNER_SCRIPT}" --seed "${seed}" --run-dir "${run_dir}")"; then
        printf '%s planning failed for seed=%s source=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${seed}" "${seed_source}" >> "${LOG_FILE}"
        exit 1
    fi
    qsub_vars="A3HT_SEED=${seed}"
    if [ -n "${CODEX_BIN_VALUE}" ]; then
        qsub_vars="${qsub_vars},A3HT_CODEX_BIN=${CODEX_BIN_VALUE}"
    fi
    if ! job_id="$("${QSUB_CMD}" -N "${JOB_NAME}" -v "${qsub_vars}" "${JOB_SCRIPT}")"; then
        printf '%s qsub failed for seed=%s source=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${seed}" "${seed_source}" >> "${LOG_FILE}"
        exit 1
    fi
    if [ "${seed_source}" = "retry_queue" ]; then
        consume_retry_seed "${seed}"
    else
        advance_next_seed "${seed}"
    fi
    printf '%s planner=%s seed=%s source=%s action=%s selected_cohort=%s active_cohort_count=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${planner_result}" "${seed}" "${seed_source}" "${A3HT_LOOP_ACTION}" "${A3HT_SELECTED_COHORT_ID}" "${A3HT_ACTIVE_COHORT_COUNT}" >> "${LOG_FILE}"
    printf '%s submitted job_id=%s seed=%s source=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${job_id}" "${seed}" "${seed_source}" >> "${LOG_FILE}"
done

printf '%s active=%s target=%s submitted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${active_jobs}" "${TARGET_JOBS}" "${submitted}" >> "${LOG_FILE}"
