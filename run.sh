#!/bin/bash
#PBS -A FoundMLIP
#PBS -l filesystems=eagle
#PBS -N a3ht
#PBS -q workq
#PBS -l select=4:ncpus=256
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

if [[ -n "${A3HT_ROOT_DIR:-}" ]]; then
    rootdir="${A3HT_ROOT_DIR}"
elif [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    rootdir="${PBS_O_WORKDIR}"
else
    rootdir="$(cd "$(dirname "$0")" && pwd)"
fi
cd "${rootdir}"

# Load paths and queue settings from config.toml
eval "$("${rootdir}/config.py" --shell-env)"
export LD_LIBRARY_PATH="$("${rootdir}/config.py" --ld-library-path)"

stage="startup"
run_dir=""
status_file=""
failure_file=""

timestamp_utc() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

fail() {
    message="$1"
    if [[ -n "${failure_file}" ]]; then
        printf 'timestamp=%s\nstage=%s\nmessage=%s\n' "$(timestamp_utc)" "${stage}" "${message}" > "${failure_file}"
        printf 'FAILED\n' > "${status_file}"
    fi
    echo "error: ${message}" >&2
    exit 1
}

finish_run() {
    exit_code=$?
    if [[ -n "${status_file}" ]]; then
        if [[ ${exit_code} -eq 0 ]]; then
            rm -f "${failure_file}"
            printf 'SUCCESS\n' > "${status_file}"
        else
            if [[ ! -f "${failure_file}" ]]; then
                printf 'timestamp=%s\nstage=%s\nmessage=run exited with code %s\n' \
                    "$(timestamp_utc)" "${stage}" "${exit_code}" > "${failure_file}"
            fi
            printf 'FAILED\n' > "${status_file}"
        fi
    fi
}

trap finish_run EXIT

LAMMPS_BIN="${LAMMPS_BIN:-${LAMMPS_DIR}/lmp}"
PLANNER_SCRIPT="${A3HT_PLANNER_SCRIPT:-${rootdir}/plan_simulation.py}"
RUNS_ROOT="${A3HT_RUNS_ROOT}"
echo "${LAMMPS_BIN}"

export PATH="${LAMMPS_DIR}:$PATH"

if command -v mpiexec >/dev/null 2>&1; then
    mpi_launcher=(mpiexec -n)
elif command -v mpirun >/dev/null 2>&1; then
    mpi_launcher=(mpirun -n)
else
    echo "error: neither mpiexec nor mpirun is available in PATH" >&2
    exit 1
fi

detect_ntasks() {
    local default_ppn=128
    if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
        local unique_nodes
        unique_nodes=$(sort -u "${PBS_NODEFILE}" | wc -l)
        if [[ "${PBS_NUM_PPN:-}" =~ ^[1-9][0-9]*$ ]]; then
            printf '%s\n' "$(( unique_nodes * PBS_NUM_PPN ))"; return
        fi
        printf '%s\n' "$(( unique_nodes * default_ppn ))"; return
    fi
    [[ "${PBS_NP:-}" =~ ^[1-9][0-9]*$ ]] && printf '%s\n' "${PBS_NP}" && return
    printf '%s\n' 128
}

ntasks=$(detect_ntasks)
lmp_bin="${LAMMPS_BIN}"
seed=""
processors=auto

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)       [[ $# -lt 2 ]] && { echo "error: --seed requires a value" >&2; exit 1; }
                      seed="$2"; shift 2 ;;
        --ntasks)     [[ $# -lt 2 ]] && { echo "error: --ntasks requires a value" >&2; exit 1; }
                      ntasks="$2"; shift 2 ;;
        --processors) [[ $# -lt 2 ]] && { echo "error: --processors requires a value" >&2; exit 1; }
                      processors="$2"; shift 2 ;;
        *)            echo "usage: $0 [--seed N] [--ntasks N] [--processors auto|Px,Py,Pz]" >&2; exit 1 ;;
    esac
done

[[ -z "${seed}" ]] && seed="${A3HT_SEED:-123}"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "--seed must be a non-negative integer"

structure_base_angle_deg="${A3HT_STRUCTURE_BASE_ANGLE_DEG:-90.0}"
structure_angle_disturb_deg="${A3HT_STRUCTURE_ANGLE_DISTURB_DEG:-20.0}"
structure_tilt_max_deg="${A3HT_STRUCTURE_TILT_MAX_DEG:-90.0}"

run_dir="${RUNS_ROOT}/${seed}"
status_file="${run_dir}/run_status.txt"
failure_file="${run_dir}/run_failure.txt"
plan_env="${run_dir}/simulation_plan.env"
plan_lmp="${run_dir}/simulation_plan.lmp"
plan_json="${run_dir}/simulation_plan.json"
mkdir -p "${run_dir}"
printf 'RUNNING\n' > "${status_file}"
rm -f "${failure_file}"

[[ "${ntasks}" =~ ^[1-9][0-9]*$ ]] || fail "--ntasks must be a positive integer"

if [[ "${processors}" == "auto" ]]; then
    procx="*"; procy="*"; procz="*"
elif [[ "${processors}" =~ ^([1-9][0-9]*),([1-9][0-9]*),([1-9][0-9]*)$ ]]; then
    procx="${BASH_REMATCH[1]}"; procy="${BASH_REMATCH[2]}"; procz="${BASH_REMATCH[3]}"
    (( procx * procy * procz != ntasks )) && fail "--processors must satisfy Px*Py*Pz = ntasks (${ntasks})"
else
    fail "--processors must be 'auto' or 'Px,Py,Pz'"
fi

stage="environment_check"
[[ -x "${LAMMPS_BIN}" ]] || fail "LAMMPS executable not found: ${LAMMPS_BIN}"
[[ -f "${rootdir}/CH.rebo" ]] || fail "REBO2 parameter file not found: ${rootdir}/CH.rebo"

cd "${run_dir}"

stage="simulation_planning"
if [[ ! -f "${plan_env}" || ! -f "${plan_lmp}" || ! -f "${plan_json}" ]]; then
    command -v python3 >/dev/null 2>&1 || fail "python3 is required to generate simulation planning artifacts"
    [[ -f "${PLANNER_SCRIPT}" ]] || fail "planner script not found: ${PLANNER_SCRIPT}"
    python3 "${PLANNER_SCRIPT}" --seed "${seed}" --run-dir "${run_dir}" --runs-root "${RUNS_ROOT}" \
        || fail "simulation planner failed for seed ${seed}"
fi
[[ -f "${plan_env}" ]] || fail "simulation plan env file not found: ${plan_env}"
[[ -f "${plan_lmp}" ]] || fail "simulation plan LAMMPS include not found: ${plan_lmp}"

# shellcheck disable=SC1090
source "${plan_env}"

if [ "${A3HT_PLANNER_STATUS:-ok}" != "ok" ]; then
    printf "Planner degraded: source=%s\n%s\n" "${A3HT_PLAN_SOURCE:-unknown}" "${A3HT_PLANNER_ERROR:-}" > planner_warning.txt
    echo "warning: Planner degraded: source=${A3HT_PLAN_SOURCE:-unknown}" >&2
fi
echo "Plan source: ${A3HT_PLAN_SOURCE}  cohort: ${A3HT_COHORT_ID}  target_kappa: ${A3HT_GOAL_TARGET_KAPPA_W_MK} W/m-K"
echo "Orientation: base=${structure_base_angle_deg} disturb=${structure_angle_disturb_deg} tilt=${structure_tilt_max_deg}"

stage="structure_generation"
"${rootdir}/generate_random_carbon.py" \
    --box "${A3HT_STRUCTURE_BOX_X_A}" "${A3HT_STRUCTURE_BOX_Y_A}" "${A3HT_STRUCTURE_BOX_Z_A}" \
    --density "${A3HT_STRUCTURE_DENSITY_G_CM3}" \
    --seed "${seed}" \
    --output random_carbon.extxyz \
    --flake-area "${A3HT_FLAKE_AREA_A2}" \
    --base-angle-deg "${structure_base_angle_deg}" \
    --angle-disturb-deg "${structure_angle_disturb_deg}" \
    --tilt-max-deg "${structure_tilt_max_deg}" \
    --format lammps
mv random_carbon.extxyz random_carbon.dat
cp -v "${rootdir}/CH.rebo" CH.rebo

run_lammps() {
    local log_file="$1" input_file="$2"
    echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log ${log_file} -in ${input_file}"
    "${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" \
        -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" \
        -log "${log_file}" -in "${input_file}"
}

stage="anneal"
run_lammps anneal.log "${rootdir}/anneal.in"
cp -v data/anneal_gc_rebo2.restart gc_rebo2.restart

stage="thermalize"
run_lammps thermalize.log "${rootdir}/thermalize.in"
cp -v data/gc_rebo2_thermalize.restart gc_rebo2.restart

stage="nemd"
run_lammps nemd.log "${rootdir}/nemd.in"
