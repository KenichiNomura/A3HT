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

stage="startup"
run_dir=""
status_file=""
failure_file=""

timestamp_utc() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

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
                printf 'timestamp=%s\nstage=%s\nmessage=run exited with code %s\n' "$(timestamp_utc)" "${stage}" "${exit_code}" > "${failure_file}"
            fi
            printf 'FAILED\n' > "${status_file}"
        fi
    fi
}

trap finish_run EXIT

default_lammps_dir="${rootdir}/lammps-30Mar2026/build-cray-rebo2"
LAMMPS_DIR="${LAMMPS_DIR:-${default_lammps_dir}}"
LAMMPS_BIN="${LAMMPS_BIN:-${LAMMPS_DIR}/lmp}"
PLANNER_SCRIPT="${A3HT_PLANNER_SCRIPT:-${rootdir}/plan_simulation.py}"
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
    local unique_nodes=""
    local default_ppn=128

    if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
        unique_nodes=$(sort -u "${PBS_NODEFILE}" | wc -l)

        if [[ "${PBS_NUM_PPN:-}" =~ ^[1-9][0-9]*$ ]]; then
            printf '%s\n' "$(( unique_nodes * PBS_NUM_PPN ))"
            return
        fi

        printf '%s\n' "$(( unique_nodes * default_ppn ))"
        return
    fi

    if [[ "${PBS_NP:-}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' "${PBS_NP}"
        return
    fi

    printf '%s\n' 128
}

ntasks=$(detect_ntasks)
lmp_bin="${LAMMPS_BIN}"

seed=""
processors=auto

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            if [[ $# -lt 2 ]]; then
                echo "error: --seed requires a value" >&2
                exit 1
            fi
            seed="$2"
            shift 2
            ;;
        --ntasks)
            if [[ $# -lt 2 ]]; then
                echo "error: --ntasks requires a value" >&2
                exit 1
            fi
            ntasks="$2"
            shift 2
            ;;
        --processors)
            if [[ $# -lt 2 ]]; then
                echo "error: --processors requires a value" >&2
                exit 1
            fi
            processors="$2"
            shift 2
            ;;
        *)
            echo "usage: $0 [--seed N] [--ntasks N] [--processors auto|Px,Py,Pz]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${seed}" ]]; then
    seed="${A3HT_SEED:-123}"
fi

if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    fail "--seed must be a non-negative integer"
fi

run_dir="${rootdir}/my_runs/${seed}"
status_file="${run_dir}/run_status.txt"
failure_file="${run_dir}/run_failure.txt"
plan_json="${run_dir}/simulation_plan.json"
plan_env="${run_dir}/simulation_plan.env"
plan_lmp="${run_dir}/simulation_plan.lmp"
mkdir -p "${run_dir}"
printf 'RUNNING\n' > "${status_file}"
rm -f "${failure_file}"

if ! [[ "${ntasks}" =~ ^[1-9][0-9]*$ ]]; then
    fail "--ntasks must be a positive integer"
fi

if [[ "${processors}" == "auto" ]]; then
    procx="*"
    procy="*"
    procz="*"
elif [[ "${processors}" =~ ^([1-9][0-9]*),([1-9][0-9]*),([1-9][0-9]*)$ ]]; then
    procx="${BASH_REMATCH[1]}"
    procy="${BASH_REMATCH[2]}"
    procz="${BASH_REMATCH[3]}"
    if (( procx * procy * procz != ntasks )); then
        fail "--processors=${processors} must satisfy Px*Py*Pz = --ntasks (${ntasks})"
    fi
else
    fail "--processors must be 'auto' or 'Px,Py,Pz'"
fi

stage="environment_check"

if [[ ! -x "${LAMMPS_BIN}" ]]; then
    fail "LAMMPS executable not found: ${LAMMPS_BIN}"
fi

runtime_lib_dirs=(
    "${LAMMPS_DIR}"
    "/lus/grand/projects/QuantMatManufact/knomura/glassycarbons/lammps-build/kim_build-prefix/lib"
    "/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib"
    "/opt/cray/pe/lib64"
    "/opt/cray/libfabric/2.2.0rc1/lib64"
    "/opt/cray/pals/1.7/lib"
    "/opt/cray/pe/cce/20.0.0/cce/x86_64/lib"
    "/opt/cray/pe/cce/20.0.0/cce-clang/x86_64/lib"
    "/usr/lib64"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${runtime_lib_dirs[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "Using runtime library directories from test_rebo2-compatible configuration"

if [[ ! -f "${rootdir}/CH.rebo" ]]; then
    fail "REBO2 parameter file not found: ${rootdir}/CH.rebo"
fi

cd "${run_dir}"

stage="simulation_planning"
if [[ ! -f "${plan_env}" || ! -f "${plan_lmp}" || ! -f "${plan_json}" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        fail "python3 is required to generate simulation planning artifacts"
    fi
    if [[ ! -f "${PLANNER_SCRIPT}" ]]; then
        fail "planner script not found: ${PLANNER_SCRIPT}"
    fi
    if ! python3 "${PLANNER_SCRIPT}" --seed "${seed}" --run-dir "${run_dir}"; then
        fail "simulation planner failed for seed ${seed}"
    fi
fi

if [[ ! -f "${plan_env}" ]]; then
    fail "simulation plan env file not found: ${plan_env}"
fi
if [[ ! -f "${plan_lmp}" ]]; then
    fail "simulation plan LAMMPS include not found: ${plan_lmp}"
fi

# shellcheck disable=SC1090
source "${plan_env}"

if [ "${A3HT_PLANNER_STATUS:-ok}" != "ok" ]; then
    printf "%s\n" "Planner degraded: source=${A3HT_PLAN_SOURCE:-unknown}" > planner_warning.txt
    if [ -n "${A3HT_PLANNER_ERROR:-}" ]; then
        printf "%s\n" "${A3HT_PLANNER_ERROR}" >> planner_warning.txt
    fi
    echo "warning: Planner degraded: source=${A3HT_PLAN_SOURCE:-unknown}" >&2
fi
echo "Using simulation plan source: ${A3HT_PLAN_SOURCE}"
echo "Cohort: id=${A3HT_COHORT_ID} target_evaluable_seeds=${A3HT_COHORT_SEED_TARGET}"
echo "Plan goal: target_kappa=${A3HT_GOAL_TARGET_KAPPA_W_MK} W/m-K max_rel_uncertainty=${A3HT_GOAL_MAX_REL_UNCERT_PCT}%"
echo "Plan summary: ${A3HT_REASONING_SUMMARY}"

stage="structure_generation"
${rootdir}/generate_random_carbon.py \
    --box "${A3HT_STRUCTURE_BOX_X_A}" "${A3HT_STRUCTURE_BOX_Y_A}" "${A3HT_STRUCTURE_BOX_Z_A}" \
    --density "${A3HT_STRUCTURE_DENSITY_G_CM3}" \
    --seed "${seed}" \
    --output random_carbon.extxyz \
    --flake-area "${A3HT_FLAKE_AREA_A2}" \
    --format lammps
mv random_carbon.extxyz random_carbon.dat

cp -v "${rootdir}/CH.rebo" CH.rebo

stage="anneal"
echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log anneal.log -in ${rootdir}/anneal.in"
"${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log anneal.log -in "${rootdir}/anneal.in"
cp -v data/anneal_gc_rebo2.restart gc_rebo2.restart

stage="thermalize"
echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log thermalize.log -in ${rootdir}/thermalize.in"
"${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log thermalize.log -in "${rootdir}/thermalize.in"
cp -v data/gc_rebo2_thermalize.restart gc_rebo2.restart

stage="nemd"
echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log nemd.log -in ${rootdir}/nemd.in"
"${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log nemd.log -in "${rootdir}/nemd.in"
