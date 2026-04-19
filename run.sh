#!/bin/bash
#PBS -A FoundMLIP
#PBS -l filesystems=eagle
#PBS -N a3ht
#PBS -q workq
#PBS -l select=4:ncpus=128
#PBS -l walltime=06:00:00
#PBS -j oe

set -euo pipefail

rootdir=/lus/eagle/projects/uMLIP-PET-FT/knomura/a3ht
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

default_lammps_dir="${rootdir}/lammps-30Mar2026/build-cray-shared"
LAMMPS_DIR="${LAMMPS_DIR:-${default_lammps_dir}}"
LAMMPS_BIN="${LAMMPS_BIN:-${LAMMPS_DIR}/lmp}"
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

if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
    ntasks=$(wc -l < "${PBS_NODEFILE}")
else
    ntasks=32
fi
lmp_bin="${LAMMPS_BIN}"

seed=""
processors=auto
kim_model_id=EDIP_LAMMPS_Marks_2000_C__MO_374144505645_000

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

kim_runtime_lib=""
ldd_kim_lib=$(ldd "${LAMMPS_BIN}" 2>/dev/null | awk '/libkim-api\.so/ && $3 ~ /^\// {print $3; exit}')
for candidate in     "${LAMMPS_DIR}/kim_build-prefix/lib"     "${ldd_kim_lib:+$(dirname "${ldd_kim_lib}")}"     "/lus/grand/projects/QuantMatManufact/knomura/glassycarbons/lammps-build/kim_build-prefix/lib"
do
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
        kim_runtime_lib="${candidate}"
        break
    fi
done

if [[ -z "${kim_runtime_lib}" ]]; then
    fail "could not locate the OpenKIM runtime library directory for ${LAMMPS_BIN}"
fi

runtime_lib_dirs=(
    "${LAMMPS_DIR}"
    "${kim_runtime_lib}"
    "/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib"
    "/opt/cray/pe/lib64"
    "/opt/cray/libfabric/2.2.0rc1/lib64"
    "/opt/cray/pals/1.7/lib"
    "/opt/cray/pe/cce/20.0.0/cce/x86_64/lib"
    "/opt/cray/pe/cce/20.0.0/cce-clang/x86_64/lib"
    "/usr/lib64"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${runtime_lib_dirs[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if ! "${LAMMPS_BIN}" -help 2>/dev/null | awk '
    /^Installed packages:/ {in_pkgs=1; next}
    /^List of individual style options included in this LAMMPS executable/ {exit !found}
    in_pkgs && /(^|[[:space:]])KIM([[:space:]]|$)/ {found=1}
    END {exit !found}
'; then
    fail "${LAMMPS_BIN} was built without the LAMMPS KIM package enabled"
fi

if ! "${LAMMPS_BIN}" -help 2>/dev/null | awk '
    /^\* Fix styles/ {in_fix=1; next}
    /^\* / && in_fix {exit !found}
    in_fix && /(^|[[:space:]])ehex([[:space:]]|$)/ {found=1}
    END {exit !found}
'; then
    fail "${LAMMPS_BIN} does not provide fix ehex, which nemd.in requires"
fi

kim_model_path=$(find "${HOME}/.kim-api" -maxdepth 3 -type d -name "${kim_model_id}" -print -quit 2>/dev/null || true)
if [[ -n "${kim_model_path}" ]]; then
    kim_portable_dir=$(dirname "${kim_model_path}")
    kim_root_dir=$(dirname "${kim_portable_dir}")
    export KIM_API_MODEL_DRIVERS_DIR="${kim_root_dir}/model-drivers-dir${KIM_API_MODEL_DRIVERS_DIR:+:${KIM_API_MODEL_DRIVERS_DIR}}"
    export KIM_API_PORTABLE_MODELS_DIR="${kim_portable_dir}${KIM_API_PORTABLE_MODELS_DIR:+:${KIM_API_PORTABLE_MODELS_DIR}}"
    export KIM_API_SIMULATOR_MODELS_DIR="${kim_root_dir}/simulator-models-dir${KIM_API_SIMULATOR_MODELS_DIR:+:${KIM_API_SIMULATOR_MODELS_DIR}}"
    echo "Using KIM model collection from ${kim_root_dir}"
else
    echo "warning: could not find ${kim_model_id} under ${HOME}/.kim-api" >&2
fi

cd "${run_dir}"

stage="structure_generation"
${rootdir}/generate_random_carbon.py --box 20 20 40 --density 1.5 --seed "${seed}" --output random_carbon.extxyz --flake-area 20 --format lammps
mv random_carbon.extxyz random_carbon.dat

stage="anneal"
echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log anneal.log -in ${rootdir}/anneal.in"
"${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log anneal.log -in "${rootdir}/anneal.in"
cp -v data/anneal_gc_edip_multistage.restart gc_edip.restart

stage="thermalize"
echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log thermalize.log -in ${rootdir}/thermalize.in"
"${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log thermalize.log -in "${rootdir}/thermalize.in"
cp -v data/gc_edip_thermalize.restart gc_edip.restart 

stage="nemd"
echo "${mpi_launcher[*]} ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log nemd.log -in ${rootdir}/nemd.in"
"${mpi_launcher[@]}" "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log nemd.log -in "${rootdir}/nemd.in"
