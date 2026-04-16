#!/bin/bash
#SBATCH -A m5047
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH --ntasks-per-node=128
#SBATCH -N 4
#SBATCH -n 512
#SBATCH --qos=preempt
#SBATCH --requeue

rootdir=${PWD}
ntasks=${SLURM_NTASKS:-32}
lmp_bin="${rootdir}/build-kim-rigid-omp-kimenv/lmp"

set -euo pipefail

seed=123
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

if ! [[ "${ntasks}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --ntasks must be a positive integer" >&2
    exit 1
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
        echo "error: --processors=${processors} must satisfy Px*Py*Pz = --ntasks (${ntasks})" >&2
        exit 1
    fi
else
    echo "error: --processors must be 'auto' or 'Px,Py,Pz'" >&2
    exit 1
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

mkdir -p my_runs/${seed} && cd my_runs/${seed}

${rootdir}/generate_random_carbon.py --box 20 20 40 --density 1.5 --seed "${seed}" --output random_carbon.extxyz --flake-area 20 --format lammps
mv random_carbon.extxyz random_carbon.dat

if [[ ! -x "${lmp_bin}" ]]; then
    echo "error: LAMMPS executable not found or not executable: ${lmp_bin}" >&2
    exit 1
fi

echo "srun -n ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log anneal.log -in ${rootdir}/anneal.in"
srun -n "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log anneal.log -in ${rootdir}/anneal.in
cp -v data/anneal_gc_edip_multistage.restart gc_edip.restart

echo "srun -n ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log thermalize.log -in ${rootdir}/thermalize.in"
srun -n "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log thermalize.log -in ${rootdir}/thermalize.in
cp -v data/gc_edip_thermalize.restart gc_edip.restart 

echo "srun -n ${ntasks} ${lmp_bin} -var procx ${procx} -var procy ${procy} -var procz ${procz} -log nemd.log -in ${rootdir}/nemd.in"
srun -n "${ntasks}" "${lmp_bin}" -var procx "${procx}" -var procy "${procy}" -var procz "${procz}" -log nemd.log -in ${rootdir}/nemd.in
