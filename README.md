# Agentic AI Accelerated High-Throughput (A3HT) Framework for Thermal Conductivity Calculations

[![LAMMPS](https://img.shields.io/badge/LAMMPS-MD%20Engine-CB2B1E?style=for-the-badge)](https://www.lammps.org/)
[![OpenKIM](https://img.shields.io/badge/OpenKIM-Interatomic%20Models-00539C?style=for-the-badge)](https://openkim.org)
[![EDIP](https://img.shields.io/badge/EDIP-Marks%202000-6C757D?style=for-the-badge)](https://link.aps.org/doi/10.1103/PhysRevB.63.035401)
[![eHEX](https://img.shields.io/badge/eHEX-NEMD%20Heat%20Exchange-0A7E8C?style=for-the-badge)](https://docs.lammps.org/fix_ehex.html)
[![Python](https://img.shields.io/badge/Python-Analysis%20%26%20ML-3776AB?style=for-the-badge)](https://www.python.org/)
![XGBoost](https://img.shields.io/badge/XGBoost-Thermal%20Conductivity%20Model-EC6B23?style=for-the-badge)

An end-to-end workflow for turning disordered carbon structures into thermal-conductivity data and machine-learning-ready descriptors.

This repository combines atomistic simulation, transport calculations, and data-driven analysis:

- simulation planning with a Codex-based MD review agent
- random glassy-carbon structure generation
- high-temperature annealing with the Marks 2000 EDIP OpenKIM potential
- 300 K equilibration and NEMD thermal conductivity calculations in LAMMPS
- structural analysis of annealed and driven configurations
- feature-table generation for downstream ML models

> In short: plan -> generate structure -> anneal -> thermalize -> drive heat flux with `eHEX` -> analyze -> train.

## A3HT At A Glance

| Component | Role |
| --- | --- |
| `generate_random_carbon.py` | Builds the initial disordered carbon network |
| `plan_simulation.py` | Proposes an in-bounds simulation plan for each run |
| `anneal.in` | Reshapes the network through staged high-temperature annealing |
| `thermalize.in` | Brings the annealed sample to a stable 300 K state |
| `nemd.in` | Imposes a heat flux and estimates thermal conductivity |
| `analyze_glassy_carbon*.py` | Extracts structural metrics, distributions, and trajectory trends |
| `build_ml_features.py` | Aggregates per-run outputs into one ML dataset |
| `train_xgboost_thermal_conductivity.py` | Learns structure-property relationships from the generated runs |

The simulation workflow in this repo is:

1. Propose a simulation plan for the next run from recent MD results and the current target goal.
2. Generate a random carbon starting structure as small graphene-like flakes.
3. Anneal the structure at high temperature with the OpenKIM EDIP carbon potential.
4. Thermalize the annealed structure at 300 K.
5. Run NEMD using the `eHEX` algorithm to impose a heat flux and estimate thermal conductivity.
6. Analyze annealed and NEMD structures.
7. Build an ML feature table and train an XGBoost regressor on the resulting dataset.

## Requirements

You will need:

- LAMMPS with:
  - `KIM` package enabled
  - `RIGID` and `OPENMP` support if you want to match the included local build names
  - `fix ehex` available for the NEMD heat-exchange step
- OpenKIM installed and configured
- The OpenKIM portable model:
  - `EDIP_LAMMPS_Marks_2000_C__MO_374144505645_000`
- Python 3.7 or newer
- Python packages:
  - `numpy`
  - `xgboost` for model training
- A PBS environment if you want to use `cron_queue.sh` unchanged

Recommended practical setup:

- a LAMMPS build linked cleanly against OpenKIM
- the Marks 2000 EDIP/C model installed in your user KIM collection
- multiple independent seeds in `my_runs/` if you want meaningful ML training data
- a working `codex exec` installation if you want AI-generated plans rather than fallback defaults
- a PBS environment if you want to use the included queue-filler unchanged

If `codex` is installed outside the default non-interactive `PATH`, set:

```bash
export A3HT_CODEX_BIN=/home/knomura/.nvm/versions/node/v24.14.1/bin/codex
```

This is especially important for cron and PBS jobs, which often do not inherit your interactive shell startup files.

The default run script expects the LAMMPS executable at:

`lammps-30Mar2026/build-cray-shared/lmp`

For the in-repo `build-cray-shared` executable to start cleanly, the runtime environment must also provide:

- `liblammps.so.0` from `lammps-30Mar2026/build-cray-shared`
- `libkim-api.so.2` from a KIM installation such as `/lus/grand/projects/QuantMatManufact/knomura/glassycarbons/lammps-build/kim_build-prefix/lib`

`run.sh` now prepends both of those locations to `LD_LIBRARY_PATH` automatically when available.

It also tries to locate the OpenKIM model under:

`$HOME/.kim-api`

and exports:

- `KIM_API_MODEL_DRIVERS_DIR`
- `KIM_API_PORTABLE_MODELS_DIR`
- `KIM_API_SIMULATOR_MODELS_DIR`

based on that installation.

## Main Files

- `run.sh`: end-to-end driver for environment checks, optional simulation planning, structure generation, annealing, thermalization, and NEMD
- `cron_queue.sh`: drives the autonomous loop by checking cohort stop/wait conditions, planning the next run before submission, and prioritizing retry seeds from `.queue_state/resubmit_seeds.txt`
- `plan_simulation.py`: uses `codex exec` or active-cohort reuse to choose per-run simulation parameters, validates hard constraints, and writes run-local plan artifacts
- `loop_status.py`: reports whether the autonomous loop should stop, wait for the active cohorts, reuse a selected cohort, or open a new cohort
- `autonomy.py`: shared cohort statistics and stop-condition logic
- `simulation_plan_schema.json`: JSON schema enforced on planner output
- `prepare_resubmits.py`: finds failed/incomplete runs, purges their run directories, and writes the retry queue for cron
- `generate_random_carbon.py`: creates a random carbon network from rotated graphene-like flakes
- `anneal.in`: high-temperature annealing schedule using the Marks 2000 EDIP/C OpenKIM model
- `thermalize.in`: minimization plus NVT/NPT/NVE equilibration before transport calculation
- `nemd.in`: thermal conductivity calculation with `fix ehex`
- `analyze_glassy_carbon.py`: analyzes a LAMMPS data file or trajectory snapshot
- `analyze_glassy_carbon_trajectory.py`: analyzes an annealing trajectory as a time series
- `build_ml_features.py`: collects analysis outputs into one ML feature table
- `train_xgboost_thermal_conductivity.py`: trains an XGBoost regressor on the feature table

## Simulation Workflow

The simulation side of A3HT is organized as a compact planned pipeline before analysis: propose the next in-bounds run, build a candidate carbon network, structurally relax it through annealing and equilibration, then measure transport under a controlled non-equilibrium heat flux.

### 1. Plan the next simulation

`cron_queue.sh` calls the planner before `qsub`, and `run.sh` calls it after environment checks pass if the plan artifacts are still missing:

```bash
python3 plan_simulation.py --seed 123 --run-dir my_runs/123
```

The planner:

- summarizes recent successful runs from `my_runs/`
- asks `codex exec` for a structured next-run plan
- reuses the active cohort parameters when repeated same-parameter seeds are still needed
- validates the result against the current hard bounds
- falls back to a conservative default plan if Codex is unavailable or the output is invalid

Each run gets:

- `simulation_plan.json`
- `simulation_plan.env`
- `simulation_plan.lmp`

Current hard geometry constraints are:

- flake area: `10-30 A^2`
- box `x`: `20-50 A`
- box `y`: `20-50 A`
- box `z`: `40-100 A`

The current target goal encoded in the planner is:

- thermal conductivity target: `3 W/m-K`
- relative uncertainty target: `< 10%`
- minimum evaluable seeds per cohort: `10`
- maximum simultaneous open cohorts: `3` by default

The same physical parameter set is repeated with different random seeds within a cohort until at least 10 evaluable seeds are available for uncertainty estimation. The autonomous loop may keep up to 3 open cohorts in flight at the same time by default.

The autonomous loop stops submitting new jobs when any cohort reaches:

- mean thermal conductivity `>= 6 W/m-K`
- relative uncertainty `< 10%`
- at least `10` evaluable seeds in that cohort

The relative uncertainty is computed from the standard error of the cohort mean thermal conductivity.

### 2. Generate the initial structure

`run.sh` calls:

```bash
./generate_random_carbon.py \
  --box "${A3HT_STRUCTURE_BOX_X_A}" "${A3HT_STRUCTURE_BOX_Y_A}" "${A3HT_STRUCTURE_BOX_Z_A}" \
  --density "${A3HT_STRUCTURE_DENSITY_G_CM3}" \
  --seed 123 \
  --output random_carbon.extxyz \
  --flake-area "${A3HT_FLAKE_AREA_A2}" \
  --format lammps
```

The generated file is then renamed to `random_carbon.dat` and used as the LAMMPS input structure.

### 3. Anneal the structure

`anneal.in`:

- includes `simulation_plan.lmp`
- reads `random_carbon.dat`
- initializes the OpenKIM model `EDIP_LAMMPS_Marks_2000_C__MO_374144505645_000`
- minimizes the initial configuration
- applies staged NVT annealing with plan-provided timestep, run length, and velocity seed

The annealing schedule is:

- 2500 K for 10 ps
- 3000 K for 10 ps
- 3500 K for 10 ps
- 4000 K for 10 ps
- 4000 K for 50 ps

This stage is where the initially random flake assembly is driven toward a more connected glassy-carbon network.

Outputs include:

- `data/anneal_gc_edip_multistage.restart`
- `data/anneal_gc_edip_multistage.data`
- `data/anneal_gc_edip_multistage.lammpstrj`
- `data/anneal_gc_edip_multistage_coordination.dat`

### 4. Thermalize the annealed structure

`thermalize.in`:

- includes `simulation_plan.lmp`
- reads `gc_edip.restart`
- shifts the periodic cell so wrapped `z` coordinates stay non-negative
- minimizes the annealed structure
- equilibrates with plan-provided temperature, timestep, stage lengths, and velocity seed

This separates structural preparation from the transport calculation so the NEMD run starts from an already relaxed state.

Outputs include:

- `data/gc_edip_thermalize.restart`
- `data/gc_edip_thermalize.data`

### 5. Run NEMD thermal conductivity

`nemd.in`:

- includes `simulation_plan.lmp`
- reads `gc_edip.restart`
- defines frozen slabs at the two ends of the box
- defines hot and cold regions next to the frozen slabs
- integrates the system with `fix nve`
- applies heat exchange with:

```lammps
fix hotflux all ehex 1000 ${nemd_eflux_ev_ps} region hot
fix coldflux all ehex 1000 -${nemd_eflux_ev_ps} region cold
```

- computes a temperature profile along `z`
- computes running hot and cold slab temperatures
- estimates the thermal conductivity from the imposed heat flux and measured temperature drop

The transport setup uses frozen boundary slabs plus hot/cold exchange regions, so the calculation is a direct non-equilibrium estimate rather than an equilibrium fluctuation method.

Important NEMD settings are now provided by the per-run plan. The default fallback plan uses:

- `dt = 0.0001 ps`
- `slabw = 5.0 A`
- `freezew = 5.0 A`
- `eflux = 0.2 eV/ps`
- `nemd_steps = 1000000`

The conductivity reported in `nemd.in` is:

`kappa = 1602.176634 * Jz * dz / dT`

where `Jz` is the imposed heat flux per cross-sectional area and `dT` is the running temperature difference between the hot and cold slabs.

Outputs include:

- `data/gc_edip_Tprofile.cont.dat`
- `data/gc_edip_hotcold.cont.dat`
- `data/gc_edip_nemd.cont.lammpstrj`
- `data/gc_edip_nemd.cont.restart`
- `data/gc_edip_nemd.cont.data`

## Running the Full Workflow

For a standard run, you only need a seed, a valid LAMMPS executable, and the OpenKIM model installed. If the environment checks pass and a per-run plan does not already exist, `run.sh` will generate one automatically.

The main driver is:

```bash
bash run.sh --seed 123 --ntasks 32 --processors auto
```

Options supported by `run.sh`:

- `--seed N`: random seed for the generated carbon structure
- `--ntasks N`: MPI task count passed to `mpiexec` or `mpirun`
- `--processors auto|Px,Py,Pz`: LAMMPS processor grid

Example with an explicit processor grid:

```bash
bash run.sh --seed 101 --ntasks 32 --processors 4,4,2
```

Each run is written under:

`my_runs/<seed>/`

with logs:

- `anneal.log`
- `thermalize.log`
- `nemd.log`
- `run_status.txt`
- `run_failure.txt` when a run exits unsuccessfully

and planning artifacts:

- `simulation_plan.json`
- `simulation_plan.env`
- `simulation_plan.lmp`

These plan artifacts record the cohort id, planner source, target conductivity, target uncertainty, and the minimum evaluable-seed count for the cohort.

and simulation outputs under:

`my_runs/<seed>/data/`

`run_status.txt` contains one of:

- `SUCCESS`
- `FAILED`
- `RUNNING`

If a run fails, `run_failure.txt` records the UTC timestamp, failing stage, and message.

## Queue Management and Resubmission

The repository includes a lightweight PBS queue-filler:

```bash
bash cron_queue.sh
```

If the planner reports that `codex` is missing in cron or PBS, export the Codex binary path before launching the queue filler:

```bash
export A3HT_CODEX_BIN=/home/knomura/.nvm/versions/node/v24.14.1/bin/codex
bash cron_queue.sh
```

`cron_queue.sh` forwards `A3HT_CODEX_BIN` into `qsub`, so the submitted batch job can resolve the same Codex executable if `run.sh` needs to regenerate planning artifacts.

To override the default number of simultaneous cohorts, set:

```bash
export A3HT_MAX_SIMULTANEOUS_COHORTS=3
```

By default it tries to keep up to `A3HT_TARGET_JOBS` jobs in the scheduler, subject to the autonomous loop stop/wait rules, and submits `run.sh` with successive seeds from:

`.queue_state/next_seed`

Before each `qsub`, `cron_queue.sh` checks the current cohort status:

- `stop`: no new jobs are submitted because a cohort already meets the target
- `wait_active_cohorts`: no new jobs are submitted because the maximum number of simultaneous cohorts is already open and each has enough running jobs to potentially reach the minimum cohort size
- `reuse_active_cohort`: the next seed reuses the selected open cohort parameters
- `plan_new_cohort`: a fresh plan is generated for a new cohort

When a submission is needed, `cron_queue.sh` creates `my_runs/<seed>/simulation_plan.*` so the submitted job already has a validated parameter set and cohort assignment.

If:

- a run crashes
- a job times out
- the environment check fails
- or you want to purge and resubmit incomplete runs

use:

```bash
python3 prepare_resubmits.py --purge-run-dirs
```

This script:

- scans `my_runs/` for non-successful runs
- queues failed seeds in `.queue_state/resubmit_seeds.txt`
- writes a manifest to `.queue_state/resubmit_manifest.json`
- removes the corresponding run directories before retry so stale partial outputs do not survive into the resubmission

`cron_queue.sh` consumes `.queue_state/resubmit_seeds.txt` before it advances `.queue_state/next_seed`, so retries are submitted ahead of brand-new seeds.

The default behavior is conservative: it queues `FAILED` runs and leaves currently `RUNNING` runs untouched. If you intentionally want to include stale `RUNNING` directories after manual inspection, use:

```bash
python3 prepare_resubmits.py --purge-run-dirs --include-running
```

## Failure Notes

One common failure mode is a LAMMPS executable that was built without the `KIM` package. In that case `run.sh` will fail during `environment_check` and write a `run_failure.txt` entry like:

```text
stage=environment_check
message=... was built without the LAMMPS KIM package enabled
```

Those runs are safe to purge and requeue after you point `LAMMPS_BIN` or `LAMMPS_DIR` at a KIM-enabled build.

Another common failure mode is a correct in-repo `lmp` executable with missing runtime libraries. In that case `run.sh` will fail during `environment_check` before the `-help` package probes can succeed. The intended default configuration is:

```bash
export LAMMPS_DIR=/lus/eagle/projects/uMLIP-PET-FT/knomura/a3ht/lammps-30Mar2026/build-cray-shared
export LAMMPS_BIN=$LAMMPS_DIR/lmp
export LD_LIBRARY_PATH=$LAMMPS_DIR:/lus/grand/projects/QuantMatManufact/knomura/glassycarbons/lammps-build/kim_build-prefix/lib:$LD_LIBRARY_PATH
```

If you keep the default `run.sh` paths, you should not need to set these manually unless your batch environment strips `LD_LIBRARY_PATH`.

## Post-Processing

Once a run finishes, the analysis scripts turn raw LAMMPS outputs into summaries that are easier to inspect, compare, and use for ML.

### Analyze a single structure or final trajectory frame

```bash
python analyze_glassy_carbon.py my_runs/123/data/anneal_gc_edip_multistage.data
```

or:

```bash
python analyze_glassy_carbon.py my_runs/123/data/gc_edip_nemd.cont.lammpstrj \
  --output-dir my_runs/123/analysis/nemd
```

This script writes JSON, CSV, and SVG summaries such as:

- `summary.json`
- `bond_length_distribution.csv`
- `bond_angle_distribution.csv`
- `rdf.csv`
- `coordination_histogram.csv`

### Analyze the annealing trajectory

```bash
python analyze_glassy_carbon_trajectory.py \
  my_runs/123/data/anneal_gc_edip_multistage.lammpstrj \
  --coordination-log my_runs/123/data/anneal_gc_edip_multistage_coordination.dat \
  --output-dir my_runs/123/analysis/anneal_timeseries
```

This produces a time-series summary in:

`my_runs/123/analysis/anneal_timeseries/trajectory_summary.csv`

## Building the ML Dataset

The ML pipeline is designed around many completed runs under `my_runs/`, where each seed acts as one structure-processing-transport sample.

After you have multiple completed runs in `my_runs/`, build the feature table with:

```bash
python build_ml_features.py --runs-root my_runs --output-csv ml_features.csv
```

If analysis outputs are missing, generate them automatically:

```bash
python build_ml_features.py \
  --runs-root my_runs \
  --generate-missing-analysis \
  --output-csv ml_features.csv \
  --summary-json ml_features_summary.json
```

The target extracted from each run is the final thermal conductivity in:

`my_runs/<seed>/data/gc_edip_hotcold.cont.dat`

The feature builder uses:

- annealed snapshot metrics
- final NEMD snapshot metrics
- annealing trajectory summary statistics
- histogram-derived descriptors
- final hot/cold slab temperatures and conductivity target

A column-by-column overview of the ML inputs is in [FEATURE_GUIDE.md](FEATURE_GUIDE.md).

## Training the XGBoost Model

With `ml_features.csv` in place, the final step is a supervised regression model that maps structural descriptors to the final NEMD conductivity target.

Train the regression model with:

```bash
python train_xgboost_thermal_conductivity.py \
  --features-csv ml_features.csv \
  --output-dir xgboost_thermal_conductivity_model
```

Outputs include:

- `xgboost_thermal_conductivity_model/xgboost_model.json`
- `xgboost_thermal_conductivity_model/feature_importance.csv`
- `xgboost_thermal_conductivity_model/train_predictions.csv`
- `xgboost_thermal_conductivity_model/test_predictions.csv`
- `xgboost_thermal_conductivity_model/training_summary.json`

## Notes and Assumptions

- `run.sh` is written for PBS and launches LAMMPS through `mpiexec` or `mpirun`.
- If `codex exec` is unavailable or fails, `plan_simulation.py` falls back to a conservative default in-bounds plan so the workflow can continue.
- Cohorts are defined by identical physical simulation parameters; random seeds differ within a cohort.
- The repository contains local LAMMPS build directories, but the documented requirement is a LAMMPS executable that supports OpenKIM and `fix ehex`.
- The NEMD method implemented here is a direct heat-flux approach using `eHEX`, not Green-Kubo.

## Typical Output Layout

```text
my_runs/
  123/
    anneal.log
    thermalize.log
    nemd.log
    gc_edip.restart
    simulation_plan.json
    simulation_plan.env
    simulation_plan.lmp
    data/
      anneal_gc_edip_multistage.data
      anneal_gc_edip_multistage.lammpstrj
      anneal_gc_edip_multistage.restart
      gc_edip_thermalize.data
      gc_edip_thermalize.restart
      gc_edip_hotcold.cont.dat
      gc_edip_Tprofile.cont.dat
      gc_edip_nemd.cont.data
      gc_edip_nemd.cont.lammpstrj
      gc_edip_nemd.cont.restart
    analysis/
      anneal/
      nemd/
      anneal_timeseries/
```
