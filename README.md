# Agentic AI Accelerated High-Throughput (A3HT) Framework for Thermal Conductivity Calculations

[![LAMMPS](https://img.shields.io/badge/LAMMPS-MD%20Engine-CB2B1E?style=for-the-badge)](https://www.lammps.org/)
[![eHEX](https://img.shields.io/badge/eHEX-NEMD%20Heat%20Exchange-0A7E8C?style=for-the-badge)](https://docs.lammps.org/fix_ehex.html)
[![Python](https://img.shields.io/badge/Python-Analysis%20%26%20ML-3776AB?style=for-the-badge)](https://www.python.org/)
![XGBoost](https://img.shields.io/badge/XGBoost-Thermal%20Conductivity%20Model-EC6B23?style=for-the-badge)

An end-to-end workflow for turning disordered carbon structures into thermal-conductivity data and machine-learning-ready descriptors.

This repository combines atomistic simulation, transport calculations, and data-driven analysis:

- simulation planning with an LLM-based MD review agent (ALCF vLLM endpoint)
- random glassy-carbon structure generation
- high-temperature annealing with the Brenner REBO2 potential
- 300 K equilibration and NEMD thermal conductivity calculations in LAMMPS
- structural analysis of annealed and driven configurations
- feature-table generation for downstream ML models

> In short: plan -> generate structure -> anneal -> thermalize -> drive heat flux with `eHEX` -> analyze -> train.

See [`flowchart.svg`](flowchart.svg) for a visual overview of the three-lane pipeline (queue loop, PBS job, post-processing).

## Repository Layout

```text
config.toml                    ← single source of truth for all parameters
flowchart.svg                  ← pipeline overview diagram
run.sh                         ← PBS job driver
cron_queue.sh                  ← autonomous queue-filler (run from cron)

src/
  config.py                    ← dependency-free TOML loader; shared by all scripts
  plan_simulation.py           ← LLM-based simulation planner (with random fallback)
  autonomy.py                  ← cohort state machine and run-record helpers
  loop_status.py               ← reports loop action (stop / wait / reuse / new cohort)
  generate_random_carbon.py    ← builds the initial disordered carbon network
  prepare_resubmits.py         ← queues failed runs for retry
  build_ml_features.py         ← aggregates per-run outputs into one ML dataset
  train_xgboost_thermal_conductivity.py  ← XGBoost regressor on the feature table
  analyze_glassy_carbon.py     ← analyzes a LAMMPS data file or trajectory snapshot
  analyze_glassy_carbon_trajectory.py   ← time-series analysis of annealing trajectory
  render_snapshots.py          ← ball-and-stick PNG snapshots for completed runs
  simulation_plan_schema.json  ← JSON schema enforced on planner output
  inference_auth_token.py      ← Globus token management for the ALCF endpoint

nemd/
  anneal.in                    ← staged high-temperature annealing schedule
  thermalize.in                ← minimization + NVT/NPT/NVE equilibration to 300 K
  nemd.in                      ← thermal conductivity via fix ehex
  CH.rebo                      ← Brenner REBO2 parameter file
```

## A3HT At A Glance

| Component | Role |
| --- | --- |
| `src/generate_random_carbon.py` | Builds the initial disordered carbon network |
| `src/plan_simulation.py` | Proposes an in-bounds simulation plan for each run |
| `nemd/anneal.in` | Reshapes the network through staged high-temperature annealing |
| `nemd/thermalize.in` | Brings the annealed sample to a stable 300 K state |
| `nemd/nemd.in` | Imposes a heat flux and estimates thermal conductivity |
| `src/analyze_glassy_carbon*.py` | Extracts structural metrics, distributions, and trajectory trends |
| `src/build_ml_features.py` | Aggregates per-run outputs into one ML dataset |
| `src/train_xgboost_thermal_conductivity.py` | Learns structure-property relationships from the generated runs |

The simulation workflow is:

1. Propose a simulation plan for the next run from recent MD results and the current target goal.
2. Generate a random carbon starting structure as small graphene-like flakes.
3. Anneal the structure at high temperature with the Brenner REBO2 carbon potential.
4. Thermalize the annealed structure at 300 K.
5. Run NEMD using the `eHEX` algorithm to impose a heat flux and estimate thermal conductivity.
6. Analyze annealed and NEMD structures.
7. Build an ML feature table and train an XGBoost regressor on the resulting dataset.

## Requirements

You will need:

- LAMMPS with the `RIGID` package enabled so `fix ehex` is available
- Python 3.7 or newer
- Python packages:
  - `numpy`
  - `openai` for the ALCF inference endpoint (the planner falls back to random if absent or unavailable)
  - `xgboost` for model training
- A PBS environment if you want to use `cron_queue.sh` unchanged

## Configuration

All tunable parameters live in **`config.toml`** at the repo root. Edit that file rather than touching any script:

```toml
[paths]
lammps_dir = "lammps-30Mar2026/build-cray-rebo2"   # relative to repo root
python     = "/home/knomura/lammps/.venv/bin/python3"

[campaign]
name         = "kappa10_base90_tilt90"
initial_seed = 1000

[goals]
target_kappa_w_mk               = 10.0    # W/m-K — stop when any cohort reaches this
target_relative_uncertainty_pct = 10.0   # % — and uncertainty is below this
min_cohort_success_seeds        = 10
max_simultaneous_cohorts        = 3

[structure]
base_angle_deg    = 90.0
angle_disturb_deg = 30.0
tilt_max_deg      = 30.0

[constraints]
flake_area_a2    = [25.0, 100.0]
box_x_a          = [40.0, 80.0]
box_y_a          = [40.0, 80.0]
box_z_a          = [80.0, 160.0]
density_g_cm3    = [1.5, 2.0]
nemd_eflux_ev_ps = [1.0, 3.0]
```

Shell scripts load config automatically via:

```bash
eval "$(python3 src/config.py --shell-env)"
```

The `--shell-env` flag emits all campaign variables (`A3HT_RUNS_ROOT`, `A3HT_STATE_DIR`, `LAMMPS_DIR`, structure angles, etc.) as `export` statements. `run.sh` and `cron_queue.sh` both call this at startup so no manual `export` commands are needed.

To authenticate with the ALCF inference endpoint (needed once before the first cron run):

```bash
python3 src/inference_auth_token.py authenticate
```

Tokens are cached in `~/.globus/` and refreshed automatically for up to 30 days. If the ALCF endpoint is unavailable, the planner falls back to random parameter exploration so jobs are never blocked.

To override the maximum number of simultaneous cohorts at runtime without editing `config.toml`:

```bash
export A3HT_MAX_SIMULTANEOUS_COHORTS=5
```

## Simulation Workflow

### 1. Plan the next simulation

`cron_queue.sh` calls the planner before `qsub`, and `run.sh` calls it after environment checks pass if plan artifacts are still missing:

```bash
python3 src/plan_simulation.py --seed 123 --run-dir my_runs/123 --runs-root my_runs
```

The planner (in priority order):

1. reuses the selected active-cohort parameters when repeated same-parameter seeds are still needed
2. tries the ALCF inference endpoint for a new-cohort plan
3. falls back to random parameter exploration if ALCF is unavailable
4. fails with a non-zero exit code only when `--disable-planner` is set and no reusable cohort exists

Each run gets:

- `simulation_plan.json`
- `simulation_plan.env`
- `simulation_plan.lmp`

Current hard geometry constraints (from `config.toml [constraints]`):

- flake area: `25–100 Å²`
- box `x`: `40–80 Å`
- box `y`: `40–80 Å`
- box `z`: `80–160 Å`
- density: `1.5–2.0 g/cm³`
- `nemd_eflux_ev_ps`: `1–3 eV/ps`

The autonomous loop stops submitting new jobs when any cohort reaches:

- mean thermal conductivity `>= 10 W/m-K`
- relative uncertainty `< 10%`
- at least `10` evaluable seeds

### 2. Generate the initial structure

`run.sh` calls `src/generate_random_carbon.py` with box, density, flake-area, and orientation parameters from the per-run plan and structure settings from `config.toml [structure]`.

`--base-angle-deg` is a fixed rotation about x applied to every flake. `--tilt-max-deg` controls the seed-dependent random x/y tilt range.

If the box is too tight to place all atoms without overlap, the packer reduces the flake area in steps and emits a warning to stderr; it returns whatever atoms it managed to place (the achieved density printed to stdout reflects the actual count).

### 3. Anneal the structure

`nemd/anneal.in`:

- includes `simulation_plan.lmp`
- reads `random_carbon.dat`
- initializes the Brenner REBO2 potential via `pair_style rebo` and `pair_coeff * * CH.rebo C`
- minimizes the initial configuration
- applies staged NVT annealing with plan-provided timestep, run length, and velocity seed

The annealing schedule (temperatures from `config.toml [anneal]`):

- 2500 K for 10 ps
- 3000 K for 10 ps
- 3500 K for 10 ps
- 4000 K for 10 ps
- 4000 K for 50 ps

Outputs include:

- `data/anneal_gc_rebo2.restart`
- `data/anneal_gc_rebo2.data`
- `data/anneal_gc_rebo2.lammpstrj`
- `data/anneal_gc_rebo2_coordination.dat`

### 4. Thermalize the annealed structure

`nemd/thermalize.in`:

- includes `simulation_plan.lmp`
- reads `gc_rebo2.restart`
- shifts the periodic cell so wrapped `z` coordinates stay non-negative
- minimizes the annealed structure
- equilibrates with plan-provided temperature, timestep, stage lengths, and velocity seed

Outputs include:

- `data/gc_rebo2_thermalize.restart`
- `data/gc_rebo2_thermalize.data`

### 5. Run NEMD thermal conductivity

`nemd/nemd.in`:

- includes `simulation_plan.lmp`
- reads `gc_rebo2.restart`
- defines frozen slabs at the two ends of the box
- defines hot and cold regions next to the frozen slabs
- integrates the system with `fix nve`
- applies heat exchange with:

```lammps
fix hotflux all ehex 1000 ${nemd_eflux_ev_ps} region hot
fix coldflux all ehex 1000 -${nemd_eflux_ev_ps} region cold
```

- computes a temperature profile along `z`
- estimates the thermal conductivity from the imposed heat flux and measured temperature drop

The conductivity reported in `nemd.in` is:

`kappa = 1602.176634 * Jz * dz / dT`

where `Jz` is the imposed heat flux per cross-sectional area and `dT` is the running temperature difference between the hot and cold slabs.

Outputs include:

- `data/gc_rebo2_Tprofile.dat`
- `data/gc_rebo2_hotcold.dat`
- `data/gc_rebo2_nemd.lammpstrj`
- `data/gc_rebo2_nemd.restart`
- `data/gc_rebo2_nemd.data`

## Running the Full Workflow

```bash
bash run.sh --seed 123 --ntasks 32 --processors auto
```

Options:

- `--seed N`: random seed for the generated carbon structure
- `--ntasks N`: MPI task count passed to `mpiexec` or `mpirun`
- `--processors auto|Px,Py,Pz`: LAMMPS processor grid

Each run is written under `${A3HT_RUNS_ROOT:-my_runs}/<seed>/` with:

- logs: `anneal.log`, `thermalize.log`, `nemd.log`
- status: `run_status.txt` (`SUCCESS` / `FAILED` / `RUNNING`)
- failure detail: `run_failure.txt` (UTC timestamp, failing stage, message)
- planning artifacts: `simulation_plan.json`, `simulation_plan.env`, `simulation_plan.lmp`
- simulation outputs under `data/`

## Queue Management and Resubmission

```bash
bash cron_queue.sh
```

At each invocation `cron_queue.sh` checks the current cohort status:

- `stop`: a cohort already meets the target — no new jobs
- `wait_active_cohorts`: all cohort slots are full and each has enough running jobs — no new jobs
- `reuse_active_cohort`: the next seed reuses the selected open cohort's parameters
- `plan_new_cohort`: a fresh LLM plan is generated for a new cohort

The loop status cache lives at `${A3HT_STATE_DIR}/.queue_state/run_records_cache.json`. Terminal `SUCCESS` and `FAILED` records are cached so repeated cron invocations do not re-parse completed run directories. If you manually edit a completed run's status or final conductivity, delete this cache so the next invocation rebuilds it.

Brand-new runs use successive seeds from `${A3HT_STATE_DIR}/next_seed`. Retry seeds from `${A3HT_STATE_DIR}/resubmit_seeds.txt` are consumed first.

To purge and requeue failed or incomplete runs:

```bash
python3 src/prepare_resubmits.py --purge-run-dirs
```

To also include stale `RUNNING` directories after manual inspection:

```bash
python3 src/prepare_resubmits.py --purge-run-dirs --include-running
```

## Failure Notes

A common failure mode is an executable or batch environment that cannot load the runtime libraries linked into the selected LAMMPS build. `run.sh` will fail at `environment_check` and write a `run_failure.txt` entry such as:

```text
stage=environment_check
message=... error while loading shared libraries: ...
```

`run.sh` builds `LD_LIBRARY_PATH` automatically from `config.toml [runtime_libs]` via `python3 src/config.py --ld-library-path`. Add or remove directories in that section rather than exporting the variable by hand.

## Post-Processing

### Analyze a single structure or final trajectory frame

```bash
python3 src/analyze_glassy_carbon.py my_runs/123/data/anneal_gc_rebo2.data
```

or:

```bash
python3 src/analyze_glassy_carbon.py my_runs/123/data/gc_rebo2_nemd.lammpstrj \
  --output-dir my_runs/123/analysis/nemd
```

### Analyze the annealing trajectory

```bash
python3 src/analyze_glassy_carbon_trajectory.py \
  my_runs/123/data/anneal_gc_rebo2.lammpstrj \
  --coordination-log my_runs/123/data/anneal_gc_rebo2_coordination.dat \
  --output-dir my_runs/123/analysis/anneal_timeseries
```

### Render ball-and-stick snapshots

```bash
python3 src/render_snapshots.py
```

Renders `snapshot.png` into each run directory that has NEMD data. Requires a separate `build-dump-image` LAMMPS build (path set in `config.toml [paths]`).

## Building the ML Dataset

```bash
python3 src/build_ml_features.py --runs-root my_runs --output-csv ml_features.csv
```

To generate missing analysis outputs automatically:

```bash
python3 src/build_ml_features.py \
  --runs-root my_runs \
  --generate-missing-analysis \
  --output-csv ml_features.csv \
  --summary-json ml_features_summary.json
```

A column-by-column overview of the ML inputs is in [FEATURE_GUIDE.md](FEATURE_GUIDE.md).

## Training the XGBoost Model

```bash
python3 src/train_xgboost_thermal_conductivity.py \
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
- If the ALCF planner is unavailable, `src/plan_simulation.py` falls back to random parameter exploration within the hard constraints so the workflow always continues.
- Cohorts are defined by identical physical simulation parameters; random seeds differ within a cohort.
- The NEMD method implemented here is a direct heat-flux approach using `eHEX`, not Green-Kubo.

## Typical Output Layout

```text
my_runs/
  123/
    anneal.log
    thermalize.log
    nemd.log
    gc_rebo2.restart
    simulation_plan.json
    simulation_plan.env
    simulation_plan.lmp
    snapshot.png
    data/
      anneal_gc_rebo2.data
      anneal_gc_rebo2.lammpstrj
      anneal_gc_rebo2.restart
      gc_rebo2_thermalize.data
      gc_rebo2_thermalize.restart
      gc_rebo2_hotcold.dat
      gc_rebo2_Tprofile.dat
      gc_rebo2_nemd.data
      gc_rebo2_nemd.lammpstrj
      gc_rebo2_nemd.restart
    analysis/
      anneal/
      nemd/
      anneal_timeseries/
```
