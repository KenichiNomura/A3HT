# Glassy Carbon Thermal Conductivity with LAMMPS and Machine Learning

This repository builds disordered carbon structures, anneals them with LAMMPS, equilibrates them, computes thermal conductivity with non-equilibrium molecular dynamics (NEMD), and then extracts structural features for machine-learning models.

The simulation workflow in this repo is:

1. Generate a random carbon starting structure as small graphene-like flakes.
2. Anneal the structure at high temperature with the OpenKIM EDIP carbon potential.
3. Thermalize the annealed structure at 300 K.
4. Run NEMD using the `eHEX` algorithm to impose a heat flux and estimate thermal conductivity.
5. Analyze annealed and NEMD structures.
6. Build an ML feature table and train an XGBoost regressor on the resulting dataset.

## Requirements

You will need:

- LAMMPS with:
  - `KIM` package enabled
  - `RIGID` and `OPENMP` support if you want to match the included local build names
  - `fix ehex` available for the NEMD heat-exchange step
- OpenKIM installed and configured
- The OpenKIM portable model:
  - `EDIP_LAMMPS_Marks_2000_C__MO_374144505645_000`
- Python 3
- Python packages:
  - `numpy`
  - `xgboost` for model training
- A Slurm environment if you want to run `run.sh` unchanged, because it uses `srun`

The default run script expects the LAMMPS executable at:

`build-kim-rigid-omp-kimenv/lmp`

It also tries to locate the OpenKIM model under:

`$HOME/.kim-api`

and exports:

- `KIM_API_MODEL_DRIVERS_DIR`
- `KIM_API_PORTABLE_MODELS_DIR`
- `KIM_API_SIMULATOR_MODELS_DIR`

based on that installation.

## Main Files

- `run.sh`: end-to-end driver for structure generation, annealing, thermalization, and NEMD
- `generate_random_carbon.py`: creates a random carbon network from rotated graphene-like flakes
- `anneal.in`: high-temperature annealing schedule using the Marks 2000 EDIP/C OpenKIM model
- `thermalize.in`: minimization plus NVT/NPT/NVE equilibration before transport calculation
- `nemd.in`: thermal conductivity calculation with `fix ehex`
- `analyze_glassy_carbon.py`: analyzes a LAMMPS data file or trajectory snapshot
- `analyze_glassy_carbon_trajectory.py`: analyzes an annealing trajectory as a time series
- `build_ml_features.py`: collects analysis outputs into one ML feature table
- `train_xgboost_thermal_conductivity.py`: trains an XGBoost regressor on the feature table
- `generate_analysis_powerpoint.py`: creates a simple `.pptx` summary for one analyzed run

## Simulation Workflow

### 1. Generate the initial structure

`run.sh` calls:

```bash
./generate_random_carbon.py \
  --box 20 20 40 \
  --density 1.5 \
  --seed 123 \
  --output random_carbon.extxyz \
  --flake-area 20 \
  --format lammps
```

The generated file is then renamed to `random_carbon.dat` and used as the LAMMPS input structure.

### 2. Anneal the structure

`anneal.in`:

- reads `random_carbon.dat`
- initializes the OpenKIM model `EDIP_LAMMPS_Marks_2000_C__MO_374144505645_000`
- minimizes the initial configuration
- applies staged NVT annealing

The annealing schedule is:

- 2500 K for 10 ps
- 3000 K for 10 ps
- 3500 K for 10 ps
- 4000 K for 10 ps
- 4000 K for 50 ps

Outputs include:

- `data/anneal_gc_edip_multistage.restart`
- `data/anneal_gc_edip_multistage.data`
- `data/anneal_gc_edip_multistage.lammpstrj`
- `data/anneal_gc_edip_multistage_coordination.dat`

### 3. Thermalize the annealed structure

`thermalize.in`:

- reads `gc_edip.restart`
- shifts the periodic cell so wrapped `z` coordinates stay non-negative
- minimizes the annealed structure
- equilibrates at 300 K with repeated NVT, NPT, and NVE stages

Outputs include:

- `data/gc_edip_thermalize.restart`
- `data/gc_edip_thermalize.data`

### 4. Run NEMD thermal conductivity

`nemd.in`:

- reads `gc_edip.restart`
- defines frozen slabs at the two ends of the box
- defines hot and cold regions next to the frozen slabs
- integrates the system with `fix nve`
- applies heat exchange with:

```lammps
fix hotflux all ehex 1000 ${eflux} region hot
fix coldflux all ehex 1000 -${eflux} region cold
```

- computes a temperature profile along `z`
- computes running hot and cold slab temperatures
- estimates the thermal conductivity from the imposed heat flux and measured temperature drop

Important NEMD settings in the current script:

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

The main driver is:

```bash
bash run.sh --seed 123 --ntasks 32 --processors auto
```

Options supported by `run.sh`:

- `--seed N`: random seed for the generated carbon structure
- `--ntasks N`: MPI task count passed to `srun`
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

and simulation outputs under:

`my_runs/<seed>/data/`

## Post-Processing

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

## Training the XGBoost Model

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

- `run.sh` is written for Slurm and uses `srun`; adapt it if you want to run without Slurm.
- The included helper script `generate_analysis_powerpoint.py` is currently hard-coded to use `my_runs/100/analysis`.
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
