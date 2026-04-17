# ML Feature Guide

This document explains the feature families written by `build_ml_features.py`, how they are calculated, and what structural signal they are intended to capture for the thermal-conductivity model.

## Naming Scheme

Most columns follow one of these prefixes:

- `anneal_...`: values from the post-anneal structure snapshot in `analysis/anneal/summary.json`
- `nemd_...`: values from the final NEMD structure snapshot in `analysis/nemd/summary.json`
- `anneal_ts_...`: statistics computed from the annealing trajectory table in `analysis/anneal_timeseries/trajectory_summary.csv`
- `delta_nemd_minus_anneal_...`: change between the final NEMD state and the annealed state
- `target_...`: final transport outputs parsed from `data/gc_edip_hotcold.cont.dat`

Each run contributes one row to `ml_features.csv`.

## Core Snapshot Features

These come directly from `analyze_glassy_carbon.py`.

| Feature family | How it is calculated | What it captures |
| --- | --- | --- |
| `*_atom_count` | Number of atoms in the analyzed snapshot | System size |
| `*_volume_angstrom3` | Simulation cell volume from box lengths | Sample volume |
| `*_density_g_cm3` | Carbon mass divided by box volume | Packing density / compactness |
| `*_bond_count` | Number of C-C pairs within the bond cutoff (`1.85 A` by default) | Overall network connectivity |
| `*_mean_coordination` | Average neighbor count in the cutoff bond graph | Mean local bonding environment |
| `*_sp2_like_fraction` | Fraction of atoms with coordination `3` | Graphitic, trigonal-like bonding content |
| `*_sp3_like_fraction` | Fraction of atoms with coordination `4` | Tetrahedral-like bonding content |
| `*_undercoordinated_fraction` | Fraction of atoms with coordination `<= 2` | Defects, dangling/chain-like environments |
| `*_overcoordinated_fraction` | Fraction of atoms with coordination `>= 5` | Highly distorted or compressed local environments |
| `*_bond_length_mean_angstrom` | Mean bond length over cutoff-defined bonds | Typical C-C spacing |
| `*_bond_length_std_angstrom` | Standard deviation of bond lengths | Bond-length disorder |
| `*_bond_angle_mean_deg` | Mean angle formed by bonded triplets | Average local angular geometry |
| `*_bond_angle_std_deg` | Standard deviation of bond angles | Angular disorder |

## Threefold-Atom Geometry Features

These are evaluated only on atoms with coordination `3`.

| Feature family | How it is calculated | What it captures |
| --- | --- | --- |
| `*_threefold_atom_count` | Count of 3-coordinated atoms | Amount of graphitic-like local motifs |
| `*_threefold_planarity_rms_mean_angstrom` | Mean RMS distance of the three neighbors from the best-fit local plane | How flat or warped 3-fold sites are |
| `*_threefold_pyramidalization_mean` | Mean out-of-plane distortion metric for 3-fold atoms | Degree of buckling away from ideal trigonal geometry |
| `*_threefold_normal_alignment_mean_abs_cos` | Mean absolute cosine between neighboring 3-fold local normals | Orientational coherence of graphitic patches |

In practice, lower planarity RMS and lower pyramidalization indicate flatter local sheets, while higher normal alignment suggests neighboring graphitic fragments are more similarly oriented.

## Coordination Histogram Features

These are derived from the cutoff-based bond graph.

| Feature family | How it is calculated | What it captures |
| --- | --- | --- |
| `*_coord_2_count`, `*_coord_3_count`, `*_coord_4_count` | Counts of atoms with 2, 3, or 4 neighbors | Population of key local hybridization classes |
| `*_coord_2_fraction`, `*_coord_3_fraction`, `*_coord_4_fraction` | The same counts divided by atom count | Size-normalized coordination makeup |

`coord_3_fraction` often acts as a compact proxy for graphitic character, while `coord_2_fraction` and `coord_4_fraction` quantify underconnected and more tetrahedral regions.

## Ring-Proxy Features

The code uses a bounded shortest-path ring proxy on the bond graph, not a full topological ring enumeration.

| Feature family | How it is calculated | What it captures |
| --- | --- | --- |
| `*_ring_3_bond_count` ... `*_ring_8_bond_count` | For each bonded pair, the code finds the shortest alternate path after removing that edge; if the resulting cycle length is 3 to 8, it increments that ring-size bin | Approximate abundance of small and medium closed network motifs |
| `*_ring_3_bond_fraction` ... `*_ring_8_bond_fraction` | Ring-proxy bond counts normalized by the total bond counts assigned to the ring proxy | Relative ring-size distribution |

Smaller rings usually indicate more strained local topology. Larger ring bins suggest a more open or irregular network.

## Histogram-Derived Descriptor Features

For each histogram CSV in `analysis/anneal` and `analysis/nemd`, `build_ml_features.py` reduces the full distribution into compact statistics.

### Generic histogram summaries

For `bond_angle_distribution`, `bond_length_distribution`, `coordination_histogram`, `ring_proxy_bond_histogram`, `threefold_normal_alignment`, and `threefold_planarity_distribution`, the feature builder computes:

- `*_mass`: total counts or total density mass in the histogram
- `*_mean`, `*_std`: weighted mean and spread
- `*_min_center`, `*_max_center`: first and last occupied bin centers
- `*_peak_center`, `*_peak_value`: mode location and its magnitude
- `*_entropy`: spread/disorder of the distribution
- `*_q10`, `*_q25`, `*_q50`, `*_q75`, `*_q90`: weighted quantiles

These summarize both the central tendency and breadth of the structural distributions without keeping every histogram bin as a model input.

### RDF-specific summaries

`rdf.csv` is summarized separately into:

- `*_rdf_peak_r`: radial location of the strongest `g(r)` peak
- `*_rdf_peak_g`: height of the strongest `g(r)` peak
- `*_rdf_mean_g`, `*_rdf_std_g`: average and spread of `g(r)`
- `*_rdf_integral_like`: cumulative positive `g(r)` signal over the sampled range
- `*_rdf_first_r_gte_1`, `*_rdf_first_r_gte_2`: first radius where `g(r)` crosses 1 or 2
- `*_rdf_last_g`: tail value at the largest sampled radius

These capture short-range order, medium-range structure, and how quickly the pair correlations relax toward a bulk-like background.

## Annealing Time-Series Features

The annealing trajectory is first reduced to a per-frame table by `analyze_glassy_carbon_trajectory.py`, then `build_ml_features.py` computes summary statistics over time.

For each tracked quantity such as `density_g_cm3`, `mean_coordination`, `sp2_like_fraction`, `bond_length_mean_angstrom`, `bond_angle_mean_deg`, `threefold_planarity_rms_mean_angstrom`, and related coordination counts, the following are generated:

- `anneal_ts_<name>_first`: value in the first analyzed frame
- `anneal_ts_<name>_last`: value in the last analyzed frame
- `anneal_ts_<name>_delta`: final minus initial change
- `anneal_ts_<name>_mean`, `std`, `min`, `max`, `range`
- `anneal_ts_<name>_q25`, `q50`, `q75`
- `anneal_ts_<name>_slope_per_frame`: linear trend versus frame index

Additional trajectory-level columns:

- `anneal_ts_frame_count`: number of analyzed frames
- `anneal_ts_timestep_start`, `anneal_ts_timestep_end`, `anneal_ts_timestep_span`: sampled time window
- `anneal_ts_coordlog_n2`, `anneal_ts_coordlog_n3`, `anneal_ts_coordlog_n4` statistics when a coordination log is available

These features are useful because they encode not just the final structure, but also how the structure evolved during annealing: stabilization, drift, fluctuation amplitude, and reorganization rate.

## Delta Features

`delta_nemd_minus_anneal_...` columns are simple differences between the NEMD snapshot and the annealed snapshot for the core summary keys.

Examples:

- `delta_nemd_minus_anneal_density_g_cm3`
- `delta_nemd_minus_anneal_mean_coordination`
- `delta_nemd_minus_anneal_sp2_like_fraction`

These columns capture how much the driven transport simulation perturbs the structure relative to the pre-drive state.

## Target Columns

These are parsed from the last line of `gc_edip_hotcold.cont.dat`.

| Target column | Meaning |
| --- | --- |
| `target_final_thermal_conductivity` | Final conductivity target used for supervised learning |
| `target_final_delta_t` | Final hot-minus-cold slab temperature difference |
| `target_final_heat_flux_jz` | Final imposed heat flux |
| `target_final_hot_temp` | Final hot slab temperature |
| `target_final_cold_temp` | Final cold slab temperature |
| `target_timestep` | Timestep of the final reported transport sample |

## Practical Interpretation

At a high level, the feature table mixes four kinds of signal:

- network topology: coordination classes, ring-proxy bins, bond counts
- local geometry: bond lengths, bond angles, planarity, pyramidalization
- structural order/disorder: RDF summaries, histogram entropy, alignment metrics
- process history: annealing trajectory trends and fluctuations

That combination is useful because thermal conductivity in disordered carbon depends on both the final static network and the path the network took to get there.
