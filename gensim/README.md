# IRP GenSim Scripts

This folder contains standalone utilities for:
- converting IRP cloth / rope datasets into robomimic-style HDF5,
- extracting single-parameter subsets,
- analyzing action collisions / distances,
- running DataRater-style meta learning and plotting score landscapes.

All scripts are designed to avoid importing IRP internals.

## Environment

Recommended Python env (as used in examples below):

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python
```

Common dependencies:
- `numpy`
- `h5py`
- `zarr` (for zarr conversion scripts)
- `torch`, `matplotlib` (for `run_datarater_meta_irp.py`)

For matplotlib cache permission issues on clusters, prepend:

```bash
MPLCONFIGDIR=/tmp/mpl
```

---

## 1) Dataset Conversion

### `convert_irp_cloth_to_robomimic.py`
Converts cloth zarr (`.zarr` or `.zarr.tar`) into robomimic-style HDF5 with action trajectories.

Typical full conversion:

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python convert_irp_cloth_to_robomimic.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_cloth.zarr \
  --output /nethome/lma326/ipl_ws/irp/data/irp_cloth_all_params.hdf5 \
  --overwrite
```

Single parameter pair:

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python convert_irp_cloth_to_robomimic.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_cloth.zarr \
  --output /nethome/lma326/ipl_ws/irp/data/irp_cloth_param_2_3.hdf5 \
  --param-indices 2 3 \
  --overwrite
```

Useful flags:
- `--only-valid` / `--include-invalid`
- `--shuffle --seed`
- `--compression {gzip,lzf,none}`
- `--extract-dir` for tar extraction location


### `convert_irp_cloth_goals_to_robomimic.py`
Goal-conditioned cloth converter.

For each sampled goal and each cloth `(size_id, density_id)` pair, selects the best action over the full cloth action grid.

Example:

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python convert_irp_cloth_goals_to_robomimic.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_cloth.zarr \
  --output /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals.hdf5 \
  --num-goals 4 \
  --num-workers 16 \
  --only-valid \
  --overwrite
```

Useful flags:
- `--num-goals`
- `--num-workers` (parallel action search)
- `--only-valid` / `--include-invalid`
- `--max-demos`


### `convert_irp_rope_goals_to_robomimic.py`
Goal-conditioned rope converter.

Uses precomputed rope control arrays (`control/best_action_coord`, `control/max_hitrate`) to select actions.

Example:

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python convert_irp_rope_goals_to_robomimic.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_rope.zarr.tar \
  --output /nethome/lma326/ipl_ws/irp/data/irp_rope_4_goals.hdf5 \
  --num-goals 4 \
  --goal-sampling linspace \
  --hitrate-threshold 0.95 \
  --num-workers 16 \
  --skip-empty-pairs \
  --overwrite
```

Notes:
- Rope stores `length` and `density` under `obs/param_size` and `obs/param_density`.
- `--skip-empty-pairs` skips `(size_id, density_id)` pairs with no valid goals at the threshold.

---

## 2) Extract One Parameter Pair

### `extract_irp_cloth_goal_pair_hdf5.py`
Extracts one cloth `(size_id, density_id)` pair into a smaller HDF5 (typically one demo per goal).

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python extract_irp_cloth_goal_pair_hdf5.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals.hdf5 \
  --output /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals_size1_density2.hdf5 \
  --size-idx 1 \
  --density-idx 2 \
  --num-goals 4 \
  --overwrite
```


### `extract_irp_rope_goal_pair_hdf5.py`
Same extraction flow for rope goal-conditioned HDF5.

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python extract_irp_rope_goal_pair_hdf5.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_rope_4_goals.hdf5 \
  --output /nethome/lma326/ipl_ws/irp/data/irp_rope_4_goals_size1_density2.hdf5 \
  --size-idx 1 \
  --density-idx 2 \
  --num-goals 4 \
  --overwrite
```

---

## 3) Analysis Utilities

### `check_irp_cloth_goal_action_collisions.py`
Checks whether different cloth parameter pairs share identical optimal actions:
- per goal,
- and across all goals (full signature match).

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python check_irp_cloth_goal_action_collisions.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals.hdf5
```


### `rank_action_distance_by_param_pair.py`
Given a reference parameter pair, ranks all other pairs by action distance across goals.

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python rank_action_distance_by_param_pair.py \
  --input /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals.hdf5 \
  --ref-size-idx 2 \
  --ref-density-idx 3 \
  --metric l2 \
  --aggregate mean \
  --top-k 10 \
  --csv-out /nethome/lma326/ipl_ws/irp/data/cloth_action_distance_rank_ref_2_3.csv
```

---

## 4) DataRater Meta Learning

### `run_datarater_meta_irp.py`
Runs a meta-gradient DataRater pipeline:
- inner model predicts `action` from `goal_points`,
- DataRater scores source training samples,
- outer objective minimizes target validation loss,
- outputs parameter-space landscape plot + CSV.

Core run:

```bash
MPLCONFIGDIR=/tmp/mpl /nethome/lma326/miniconda3/envs/ipl/bin/python run_datarater_meta_irp.py \
  --source-hdf5 /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals.hdf5 \
  --target-hdf5 /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals_size1_density2.hdf5 \
  --output-dir /nethome/lma326/ipl_ws/irp/data/datarater_meta_cloth \
  --outer-steps 500 \
  --inner-steps 1 \
  --n-models 10 \
  --batch-size 0 \
  --normalize \
  --device cpu
```

Sim-and-real style co-training mix (example 90/10 source/target in inner train):

```bash
MPLCONFIGDIR=/tmp/mpl /nethome/lma326/miniconda3/envs/ipl/bin/python run_datarater_meta_irp.py \
  --source-hdf5 /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals.hdf5 \
  --target-hdf5 /nethome/lma326/ipl_ws/irp/data/irp_cloth_4_goals_size1_density2.hdf5 \
  --output-dir /nethome/lma326/ipl_ws/irp/data/datarater_meta_cloth_mix \
  --target-train-ratio 0.1 \
  --outer-steps 500 \
  --normalize
```

Anti-collapse knobs (recommended when weights become too peaky):

```bash
--score-temperature 2.0 \
--weight-uniform-mix 0.1 \
--meta-entropy-reg 0.01 \
--score-clip 5.0
```

Important output files:
- `datarater_landscape_size_density.png`
- `datarater_pair_scores.csv`
- `run_summary.json`

Landscape behavior:
- computed from **source samples only** (each source sample scored once),
- `n=` in each cell is the number of source samples in that `(size_id, density_id)` bin.

---

## Quick Help

For full argument details on any script:

```bash
/nethome/lma326/miniconda3/envs/ipl/bin/python <script_name>.py --help
```
