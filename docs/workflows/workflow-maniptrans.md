# Workflow: ManipTrans (IsaacGym)

SPIDER MPC for dexterous hand-object manipulation using [IsaacGym](https://developer.nvidia.com/isaac-gym) and [HO-Tracker-Baseline](https://github.com/KailinLi/HO-Tracker-Baseline) (ManipTrans).

## Quick Start (Reproducible Setup)

### 1. Download Data

All required data is hosted on HuggingFace. Download it into the spider directory:

```bash
sudo apt install git-lfs
git lfs install
cd /path/to/spider
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

This downloads `example_datasets/` containing:
- `raw/maniptrans/HO-Tracker/` — 16 trajectory demos (8 left-hand, 8 right-hand)
- `raw/maniptrans/assets/` — pre-trained base imitator checkpoints
- `raw/maniptrans/rl_runs/` — 16 trained RL baseline checkpoints
- `raw/maniptrans/retargeting/` — optional retargeting initialization
- `processed/maniptrans/` — pre-computed rollouts, videos, and metrics (160 runs)

### 2. Create Data Symlink

ManipTrans uses relative paths (`data/HO-Tracker/...`). Create a symlink:

```bash
cd /path/to/spider
ln -sf example_datasets/raw/maniptrans data
```

### 3. Install Dependencies

Requires Python 3.8 + PyTorch 1.13 + IsaacGym (older than SPIDER's default Python 3.12+):

```bash
# Create conda environment
conda create -n maniptrans python=3.8
conda activate maniptrans

# Install IsaacGym (requires NVIDIA GPU)
cd /path/to/isaacgym/python && pip install -e .

# Install HO-Tracker-Baseline
cd /path/to/HO-Tracker-Baseline && pip install -e .

# Install SPIDER (minimal, no-deps to avoid conflicts)
cd /path/to/spider
pip install --ignore-requires-python --no-deps -e .
pip install loguru mujoco
```

### 4. Set Up HO-Tracker-Baseline Data

HO-Tracker-Baseline also needs access to the trajectory data and imitator checkpoints. Point its data paths to spider's `example_datasets/`:

```bash
# Link HO-Tracker data
cd /path/to/HO-Tracker-Baseline
rm -f data/HO-Tracker
ln -sf /path/to/spider/example_datasets/raw/maniptrans/HO-Tracker data/HO-Tracker

# Link retargeting data
rm -rf data/retargeting/HO-Tracker
ln -sf /path/to/spider/example_datasets/raw/maniptrans/retargeting/HO-Tracker data/retargeting/HO-Tracker

# Copy imitator checkpoints
cp /path/to/spider/example_datasets/raw/maniptrans/assets/imitator_*.pth assets/
```

### 5. Run SPIDER

```bash
conda activate maniptrans
cd /path/to/spider
export LD_LIBRARY_PATH="/path/to/miniconda3/envs/maniptrans/lib:${LD_LIBRARY_PATH:-}"

# Default: 0f900@0, inspire hand, left hand
python examples/run_maniptrans.py

# Different trajectory:
python examples/run_maniptrans.py task=81a95@5

# Right hand:
python examples/run_maniptrans.py task=07bb1@1 embodiment_type=right

# Fewer iterations (faster):
python examples/run_maniptrans.py max_num_iterations=16
```

Output is saved to `example_datasets/processed/maniptrans/inspire/{side}/{task}/{data_id}/`:
- `rollout_isaac.npz` — trajectory data
- `rollout_isaac.mp4` — video
- `metrics_isaac.json` — evaluation metrics

## Available Trajectories

16 trajectories from HO-Tracker dataset (8 per hand):

| Side | Task IDs |
|------|----------|
| Left | `0f900@0`, `81a95@5`, `0f4a7@1`, `70ca2@0`, `39e5e@5`, `380c7@0`, `2ddc8@2`, `0b3a0@6` |
| Right | `07bb1@1`, `235ac@0`, `07301@0`, `49793@0`, `01857@10`, `031b7@0`, `03ac9@0`, `155cb@0` |

## Configuration

Key parameters in `examples/config/maniptrans.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | `0f900@0` | HO-Tracker data sequence |
| `embodiment_type` | `left` | `left` or `right` |
| `num_samples` | `1024` | Parallel CEM samples (IsaacGym envs) |
| `max_num_iterations` | `32` | CEM iterations per MPC step |
| `horizon` | `0.6` | Planning horizon (seconds) |
| `ctrl_dt` | `0.2` | Control timestep |
| `seed` | `0` | Random seed |
| `data_id` | `0` | Output subdirectory index |

## RL Baseline Evaluation

Trained RL checkpoints are included for all 16 trajectories. To evaluate:

```bash
cd /path/to/HO-Tracker-Baseline

# Example: evaluate left-hand trajectory
python main/rl/train.py \
    task=ResDexHand dexhand=inspire side=lh \
    headless=true num_envs=256 test=true \
    +forceFullLength=true seed=0 \
    rolloutStateInit=false randomStateInit=false \
    "dataIndices=[0f900@0]" \
    checkpoint=/path/to/spider/example_datasets/raw/maniptrans/rl_runs/baseline_0f900@0_inspire_lh/nn/baseline_0f900@0.pth \
    actionsMovingAverage=0.4 \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth
```

Side tags: `lh` for left-hand trajectories, `rh` for right-hand trajectories.

## Evaluation Metrics

Four metrics computed against ground-truth demo data:

| Metric | Description |
|--------|-------------|
| `e_t` | Object translation error (cm) |
| `e_r` | Object rotation error (degrees, geodesic) |
| `e_j` | Joint position error (cm) |
| `e_ft` | Fingertip position error (cm) |

Run evaluation across all trajectories:

```bash
# Single-seed evaluation (data_id=0 only)
python spider/postprocess/evaluate_maniptrans.py

# Multi-seed evaluation (all data_ids, mean +/- std)
python spider/postprocess/evaluate_maniptrans.py --multiseed
```

## Important Notes

### Python 3.8 Compatibility

ManipTrans requires Python 3.8 + PyTorch 1.13 + IsaacGym. SPIDER handles this via:
- `from __future__ import annotations` in `spider/config.py` (for `tuple[float, float]` syntax)
- `hasattr(torch, "compile")` guards in `spider/optimizers/sampling.py`

### Action Space

SPIDER optimizes 18-dim residual actions (6 wrist force/torque + 12 DOF positions). The base imitator provides nominal tracking actions. These combine in the env's `pre_physics_step` to produce 36-dim actions for the full base+residual architecture.

### ManipTrans Config Fields

ManipTrans-specific fields (e.g., `maniptrans_episode_length`) are **not** part of the SPIDER `Config` dataclass. In `examples/run_maniptrans.py`, they are extracted from the Hydra config and attached as attributes after `Config` construction.
