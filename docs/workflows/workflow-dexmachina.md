# Workflow: DexMachina (Genesis)

The DexMachina workflow integrates SPIDER with [Genesis](https://genesis.github.io/) simulator and [DexMachina](https://github.com/MandiZhao/dexmachina) framework for dexterous manipulation with downstream RL training.

## Quick Start (Reproducible Setup)

### 1. Download Data

All required data is hosted on HuggingFace. Download it into the spider directory:

```bash
sudo apt install git-lfs
git lfs install
cd /path/to/spider
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

### 2. Install DexMachina Environment

```bash
# Create conda environment
conda create -n dexmachina python=3.10
conda activate dexmachina

# Install PyTorch
pip install torch==2.5.1

# Clone and install the Genesis fork (use the 'older' branch for API compatibility)
git clone https://github.com/MandiZhao/Genesis.git
cd Genesis
git checkout older
pip install -e .
pip install libigl==2.5.1  # Required: fixes igl.signed_distance API mismatch
cd ..

# Clone and install rl_games fork
git clone https://github.com/MandiZhao/rl_games.git
cd rl_games
pip install -e .
cd ..

# Clone and install dexmachina
git clone https://github.com/MandiZhao/dexmachina.git
cd dexmachina
pip install -e .
cd ..

# Install additional packages for video export and RL training
pip install moviepy==1.0.3 gymnasium ray seaborn
```

### 3. Install SPIDER (Minimal)

Install SPIDER without MuJoCo Warp (only need optimization components):

```bash
cd /path/to/spider
conda activate dexmachina

# Install SPIDER without dependencies (DexMachina has its own)
pip install --ignore-requires-python --no-deps -e .
# Install other dependencies
pip install loguru tyro hydra-core rerun-sdk==0.26.2
```

### 4. Copy Pre-Generated Data to DexMachina Assets

Pre-generated inspire_hand data (7 clips) is stored in the HuggingFace dataset so you don't need to re-run the retargeting pipeline.

```
example_datasets/raw/dexmachina/
  contact_retarget/inspire_hand/s01/   # contact mapping (.npy) + videos (.mp4)
  retargeted/inspire_hand/s01/         # retargeted kinematics (.pt)
  retargeter_results/inspire_hand/s01/ # retargeter results (.npy)
```

Clips: `box_use_01`, `ketchup_use_01`, `ketchup_use_02`, `laptop_use_01`, `mixer_use_01`, `notebook_use_01`, `waffleiron_use_01`

After installing DexMachina, copy the data into its assets directory:

```bash
DEXMACHINA_ASSETS=$(python -c "from dexmachina.asset_utils import get_asset_path; print(get_asset_path(''))")
SPIDER_DATA=/path/to/spider/example_datasets/raw/dexmachina

cp -r $SPIDER_DATA/retargeted/inspire_hand/        $DEXMACHINA_ASSETS/retargeted/inspire_hand/
cp -r $SPIDER_DATA/contact_retarget/inspire_hand/   $DEXMACHINA_ASSETS/contact_retarget/inspire_hand/
cp -r $SPIDER_DATA/retargeter_results/inspire_hand/  $DEXMACHINA_ASSETS/retargeter_results/inspire_hand/
```

If the data is missing at runtime, the error message will print the exact copy commands for your paths.

### 5. Run SPIDER

```bash
conda activate dexmachina
cd /path/to/spider

# Run with default config (inspire hand, box-30-230 task)
python examples/run_dexmachina.py
```

This will:
1. Initialize Genesis environment with DexMachina task
2. Load reference motion from task
3. Optimize trajectory with Sampling
4. Save optimized trajectory and video

Output is saved to `example_datasets/processed/arctic/inspire_hand/bimanual/{task}/{data_id}/`:
- `trajectory_dexmachina.npz` --- trajectory data
- `trajectory_dexmachina.mp4` --- video

## Evaluate Trajectories

After running `run_dexmachina.py`, evaluate object tracking metrics (position, rotation, articulation distances) with:

```bash
conda activate dexmachina
cd /path/to/spider

# Evaluate all tasks under the default data directory
python spider/postprocess/evaluate_dexmachina.py

# Evaluate specific tasks
python spider/postprocess/evaluate_dexmachina.py --tasks box-30-230 laptop-30-230

# Use a different dataset/robot configuration
python spider/postprocess/evaluate_dexmachina.py --dataset_name arctic --robot_type inspire_hand --embodiment_type bimanual
```

The script uses the same dataset path resolution as `config.py` (`dataset_dir/processed/{dataset_name}/{robot_type}/{embodiment_type}/{task}/{data_id}/`). You can also pass `--data_dir` to override with an explicit path.

## Compare SPIDER vs RL (Multi-Seed)

To compare SPIDER (MPC) against RL rollouts across multiple seeds, use `--compare` mode. This loads both `trajectory_dexmachina.npz` (SPIDER) and `rollout_rl.npz` (RL) from each `{task}/{seed}/` directory and prints mean +/- std tables.

```bash
# Compare across seeds 0-4 (default), print Markdown tables
python spider/postprocess/evaluate_dexmachina.py --compare

# Compare with LaTeX tables and per-seed detail
python spider/postprocess/evaluate_dexmachina.py --compare --latex --detail

# Compare specific tasks / seeds
python spider/postprocess/evaluate_dexmachina.py --compare --tasks box-30-230 laptop-30-230 --seeds 0 1 2
```

## Configuration

Key parameters in `examples/config/dexmachina.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | `box-30-230` | Object task (format: `object_name-start_frame-end_frame`) |
| `robot_type` | `inspire_hand` | Dexterous hand model |
| `embodiment_type` | `bimanual` | Hand configuration |
| `num_samples` | `1024` | Parallel CEM samples (Genesis envs) |
| `max_num_iterations` | `32` | CEM iterations per MPC step |
| `horizon` | `0.40` | Planning horizon (seconds) |
| `ctrl_dt` | `0.20` | Control timestep |

## Common Issues

1. **Missing DexMachina data**: Copy pre-generated data from `example_datasets/raw/dexmachina/` into the DexMachina assets directory (see "Copy Pre-Generated Data" section above).

2. **Genesis branch compatibility**: You must use the `older` branch of the Genesis fork (`git checkout older`). The `main` and `0122` branches have upstream merges that break API compatibility with DexMachina (e.g., `set_dofs_kp` input shape broadcasting, `inertial_quat` None handling).

3. **`igl.signed_distance` unpacking error** (`ValueError: too many values to unpack`): Install `libigl==2.5.1` explicitly. Newer versions of libigl change the return signature.

4. **`moviepy.editor` not found**: Install `moviepy==1.0.3` (not the newer 2.x which removed `moviepy.editor`). This is needed for video export.

5. **torch weights loading error**: If failed to load trajectory from dexmachina, ensure the data is loaded with `torch.load(data_fname, weights_only=False)`.

6. **numpy version**: Use `numpy==1.26.4` to avoid compatibility issues with Genesis and taichi.

7. **First run is slow**: The first run compiles taichi kernels (JIT) which can appear to hang for several minutes with no output. Subsequent runs use cached kernels and are faster.
