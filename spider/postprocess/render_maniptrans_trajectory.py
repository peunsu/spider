# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Render a video from an RL rollout (HDF5) or run the RL policy to generate one.

Usage:
    # Step 1: Run RL evaluation to produce rollout HDF5
    python examples/render_rl_trajectory.py rollout \
        --checkpoint runs/baseline_0f900@0_inspire_lh__02-14-20-02-07/nn/baseline.pth \
        --data-index 0f900@0 --side lh --dexhand inspire

    # Step 2: Convert HDF5 rollout to video
    python examples/render_rl_trajectory.py render \
        --hdf5 dumps/dump_baseline_0f900@0_inspire_lh_*/rollouts.hdf5 \
        --data-index 0f900@0 --side lh --dexhand inspire

    # Or do both in one go:
    python examples/render_rl_trajectory.py both \
        --checkpoint runs/baseline_0f900@0_inspire_lh__02-14-20-02-07/nn/baseline.pth \
        --data-index 0f900@0 --side lh --dexhand inspire

Author: Chaoyi Pan
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np


def run_rl_rollout(args):
    """Run RL policy in test mode to produce rollout HDF5."""
    side_tag = {"lh": "LH", "rh": "RH", "bih": "BiH"}[args.side]
    imitator_side = "lh" if args.side == "lh" else "rh"

    ckpt = os.path.abspath(args.checkpoint)

    cmd = (
        f"PYTHONPATH=/home/ubuntu/HO-Tracker-Baseline:$PYTHONPATH "
        f"python main/rl/train.py "
        f"task=ResDexHand dexhand={args.dexhand} side={side_tag} "
        f"headless=true num_envs={args.num_envs} test=true "
        f"rolloutStateInit=false randomStateInit=false "
        f"dataIndices=[{args.data_index}] "
        f"checkpoint={ckpt} "
        f"actionsMovingAverage={args.smooth} "
        f"experiment=baseline "
        f"save_rollouts=true "
        f"num_rollouts_to_save={args.num_envs} "
        f"num_rollouts_to_run={args.num_envs * 4} "
        f"save_successful_rollouts_only=false "
        f"rh_base_model_checkpoint=assets/imitator_rh_{args.dexhand}.pth "
        f"lh_base_model_checkpoint=assets/imitator_lh_{args.dexhand}.pth"
    )

    print(f"Running RL evaluation:\n  {cmd}\n")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd="/home/ubuntu/HO-Tracker-Baseline",
        env={
            **os.environ,
            "PYTHONPATH": "/home/ubuntu/HO-Tracker-Baseline:" + os.environ.get("PYTHONPATH", ""),
        },
    )
    if result.returncode != 0:
        print(f"ERROR: RL evaluation failed (exit code {result.returncode})")
        sys.exit(1)

    # Find the generated HDF5 (train.py prepends "dump_" to experiment name)
    pattern = f"/home/ubuntu/HO-Tracker-Baseline/dumps/dump_*baseline*{args.data_index}_{args.dexhand}_{args.side}*/rollouts.hdf5"
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        print(f"ERROR: no rollout HDF5 found matching {pattern}")
        sys.exit(1)
    hdf5_path = matches[-1]  # most recent
    print(f"Rollout saved to: {hdf5_path}")
    return hdf5_path


def hdf5_to_npz(hdf5_path, side):
    """Convert RL rollout HDF5 to NPZ format compatible with SPIDER video recording.

    SPIDER's record_video expects:
        qpos: (T, 19) = obj_pos(3) + obj_quat(4) + dof_pos(12)
        base_state: (T, 13)
    """
    import h5py

    side_key = "lh" if side == "lh" else "rh"

    with h5py.File(hdf5_path, "r") as f:
        # Try successful rollouts first, fall back to failed
        grp = None
        for category in ["successful", "failed", "all"]:
            cat_path = f"rollouts/{category}"
            if cat_path in f:
                keys = list(f[cat_path].keys())
                if keys:
                    # Pick the longest rollout for best visualization
                    best_key = max(
                        keys,
                        key=lambda k: len(f[f"{cat_path}/{k}/reward"]),
                    )
                    n_steps = len(f[f"{cat_path}/{best_key}/reward"])
                    print(f"Found {len(keys)} {category} rollouts, using {best_key} ({n_steps} steps)")
                    grp = f[f"{cat_path}/{best_key}"]
                    break
        if grp is None:
            print("ERROR: no rollouts found in HDF5")
            sys.exit(1)

        # Object state: (T, 13) — pos(3) + quat(4) + vel(3) + ang_vel(3)
        obj_state = np.array(grp[f"state_manip_obj_{side_key}"])
        obj_pos = obj_state[:, :3]
        obj_quat = obj_state[:, 3:7]

        # Base state (wrist): (T, 13)
        base_state = np.array(grp[f"state_{side_key}"])

        # DOF positions: (T, n_dofs)
        q = np.array(grp[f"q_{side_key}"])

        # Joint state for metrics: (T, n_bodies * 13)
        joint_state_flat = np.array(grp[f"joint_state_{side_key}"])
        T = len(obj_pos)
        n_bodies = joint_state_flat.shape[1] // 13
        joint_state = joint_state_flat.reshape(T, n_bodies, 13)

    # Assemble qpos: obj_pos(3) + obj_quat(4) + dof_pos(n_dofs)
    qpos = np.concatenate([obj_pos, obj_quat, q], axis=1)

    # Save NPZ
    npz_path = hdf5_path.replace("rollouts.hdf5", "rollout_rl.npz")
    np.savez(
        npz_path,
        qpos=qpos,
        base_state=base_state,
        obj_pos=obj_pos,
        obj_quat=obj_quat,
        joint_state=joint_state[:, :, :3],  # positions only
    )
    print(f"Converted to NPZ: {npz_path} ({T} frames)")
    return npz_path


def render_video(npz_path, args):
    """Render video from NPZ trajectory using SPIDER's record_video subprocess."""
    output_path = npz_path.replace(".npz", ".mp4")
    if args.output:
        output_path = args.output

    # Build a minimal subprocess script that uses the SPIDER video infrastructure
    script = f'''
import sys, os
sys.path.insert(0, "/home/ubuntu/HO-Tracker-Baseline")
os.chdir("{os.getcwd()}")
from isaacgym import gymapi
import copy
from spider.config import Config
from spider.simulators.maniptrans import setup_env, _render_frames

config = Config(
    dataset_dir="example_datasets",
    dataset_name="maniptrans",
    robot_type="{args.dexhand}",
    embodiment_type="{'left' if args.side == 'lh' else 'right'}",
    task="{args.data_index}",
    data_id=0,
    device="cuda:0",
    output_dir=".",
    simulator="maniptrans",
    sim_dt=1/60.0, ctrl_dt=0.1, knot_dt=0.1, ref_dt=1/60.0, trace_dt=1/60.0,
    horizon=0.2, show_viewer=False, viewer="none",
    save_video=True, save_info=True,
    num_samples=2, num_dr=1, max_sim_steps=-1,
)
config.maniptrans_data_indices = ["{args.data_index}"]
config.maniptrans_episode_length = 1200
config.maniptrans_tighten_method = "None"
config.maniptrans_tighten_factor = 1.0
config.maniptrans_actions_moving_average = 0.4
config.maniptrans_use_pid = False
config.maniptrans_use_quat_rot = False
config.maniptrans_ctrl_freq_inv = 1
config.maniptrans_record_video = True
config.headless = True

import numpy as np
traj = np.load("{npz_path}")
qpos = traj["qpos"]
base_state = traj["base_state"]
total_steps = len(qpos)
print(f"Recording video: {{total_steps}} frames")

env = setup_env(config, None)
_render_frames(env, config.device, qpos, base_state, total_steps,
               "{output_path}", 60)
'''

    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/ubuntu/HO-Tracker-Baseline:" + env.get("PYTHONPATH", "")

    print(f"Rendering video to {output_path}...")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace"), end="")
    if result.returncode != 0:
        print(f"ERROR: rendering failed (exit code {result.returncode})")
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace")[-2000:])
        sys.exit(1)
    print(f"Video saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render RL trajectory video")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-index", default="0f900@0")
    common.add_argument("--side", default="lh", choices=["lh", "rh"])
    common.add_argument("--dexhand", default="inspire")
    common.add_argument("--output", default=None, help="Output video path")

    # Subcommand: rollout
    p_rollout = subparsers.add_parser("rollout", parents=[common],
                                       help="Run RL eval to generate HDF5")
    p_rollout.add_argument("--checkpoint", required=True)
    p_rollout.add_argument("--num-envs", type=int, default=256)
    p_rollout.add_argument("--smooth", type=float, default=0.4)

    # Subcommand: render
    p_render = subparsers.add_parser("render", parents=[common],
                                      help="Convert HDF5 to video")
    p_render.add_argument("--hdf5", required=True,
                          help="Path to rollouts.hdf5 (supports glob)")

    # Subcommand: both
    p_both = subparsers.add_parser("both", parents=[common],
                                    help="Run RL eval then render video")
    p_both.add_argument("--checkpoint", required=True)
    p_both.add_argument("--num-envs", type=int, default=256)
    p_both.add_argument("--smooth", type=float, default=0.4)

    args = parser.parse_args()

    if args.command == "rollout":
        run_rl_rollout(args)

    elif args.command == "render":
        hdf5_path = args.hdf5
        if "*" in hdf5_path:
            matches = sorted(glob.glob(hdf5_path), key=os.path.getmtime)
            if not matches:
                print(f"ERROR: no files matching {hdf5_path}")
                sys.exit(1)
            hdf5_path = matches[-1]
        npz_path = hdf5_to_npz(hdf5_path, args.side)
        render_video(npz_path, args)

    elif args.command == "both":
        hdf5_path = run_rl_rollout(args)
        npz_path = hdf5_to_npz(hdf5_path, args.side)
        render_video(npz_path, args)


if __name__ == "__main__":
    main()
