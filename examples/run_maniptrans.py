# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run SPIDER MPC with ManipTrans (IsaacGym) backend.

Usage:
    # Default (0f900@0, inspire, left hand):
    python examples/run_maniptrans.py

    # Different data sequence:
    python examples/run_maniptrans.py task=0f900@0

    # With video recording:
    python examples/run_maniptrans.py save_video=true

    # Fewer MPC iterations (faster):
    python examples/run_maniptrans.py max_num_iterations=16

Author: Chaoyi Pan
"""

from __future__ import annotations

# IsaacGym must be imported before PyTorch (loaded by spider.config)
from isaacgym import gymapi as _gymapi  # noqa: F401

import json
import os
import time

import hydra
import loguru
import numpy as np
import torch
from omegaconf import DictConfig

from spider.config import Config, process_config
from spider.interp import get_slice
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.simulators.maniptrans import (
    copy_sample_state,
    get_fingertip_pos_dist,
    get_joint_pos_dist,
    get_obj_pos_dist,
    get_obj_quat_dist,
    get_qpos,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    record_video,
    save_env_params,
    save_state,
    setup_env,
    step_env,
    sync_env,
)
# Keys in the Hydra config that are ManipTrans-specific and not part of
# the Config dataclass.  They are extracted before constructing Config and
# attached afterwards as extra attributes.
_MANIPTRANS_KEYS = [
    "maniptrans_data_indices",
    "maniptrans_episode_length",
    "maniptrans_tighten_method",
    "maniptrans_tighten_factor",
    "maniptrans_tighten_steps",
    "maniptrans_actions_moving_average",
    "maniptrans_use_pid",
    "maniptrans_use_quat_rot",
    "maniptrans_ctrl_freq_inv",
    "maniptrans_record_video",
]


def main(config: Config):
    """Run the SPIDER MPC using ManipTrans (IsaacGym) backend."""
    # Setup env (ref_data set to None since environment has built-in reference)
    env = setup_env(config, None)
    # ManipTrans uses base+residual action architecture.
    # The nominal base actions (DOF targets from demo) are fixed inside step_env.
    # SPIDER only optimizes the residual (18 dims).
    config.nu = env.num_actions

    # Save output_dir before process_config (which sets output_dir to
    # the standard processed directory for MJWP).
    output_dir = config.output_dir

    # process config: compute steps, noise schedule, output paths
    config = process_config(config)

    # Restore ManipTrans-specific output_dir if explicitly set
    if output_dir:
        config.output_dir = output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    loguru.logger.info(f"Output directory: {config.output_dir}")
    loguru.logger.info(
        f"nu={config.nu}, num_envs={env.num_envs}, "
        f"horizon_steps={config.horizon_steps}, ctrl_steps={config.ctrl_steps}"
    )

    # Set max_sim_steps from environment episode length
    max_episode_length = int(env.demo_data["seq_len"].min().item())
    if config.max_sim_steps == -1:
        config.max_sim_steps = max_episode_length
    else:
        config.max_sim_steps = min(max_episode_length, config.max_sim_steps)
    loguru.logger.info(f"max_sim_steps={config.max_sim_steps}")

    # Create placeholder reference data for compatibility
    ref_data = (
        torch.zeros(
            config.max_sim_steps + config.horizon_steps + config.ctrl_steps,
            config.nu,
            device=config.device,
        ),
    )

    # Setup env params (no curriculum for ManipTrans evaluation)
    env_params_list = []
    for i in range(config.max_num_iterations):
        env_params_list.append([{} for _ in range(config.num_dr)])
    config.env_params_list = env_params_list

    # setup optimizer
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)

    # initial controls
    ctrls = torch.zeros(
        config.horizon_steps,
        config.nu,
        device=config.device,
    )

    # buffers for saving info and trajectory
    info_list = []

    # MPC loop
    t_start = time.perf_counter()
    sim_step = 0
    while sim_step < config.max_sim_steps:
        t0 = time.perf_counter()

        # optimize using future reference window at control-rate (+1 lookahead)
        ref_slice = get_slice(
            ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
        )
        ctrls, infos = optimize(config, env, ctrls, ref_slice)
        infos["sim_step"] = sim_step

        # step environment for ctrl_steps
        step_info = {
            "obj_pos_dist": [],
            "obj_quat_dist": [],
            "joint_pos_dist": [],
            "fingertip_pos_dist": [],
            "qpos": [],
            "obj_pos": [],
            "obj_quat": [],
            "joint_state": [],
            "base_state": [],
        }
        for i in range(config.ctrl_steps):
            ctrl = ctrls[i]
            ctrl_repeat = ctrl.unsqueeze(0).repeat(
                int(config.num_samples), 1
            )
            step_env(config, env, ctrl_repeat)
            step_info["obj_pos_dist"].append(
                get_obj_pos_dist(env)[0].item()
            )
            step_info["obj_quat_dist"].append(
                get_obj_quat_dist(env)[0].item()
            )
            step_info["joint_pos_dist"].append(
                get_joint_pos_dist(env)[0].item()
            )
            step_info["fingertip_pos_dist"].append(
                get_fingertip_pos_dist(env)[0].item()
            )
            step_info["qpos"].append(
                get_qpos(env)[0].detach().cpu().numpy()
            )
            step_info["obj_pos"].append(
                env.states["manip_obj_pos"][0].detach().cpu().numpy()
            )
            step_info["obj_quat"].append(
                env.states["manip_obj_quat"][0].detach().cpu().numpy()
            )
            step_info["joint_state"].append(
                env.states["joints_state"][0, :, :3]
                .detach()
                .cpu()
                .numpy()
            )
            step_info["base_state"].append(
                env.states["base_state"][0].detach().cpu().numpy()
            )
            sim_step += 1

        for k in step_info:
            step_info[k] = np.stack(step_info[k], axis=0)
        infos.update(step_info)

        # sync env state
        env = sync_env(config, env)

        # receding horizon update
        prev_ctrl = ctrls[config.ctrl_steps :]
        new_ctrl = torch.zeros(
            (config.ctrl_steps, config.nu),
            device=config.device,
        )
        ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

        # progress
        t1 = time.perf_counter()
        rtr = config.ctrl_dt / (t1 - t0)
        print(
            f"  step {sim_step:>4d}/{config.max_sim_steps} | "
            f"e_t={step_info['obj_pos_dist'].mean():.4f}m | "
            f"e_r={step_info['obj_quat_dist'].mean():.1f}deg | "
            f"e_j={step_info['joint_pos_dist'].mean():.4f}m | "
            f"e_ft={step_info['fingertip_pos_dist'].mean():.4f}m | "
            f"opt={infos['opt_steps'][0]} | "
            f"RTR={rtr:.2f}x"
        )

        # record info/trajectory at control tick
        info_list.append(
            {k: v for k, v in infos.items() if k != "trace_sample"}
        )

    t_end = time.perf_counter()

    # Aggregate all per-step metrics
    obj_pos_dists = np.concatenate(
        [info["obj_pos_dist"] for info in info_list]
    )
    obj_quat_dists = np.concatenate(
        [info["obj_quat_dist"] for info in info_list]
    )
    joint_pos_dists = np.concatenate(
        [info["joint_pos_dist"] for info in info_list]
    )
    fingertip_pos_dists = np.concatenate(
        [info["fingertip_pos_dist"] for info in info_list]
    )

    # RL-compatible metrics (matching ManipTrans eval_score.py):
    #   e_t  = mean object translation error (m)
    #   e_r  = mean object rotation error (deg)
    #   e_j  = mean joint position error (m)
    #   e_ft = mean fingertip position error (m)
    e_t = float(obj_pos_dists.mean())
    e_r = float(obj_quat_dists.mean())
    e_j = float(joint_pos_dists.mean())
    e_ft = float(fingertip_pos_dists.mean())

    loguru.logger.info(
        f"Total time: {t_end - t_start:.1f}s | "
        f"e_t={e_t * 100:.2f}cm | e_r={e_r:.2f}deg | "
        f"e_j={e_j * 100:.2f}cm | e_ft={e_ft * 100:.2f}cm"
    )

    # save evaluation metrics
    if config.save_info:
        metrics = {
            # RL-compatible metrics (same keys as ManipTrans eval.json)
            "e_t": e_t,
            "e_r": e_r,
            "e_j": e_j,
            "e_ft": e_ft,
            # Extended statistics
            "e_t_std": float(obj_pos_dists.std()),
            "e_r_std": float(obj_quat_dists.std()),
            "e_j_std": float(joint_pos_dists.std()),
            "e_ft_std": float(fingertip_pos_dists.std()),
            # Run info
            "total_sim_steps": sim_step,
            "total_time": t_end - t_start,
            "data_index": config.maniptrans_data_indices[0],
            "robot_type": config.robot_type,
            "embodiment_type": config.embodiment_type,
        }
        metrics_path = f"{config.output_dir}/metrics_isaac.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        loguru.logger.info(f"Saved metrics to {metrics_path}")

    # save retargeted trajectory
    traj_path = f"{config.output_dir}/rollout_isaac.npz"
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            try:
                info_aggregated[k] = np.stack(
                    [info[k] for info in info_list], axis=0
                )
            except (ValueError, TypeError):
                pass
        np.savez(traj_path, **info_aggregated)
        loguru.logger.info(f"Saved trajectory to {traj_path}")

    # Record video by replaying with a separate small environment.
    # Destroy MPC environment first — IsaacGym GPU pipeline can't have two sims.
    if config.save_video and config.save_info:
        video_path = f"{config.output_dir}/rollout_isaac.mp4"
        record_video(config, video_path, prev_env=env, traj_path=traj_path)
        loguru.logger.info(f"Saved video to {video_path}")

    return


@hydra.main(version_base=None, config_path="config", config_name="maniptrans")
def run_main(cfg: DictConfig) -> None:
    config_dict = dict(cfg)

    # Extract ManipTrans-specific keys (not in Config dataclass)
    maniptrans_extras = {}
    for key in _MANIPTRANS_KEYS:
        if key in config_dict:
            maniptrans_extras[key] = config_dict.pop(key)

    # Handle special conversions
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    # Convert list-like OmegaConf objects to plain Python types
    if "maniptrans_data_indices" in maniptrans_extras:
        maniptrans_extras["maniptrans_data_indices"] = list(
            maniptrans_extras["maniptrans_data_indices"]
        )

    # Use data index as task name (e.g., "0f900@0") instead of generic "maniptrans".
    # This puts output under .../0f900@0/0/ instead of .../maniptrans/0/.
    data_indices = maniptrans_extras.get("maniptrans_data_indices", [])
    if data_indices and config_dict.get("task") == "maniptrans":
        config_dict["task"] = data_indices[0]

    # If maniptrans_data_indices not explicitly set, derive from task name
    if "maniptrans_data_indices" not in maniptrans_extras:
        maniptrans_extras["maniptrans_data_indices"] = [config_dict["task"]]

    config = Config(**config_dict)

    # Attach ManipTrans-specific fields
    for key, value in maniptrans_extras.items():
        setattr(config, key, value)
    config.headless = True

    main(config)


if __name__ == "__main__":
    run_main()
