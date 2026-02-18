# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator for sampling with ManipTrans based on IsaacGym.

This module provides a ManipTrans (HO-Tracker-Baseline) backend that matches
the sampling API used by the generic optimizer pipeline. It wraps the IsaacGym
dexterous hand manipulation environment and exposes reward/termination from the
RL training pipeline.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul

from spider.config import Config

# ---------------------------------------------------------------------------
# ManipTrans environment imports
# ---------------------------------------------------------------------------
from maniptrans_envs.lib.envs.tasks.dexhandmanip_sh import (
    DexHandManipLHEnv,
    DexHandManipRHEnv,
    compute_imitation_reward,
    quat_to_angle_axis,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def _ensure_hands_registered():
    """Trigger dexhand auto-registration if it hasn't happened yet.

    Due to circular imports at module init time, the auto-registration in
    ``dexhands/__init__.py`` may not have completed when this module is first
    loaded.  Calling this function lazily (at setup_env time) avoids the issue.
    """
    if not DexHandFactory._registry:
        import importlib
        import os

        dexhands_dir = os.path.dirname(
            importlib.import_module(
                "maniptrans_envs.lib.envs.dexhands"
            ).__file__
        )
        DexHandFactory.auto_register_hands(
            dexhands_dir, "maniptrans_envs.lib.envs.dexhands"
        )
from main.dataset.transform import aa_to_quat, rotmat_to_quat

try:
    from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import DexHandManipBiHEnv
except ImportError:
    DexHandManipBiHEnv = None

# Map SPIDER side names to ManipTrans env classes
ENV_MAP = {
    "right": DexHandManipRHEnv,
    "left": DexHandManipLHEnv,
}
if DexHandManipBiHEnv is not None:
    ENV_MAP["bimanual"] = DexHandManipBiHEnv


# ---------------------------------------------------------------------------
# Base imitator model (pre-trained first-stage policy)
# ---------------------------------------------------------------------------


class _BaseImitatorModel(nn.Module):
    """Self-contained base imitator model matching ManipTrans checkpoint.

    Architecture (from imitator_lh_inspire.pth / imitator_rh_inspire.pth):
      - RunningMeanStd normalization per obs key (privileged, proprioception, target)
      - Feature encoder: concat(priv, prop, tgt) -> MLP 237->512->512->512->256 (SiLU)
      - Actor MLP: 256->256->512->128->64 (ELU)
      - mu: Linear(64->18), sigma: Parameter(18)

    The observation dimensions are hand-specific:
      - privileged: n_dofs (e.g., 12 for inspire)
      - proprioception: 13 + n_dofs*3 (e.g., 49 for inspire)
      - target: 3+3+3+4+4+3+3+(n_bodies-1)*3*3 (e.g., 176 for inspire)
    """

    def __init__(self, checkpoint_path, device):
        super().__init__()
        ckp = torch.load(checkpoint_path, map_location=device)
        sd = ckp["model"]

        # Infer dimensions from checkpoint
        priv_dim = sd[
            "running_mean_std.running_mean_std.privileged.running_mean"
        ].shape[0]
        prop_dim = sd[
            "running_mean_std.running_mean_std.proprioception.running_mean"
        ].shape[0]
        tgt_dim = sd[
            "running_mean_std.running_mean_std.target.running_mean"
        ].shape[0]
        feat_in = priv_dim + prop_dim + tgt_dim
        action_dim = sd["a2c_network.mu.bias"].shape[0]

        # Store obs dims for slicing env observations
        self.priv_dim = priv_dim
        self.prop_dim = prop_dim
        self.tgt_dim = tgt_dim

        # Normalization buffers
        self.register_buffer(
            "priv_mean",
            sd["running_mean_std.running_mean_std.privileged.running_mean"].float(),
        )
        self.register_buffer(
            "priv_var",
            sd["running_mean_std.running_mean_std.privileged.running_var"].float(),
        )
        self.register_buffer(
            "prop_mean",
            sd[
                "running_mean_std.running_mean_std.proprioception.running_mean"
            ].float(),
        )
        self.register_buffer(
            "prop_var",
            sd[
                "running_mean_std.running_mean_std.proprioception.running_var"
            ].float(),
        )
        self.register_buffer(
            "tgt_mean",
            sd["running_mean_std.running_mean_std.target.running_mean"].float(),
        )
        self.register_buffer(
            "tgt_var",
            sd["running_mean_std.running_mean_std.target.running_var"].float(),
        )

        # Feature encoder: Linear-Identity-SiLU blocks + final Linear
        # hidden_depth=3, hidden_dim=512, output_dim=256
        self.feat_enc = nn.Sequential(
            nn.Linear(feat_in, 512),
            nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Linear(512, 512),
            nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Linear(512, 512),
            nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Linear(512, 256),
        )

        # Actor MLP: units [256, 512, 128, 64] with ELU
        self.actor_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
        )

        # Output
        self.mu = nn.Linear(64, action_dim)
        self.sigma = nn.Parameter(torch.zeros(action_dim))

        # Load weights
        for i in [0, 3, 6, 9]:
            self.feat_enc[i].weight.data.copy_(
                sd[f"a2c_network.dict_feature_encoder._head.{i}.weight"]
            )
            self.feat_enc[i].bias.data.copy_(
                sd[f"a2c_network.dict_feature_encoder._head.{i}.bias"]
            )
        for i in [0, 2, 4, 6]:
            self.actor_mlp[i].weight.data.copy_(
                sd[f"a2c_network.actor_mlp.{i}.weight"]
            )
            self.actor_mlp[i].bias.data.copy_(
                sd[f"a2c_network.actor_mlp.{i}.bias"]
            )
        self.mu.weight.data.copy_(sd["a2c_network.mu.weight"])
        self.mu.bias.data.copy_(sd["a2c_network.mu.bias"])
        self.sigma.data.copy_(sd["a2c_network.sigma"])

        self.to(device)
        self.eval()

    def _normalize(self, x, mean, var):
        eps = 1e-5
        return torch.clamp((x - mean) / torch.sqrt(var + eps), -20.0, 20.0)

    @torch.no_grad()
    def forward(self, privileged, proprioception, target):
        """Return deterministic base action (mu) of shape (N, action_dim)."""
        p = self._normalize(privileged, self.priv_mean, self.priv_var)
        r = self._normalize(proprioception, self.prop_mean, self.prop_var)
        t = self._normalize(target, self.tgt_mean, self.tgt_var)
        # Concatenate sorted by key: privileged, proprioception, target
        x = torch.cat([p, r, t], dim=-1)
        x = self.feat_enc(x)
        x = self.actor_mlp(x)
        return self.mu(x)


def _load_base_imitator(config, env):
    """Load the pre-trained base imitator and store on env."""
    side = config.embodiment_type
    robot = config.robot_type

    # Determine checkpoint path
    ckpt_path = getattr(config, "maniptrans_base_ckpt", None)
    if ckpt_path is None:
        side_tag = "lh" if side == "left" else "rh"
        # Default path relative to HO-Tracker-Baseline
        ho_tracker_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        # Try common locations
        candidates = [
            f"assets/imitator_{side_tag}_{robot}.pth",
            os.path.join(
                ho_tracker_root,
                "..",
                "HO-Tracker-Baseline",
                "assets",
                f"imitator_{side_tag}_{robot}.pth",
            ),
            f"/home/ubuntu/HO-Tracker-Baseline/assets/imitator_{side_tag}_{robot}.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Base imitator checkpoint not found. "
            f"Tried: {candidates}. "
            f"Set config.maniptrans_base_ckpt to the correct path."
        )

    print(f"Loading base imitator from {ckpt_path}")
    env._spider_base_model = _BaseImitatorModel(ckpt_path, env.device)
    return env


def _get_base_action(env):
    """Compute 18-dim base action from the pre-trained imitator.

    Calls env.compute_observations() to populate obs_dict, then extracts
    the base model's portion of observations and runs the forward pass.
    """
    model = env._spider_base_model

    # Populate env.obs_dict from current state + demo targets
    env.compute_observations()

    # Extract base model observations (first N dims from each key)
    privileged = env.obs_dict["privileged"][:, : model.priv_dim]
    proprioception = env.obs_dict["proprioception"][:, : model.prop_dim]
    target = env.obs_dict["target"][:, : model.tgt_dim]

    # Forward pass (deterministic: use mu directly)
    base_action = model(privileged, proprioception, target)  # (N, 18)
    return torch.clamp(base_action, -1.0, 1.0)


# --
# Helper: build ManipTrans task config dict
# --


def _build_task_config(config: Config) -> dict:
    """Build the IsaacGym task config dict from SPIDER Config."""
    _ensure_hands_registered()
    side = config.embodiment_type  # "right", "left", or "bimanual"
    dexhand_name = config.robot_type  # "inspire", "shadow", "allegro", ...

    # Determine task name
    side_suffix = {"right": "RH", "left": "LH", "bimanual": "BiH"}[side]
    task_name = f"ResDexHand{side_suffix}"

    # Determine DOFs from dexhand
    dexhand = DexHandFactory.create_hand(dexhand_name, "right")
    n_dofs = dexhand.n_dofs

    # Parse data indices from config
    data_indices = getattr(config, "maniptrans_data_indices", [])
    if isinstance(data_indices, str):
        data_indices = [data_indices]

    use_pid = getattr(config, "maniptrans_use_pid", False)
    use_quat_rot = getattr(config, "maniptrans_use_quat_rot", False)

    num_actions = (1 + 6 + n_dofs) if use_quat_rot else (6 + n_dofs)
    prop_obs_dim = 13 + n_dofs * 3
    if side == "bimanual":
        prop_obs_dim *= 2
        num_actions *= 2
    privileged_obs_dim = n_dofs + 13 + 5 * 4 + 3 + 1
    if side == "bimanual":
        privileged_obs_dim *= 2

    tighten_method = getattr(config, "maniptrans_tighten_method", "None")
    tighten_factor = getattr(config, "maniptrans_tighten_factor", 1.0)
    tighten_steps = getattr(config, "maniptrans_tighten_steps", 3200)
    actions_moving_avg = getattr(config, "maniptrans_actions_moving_average", 0.4)
    episode_length = getattr(config, "maniptrans_episode_length", 1200)

    cfg = {
        "name": task_name,
        "physics_engine": "physx",
        "seed": getattr(config, "seed", 42),
        "env": {
            "numEnvs": int(config.num_samples),
            "dexhand": dexhand_name,
            "episodeLength": episode_length,
            "training": False,
            "usePIDControl": use_pid,
            "rolloutStateInit": False,
            "randomStateInit": False,
            "dataIndices": data_indices,
            "obsFutureLength": 1,
            "rolloutLen": None,
            "rolloutBegin": None,
            "clipObservations": 5.0,
            "clipActions": 1.0,
            "aggregateMode": 3,
            "tightenMethod": tighten_method,
            "tightenFactor": tighten_factor,
            "tightenSteps": tighten_steps,
            "actionScale": 1.0,
            "useQuatRot": use_quat_rot,
            "propDumpInfo": {},
            "actionsMovingAverage": actions_moving_avg,
            "translationScale": 1.0,
            "orientationScale": 0.1,
            "bimanual_mode": "united",
            "numActions": num_actions,
            "propObsDim": prop_obs_dim,
            "obsKeys": ["q", "cos_q", "sin_q", "base_state"],
            "privilegedObsDim": privileged_obs_dim,
            "privilegedObsKeys": [
                "dq",
                "manip_obj_pos",
                "manip_obj_quat",
                "manip_obj_vel",
                "manip_obj_ang_vel",
                "tip_force",
                "manip_obj_com",
                "manip_obj_weight",
            ],
            "enableCameraSensors": getattr(config, "maniptrans_record_video", False),
            "controlFrequencyInv": getattr(
                config, "maniptrans_ctrl_freq_inv", 1
            ),
        },
        "sim": {
            "dt": 0.0166667,  # 1/60
            "substeps": 2,
            "up_axis": "z",
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "physx": {
                "num_threads": 4,
                "solver_type": 1,
                "use_gpu": True,
                "num_position_iterations": 8,
                "num_velocity_iterations": 1,
                "contact_offset": 0.005,
                "rest_offset": 0.0,
                "bounce_threshold_velocity": 0.2,
                "max_depenetration_velocity": 1000.0,
                "default_buffer_size_multiplier": 5.0,
                "max_gpu_contact_pairs": 4194304,
                "num_subscenes": 4,
                "contact_collection": 1,
            },
        },
        "task": {
            "randomize": False,
            "randomization_params": {},
        },
    }
    return cfg


def _attach_single_camera(env, graphics_device_id: int):
    """Create a single 1280x720 camera on env-0 for video recording.

    This avoids the overhead of creating cameras for all N environments
    when only env-0's viewpoint is needed.
    """
    from isaacgym import gymapi as _gymapi

    cam_props = _gymapi.CameraProperties()
    cam_props.enable_tensors = True
    cam_props.width = 1280
    cam_props.height = 720
    cam_props.horizontal_fov = 69.4

    cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
    cam_pos = _gymapi.Vec3(0.80, -0.00, 0.7)
    cam_target = _gymapi.Vec3(-1, -0.00, 0.3)
    env.gym.set_camera_location(cam_handle, env.envs[0], cam_pos, cam_target)

    # Wrap the GPU tensor so we can read it later
    cam_tensor = env.gym.get_camera_image_gpu_tensor(
        env.sim, env.envs[0], cam_handle, _gymapi.IMAGE_COLOR
    )
    env._spider_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
    env._spider_cam_handle = cam_handle


# --
# Key functions (SPIDER simulator API)
# --


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]):
    """Setup and reset the ManipTrans IsaacGym environment.

    Returns the VecTask environment instance.
    """
    cfg = _build_task_config(config)

    device_str = str(config.device)
    device_id = 0
    if ":" in device_str:
        device_id = int(device_str.split(":")[1])

    side = config.embodiment_type
    env_cls = ENV_MAP[side]

    record = getattr(config, "maniptrans_record_video", False)
    use_isaac_viewer = getattr(config, "viewer", "none") == "isaac"

    # Determine graphics device: enabled for viewer, recording, or non-headless
    need_graphics = record or use_isaac_viewer or not getattr(config, "headless", True)

    # Don't pass record=True to env constructor — that creates a camera for
    # every environment.  Instead we enable the graphics device and manually
    # attach a single camera to env-0 after construction.
    env = env_cls(
        cfg,
        rl_device=device_str,
        sim_device=device_str,
        graphics_device_id=device_id if need_graphics else -1,
        display=use_isaac_viewer,
        record=False,
        headless=not use_isaac_viewer,
    )

    # Store dexhand for reward computation
    env._spider_dexhand = env.dexhand
    env._spider_side = side
    env._spider_isaac_viewer = use_isaac_viewer

    # Manually create a single camera for env-0 when video is requested
    if record:
        _attach_single_camera(env, device_id)

    # Load pre-trained base imitator model
    _load_base_imitator(config, env)

    return env


def step_env(config: Config, env, ctrl: torch.Tensor):
    """Step all environments with provided controls of shape (N, nu).

    ctrl has shape (N, nu) where nu = env.num_actions (18, residual only).
    The pre-trained base imitator is called at each step to produce the
    first 18 dims (base actions), then the full 36-dim action is formed.
    """
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).repeat(env.num_envs, 1)

    # Clamp residual actions to valid range
    # ctrl = torch.clamp(ctrl, -1.0, 1.0)

    # Get base action from pre-trained imitator (uses current obs)
    nominal_base = _get_base_action(env)  # (N, 18)

    # Concatenate: [nominal_base(18) | residual(18)] = 36 dims
    full_action = torch.cat([nominal_base, ctrl], dim=-1)

    # Store full actions for reward computation
    env.actions = full_action

    # Apply actions (pre_physics_step applies forces and sets DOF targets)
    env.pre_physics_step(full_action)

    # Step physics
    for _i in range(env.control_freq_inv):
        env.gym.simulate(env.sim)

    # Render camera if the render flag is set (only during actual execution,
    # not during MPC optimization rollouts).
    do_render = getattr(env, "_spider_render_camera", False)
    if do_render:
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)

    # Draw Isaac viewer if enabled (handles draw_viewer, poll_events, sync)
    if getattr(env, "_spider_isaac_viewer", False):
        env.render()

    # Refresh state tensors
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_force_sensor_tensor(env.sim)
    env.gym.refresh_dof_force_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    # Update internal states
    env._update_states()

    if do_render:
        env.gym.end_access_image_tensors(env.sim)

    # Increment progress counters
    env.progress_buf += 1
    env.running_progress_buf += 1

    return env


def get_camera_frame(env) -> torch.Tensor:
    """Return the latest RGBA frame from the env-0 camera.

    Returns (H, W, 4) uint8 tensor, or None if recording is not enabled.
    """
    if not hasattr(env, "_spider_cam_tensor"):
        return None
    return env._spider_cam_tensor.clone()


def _compute_target_state(env) -> dict:
    """Compute the target state dict for reward computation."""
    target_state = {}
    # Clamp index to valid range to prevent out-of-bounds access during
    # MPC horizon rollout (optimizer looks ahead beyond seq_len)
    max_idx = env.demo_data["seq_len"].min().long().item() - 1
    cur_idx = torch.clamp(env.progress_buf, max=max_idx)

    cur_wrist_pos = env.demo_data["wrist_pos"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_state["wrist_pos"] = cur_wrist_pos

    cur_wrist_rot = env.demo_data["wrist_rot"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

    target_state["wrist_vel"] = env.demo_data["wrist_velocity"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_state["wrist_ang_vel"] = env.demo_data["wrist_angular_velocity"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]

    target_state["tips_distance"] = env.demo_data["tips_distance"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]

    cur_joints_pos = env.demo_data["mano_joints"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_state["joints_pos"] = cur_joints_pos.reshape(env.num_envs, -1, 3)

    target_state["joints_vel"] = env.demo_data["mano_joints_velocity"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ].reshape(env.num_envs, -1, 3)

    cur_obj_transf = env.demo_data["obj_trajectory"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
    target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[
        :, [1, 2, 3, 0]
    ]

    target_state["manip_obj_vel"] = env.demo_data["obj_velocity"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_state["manip_obj_ang_vel"] = env.demo_data["obj_angular_velocity"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]

    # Compute tip forces
    target_state["tip_force"] = torch.stack(
        [
            env.net_cf[:, env.dexhand_handles[k], :]
            for k in env.dexhand.contact_body_names
        ],
        axis=1,
    )

    # Update contact history
    env.tips_contact_history = torch.cat(
        [
            env.tips_contact_history[:, 1:],
            (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
        ],
        dim=1,
    )
    target_state["tip_contact_state"] = env.tips_contact_history

    # Power / wrist_power / tip_force: set to constants so that
    # reward_power, reward_wrist_power, and reward_finger_tip_force
    # are identical across all parallel samples and do not influence
    # MPC sample selection.  These regularization terms are useful for
    # RL training but add noise to the sampling-based MPC optimizer.
    target_state["power"] = torch.zeros(
        env.num_envs, device=env.device
    )
    target_state["wrist_power"] = torch.zeros(
        env.num_envs, device=env.device
    )
    target_state["tip_force"] = torch.zeros_like(
        target_state["tip_force"]
    )

    return target_state


def get_reward(
    config: Config,
    env,
    ref: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Compute the RL reward from ManipTrans.

    Uses the same compute_imitation_reward as RL training.
    Returns (N,) reward tensor and info dict.
    """
    target_state = _compute_target_state(env)
    max_length = torch.clip(
        env.demo_data["seq_len"], 0, env.max_episode_length
    ).float()

    scale_factor = 1.0  # evaluation mode: no curriculum tightening

    (
        reward,
        reset_buf,
        succeeded,
        failed,
        reward_dict,
        error_buf,
    ) = compute_imitation_reward(
        env.reset_buf,
        env.progress_buf,
        env.running_progress_buf,
        env.actions,
        env.states,
        target_state,
        max_length,
        scale_factor,
        env.dexhand.weight_idx,
        False,  # training=False
    )

    # Remove actuator energy / contact force terms that are meaningful for
    # RL training but not for MPC residual search.
    #   reward_finger_tip_force * 1.0
    #   reward_power            * 0.5
    #   reward_wrist_power      * 0.5
    reward = (
        reward
        - 1.0 * reward_dict["reward_finger_tip_force"]
        - 0.5 * reward_dict["reward_power"]
        - 0.5 * reward_dict["reward_wrist_power"]
    )

    # Cache for get_terminate
    env._spider_last_failed = failed
    env._spider_last_succeeded = succeeded
    env._spider_last_error = error_buf

    info = {
        "reward": reward.detach(),
        "succeeded": succeeded.float(),
        "failed": failed.float(),
    }
    for k, v in reward_dict.items():
        info[k] = v.detach() if isinstance(v, torch.Tensor) else v

    return reward.detach(), info


def get_terminate(
    config: Config, env, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Return termination signal per environment.

    Uses the RL failure detection from ManipTrans.
    Returns (N,) float tensor (1.0 = terminated).
    """
    if hasattr(env, "_spider_last_failed"):
        term = env._spider_last_failed.float()
    else:
        # If get_reward hasn't been called yet, compute it
        _, _ = get_reward(config, env, ref_slice)
        term = env._spider_last_failed.float()

    # Mark NaN/Inf envs as terminated so they are resampled during rollout
    # instead of wasting samples.
    nan_mask = _detect_nan_envs(env)
    if nan_mask.any():
        term = torch.where(nan_mask, torch.ones_like(term), term)

    return term


def get_terminal_reward(
    config: Config,
    env,
    ref_slice: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Terminal reward scaled by terminal_rew_scale."""
    rew, info = get_reward(config, env, ref_slice)
    return config.terminal_rew_scale * rew, info


def get_trace(config: Config, env) -> torch.Tensor:
    """Return per-environment trace points for visualization.

    Returns shape (N, num_trace_points, 3) with:
    - Object position
    - Joint (fingertip) positions
    - Wrist position
    """
    obj_pos = env.states["manip_obj_pos"]  # (N, 3)
    joints = env.states["joints_state"][:, :, :3]  # (N, n_bodies, 3)
    wrist_pos = env.states["base_state"][:, :3]  # (N, 3)

    trace = torch.cat(
        [
            obj_pos.unsqueeze(1),
            wrist_pos.unsqueeze(1),
            joints,
        ],
        dim=1,
    )
    return trace


def get_qpos(env) -> torch.Tensor:
    """Get the qpos of the environment.

    Returns object pos + object quat + DOF positions.
    """
    obj_pos = env.states["manip_obj_pos"]  # (N, 3)
    obj_quat = env.states["manip_obj_quat"]  # (N, 4)
    dof_pos = env.states["q"]  # (N, n_dofs)
    return torch.cat([obj_pos, obj_quat, dof_pos], dim=1)


def get_obj_pos_dist(env) -> torch.Tensor:
    """Get the position distance between the object and the demo object."""
    max_idx = env.demo_data["seq_len"].min().long().item() - 1
    cur_idx = torch.clamp(env.progress_buf, max=max_idx)
    cur_obj_transf = env.demo_data["obj_trajectory"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_obj_pos = cur_obj_transf[:, :3, 3]
    current_obj_pos = env.states["manip_obj_pos"]
    return torch.norm(target_obj_pos - current_obj_pos, dim=-1)


def get_obj_quat_dist(env) -> torch.Tensor:
    """Get the rotation distance between the object and the demo object."""
    max_idx = env.demo_data["seq_len"].min().long().item() - 1
    cur_idx = torch.clamp(env.progress_buf, max=max_idx)
    cur_obj_transf = env.demo_data["obj_trajectory"][
        torch.arange(env.num_envs, device=env.device), cur_idx
    ]
    target_obj_quat = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
    current_obj_quat = env.states["manip_obj_quat"]
    diff_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    angle, _ = quat_to_angle_axis(diff_rot)
    return angle.abs() / math.pi * 180.0  # degrees


def get_joint_pos_dist(env) -> torch.Tensor:
    """Get joint position error (all bodies including wrist).

    Matches the RL eval ``e_j`` metric: mean L2 distance between GT and
    predicted joint positions averaged over all bodies.

    Returns (N,) tensor.
    """
    max_idx = env.demo_data["seq_len"].min().long().item() - 1
    cur_idx = torch.clamp(env.progress_buf, max=max_idx)
    arange = torch.arange(env.num_envs, device=env.device)

    # GT joints (excluding wrist)
    gt_joints = env.demo_data["mano_joints"][arange, cur_idx].reshape(
        env.num_envs, -1, 3
    )
    # GT wrist
    gt_wrist = env.demo_data["wrist_pos"][arange, cur_idx].unsqueeze(1)
    # Full GT: wrist + joints  (N, 18, 3)
    gt_all = torch.cat([gt_wrist, gt_joints], dim=1)

    # Predicted joint positions (all bodies)
    pred_all = env.states["joints_state"][:, :, :3]  # (N, 18, 3)

    return torch.norm(gt_all - pred_all, dim=2).mean(dim=1)


def get_fingertip_pos_dist(env) -> torch.Tensor:
    """Get fingertip position error.

    Matches the RL eval ``e_ft`` metric: mean L2 distance between GT and
    predicted fingertip positions.

    Returns (N,) tensor.
    """
    max_idx = env.demo_data["seq_len"].min().long().item() - 1
    cur_idx = torch.clamp(env.progress_buf, max=max_idx)
    arange = torch.arange(env.num_envs, device=env.device)

    # GT joints (excluding wrist)
    gt_joints = env.demo_data["mano_joints"][arange, cur_idx].reshape(
        env.num_envs, -1, 3
    )
    # GT wrist
    gt_wrist = env.demo_data["wrist_pos"][arange, cur_idx].unsqueeze(1)
    # Full GT: wrist + joints  (N, 18, 3)
    gt_all = torch.cat([gt_wrist, gt_joints], dim=1)

    # Predicted joint positions (all bodies)
    pred_all = env.states["joints_state"][:, :, :3]  # (N, 18, 3)

    # Fingertip indices from dexhand.weight_idx
    tip_idx = [v[0] for k, v in env.dexhand.weight_idx.items() if "tip" in k]
    return torch.norm(
        gt_all[:, tip_idx] - pred_all[:, tip_idx], dim=2
    ).mean(dim=1)


def save_state(env):
    """Save the complete state of all environments."""
    return {
        "root_state": env._root_state.clone(),
        "dof_state": env._dof_state.clone(),
        "progress_buf": env.progress_buf.clone(),
        "running_progress_buf": env.running_progress_buf.clone(),
        "reset_buf": env.reset_buf.clone(),
        "prev_targets": env.prev_targets.clone(),
        "curr_targets": env.curr_targets.clone(),
        "apply_forces": env.apply_forces.clone(),
        "apply_torque": env.apply_torque.clone(),
        "tips_contact_history": env.tips_contact_history.clone(),
    }


def load_state(env, state: dict):
    """Load the complete state to all environments."""
    env._root_state[:] = state["root_state"]
    env._dof_state[:] = state["dof_state"]
    env.progress_buf[:] = state["progress_buf"]
    env.running_progress_buf[:] = state["running_progress_buf"]
    env.reset_buf[:] = state["reset_buf"]
    env.prev_targets[:] = state["prev_targets"]
    env.curr_targets[:] = state["curr_targets"]
    env.apply_forces[:] = state["apply_forces"]
    env.apply_torque[:] = state["apply_torque"]
    env.tips_contact_history[:] = state["tips_contact_history"]

    # Push state back to IsaacGym using indexed API.
    # Must use indexed (not full-tensor) because the env has many actors
    # (table, scene objects, mano_joint markers) that should NOT be touched.
    dexhand_indices = env._global_dexhand_indices.flatten()
    manip_obj_indices = env._global_manip_obj_indices.flatten()
    all_indices = torch.cat([dexhand_indices, manip_obj_indices])

    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._dof_state),
        gymtorch.unwrap_tensor(dexhand_indices),
        len(dexhand_indices),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_state),
        gymtorch.unwrap_tensor(all_indices),
        len(all_indices),
    )

    # Refresh
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env._update_states()

    return env


# ---------------------------------------------------------------------------
# NaN detection & recovery helpers
# ---------------------------------------------------------------------------


def _detect_nan_envs(env):
    """Return (num_envs,) bool tensor — True where state contains NaN/Inf."""
    root_flat = env._root_state.reshape(env.num_envs, -1)
    dof_flat = env._dof_state.reshape(env.num_envs, -1)
    return (
        torch.any(torch.isnan(root_flat) | torch.isinf(root_flat), dim=1)
        | torch.any(torch.isnan(dof_flat) | torch.isinf(dof_flat), dim=1)
    )


def _hard_reset_to_demo(env, env_ids):
    """Reset *env_ids* to the demo trajectory at their current progress.

    Writes the reference state from ``env.demo_data`` into the IsaacGym
    tensor buffers.  Mirrors the RL ``_reset_default`` logic.
    The caller is responsible for pushing to IsaacGym afterwards (the normal
    ``sync_env`` path handles this).
    """
    seq_len = env.demo_data["seq_len"][env_ids].long()
    max_idx = seq_len - 1
    seq_idx = torch.clamp(env.progress_buf[env_ids].long(), min=0)
    seq_idx = torch.min(seq_idx, max_idx)

    # --- DOF state ---
    dof_pos = env.demo_data["opt_dof_pos"][env_ids, seq_idx]
    dof_pos = torch.clamp(
        dof_pos,
        env.dexhand_dof_lower_limits.unsqueeze(0),
        env.dexhand_dof_upper_limits.unsqueeze(0),
    )
    dof_vel = env.demo_data["opt_dof_velocity"][env_ids, seq_idx]
    if hasattr(env, "_dexhand_dof_speed_limits"):
        dof_vel = torch.clamp(
            dof_vel,
            -env._dexhand_dof_speed_limits.unsqueeze(0),
            env._dexhand_dof_speed_limits.unsqueeze(0),
        )

    env._q[env_ids] = dof_pos
    env._qd[env_ids] = dof_vel
    env._pos_control[env_ids] = dof_pos

    # --- Hand wrist (base) state ---
    wrist_pos = env.demo_data["opt_wrist_pos"][env_ids, seq_idx]
    wrist_rot = aa_to_quat(env.demo_data["opt_wrist_rot"][env_ids, seq_idx])
    wrist_rot = wrist_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw (IsaacGym)
    wrist_vel = env.demo_data["opt_wrist_velocity"][env_ids, seq_idx]
    wrist_ang_vel = env.demo_data["opt_wrist_angular_velocity"][
        env_ids, seq_idx
    ]
    env._base_state[env_ids] = torch.cat(
        [wrist_pos, wrist_rot, wrist_vel, wrist_ang_vel], dim=-1
    )

    # --- Object state ---
    obj_transf = env.demo_data["obj_trajectory"][env_ids, seq_idx]
    obj_pos = obj_transf[:, :3, 3]
    obj_rot = rotmat_to_quat(obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
    obj_vel = env.demo_data["obj_velocity"][env_ids, seq_idx]
    obj_ang_vel = env.demo_data["obj_angular_velocity"][env_ids, seq_idx]

    env._manip_obj_root_state[env_ids, :3] = obj_pos
    env._manip_obj_root_state[env_ids, 3:7] = obj_rot
    env._manip_obj_root_state[env_ids, 7:10] = obj_vel
    env._manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

    # --- Clear auxiliary buffers ---
    env.reset_buf[env_ids] = 0
    env.running_progress_buf[env_ids] = 0
    env.apply_forces[env_ids] = 0
    env.apply_torque[env_ids] = 0
    env.curr_targets[env_ids] = 0
    env.prev_targets[env_ids] = 0
    env.tips_contact_history[env_ids] = True


def sync_env(config: Config, env, mj_data=None):
    """Broadcast the state from first environment to all environments."""
    # --- NaN detection and recovery (before broadcasting env 0) ---
    nan_mask = _detect_nan_envs(env)
    if nan_mask.any():
        nan_count = nan_mask.sum().item()
        print(
            f"WARNING: {nan_count}/{env.num_envs} envs have NaN state, "
            f"recovering"
        )
        if nan_mask[0]:
            # Env 0 is broken — try to salvage from a clean env
            clean_ids = torch.where(~nan_mask)[0]
            if len(clean_ids) > 0:
                src = clean_ids[0].item()
                print(f"  Copying clean env {src} -> env 0")
                env._root_state[0] = env._root_state[src].clone()
                env._dof_state[0] = env._dof_state[src].clone()
                env.progress_buf[0] = env.progress_buf[src].clone()
                env.running_progress_buf[0] = (
                    env.running_progress_buf[src].clone()
                )
                env.prev_targets[0] = env.prev_targets[src].clone()
                env.curr_targets[0] = env.curr_targets[src].clone()
                env._pos_control[0] = env._pos_control[src].clone()
                env.apply_forces[0] = env.apply_forces[src].clone()
                env.apply_torque[0] = env.apply_torque[src].clone()
                env.tips_contact_history[0] = (
                    env.tips_contact_history[src].clone()
                )
            else:
                # All envs are broken — hard-reset env 0 to demo data
                print("  All envs NaN — resetting env 0 to demo trajectory")
                _hard_reset_to_demo(
                    env,
                    torch.tensor([0], device=env.device, dtype=torch.long),
                )

    # Root states: copy env 0 to all
    env._root_state[1:] = env._root_state[0:1].clone()
    # DOF states: copy env 0 to all
    env._dof_state[1:] = env._dof_state[0:1].clone()
    # Progress buffers
    env.progress_buf[1:] = env.progress_buf[0:1].clone()
    env.running_progress_buf[1:] = env.running_progress_buf[0:1].clone()
    env.reset_buf[:] = 0
    # Targets
    env.prev_targets[1:] = env.prev_targets[0:1].clone()
    env.curr_targets[1:] = env.curr_targets[0:1].clone()
    env._pos_control[1:] = env._pos_control[0:1].clone()
    # Forces
    env.apply_forces[1:] = env.apply_forces[0:1].clone()
    env.apply_torque[1:] = env.apply_torque[0:1].clone()
    # Contact
    env.tips_contact_history[1:] = env.tips_contact_history[0:1].clone()

    # Push to IsaacGym using indexed API — only update dexhand and manip_obj.
    # The env has additional actors (table, scene objects, mano_joint markers
    # when viewer is enabled) that must NOT be overwritten.
    dexhand_indices = env._global_dexhand_indices.flatten()
    manip_obj_indices = env._global_manip_obj_indices.flatten()
    all_indices = torch.cat([dexhand_indices, manip_obj_indices])

    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._dof_state),
        gymtorch.unwrap_tensor(dexhand_indices),
        len(dexhand_indices),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_state),
        gymtorch.unwrap_tensor(all_indices),
        len(all_indices),
    )

    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env._update_states()

    return env


def copy_sample_state(
    config: Config,
    env,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
):
    """Copy simulation state from source samples to destination samples."""
    src_idx = src_indices.long()
    dst_idx = dst_indices.long()

    # Root states
    env._root_state[dst_idx] = env._root_state[src_idx].clone()
    # DOF states
    env._dof_state[dst_idx] = env._dof_state[src_idx].clone()
    # Progress buffers
    env.progress_buf[dst_idx] = env.progress_buf[src_idx].clone()
    env.running_progress_buf[dst_idx] = env.running_progress_buf[src_idx].clone()
    # Targets
    env.prev_targets[dst_idx] = env.prev_targets[src_idx].clone()
    env.curr_targets[dst_idx] = env.curr_targets[src_idx].clone()
    env._pos_control[dst_idx] = env._pos_control[src_idx].clone()
    # Forces
    env.apply_forces[dst_idx] = env.apply_forces[src_idx].clone()
    env.apply_torque[dst_idx] = env.apply_torque[src_idx].clone()
    # Contact history
    env.tips_contact_history[dst_idx] = env.tips_contact_history[src_idx].clone()

    # Push changes to IsaacGym for the affected environments
    dst_dexhand = env._global_dexhand_indices[dst_idx].flatten()
    dst_obj = env._global_manip_obj_indices[dst_idx].flatten()
    dst_all = torch.cat([dst_dexhand, dst_obj])

    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._dof_state),
        gymtorch.unwrap_tensor(dst_dexhand),
        len(dst_dexhand),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_state),
        gymtorch.unwrap_tensor(dst_all),
        len(dst_all),
    )


def record_video(
    config: Config,
    output_path: str,
    fps: int = 60,
    prev_env=None,
    traj_path: str | None = None,
):
    """Record a video by setting env state from saved trajectory (no physics).

    Creates a separate lightweight environment (2 envs to avoid the squeeze bug
    in ManipTrans ``pack_data``) with a camera, then directly writes saved
    trajectory states into the sim buffers and captures frames.

    Args:
        config: SPIDER config.
        output_path: Path to save the mp4 video.
        fps: Frame rate for the video.
        prev_env: If given, destroy this env's sim before creating the new one
            (IsaacGym GPU pipeline doesn't support two concurrent sims).
        traj_path: Path to the trajectory .npz file. If None, defaults to
            ``{config.output_dir}/trajectory_maniptrans.npz``.
    """
    import copy

    import imageio
    import numpy as np

    if traj_path is None:
        traj_path = f"{config.output_dir}/trajectory_maniptrans.npz"
    if not os.path.exists(traj_path):
        print(f"WARNING: trajectory not found at {traj_path}, skipping video")
        return

    traj = np.load(traj_path)

    # Flatten (n_ticks, ctrl_steps, ...) → (total_steps, ...)
    qpos = traj["qpos"].reshape(-1, traj["qpos"].shape[-1])  # (T, 19)
    base_state = traj["base_state"].reshape(
        -1, traj["base_state"].shape[-1]
    )  # (T, 13)
    total_steps = len(qpos)
    print(f"Recording video: {total_steps} frames from {traj_path}")

    # If a previous env exists, we can't safely create a second IsaacGym sim
    # in the same process (GPU pipeline segfaults). Destroy the existing sim
    # first, then spawn a subprocess with a fresh IsaacGym context.
    if prev_env is not None:
        prev_env.gym.destroy_sim(prev_env.sim)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        _record_video_subprocess(config, traj_path, output_path, fps)
        return

    vid_config = copy.copy(config)
    vid_config.num_samples = 2
    vid_config.maniptrans_record_video = True
    vid_config.headless = True

    env = setup_env(vid_config, None)
    _render_frames(env, config.device, qpos, base_state, total_steps,
                   output_path, fps)


def _render_frames(env, device, qpos, base_state, total_steps,
                   output_path, fps):
    """Render video frames by writing trajectory state into sim buffers."""
    import imageio
    import numpy as np

    t_qpos = torch.tensor(qpos, dtype=torch.float32, device=device)
    t_base = torch.tensor(base_state, dtype=torch.float32, device=device)
    t_obj_pos = t_qpos[:, :3]
    t_obj_quat = t_qpos[:, 3:7]
    t_dof_pos = t_qpos[:, 7:]

    dexhand_idx = env._global_dexhand_indices[0:1].flatten()
    obj_idx = env._global_manip_obj_indices[0:1].flatten()
    all_idx = torch.cat([dexhand_idx, obj_idx])

    video_frames = []
    for step in range(total_steps):
        env._base_state[0, :] = t_base[step]
        env._manip_obj_root_state[0, :3] = t_obj_pos[step]
        env._manip_obj_root_state[0, 3:7] = t_obj_quat[step]
        env._manip_obj_root_state[0, 7:] = 0.0
        env._q[0, :] = t_dof_pos[step]
        env._qd[0, :] = 0.0

        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env._dof_state),
            gymtorch.unwrap_tensor(dexhand_idx),
            len(dexhand_idx),
        )
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env._root_state),
            gymtorch.unwrap_tensor(all_idx),
            len(all_idx),
        )

        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
        video_frames.append(env._spider_cam_tensor.clone().cpu().numpy())
        env.gym.end_access_image_tensors(env.sim)

        if (step + 1) % 200 == 0:
            print(f"  frame {step + 1}/{total_steps}")

    if len(video_frames) > 0:
        rgb_frames = [f[..., :3] for f in video_frames]
        imageio.mimsave(output_path, rgb_frames, fps=fps)
        print(f"Saved video ({len(video_frames)} frames) to {output_path}")
    else:
        print("WARNING: no frames captured, skipping video save")


def _record_video_subprocess(config, traj_path, output_path, fps):
    """Record video in a separate process to avoid IsaacGym GPU pipeline conflicts."""
    import subprocess
    import sys
    import json as _json

    # Serialize the config fields needed for video recording
    vid_args = _json.dumps({
        "dataset_dir": config.dataset_dir,
        "dataset_name": config.dataset_name,
        "robot_type": config.robot_type,
        "embodiment_type": config.embodiment_type,
        "task": config.task,
        "data_id": config.data_id,
        "device": str(config.device),
        "output_dir": config.output_dir,
        "maniptrans_data_indices": getattr(config, "maniptrans_data_indices", [config.task]),
        "maniptrans_episode_length": getattr(config, "maniptrans_episode_length", 1200),
        "maniptrans_tighten_method": getattr(config, "maniptrans_tighten_method", "None"),
        "maniptrans_tighten_factor": getattr(config, "maniptrans_tighten_factor", 1.0),
        "maniptrans_actions_moving_average": getattr(config, "maniptrans_actions_moving_average", 0.4),
        "maniptrans_use_pid": getattr(config, "maniptrans_use_pid", False),
        "maniptrans_use_quat_rot": getattr(config, "maniptrans_use_quat_rot", False),
        "maniptrans_ctrl_freq_inv": getattr(config, "maniptrans_ctrl_freq_inv", 1),
        "traj_path": traj_path,
        "output_path": output_path,
        "fps": fps,
    })

    script = f'''
import sys, os, json
sys.path.insert(0, "/home/ubuntu/HO-Tracker-Baseline")
os.chdir("{os.getcwd()}")
from isaacgym import gymapi
args = json.loads({repr(vid_args)})

from spider.config import Config
from spider.simulators.maniptrans import record_video

config = Config(
    dataset_dir=args["dataset_dir"],
    dataset_name=args["dataset_name"],
    robot_type=args["robot_type"],
    embodiment_type=args["embodiment_type"],
    task=args["task"],
    data_id=args["data_id"],
    device=args["device"],
    output_dir=args["output_dir"],
    simulator="maniptrans",
    sim_dt=1/60.0, ctrl_dt=0.1, knot_dt=0.1, ref_dt=1/60.0, trace_dt=1/60.0,
    horizon=0.2, show_viewer=False, viewer="none",
    save_video=True, save_info=True,
    num_samples=2, num_dr=1, max_sim_steps=-1,
)
for k in ["maniptrans_data_indices", "maniptrans_episode_length",
           "maniptrans_tighten_method", "maniptrans_tighten_factor",
           "maniptrans_actions_moving_average", "maniptrans_use_pid",
           "maniptrans_use_quat_rot", "maniptrans_ctrl_freq_inv"]:
    setattr(config, k, args[k])
config.headless = True

record_video(config, args["output_path"], fps=args["fps"],
             traj_path=args["traj_path"])
'''

    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/ubuntu/HO-Tracker-Baseline:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout.decode("utf-8", errors="replace"), end="")
    if result.returncode != 0:
        print(f"WARNING: video recording subprocess failed (exit {result.returncode})")
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace")[-500:])


def save_env_params(config: Config, env):
    """Save the current simulation parameters (friction, gravity)."""
    sim_params = env.gym.get_sim_params(env.sim)
    return {
        "gravity_z": sim_params.gravity.z,
    }


def load_env_params(config: Config, env, env_param: dict):
    """Load the simulation parameters."""
    if "gravity_z" in env_param:
        sim_params = env.gym.get_sim_params(env.sim)
        sim_params.gravity.z = env_param["gravity_z"]
        env.gym.set_sim_params(env.sim, sim_params)
    return env
