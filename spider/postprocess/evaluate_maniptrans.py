# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute tracking metrics (et, er, ej, eft) from rollout_isaac.npz and
rollout_rl.npz files, and output comparison between SPIDER (MPC) and RL
baseline with mean and std.

Auto-discovers all trajectories under the processed maniptrans directory,
handling both left-hand and right-hand trajectories.

Supports multi-seed evaluation (--multiseed): scans data_id directories 0..N,
groups results by (trajectory, method), and reports mean +/- std across seeds.

Metrics (matching ManipTrans eval_score.py):
    et  - Object translation error (m, reported in cm)
    er  - Object rotation error (degrees)
    ej  - Hand joint position error (m, reported in cm)
    eft - Hand fingertip position error (m, reported in cm)

Usage:
    # Evaluate all trajectories (auto-discover left and right, seed 0 only)
    python spider/postprocess/evaluate_maniptrans.py

    # Multi-seed evaluation (scans all data_id dirs 0..N)
    python spider/postprocess/evaluate_maniptrans.py --multiseed

    # Evaluate specific base directory
    python spider/postprocess/evaluate_maniptrans.py \
        --base_dir example_datasets/processed/maniptrans/inspire

    # Evaluate specific trajectories only
    python spider/postprocess/evaluate_maniptrans.py --data_indices 0f900@0 03ac9@0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# IsaacGym must be imported before torch
sys.path.insert(0, "/home/ubuntu/HO-Tracker-Baseline")
from isaacgym import gymapi  # noqa: E402, F401

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle to 3x3 rotation matrix (numpy, batched)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    axis = np.where(angle > 1e-8, aa / angle, np.zeros_like(aa))
    angle = angle[..., 0]  # (...,)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    R = np.stack(
        [
            t * x * x + c,
            t * x * y - z * s,
            t * x * z + y * s,
            t * x * y + z * s,
            t * y * y + c,
            t * y * z - x * s,
            t * x * z - y * s,
            t * y * z + x * s,
            t * z * z + c,
        ],
        axis=-1,
    ).reshape(aa.shape[:-1] + (3, 3))
    return R


def _rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to axis-angle (numpy, batched)."""
    trace = np.trace(R, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    axis = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(norm > 1e-8, axis / norm, np.zeros_like(axis))
    return axis * angle[..., None]


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix (numpy, batched)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def rotation_distance_deg(gt_rotmat: np.ndarray, pred_quat_xyzw: np.ndarray) -> float:
    """Geodesic rotation distance in degrees."""
    pred_rotmat = _quat_xyzw_to_rotmat(pred_quat_xyzw)
    diff = gt_rotmat @ np.swapaxes(pred_rotmat, -1, -2)
    aa = _rotmat_to_aa(diff)
    angle = np.linalg.norm(aa, axis=-1)
    angle = np.minimum(angle, 2 * np.pi - angle)
    return float((angle / np.pi * 180.0).mean())


def translation_distance(gt_pos: np.ndarray, pred_pos: np.ndarray) -> float:
    """Mean L2 translation distance in meters."""
    return float(np.linalg.norm(gt_pos - pred_pos, axis=-1).mean())


def joint_distance(gt_joints: np.ndarray, pred_joints: np.ndarray) -> float:
    """Mean L2 joint position error in meters."""
    return float(np.linalg.norm(gt_joints - pred_joints, axis=-1).mean())


def fingertip_distance(
    gt_joints: np.ndarray, pred_joints: np.ndarray, tip_indices: list[int]
) -> float:
    """Mean L2 fingertip position error in meters."""
    gt_tips = gt_joints[:, tip_indices]
    pred_tips = pred_joints[:, tip_indices]
    return float(np.linalg.norm(gt_tips - pred_tips, axis=-1).mean())


# ---------------------------------------------------------------------------
# Trajectory evaluation (works for both SPIDER and RL NPZ files)
# ---------------------------------------------------------------------------


def _build_gt_joint_positions(demo_data: dict, dexhand, T: int) -> np.ndarray:
    """Build ground-truth joint position array from demo data.

    The demo_data["mano_joints"] is a dict keyed by joint name (e.g.
    "index_proximal"), each value a (T_total, 3) tensor. We stack them
    in the same order as dexhand.body_names, prepend wrist, and trim to T.

    Returns: (T, n_bodies, 3) numpy array.
    """
    mano_joints = demo_data["mano_joints"]

    if isinstance(mano_joints, dict):
        # Stack joints in body_names order (skip wrist)
        joint_list = []
        for j_name in dexhand.body_names:
            mapped_name = dexhand.to_hand(j_name)[0]
            if mapped_name == "wrist":
                continue
            joint_list.append(mano_joints[mapped_name][:T])
        gt_joints = np.stack(joint_list, axis=1)  # (T, n_joints, 3)
    else:
        # Already a (T_total, n_joints, 3) array
        gt_joints = mano_joints[:T]

    gt_wrist = demo_data["wrist_pos"][:T]
    if gt_wrist.ndim == 1:
        gt_wrist = gt_wrist[None, :]
    gt_full = np.concatenate([gt_wrist[:, None, :], gt_joints], axis=1)
    return gt_full


def evaluate_trajectory_npz(
    npz_path: str,
    demo_data: dict,
    dexhand_weight_idx: dict,
    dexhand=None,
) -> dict:
    """Load a rollout NPZ and compute ManipTrans-compatible metrics.

    Works for both rollout_isaac.npz (SPIDER) and rollout_rl.npz (RL).
    """
    data = np.load(npz_path, allow_pickle=True)
    metrics = {}

    # --- Object metrics (et, er) ---
    if "obj_pos" in data and "obj_quat" in data:
        pred_obj_pos = data["obj_pos"].reshape(-1, 3)
        pred_obj_quat = data["obj_quat"].reshape(-1, 4)
        T_pred = len(pred_obj_pos)

        gt_obj_traj = demo_data["obj_trajectory"]  # (T_gt, 4, 4)
        T = min(T_pred, len(gt_obj_traj))

        gt_obj_pos = gt_obj_traj[:T, :3, 3]
        gt_obj_rotmat = gt_obj_traj[:T, :3, :3]
        pred_obj_pos = pred_obj_pos[:T]
        pred_obj_quat = pred_obj_quat[:T]

        metrics["e_t"] = translation_distance(gt_obj_pos, pred_obj_pos)
        metrics["e_r"] = rotation_distance_deg(gt_obj_rotmat, pred_obj_quat)
        metrics["steps"] = T
    elif "obj_pos_dist" in data and "obj_quat_dist" in data:
        metrics["e_t"] = float(data["obj_pos_dist"].flatten().mean())
        metrics["e_r"] = float(data["obj_quat_dist"].flatten().mean())

    # --- Joint metrics (ej, eft) ---
    if "joint_state" in data and "base_state" in data and dexhand is not None:
        pred_joints = data["joint_state"].reshape(-1, *data["joint_state"].shape[-2:])
        pred_wrist_pos = data["base_state"].reshape(-1, data["base_state"].shape[-1])[
            :, :3
        ]
        T_pred = len(pred_joints)

        # Build GT joint positions
        gt_full_joints = _build_gt_joint_positions(demo_data, dexhand, T_pred)
        T = min(T_pred, len(gt_full_joints))
        gt_full_joints = gt_full_joints[:T]

        pred_full_joints = np.concatenate(
            [pred_wrist_pos[:T, None, :], pred_joints[:T]], axis=1
        )

        n_joints = min(gt_full_joints.shape[1], pred_full_joints.shape[1])
        gt_full_joints = gt_full_joints[:, :n_joints]
        pred_full_joints = pred_full_joints[:, :n_joints]

        metrics["e_j"] = joint_distance(gt_full_joints, pred_full_joints)

        tip_indices = []
        for key in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]:
            if key in dexhand_weight_idx:
                tip_indices.extend(dexhand_weight_idx[key])
        if tip_indices:
            valid_tips = [i for i in tip_indices if i < n_joints]
            if valid_tips:
                metrics["e_ft"] = fingertip_distance(
                    gt_full_joints, pred_full_joints, valid_tips
                )
            else:
                metrics["e_ft"] = 0.0
        else:
            metrics["e_ft"] = 0.0
    else:
        metrics["e_j"] = 0.0
        metrics["e_ft"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Demo data loading
# ---------------------------------------------------------------------------


def load_demo_data_np(
    data_id: str,
    side: str,
    dexhand_name: str,
    max_seq_len: int = 1200,
) -> dict:
    """Load demo data using ManipTrans's ManipDataFactory."""
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    from main.dataset.factory import ManipDataFactory
    from main.dataset.transform import aa_to_rotmat

    dexhand = DexHandFactory.create_hand(dexhand_name, side)

    table_half_height = 0.015
    table_pos_z = 0.4
    table_surface_z = table_pos_z + table_half_height
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(
        np.array([0, 0, -np.pi / 2])
    ) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    mujoco2gym_transf[:3, 3] = np.array([0, 0, table_surface_z])
    mujoco2gym_transf = torch.tensor(
        mujoco2gym_transf, device="cuda:0", dtype=torch.float32
    )

    demo_dataset = ManipDataFactory.create_data(
        manipdata_type="hotracker",
        side=side,
        device="cuda:0",
        mujoco2gym_transf=mujoco2gym_transf,
        max_seq_len=max_seq_len,
        dexhand=dexhand,
    )

    demo = demo_dataset[data_id]

    result = {}
    for k, v in demo.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().numpy()
        elif isinstance(v, dict):
            # Convert dict of tensors (e.g. mano_joints)
            result[k] = {
                dk: dv.cpu().numpy() if isinstance(dv, torch.Tensor) else dv
                for dk, dv in v.items()
            }
        else:
            result[k] = v

    return result, dexhand


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------


def discover_trajectories(base_dir: str) -> list[dict]:
    """Discover all trajectories under base_dir/left/*/0/ and base_dir/right/*/0/.

    Returns a list of dicts with keys: data_id, side, dir, has_spider, has_rl.
    """
    trajectories = []
    for side in ["left", "right"]:
        side_dir = os.path.join(base_dir, side)
        if not os.path.isdir(side_dir):
            continue
        for traj_name in sorted(os.listdir(side_dir)):
            traj_dir = os.path.join(side_dir, traj_name, "0")
            if not os.path.isdir(traj_dir):
                continue
            spider_npz = os.path.join(traj_dir, "rollout_isaac.npz")
            rl_npz = os.path.join(traj_dir, "rollout_rl.npz")
            trajectories.append({
                "data_id": traj_name,
                "side": side,
                "dir": traj_dir,
                "has_spider": os.path.exists(spider_npz),
                "has_rl": os.path.exists(rl_npz),
            })
    return trajectories


def discover_trajectories_multiseed(base_dir: str) -> list[dict]:
    """Discover all (trajectory, seed) pairs under base_dir/{side}/{traj}/{seed}/.

    Scans all numeric subdirectories (data_id = 0, 1, 2, ...) for each trajectory.

    Returns a list of dicts with keys: data_id, side, seed, dir, has_spider, has_rl.
    """
    trajectories = []
    for side in ["left", "right"]:
        side_dir = os.path.join(base_dir, side)
        if not os.path.isdir(side_dir):
            continue
        for traj_name in sorted(os.listdir(side_dir)):
            traj_base = os.path.join(side_dir, traj_name)
            if not os.path.isdir(traj_base):
                continue
            for seed_name in sorted(os.listdir(traj_base)):
                if not seed_name.isdigit():
                    continue
                seed_dir = os.path.join(traj_base, seed_name)
                if not os.path.isdir(seed_dir):
                    continue
                spider_npz = os.path.join(seed_dir, "rollout_isaac.npz")
                rl_npz = os.path.join(seed_dir, "rollout_rl.npz")
                trajectories.append({
                    "data_id": traj_name,
                    "side": side,
                    "seed": int(seed_name),
                    "dir": seed_dir,
                    "has_spider": os.path.exists(spider_npz),
                    "has_rl": os.path.exists(rl_npz),
                })
    return trajectories


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(val, unit=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>10}"
    return f"{val:>10.4f}{unit}"


def _fmt_mean_std(values, scale=1.0):
    """Format mean +/- std."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    if not valid:
        return f"{'N/A':>10}"
    mean = np.mean(valid) * scale
    std = np.std(valid) * scale
    return f"{mean:>7.2f} +/- {std:<5.2f}"


def _print_results_table(
    title: str,
    results: list[dict],
    metric_keys: list[str] = None,
):
    """Print a results table with mean and std."""
    if metric_keys is None:
        metric_keys = ["e_t", "e_r", "e_j", "e_ft"]

    print(f"\n{title}")
    header = (
        f"  {'Data ID':<15} {'Side':<6}"
        f"  {'et (cm)':>10}  {'er (deg)':>10}  {'ej (cm)':>10}  {'eft (cm)':>10}"
        f"  {'Steps':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in sorted(results, key=lambda x: (x.get("side", ""), x.get("data_id", ""))):
        steps = r.get("steps", 0)
        row = (
            f"  {r.get('data_id', 'N/A'):<15} {r.get('side', 'N/A'):<6}"
            f"  {_fmt(r.get('e_t', 0) * 100)}"
            f"  {_fmt(r.get('e_r', 0))}"
            f"  {_fmt(r.get('e_j', 0) * 100)}"
            f"  {_fmt(r.get('e_ft', 0) * 100)}"
            f"  {steps:>7}"
        )
        print(row)

    n = len(results)
    if n > 0:
        print("  " + "-" * (len(header) - 2))
        # Mean +/- Std
        et_vals = [r.get("e_t", 0) for r in results]
        er_vals = [r.get("e_r", 0) for r in results]
        ej_vals = [r.get("e_j", 0) for r in results]
        eft_vals = [r.get("e_ft", 0) for r in results]
        print(
            f"  {'MEAN +/- STD':<15} {'':>6}"
            f"  {_fmt_mean_std(et_vals, 100):>10}"
            f"  {_fmt_mean_std(er_vals):>10}"
            f"  {_fmt_mean_std(ej_vals, 100):>10}"
            f"  {_fmt_mean_std(eft_vals, 100):>10}"
        )
        print(f"  N = {n} trajectories")


def _print_comparison_table(spider_results: list[dict], rl_results: list[dict]):
    """Print side-by-side comparison with mean and std."""
    spider_by_id = {(r["data_id"], r["side"]): r for r in spider_results}
    rl_by_id = {(r["data_id"], r["side"]): r for r in rl_results}
    common_keys = sorted(set(spider_by_id.keys()) & set(rl_by_id.keys()))

    if not common_keys:
        print("\n  No overlapping trajectories to compare.")
        return

    print(f"\n{'=' * 120}")
    print("COMPARISON: SPIDER (MPC) vs RL Baseline")
    print(f"{'=' * 120}")
    header = (
        f"  {'Data ID':<15} {'Side':<6} {'Method':<8}"
        f"  {'et (cm)':>10}  {'er (deg)':>10}  {'ej (cm)':>10}  {'eft (cm)':>10}"
        f"  {'Steps':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    spider_et, spider_er, spider_ej, spider_eft = [], [], [], []
    rl_et, rl_er, rl_ej, rl_eft = [], [], [], []

    for key in common_keys:
        data_id, side = key
        sp = spider_by_id[key]
        rl = rl_by_id[key]

        print(
            f"  {data_id:<15} {side:<6} {'SPIDER':<8}"
            f"  {_fmt(sp.get('e_t', 0) * 100)}"
            f"  {_fmt(sp.get('e_r', 0))}"
            f"  {_fmt(sp.get('e_j', 0) * 100)}"
            f"  {_fmt(sp.get('e_ft', 0) * 100)}"
            f"  {sp.get('steps', 0):>7}"
        )
        print(
            f"  {'':<15} {'':<6} {'RL':<8}"
            f"  {_fmt(rl.get('e_t', 0) * 100)}"
            f"  {_fmt(rl.get('e_r', 0))}"
            f"  {_fmt(rl.get('e_j', 0) * 100)}"
            f"  {_fmt(rl.get('e_ft', 0) * 100)}"
            f"  {rl.get('steps', 0):>7}"
        )

        # Compute improvement
        def _improvement(sp_v, rl_v):
            if sp_v is None or rl_v is None or rl_v == 0:
                return ""
            pct = (rl_v - sp_v) / rl_v * 100
            return f"({pct:+.1f}%)"

        imp_et = _improvement(sp.get("e_t"), rl.get("e_t"))
        imp_er = _improvement(sp.get("e_r"), rl.get("e_r"))
        imp_ej = _improvement(sp.get("e_j"), rl.get("e_j"))
        imp_eft = _improvement(sp.get("e_ft"), rl.get("e_ft"))
        print(
            f"  {'':<15} {'':<6} {'delta':<8}"
            f"  {imp_et:>10}"
            f"  {imp_er:>10}"
            f"  {imp_ej:>10}"
            f"  {imp_eft:>10}"
        )
        print()

        spider_et.append(sp.get("e_t", 0))
        spider_er.append(sp.get("e_r", 0))
        spider_ej.append(sp.get("e_j", 0))
        spider_eft.append(sp.get("e_ft", 0))
        rl_et.append(rl.get("e_t", 0))
        rl_er.append(rl.get("e_r", 0))
        rl_ej.append(rl.get("e_j", 0))
        rl_eft.append(rl.get("e_ft", 0))

    # Summary statistics
    print("  " + "=" * (len(header) - 2))
    print(
        f"  {'SPIDER':<15} {'':<6} {'mean':>8}"
        f"  {_fmt_mean_std(spider_et, 100):>10}"
        f"  {_fmt_mean_std(spider_er):>10}"
        f"  {_fmt_mean_std(spider_ej, 100):>10}"
        f"  {_fmt_mean_std(spider_eft, 100):>10}"
    )
    print(
        f"  {'RL':<15} {'':<6} {'mean':>8}"
        f"  {_fmt_mean_std(rl_et, 100):>10}"
        f"  {_fmt_mean_std(rl_er):>10}"
        f"  {_fmt_mean_std(rl_ej, 100):>10}"
        f"  {_fmt_mean_std(rl_eft, 100):>10}"
    )

    # Aggregate improvement
    n = len(common_keys)
    if n > 0:
        sp_mean_et = np.mean(spider_et) * 100
        rl_mean_et = np.mean(rl_et) * 100
        sp_mean_er = np.mean(spider_er)
        rl_mean_er = np.mean(rl_er)
        sp_mean_ej = np.mean(spider_ej) * 100
        rl_mean_ej = np.mean(rl_ej) * 100
        sp_mean_eft = np.mean(spider_eft) * 100
        rl_mean_eft = np.mean(rl_eft) * 100

        def _delta_str(sp, rl):
            if rl == 0:
                return "N/A"
            pct = (rl - sp) / rl * 100
            return f"{pct:+.1f}%"

        print(
            f"\n  Improvement (SPIDER vs RL): "
            f"et: {_delta_str(sp_mean_et, rl_mean_et)}, "
            f"er: {_delta_str(sp_mean_er, rl_mean_er)}, "
            f"ej: {_delta_str(sp_mean_ej, rl_mean_ej)}, "
            f"eft: {_delta_str(sp_mean_eft, rl_mean_eft)}"
        )
    print(f"  N = {n} paired trajectories")


def _print_multiseed_comparison(
    spider_results: list[dict],
    rl_results: list[dict],
):
    """Print per-trajectory comparison with mean +/- std across seeds.

    Each result dict must have 'data_id', 'side', 'seed', and metric keys.
    Groups by (data_id, side), computes mean +/- std per metric, and prints
    a side-by-side comparison table.
    """
    from collections import defaultdict

    def _group_results(results):
        grouped = defaultdict(list)
        for r in results:
            key = (r["data_id"], r["side"])
            grouped[key].append(r)
        return grouped

    spider_grouped = _group_results(spider_results)
    rl_grouped = _group_results(rl_results)
    all_keys = sorted(set(spider_grouped.keys()) | set(rl_grouped.keys()))

    if not all_keys:
        print("\n  No trajectories found for multi-seed comparison.")
        return

    # Keys with both methods available
    common_keys = sorted(set(spider_grouped.keys()) & set(rl_grouped.keys()))

    print(f"\n{'=' * 130}")
    print("MULTI-SEED COMPARISON: SPIDER (MPC) vs RL Baseline")
    print(f"{'=' * 130}")

    header = (
        f"  {'Data ID':<15} {'Side':<6} {'Method':<8} {'Seeds':>5}"
        f"  {'et (cm)':>18}  {'er (deg)':>18}"
        f"  {'ej (cm)':>18}  {'eft (cm)':>18}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Accumulators for overall mean across trajectories
    # Store per-trajectory means (one mean per trajectory, computed from seeds)
    spider_traj_means = {"e_t": [], "e_r": [], "e_j": [], "e_ft": []}
    rl_traj_means = {"e_t": [], "e_r": [], "e_j": [], "e_ft": []}

    for key in all_keys:
        data_id, side = key
        first_in_pair = True

        for method_name, grouped, traj_means in [
            ("SPIDER", spider_grouped, spider_traj_means),
            ("RL", rl_grouped, rl_traj_means),
        ]:
            if key not in grouped:
                continue
            results = grouped[key]
            n_seeds = len(results)

            # Compute mean +/- std across seeds for each metric
            row_parts = []
            for mk in ["e_t", "e_r", "e_j", "e_ft"]:
                vals = [r.get(mk, 0) for r in results]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                scale = 100.0 if mk in ("e_t", "e_j", "e_ft") else 1.0
                row_parts.append(
                    f"{mean_val * scale:>7.2f} +/- {std_val * scale:<5.2f}"
                )
                traj_means[mk].append(mean_val)

            if first_in_pair:
                print(
                    f"  {data_id:<15} {side:<6} {method_name:<8} {n_seeds:>5}"
                    f"  {row_parts[0]:>18}  {row_parts[1]:>18}"
                    f"  {row_parts[2]:>18}  {row_parts[3]:>18}"
                )
                first_in_pair = False
            else:
                print(
                    f"  {'':<15} {'':<6} {method_name:<8} {n_seeds:>5}"
                    f"  {row_parts[0]:>18}  {row_parts[1]:>18}"
                    f"  {row_parts[2]:>18}  {row_parts[3]:>18}"
                )

        # Print delta for common trajectories
        if key in common_keys:
            sp_results = spider_grouped[key]
            rl_results_here = rl_grouped[key]
            sp_mean_et = np.mean([r.get("e_t", 0) for r in sp_results])
            rl_mean_et = np.mean([r.get("e_t", 0) for r in rl_results_here])
            sp_mean_er = np.mean([r.get("e_r", 0) for r in sp_results])
            rl_mean_er = np.mean([r.get("e_r", 0) for r in rl_results_here])
            sp_mean_ej = np.mean([r.get("e_j", 0) for r in sp_results])
            rl_mean_ej = np.mean([r.get("e_j", 0) for r in rl_results_here])
            sp_mean_eft = np.mean([r.get("e_ft", 0) for r in sp_results])
            rl_mean_eft = np.mean([r.get("e_ft", 0) for r in rl_results_here])

            def _delta(sp_v, rl_v, scale=1.0):
                if rl_v == 0:
                    return f"{'N/A':>18}"
                pct = (rl_v - sp_v) / rl_v * 100
                return f"{'(' + f'{pct:+.1f}%' + ')':>18}"

            print(
                f"  {'':<15} {'':<6} {'delta':<8} {'':>5}"
                f"  {_delta(sp_mean_et, rl_mean_et)}"
                f"  {_delta(sp_mean_er, rl_mean_er)}"
                f"  {_delta(sp_mean_ej, rl_mean_ej)}"
                f"  {_delta(sp_mean_eft, rl_mean_eft)}"
            )
        print()

    # Overall summary: mean of per-trajectory means
    print("  " + "=" * (len(header) - 2))
    for method_name, traj_means in [
        ("SPIDER", spider_traj_means),
        ("RL", rl_traj_means),
    ]:
        if not traj_means["e_t"]:
            continue
        n = len(traj_means["e_t"])
        parts = []
        for mk in ["e_t", "e_r", "e_j", "e_ft"]:
            vals = traj_means[mk]
            scale = 100.0 if mk in ("e_t", "e_j", "e_ft") else 1.0
            mean_val = np.mean(vals) * scale
            std_val = np.std(vals) * scale
            parts.append(f"{mean_val:>7.2f} +/- {std_val:<5.2f}")
        print(
            f"  {'OVERALL':<15} {'':<6} {method_name:<8} {n:>5}"
            f"  {parts[0]:>18}  {parts[1]:>18}"
            f"  {parts[2]:>18}  {parts[3]:>18}"
        )

    # Overall improvement (using common trajectories only)
    if common_keys:
        sp_means = {
            mk: [
                np.mean([r.get(mk, 0) for r in spider_grouped[k]])
                for k in common_keys
            ]
            for mk in ["e_t", "e_r", "e_j", "e_ft"]
        }
        rl_means = {
            mk: [
                np.mean([r.get(mk, 0) for r in rl_grouped[k]])
                for k in common_keys
            ]
            for mk in ["e_t", "e_r", "e_j", "e_ft"]
        }

        def _overall_delta(sp_list, rl_list, scale=1.0):
            sp_m = np.mean(sp_list) * scale
            rl_m = np.mean(rl_list) * scale
            if rl_m == 0:
                return "N/A"
            pct = (rl_m - sp_m) / rl_m * 100
            return f"{pct:+.1f}%"

        print(
            f"\n  Improvement (SPIDER vs RL, {len(common_keys)} paired trajectories): "
            f"et: {_overall_delta(sp_means['e_t'], rl_means['e_t'], 100)}, "
            f"er: {_overall_delta(sp_means['e_r'], rl_means['e_r'])}, "
            f"ej: {_overall_delta(sp_means['e_j'], rl_means['e_j'], 100)}, "
            f"eft: {_overall_delta(sp_means['e_ft'], rl_means['e_ft'], 100)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_single_seed_eval(args):
    """Original single-seed evaluation (data_id=0 only)."""
    base_dir = os.path.abspath(args.base_dir)
    dexhand_name = args.dexhand

    trajectories = discover_trajectories(base_dir)
    if not trajectories:
        print(f"No trajectories found under {base_dir}")
        return

    if args.data_indices:
        trajectories = [t for t in trajectories if t["data_id"] in args.data_indices]

    n_spider = sum(1 for t in trajectories if t["has_spider"])
    n_rl = sum(1 for t in trajectories if t["has_rl"])
    print(f"Found {len(trajectories)} trajectories ({n_spider} with SPIDER, {n_rl} with RL)")
    for t in trajectories:
        print(f"  {t['side']}/{t['data_id']}: SPIDER={'Y' if t['has_spider'] else 'N'} RL={'Y' if t['has_rl'] else 'N'}")
    print()

    spider_results = []
    rl_results = []
    demo_cache = {}

    for traj in trajectories:
        data_id = traj["data_id"]
        side = traj["side"]
        traj_dir = traj["dir"]

        cache_key = (data_id, side)
        if cache_key not in demo_cache:
            try:
                demo_data, dexhand = load_demo_data_np(data_id, side, dexhand_name)
                demo_cache[cache_key] = (demo_data, dexhand)
            except Exception as e:
                print(f"  WARNING: Failed to load demo data for {side}/{data_id}: {e}")
                continue
        demo_data, dexhand = demo_cache[cache_key]

        if traj["has_spider"]:
            spider_npz = os.path.join(traj_dir, "rollout_isaac.npz")
            print(f"Evaluating SPIDER: {side}/{data_id}")
            try:
                metrics = evaluate_trajectory_npz(
                    spider_npz, demo_data, dexhand.weight_idx, dexhand=dexhand
                )
                metrics["data_id"] = data_id
                metrics["side"] = side
                spider_results.append(metrics)
            except Exception as e:
                print(f"  WARNING: Failed to evaluate SPIDER {side}/{data_id}: {e}")

        if traj["has_rl"]:
            rl_npz = os.path.join(traj_dir, "rollout_rl.npz")
            print(f"Evaluating RL:     {side}/{data_id}")
            try:
                metrics = evaluate_trajectory_npz(
                    rl_npz, demo_data, dexhand.weight_idx, dexhand=dexhand
                )
                metrics["data_id"] = data_id
                metrics["side"] = side
                rl_results.append(metrics)
            except Exception as e:
                print(f"  WARNING: Failed to evaluate RL {side}/{data_id}: {e}")

    if spider_results:
        _print_results_table("SPIDER (MPC) Results", spider_results)
    if rl_results:
        _print_results_table("RL Baseline Results", rl_results)
    if spider_results and rl_results:
        _print_comparison_table(spider_results, rl_results)

    all_results = {"spider": spider_results, "rl": rl_results}
    output_path = os.path.join(base_dir, "eval_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")


def _run_multiseed_eval(args):
    """Multi-seed evaluation: scan all data_id dirs, group by trajectory."""
    base_dir = os.path.abspath(args.base_dir)
    dexhand_name = args.dexhand

    trajectories = discover_trajectories_multiseed(base_dir)
    if not trajectories:
        print(f"No multi-seed trajectories found under {base_dir}")
        return

    if args.data_indices:
        trajectories = [t for t in trajectories if t["data_id"] in args.data_indices]

    # Summary
    from collections import defaultdict
    traj_seeds = defaultdict(set)
    for t in trajectories:
        traj_seeds[(t["data_id"], t["side"])].add(t["seed"])

    n_spider = sum(1 for t in trajectories if t["has_spider"])
    n_rl = sum(1 for t in trajectories if t["has_rl"])
    print(
        f"Found {len(trajectories)} (trajectory, seed) pairs "
        f"across {len(traj_seeds)} unique trajectories"
    )
    print(f"  SPIDER: {n_spider}, RL: {n_rl}")
    for key in sorted(traj_seeds.keys()):
        data_id, side = key
        seeds = sorted(traj_seeds[key])
        seed_str = ",".join(str(s) for s in seeds)
        n_sp = sum(
            1 for t in trajectories
            if t["data_id"] == data_id and t["side"] == side and t["has_spider"]
        )
        n_rl_t = sum(
            1 for t in trajectories
            if t["data_id"] == data_id and t["side"] == side and t["has_rl"]
        )
        print(
            f"  {side}/{data_id}: seeds=[{seed_str}] "
            f"SPIDER={n_sp} RL={n_rl_t}"
        )
    print()

    # Evaluate
    spider_results = []
    rl_results = []
    demo_cache = {}

    for traj in trajectories:
        data_id = traj["data_id"]
        side = traj["side"]
        seed = traj["seed"]
        traj_dir = traj["dir"]

        cache_key = (data_id, side)
        if cache_key not in demo_cache:
            try:
                demo_data, dexhand = load_demo_data_np(data_id, side, dexhand_name)
                demo_cache[cache_key] = (demo_data, dexhand)
            except Exception as e:
                print(f"  WARNING: Failed to load demo data for {side}/{data_id}: {e}")
                continue
        demo_data, dexhand = demo_cache[cache_key]

        if traj["has_spider"]:
            spider_npz = os.path.join(traj_dir, "rollout_isaac.npz")
            print(f"Evaluating SPIDER: {side}/{data_id} seed={seed}")
            try:
                metrics = evaluate_trajectory_npz(
                    spider_npz, demo_data, dexhand.weight_idx, dexhand=dexhand
                )
                metrics["data_id"] = data_id
                metrics["side"] = side
                metrics["seed"] = seed
                spider_results.append(metrics)
            except Exception as e:
                print(
                    f"  WARNING: Failed to evaluate SPIDER "
                    f"{side}/{data_id} seed={seed}: {e}"
                )

        if traj["has_rl"]:
            rl_npz = os.path.join(traj_dir, "rollout_rl.npz")
            print(f"Evaluating RL:     {side}/{data_id} seed={seed}")
            try:
                metrics = evaluate_trajectory_npz(
                    rl_npz, demo_data, dexhand.weight_idx, dexhand=dexhand
                )
                metrics["data_id"] = data_id
                metrics["side"] = side
                metrics["seed"] = seed
                rl_results.append(metrics)
            except Exception as e:
                print(
                    f"  WARNING: Failed to evaluate RL "
                    f"{side}/{data_id} seed={seed}: {e}"
                )

    # Print per-seed detail tables
    if spider_results:
        _print_results_table(
            f"SPIDER (MPC) Results ({len(spider_results)} seed-runs)",
            spider_results,
        )
    if rl_results:
        _print_results_table(
            f"RL Baseline Results ({len(rl_results)} seed-runs)",
            rl_results,
        )

    # Print multi-seed grouped comparison
    if spider_results or rl_results:
        _print_multiseed_comparison(spider_results, rl_results)

    # Save all per-seed results to JSON
    all_results = {
        "spider": spider_results,
        "rl": rl_results,
        "mode": "multiseed",
    }
    output_path = os.path.join(base_dir, "eval_multiseed.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved multi-seed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ManipTrans trajectory metrics (SPIDER vs RL)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="example_datasets/processed/maniptrans/inspire",
        help="Base directory containing left/ and right/ subdirectories",
    )
    parser.add_argument(
        "--dexhand",
        type=str,
        default="inspire",
    )
    parser.add_argument(
        "--data_indices",
        nargs="*",
        default=None,
        help="Specific data indices to evaluate (default: all found)",
    )
    parser.add_argument(
        "--multiseed",
        action="store_true",
        default=False,
        help="Enable multi-seed evaluation (scan all data_id dirs, not just 0)",
    )
    args = parser.parse_args()

    if args.multiseed:
        _run_multiseed_eval(args)
    else:
        _run_single_seed_eval(args)


if __name__ == "__main__":
    main()
