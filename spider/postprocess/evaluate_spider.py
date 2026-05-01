# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate SPIDER retargeted trajectories using ManipTrans-style metrics.

Compares retargeted robot trajectories against MANO ground-truth keypoints
in Cartesian space.

Metrics:
    E_t  - Object translation error (mean L2, reported in cm)
    E_r  - Object rotation error (mean geodesic angle, degrees)
    E_j  - Hand keypoint position error: palm + 5 fingertips (cm)
    E_ft - Fingertip position error: 5 fingertips only (cm)
    SR   - Success rate: fraction of trajectories where E_t < threshold

Usage:
    python spider/postprocess/evaluate_spider.py \\
        --dataset-name hocap \\
        --robot-type xhand \\
        --embodiment-type right \\
        --contact-guidance

    python spider/postprocess/evaluate_spider.py \\
        --dataset-name hocap \\
        --robot-type xhand \\
        --embodiment-type right \\
        --base-dir example_datasets \\
        --et-threshold 0.10 \\
        --er-threshold 30.0 \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from spider import ROOT

DEFAULT_BASE_DIR = os.path.join(ROOT, "..", "example_datasets")
DEFAULT_FREQ = 50.0


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz → (N, 3, 3)."""
    xyzw = np.concatenate([q[:, 1:], q[:, :1]], axis=1)
    return Rotation.from_quat(xyzw).as_matrix()


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between two (N, 3, 3) rotation matrix arrays."""
    R_diff = R1 @ R2.swapaxes(-1, -2)
    trace = np.trace(R_diff, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle = np.minimum(angle, np.pi - angle)
    return np.degrees(angle)


# ---------------------------------------------------------------------------
# Reference frequency loading
# ---------------------------------------------------------------------------


def _load_ref_freq(data_dir: str, contact_guidance: bool) -> float:
    """Try to read reference frequency from the kinematic NPZ."""
    suffix = "_act" if contact_guidance else ""
    kin_path = os.path.join(data_dir, f"trajectory_kinematic{suffix}.npz")
    if os.path.exists(kin_path):
        try:
            d = np.load(kin_path)
            if "frequency" in d:
                return float(d["frequency"])
        except Exception:
            pass
    return DEFAULT_FREQ


# ---------------------------------------------------------------------------
# FK precomputation
# ---------------------------------------------------------------------------


def _precompute_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_full: np.ndarray,
    site_ids: dict[str, int],
    obj_body_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run FK for every frame in qpos_full and collect site positions.

    Returns:
        obj_pos:    (T, 3) object site world positions
        obj_rotmat: (T, 3, 3) object body rotation matrices
        palm_pos:   (T, 3) palm site positions
        tip_pos:    (T, 5, 3) fingertip site positions [thumb, index, middle, ring, pinky]
    """
    T = len(qpos_full)
    obj_pos = np.empty((T, 3))
    obj_rotmat = np.empty((T, 3, 3))
    palm_pos = np.empty((T, 3))
    tip_pos = np.empty((T, 5, 3))

    tip_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

    for i in range(T):
        data.qpos[:] = qpos_full[i]
        mujoco.mj_kinematics(model, data)
        obj_pos[i] = data.site_xpos[site_ids["obj"]]
        obj_rotmat[i] = data.xmat[obj_body_id].reshape(3, 3)
        palm_pos[i] = data.site_xpos[site_ids["palm"]]
        for j, name in enumerate(tip_names):
            tip_pos[i, j] = data.site_xpos[site_ids[name]]

    return obj_pos, obj_rotmat, palm_pos, tip_pos


# ---------------------------------------------------------------------------
# Single trajectory evaluation
# ---------------------------------------------------------------------------


def evaluate_one(
    traj_path: str,
    ref_path: str,
    xml_path: str,
    embodiment_type: str,
    side: str,
) -> dict | None:
    """Evaluate one retargeted trajectory against MANO ground truth.

    Args:
        traj_path:       Path to trajectory_mjwp[_act].npz
        ref_path:        Path to trajectory_keypoints.npz
        xml_path:        Path to scene[_act].xml
        embodiment_type: 'right', 'left', or 'bimanual'
        side:            'right' or 'left' (the dominant side for bimanual)

    Returns:
        Dict with keys e_t, e_r, e_j, e_ft, n_frames, or None on failure.
    """
    try:
        traj = np.load(traj_path)
        ref = np.load(ref_path)
    except Exception as e:
        warnings.warn(f"Failed to load files: {e}")
        return None

    qpos_full = traj["qpos"].reshape(-1, traj["qpos"].shape[-1])
    time_full = traj["time"].reshape(-1)

    contact_guidance = "_act" in os.path.basename(traj_path)
    freq = _load_ref_freq(os.path.dirname(traj_path), contact_guidance)

    ref_wrist = ref[f"qpos_wrist_{side}"]        # (N, 7) pos_xyz + quat_wxyz
    ref_finger = ref[f"qpos_finger_{side}"]       # (N, 5, 7)
    ref_obj = ref[f"qpos_obj_{side}"]             # (N, 7) pos_xyz + quat_wxyz

    N_ref = len(ref_wrist)
    time_ref = np.arange(N_ref) / freq
    max_sim_time = time_full[-1]

    # Keep only ref frames within simulation time range
    valid_mask = time_ref <= max_sim_time
    time_ref = time_ref[valid_mask]
    ref_wrist = ref_wrist[valid_mask]
    ref_finger = ref_finger[valid_mask]
    ref_obj = ref_obj[valid_mask]
    N = len(time_ref)

    if N == 0:
        warnings.warn(f"No reference frames within simulation time range: {traj_path}")
        return None

    # Load MuJoCo model
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        warnings.warn(f"Failed to load XML {xml_path}: {e}")
        return None

    # Resolve site and body IDs
    prefix = side
    try:
        site_ids = {
            "obj": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_object"),
            "palm": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_palm"),
            "thumb_tip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_thumb_tip"),
            "index_tip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_index_tip"),
            "middle_tip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_middle_tip"),
            "ring_tip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_ring_tip"),
            "pinky_tip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_pinky_tip"),
        }
        obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_object")
    except Exception as e:
        warnings.warn(f"Failed to resolve site/body IDs: {e}")
        return None

    # Precompute FK for all sim frames
    sim_obj_pos, sim_obj_rot, sim_palm_pos, sim_tip_pos = _precompute_fk(
        m, d, qpos_full, site_ids, obj_body_id
    )

    # Align reference to nearest sim frames
    sim_indices = np.array([np.argmin(np.abs(time_full - t)) for t in time_ref])

    matched_obj_pos = sim_obj_pos[sim_indices]        # (N, 3)
    matched_obj_rot = sim_obj_rot[sim_indices]        # (N, 3, 3)
    matched_palm_pos = sim_palm_pos[sim_indices]      # (N, 3)
    matched_tip_pos = sim_tip_pos[sim_indices]        # (N, 5, 3)

    # E_t: object translation error (m)
    ref_obj_pos = ref_obj[:, :3]
    e_t = float(np.linalg.norm(matched_obj_pos - ref_obj_pos, axis=-1).mean())

    # E_r: object rotation error (degrees)
    ref_obj_quat_wxyz = ref_obj[:, 3:]
    ref_obj_rot = _quat_wxyz_to_rotmat(ref_obj_quat_wxyz)
    e_r = float(_geodesic_deg(matched_obj_rot, ref_obj_rot).mean())

    # E_j: palm + all 5 fingertips (m)
    ref_palm_pos = ref_wrist[:, :3]                   # (N, 3)
    ref_tip_pos = ref_finger[:, :, :3]                # (N, 5, 3)

    pred_joints = np.concatenate(
        [matched_palm_pos[:, None, :], matched_tip_pos], axis=1
    )  # (N, 6, 3)
    ref_joints = np.concatenate(
        [ref_palm_pos[:, None, :], ref_tip_pos], axis=1
    )  # (N, 6, 3)

    e_j = float(np.linalg.norm(pred_joints - ref_joints, axis=-1).mean())

    # E_ft: fingertip-only error (m)
    e_ft = float(np.linalg.norm(matched_tip_pos - ref_tip_pos, axis=-1).mean())

    return {
        "e_t": e_t,
        "e_r": e_r,
        "e_j": e_j,
        "e_ft": e_ft,
        "n_frames": N,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_trajectories(
    base_dir: str,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    contact_guidance: bool,
) -> list[dict]:
    """Find all retargeted trajectories under the processed data directory."""
    npz_name = "trajectory_mjwp_act.npz" if contact_guidance else "trajectory_mjwp.npz"
    xml_name = "scene_act.xml" if contact_guidance else "scene.xml"
    robot_base = os.path.join(
        base_dir, "processed", dataset_name, robot_type, embodiment_type
    )
    mano_base = os.path.join(
        base_dir, "processed", dataset_name, "mano", embodiment_type
    )

    if not os.path.isdir(robot_base):
        return []

    trajectories = []
    for task in sorted(os.listdir(robot_base)):
        task_dir = os.path.join(robot_base, task)
        if not os.path.isdir(task_dir):
            continue
        xml_path = os.path.join(task_dir, xml_name)
        if not os.path.exists(xml_path):
            xml_path = os.path.join(task_dir, "scene_act.xml" if not contact_guidance else "scene.xml")
            if not os.path.exists(xml_path):
                warnings.warn(f"No scene XML found in {task_dir}, skipping task")
                continue

        for data_id in sorted(
            d for d in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, d)) and d.isdigit()
        ):
            traj_path = os.path.join(task_dir, data_id, npz_name)
            if not os.path.exists(traj_path):
                continue

            side = embodiment_type if embodiment_type in ("right", "left") else "right"
            ref_path = os.path.join(mano_base, task, data_id, "trajectory_keypoints.npz")
            if not os.path.exists(ref_path):
                warnings.warn(f"Reference not found: {ref_path}, skipping")
                continue

            trajectories.append({
                "task": task,
                "data_id": data_id,
                "traj_path": traj_path,
                "ref_path": ref_path,
                "xml_path": xml_path,
                "side": side,
            })
    return trajectories


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>10}"
    return f"{val:>10.4f}"


def _mean_std(vals: list[float], scale: float = 1.0) -> str:
    valid = [v for v in vals if v is not None and not np.isnan(v)]
    if not valid:
        return f"{'N/A':>10}"
    m = np.mean(valid) * scale
    s = np.std(valid) * scale
    return f"{m:>6.2f} ± {s:<5.2f}"


def _print_results(
    results: list[dict],
    et_threshold: float,
    er_threshold: float,
    ej_threshold: float,
    eft_threshold: float,
) -> None:
    header = (
        f"  {'Task':<45} {'ID':>6}"
        f"  {'et(cm)':>10}  {'er(°)':>10}  {'ej(cm)':>10}  {'eft(cm)':>10}  {'N':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        print(
            f"  {r['task']:<45} {r['data_id']:>6}"
            f"  {_fmt(r.get('e_t', None) * 100 if r.get('e_t') is not None else None)}"
            f"  {_fmt(r.get('e_r'))}"
            f"  {_fmt(r.get('e_j', None) * 100 if r.get('e_j') is not None else None)}"
            f"  {_fmt(r.get('e_ft', None) * 100 if r.get('e_ft') is not None else None)}"
            f"  {r.get('n_frames', 0):>6}"
        )

    valid = [r for r in results if r.get("e_t") is not None]
    if not valid:
        return

    et_vals = [r["e_t"] for r in valid]
    er_vals = [r["e_r"] for r in valid]
    ej_vals = [r["e_j"] for r in valid]
    eft_vals = [r["e_ft"] for r in valid]
    
    n_success_t = sum(1 for v in et_vals if v < et_threshold)
    n_success_r = sum(1 for v in er_vals if v < er_threshold)
    n_success_j = sum(1 for v in ej_vals if v < ej_threshold)
    n_success_ft = sum(1 for v in eft_vals if v < eft_threshold)
    n_success = sum(
        1
        for et, er, ej, eft in zip(et_vals, er_vals, ej_vals, eft_vals)
        if (
            et < et_threshold
            and er < er_threshold
            and ej < ej_threshold
            and eft < eft_threshold
        )
    )

    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'MEAN ± STD':<45} {len(valid):>6}"
        f"  {_mean_std(et_vals, 100):>10}"
        f"  {_mean_std(er_vals):>10}"
        f"  {_mean_std(ej_vals, 100):>10}"
        f"  {_mean_std(eft_vals, 100):>10}"
    )
    print(f"\n  Success Rate (et<{et_threshold*100:.0f}cm):          {n_success_t}/{len(valid)} = {n_success_t/len(valid)*100:.1f}%")
    print(f"  Success Rate (er<{er_threshold:.0f}°):              {n_success_r}/{len(valid)} = {n_success_r/len(valid)*100:.1f}%")
    print(f"  Success Rate (ej<{ej_threshold*100:.0f}cm):          {n_success_j}/{len(valid)} = {n_success_j/len(valid)*100:.1f}%")
    print(f"  Success Rate (eft<{eft_threshold*100:.0f}cm):        {n_success_ft}/{len(valid)} = {n_success_ft/len(valid)*100:.1f}%")
    print(
        f"  Success Rate (et<{et_threshold*100:.0f}cm AND er<{er_threshold:.0f}° AND "
        f"ej<{ej_threshold*100:.0f}cm AND eft<{eft_threshold*100:.0f}cm): "
        f"{n_success}/{len(valid)} = {n_success/len(valid)*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SPIDER retargeted trajectories (ManipTrans metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name (e.g. hocap)")
    parser.add_argument("--robot-type", required=True, help="Robot type (e.g. xhand, shadow_hand)")
    parser.add_argument(
        "--embodiment-type",
        required=True,
        choices=["right", "left", "bimanual"],
        help="Hand embodiment type",
    )
    parser.add_argument(
        "--contact-guidance",
        action="store_true",
        default=False,
        help="Use trajectory_mjwp_act.npz (contact guidance ON) instead of trajectory_mjwp.npz",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Base directory containing the 'processed' subdirectory",
    )
    parser.add_argument(
        "--et-threshold",
        type=float,
        default=0.03,
        help="Object translation threshold for success rate (meters)",
    )
    parser.add_argument(
        "--er-threshold",
        type=float,
        default=30.0,
        help="Object rotation threshold for success rate (degrees)",
    )
    parser.add_argument(
        "--ej-threshold",
        type=float,
        default=0.08,
        help="Hand keypoint position threshold for success rate (meters)",
    )
    parser.add_argument(
        "--eft-threshold",
        type=float,
        default=0.06,
        help="Fingertip position threshold for success rate (meters)",
    )
    parser.add_argument("--output", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    npz_label = "trajectory_mjwp_act.npz" if args.contact_guidance else "trajectory_mjwp.npz"

    print(f"Dataset:      {args.dataset_name}")
    print(f"Robot:        {args.robot_type}")
    print(f"Embodiment:   {args.embodiment_type}")
    print(f"Trajectory:   {npz_label}")
    print(f"Base dir:     {base_dir}")
    print()

    trajs = discover_trajectories(
        base_dir,
        args.dataset_name,
        args.robot_type,
        args.embodiment_type,
        args.contact_guidance,
    )

    if not trajs:
        print(f"No trajectories found under {base_dir}/processed/{args.dataset_name}/{args.robot_type}/{args.embodiment_type}/")
        return

    print(f"Found {len(trajs)} trajectories\n")

    results = []
    for i, traj_info in enumerate(trajs):
        tag = f"[{i+1}/{len(trajs)}] {traj_info['task']} / {traj_info['data_id']}"
        print(f"{tag} ...", end="", flush=True)

        metrics = evaluate_one(
            traj_info["traj_path"],
            traj_info["ref_path"],
            traj_info["xml_path"],
            args.embodiment_type,
            traj_info["side"],
        )

        if metrics is None:
            print("  FAILED")
            results.append({
                "task": traj_info["task"],
                "data_id": traj_info["data_id"],
                "e_t": None,
                "e_r": None,
                "e_j": None,
                "e_ft": None,
                "n_frames": 0,
            })
        else:
            print(
                f"  et={metrics['e_t']*100:.2f}cm  er={metrics['e_r']:.1f}°"
                f"  ej={metrics['e_j']*100:.2f}cm  eft={metrics['e_ft']*100:.2f}cm"
                f"  N={metrics['n_frames']}"
            )
            results.append({
                "task": traj_info["task"],
                "data_id": traj_info["data_id"],
                **metrics,
            })

    print(f"\n{'='*100}")
    print(
        f"Results — {args.dataset_name} / {args.robot_type} / {args.embodiment_type} / {npz_label}"
    )
    print(f"{'='*100}\n")
    _print_results(
        results,
        args.et_threshold,
        args.er_threshold,
        args.ej_threshold,
        args.eft_threshold,
    )

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset_name": args.dataset_name,
                    "robot_type": args.robot_type,
                    "embodiment_type": args.embodiment_type,
                    "contact_guidance": args.contact_guidance,
                    "et_threshold": args.et_threshold,
                    "er_threshold": args.er_threshold,
                    "ej_threshold": args.ej_threshold,
                    "eft_threshold": args.eft_threshold,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
