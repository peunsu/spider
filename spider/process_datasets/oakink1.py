# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process OakInk-Image dataset (intent=use/0001) to SPIDER format.

Process:
1. Load 21 hand joints in camera space, transform to world space via cam_extr
2. Extract wrist pose (position + landmark-based rotation) and fingertip positions
3. Extract object world-space pose from obj_anno (T_w_o) in general_info
4. Center object mesh by subtracting centroid, adjust trajectory accordingly
5. Apply global rotation to convert OakInk world space to SPIDER sim space

Input: OAKINK_DIR/image/ (OakInk-Image dataset, intent_id=0001 only)
Output: processed/oakink1/mano/{embodiment_type}/{task}/{data_id}/trajectory_keypoints.npz

seq_id format: {obj_id}_{intent_id}_{subject_id}
  intent 0001 = use (only this intent is processed)

Author: SPIDER Team
"""

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import pymeshlab
import tyro
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
USE_INTENT_ID = "0001"
REF_DT = 1.0 / 30.0  # OakInk-Image is annotated at 30 FPS


def moving_average_filter(signal, window_size=5):
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode="edge")
    kernel = np.ones(window_size) / window_size
    smoothed = np.array(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(signal.shape[1])]
    ).T
    return smoothed.squeeze()


def compute_mesh_centroid(mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    return np.mean(ms.current_mesh().vertex_matrix(), axis=0)


def find_obj_mesh(obj_root, obj_id):
    """Return path to obj mesh (.obj or .ply), or None if not found."""
    for ext in (".obj", ".ply"):
        p = obj_root / f"{obj_id}{ext}"
        if p.exists():
            return p
    return None


def extract_wrist_rotation(joints_world):
    """Compute wrist rotation matrix from 21 world-space hand landmarks.

    Uses middle-finger MCP (9), index MCP (5), ring MCP (13) and wrist (0)
    to define a local frame, matching the convention in gigahand.py.
    """
    z_axis = joints_world[9] - joints_world[0]
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    y_aux = joints_world[5] - joints_world[13]
    y_aux /= np.linalg.norm(y_aux) + 1e-8
    x_axis = np.cross(y_aux, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    return np.stack([x_axis, y_axis, z_axis], axis=1)


VIEW_ID = 0  # only view 0 is used


def load_sequence_frames(anno_path, seq_id_ts, sbj_flag):
    """Return sorted list of frame indices for a given sequence (view 0 only)."""
    with open(anno_path / "seq_all.json") as f:
        seq_all = json.load(f)
    return sorted(
        item[2]
        for item in seq_all
        if item[0] == seq_id_ts and item[3] == VIEW_ID and item[1] == sbj_flag
    )


def process_sequence(
    seq_id_ts,
    dataset_dir,
    data_id=0,
    embodiment_type="right",
    show_viewer=False,
    smooth=True,
):
    """Process one OakInk-Image sequence and save SPIDER-format outputs.

    Args:
        seq_id_ts: "A01001_0001_0000/2021-09-26-19-59-58"
        dataset_dir: root of the datasets directory (contains raw/ and processed/)
        data_id: integer index used when multiple timestamps share the same seq_id
        cam_id: primary camera to use for extracting per-frame annotations
        embodiment_type: "right" (OakInk-Image only contains right-hand data)
        show_viewer: launch interactive MuJoCo viewer after saving
        smooth: apply light moving-average smoothing to hand and object trajectories
    """
    oakink_path = Path(dataset_dir) / "raw" / "oakink1"
    anno_path = oakink_path / "image" / "anno"
    obj_mesh_root = oakink_path / "image" / "obj"

    seq_id, ts = seq_id_ts.split("/")
    obj_id = seq_id.split("_")[0]
    sbj_flag = 0  # use-intent is always single subject (flag=0)
    task_name = seq_id.replace("_", "-")

    frames = load_sequence_frames(anno_path, seq_id_ts, sbj_flag)
    if not frames:
        loguru.logger.warning(f"No frames found for {seq_id_ts} view={VIEW_ID}, skipping.")
        return False

    obj_mesh_path = find_obj_mesh(obj_mesh_root, obj_id)
    if obj_mesh_path is None:
        loguru.logger.warning(f"No mesh file for obj_id={obj_id}, skipping {seq_id}.")
        return False

    N = len(frames)
    wrist_pos_raw = np.zeros((N, 3))
    wrist_rot_raw = np.zeros((N, 3, 3))
    fingertip_raw = np.zeros((N, 5, 3))
    obj_T_w_o = np.zeros((N, 4, 4))
    valid = np.ones(N, dtype=bool)

    for i, frame in enumerate(frames):
        fname = f"{seq_id}__{ts}__{sbj_flag}__{frame}__{VIEW_ID}.pkl"
        hj_path = anno_path / "hand_j" / fname
        gi_path = anno_path / "general_info" / fname

        if not hj_path.exists() or not gi_path.exists():
            valid[i] = False
            continue

        with open(hj_path, "rb") as f:
            hj_cam = pickle.load(f)  # (21, 3) camera space
        with open(gi_path, "rb") as f:
            gi = pickle.load(f)

        T_w_c = np.linalg.inv(gi["cam_extr"].numpy())  # world←camera
        hj_h = np.concatenate([hj_cam, np.ones((21, 1))], axis=1)
        joints_world = (T_w_c @ hj_h.T).T[:, :3]

        wrist_pos_raw[i] = joints_world[0]
        wrist_rot_raw[i] = extract_wrist_rotation(joints_world)
        fingertip_raw[i] = joints_world[FINGERTIP_INDICES]
        obj_T_w_o[i] = gi["obj_anno"].numpy()  # T_w_o (world←obj canonical)

    # Fill missing frames by nearest-neighbour
    if not valid.all():
        n_missing = (~valid).sum()
        loguru.logger.warning(f"{seq_id}: {n_missing}/{N} frames missing, filling by nearest neighbour.")
        valid_idxs = np.where(valid)[0]
        if len(valid_idxs) == 0:
            loguru.logger.error(f"{seq_id}: no valid frames at all, skipping.")
            return False
        for m in np.where(~valid)[0]:
            nn = valid_idxs[np.argmin(np.abs(valid_idxs - m))]
            wrist_pos_raw[m] = wrist_pos_raw[nn]
            wrist_rot_raw[m] = wrist_rot_raw[nn]
            fingertip_raw[m] = fingertip_raw[nn]
            obj_T_w_o[m] = obj_T_w_o[nn]

    if smooth:
        wrist_pos_raw = moving_average_filter(wrist_pos_raw, window_size=5)
        fingertip_raw = moving_average_filter(fingertip_raw.reshape(N, 15), window_size=5).reshape(N, 5, 3)
        obj_trans = moving_average_filter(obj_T_w_o[:, :3, 3], window_size=5)
        obj_T_w_o[:, :3, 3] = obj_trans

    # OakInk world space → SPIDER sim space (Z-up → Y-up with 90° pitch)
    r_global = Rotation.from_euler("xyz", [0, 0, 0])

    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))   # unused (right hand only)
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))

    for i in range(N):
        # Wrist
        qpos_wrist_right[i, :3] = r_global.apply(wrist_pos_raw[i])
        r_wrist = r_global * Rotation.from_matrix(wrist_rot_raw[i])
        xyzw = r_wrist.as_quat()
        qpos_wrist_right[i, 3:] = xyzw[[3, 0, 1, 2]]  # wxyz

        # Fingertips (identity orientation — only position is used)
        for j in range(5):
            qpos_finger_right[i, j, :3] = r_global.apply(fingertip_raw[i, j])
            qpos_finger_right[i, j, 3:] = [1, 0, 0, 0]

        # Object
        T = obj_T_w_o[i]
        qpos_obj_right[i, :3] = r_global.apply(T[:3, 3])
        r_obj = r_global * Rotation.from_matrix(T[:3, :3])
        xyzw = r_obj.as_quat()
        qpos_obj_right[i, 3:] = xyzw[[3, 0, 1, 2]]

    # Centre the mesh and adjust object trajectory by the centroid offset.
    # After subtracting the centroid from mesh vertices, the tracked position
    # shifts by R_sim @ centroid_canonical (where R_sim already includes r_global).
    mesh_centroid = compute_mesh_centroid(str(obj_mesh_path))
    loguru.logger.info(f"{obj_id} mesh centroid: {mesh_centroid}")

    mesh_transform = np.eye(4)
    mesh_transform[:3, 3] = mesh_centroid
    for i in range(N):
        q_wxyz = qpos_obj_right[i, 3:]
        R_sim = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
        T_sim = np.eye(4)
        T_sim[:3, :3] = R_sim
        T_sim[:3, 3] = qpos_obj_right[i, :3]
        T_corrected = T_sim @ mesh_transform
        qpos_obj_right[i, :3] = T_corrected[:3, 3]

    # Save centred mesh
    mesh_dir = get_mesh_dir(dataset_dir, "oakink1", obj_id)
    os.makedirs(mesh_dir, exist_ok=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(obj_mesh_path))
    ms.apply_filter(
        "compute_coord_by_function",
        x=f"x-({mesh_centroid[0]})",
        y=f"y-({mesh_centroid[1]})",
        z=f"z-({mesh_centroid[2]})",
    )
    ms.save_current_mesh(f"{mesh_dir}/visual.obj")

    # Persist trajectory
    output_dir = get_processed_data_dir(
        dataset_dir, "oakink1", "mano", embodiment_type, task_name, data_id
    )
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
        contact=np.zeros((N, 10)),
    )

    mesh_dir_rel = str(Path(mesh_dir).relative_to(dataset_dir))
    task_info = {
        "task": task_name,
        "dataset_name": "oakink1",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": mesh_dir_rel,
        "left_object_mesh_dir": None,
        "ref_dt": REF_DT,
    }
    with open(f"{output_dir}/../task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    loguru.logger.info(f"Saved {N} frames → {output_dir}")

    if show_viewer:
        visualize(
            qpos_wrist_right, qpos_finger_right,
            qpos_wrist_left, qpos_finger_left,
            qpos_obj_right, f"{mesh_dir}/visual.obj",
        )

    return True


def visualize(q_wr, q_fr, q_wl, q_fl, q_or, mesh_path):
    N = q_wr.shape[0]
    qpos_list = np.concatenate(
        [q_wr[:, None], q_fr, q_wl[:, None], q_fl, q_or[:, None], np.zeros_like(q_or[:, None])],
        axis=1,
    )
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")

    obj_handle = mj_spec.worldbody.add_body(name="right_object", mocap=True)
    obj_handle.add_site(
        name="right_object_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[1, 0, 0, 0.5],
        group=0,
    )
    mj_spec.add_mesh(name="right_object_mesh", file=str(mesh_path))
    obj_handle.add_geom(
        name="right_object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="right_object_mesh",
        pos=[0, 0, 0],
        quat=[1, 0, 0, 0],
        group=0,
        condim=1,
    )
    left_handle = mj_spec.worldbody.add_body(name="left_object", mocap=True)
    left_handle.add_site(
        name="left_object_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[0, 1, 0, 0.2],
        group=0,
    )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)
    num_mocap = mj_model.nmocap
    qpos_list = qpos_list[:, :num_mocap, :]

    rate_limiter = RateLimiter(60.0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        cnt = 0
        while viewer.is_running():
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            cnt = (cnt + 1) % N
            rate_limiter.sleep()


def collect_use_sequences(anno_path):
    """Return {seq_id: [seq_id_ts, ...]} for all use-intent single-subject sequences."""
    with open(anno_path / "seq_all.json") as f:
        seq_all = json.load(f)

    seen = set()
    result = defaultdict(list)
    for item in seq_all:
        seq_id_ts = item[0]
        if seq_id_ts in seen:
            continue
        seq_id = seq_id_ts.split("/")[0]
        parts = seq_id.split("_")
        if len(parts) == 3 and parts[1] == USE_INTENT_ID:
            seen.add(seq_id_ts)
            result[seq_id].append(seq_id_ts)
    return result


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    seq_id_ts: str = "",
    embodiment_type: str = "right",
    show_viewer: bool = False,
    smooth: bool = True,
):
    """Process OakInk-Image use-intent sequences into SPIDER format.

    Only view_id=0 annotations are used. By default processes all available
    use-intent (intent_id=0001) single-subject sequences.
    Pass --seq-id-ts to restrict to one sequence, e.g.:

        uv run spider/process_datasets/oakink1.py \\
            --seq-id-ts "A01001_0001_0000/2021-09-26-19-59-58"
    """
    dataset_dir = os.path.abspath(dataset_dir)
    anno_path = Path(dataset_dir) / "raw" / "oakink1" / "image" / "anno"

    if seq_id_ts:
        # Single sequence mode
        seq_id = seq_id_ts.split("/")[0]
        parts = seq_id.split("_")
        if len(parts) != 3 or parts[1] != USE_INTENT_ID:
            loguru.logger.error(
                f"seq_id_ts must be a use-intent (0001) single-subject sequence, got: {seq_id_ts}"
            )
            return
        process_sequence(
            seq_id_ts, dataset_dir, data_id=0,
            embodiment_type=embodiment_type, show_viewer=show_viewer, smooth=smooth,
        )
        return

    # Batch mode: all use-intent sequences
    use_seqs = collect_use_sequences(anno_path)
    total = sum(len(v) for v in use_seqs.values())
    loguru.logger.info(f"Found {total} use-intent sequences across {len(use_seqs)} unique seq_ids.")

    n_ok, n_skip = 0, 0
    for seq_id, seq_id_ts_list in sorted(use_seqs.items()):
        for data_id, s in enumerate(sorted(seq_id_ts_list)):
            ok = process_sequence(
                s, dataset_dir, data_id=data_id,
                embodiment_type=embodiment_type, show_viewer=show_viewer, smooth=smooth,
            )
            if ok:
                n_ok += 1
            else:
                n_skip += 1

    loguru.logger.info(f"Done. Processed={n_ok}, skipped={n_skip}.")


if __name__ == "__main__":
    tyro.cli(main)
