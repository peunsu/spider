"""Run IK for the given hand type and mode.

This is a simplified version of IK based on mink.
No open-hand or contact support.

Input: mujoco scene xml file + hand keypoints + object trajectories.
Output: npz file which contains qpos for key frames in xml.

Strategy:
1. Frame 0: set wrist targets first, integrate many steps, then add finger
   tip targets and integrate more steps.
2. Subsequent frames: solve IK with all targets, integrating sim_dt steps
   for each ref_dt interval.
3. Object qpos is set directly (no retargeting).

Author: Chaoyi Pan
Date: 2026-03-07
"""

import os

import loguru
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "xhand",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = False,
    save_video: bool = True,
    data_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.005,
    ref_dt: float = 0.02,
    wrist_pos_cost: float = 0.3,
    wrist_ori_cost: float = 3.0,
    finger_pos_cost: float = 10.0,
    posture_cost: float = 1e-2,
    wrist_init_steps: int = 200,
    finger_init_steps: int = 300,
    average_frame_size: int = 3,
    z_offset: float = 0.0,
    act_scene: bool = False,
):
    # Resolve directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    model_path = (
        f"{processed_dir_robot}/../scene_act.xml"
        if act_scene
        else f"{processed_dir_robot}/../scene.xml"
    )

    # Load reference keypoints
    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    def _ensure_quat_continuity(arr):
        """Flip quaternion signs so consecutive frames stay in the same hemisphere."""
        out = arr.copy()
        for i in range(1, len(out)):
            if np.dot(out[i], out[i - 1]) < 0:
                out[i] = -out[i]
        return out

    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left = loaded_data["qpos_obj_left"][start_idx:end_idx]

    # Enforce sign continuity: q and -q are the same rotation, but a sign flip
    # between frames causes Rotation.as_euler() to jump by ~180 degrees.
    qpos_wrist_right[:, 3:] = _ensure_quat_continuity(qpos_wrist_right[:, 3:])
    qpos_wrist_left[:, 3:] = _ensure_quat_continuity(qpos_wrist_left[:, 3:])
    qpos_obj_right[:, 3:] = _ensure_quat_continuity(qpos_obj_right[:, 3:])
    qpos_obj_left[:, 3:] = _ensure_quat_continuity(qpos_obj_left[:, 3:])
    for j in range(qpos_finger_right.shape[1]):
        qpos_finger_right[:, j, 3:] = _ensure_quat_continuity(qpos_finger_right[:, j, 3:])
        qpos_finger_left[:, j, 3:] = _ensure_quat_continuity(qpos_finger_left[:, j, 3:])
    try:
        contact_left = loaded_data["contact_left"][start_idx:end_idx]
        contact_right = loaded_data["contact_right"][start_idx:end_idx]
    except KeyError:
        try:
            # Some datasets store combined contact as single "contact" key (N, 10)
            contact_combined = loaded_data["contact"][start_idx:end_idx]
            contact_right = contact_combined[:, :5]
            contact_left = contact_combined[:, 5:]
        except KeyError:
            loguru.logger.warning("No contact data found, using all ones")
            contact_left = np.ones((qpos_finger_right.shape[0], 5))
            contact_right = np.ones((qpos_finger_left.shape[0], 5))

    # Build ik.py-compatible contact structure: [hand_zeros | object_contact_values]
    # contact_offset in run_mjwp = max(total_cols - contact_len, 0) skips the zero half
    # and reads the per-finger values from the trailing half.
    _nf = 5  # fingers per side
    _cr = contact_right[:, :_nf].astype(np.float32)
    _cl = contact_left[:, :_nf].astype(np.float32)
    _N = _cr.shape[0]
    if embodiment_type == "bimanual":
        # [zeros(10) | right_5 | left_5] → (N, 20)
        contact_ref = np.concatenate(
            [np.zeros((_N, _nf * 2), dtype=np.float32), _cr, _cl], axis=1
        )
    elif embodiment_type == "right":
        # [zeros(5) | right_5] → (N, 10)
        contact_ref = np.concatenate(
            [np.zeros((_N, _nf), dtype=np.float32), _cr], axis=1
        )
    else:  # left
        # [zeros(5) | left_5] → (N, 10)
        contact_ref = np.concatenate(
            [np.zeros((_N, _nf), dtype=np.float32), _cl], axis=1
        )

    # Build reference array: (H, num_sites, 7) where 7 = [x, y, z, qw, qx, qy, qz]
    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset

    num_frames = qpos_finger_right.shape[0]

    # Reference index mapping:
    # 0: right_palm, 1-5: right fingers, 6: left_palm, 7-11: left fingers,
    # 12: right_object, 13: left_object
    ref_idx = {}
    ref_idx["right_palm"] = 0
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"right_{name}"] = i + 1
    ref_idx["left_palm"] = 6
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"left_{name}"] = i + 7
    ref_idx["right_object"] = 12
    ref_idx["left_object"] = 13

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)

    # Contact site IDs for world-space position recording.
    # Order matches config.build_hand_contact_site_ids: right then left, thumb→pinky.
    _contact_finger_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    _contact_sites_ordered = []
    if embodiment_type in ["right", "bimanual"]:
        _contact_sites_ordered.extend([f"right_{f}" for f in _contact_finger_names])
    if embodiment_type in ["left", "bimanual"]:
        _contact_sites_ordered.extend([f"left_{f}" for f in _contact_finger_names])
    contact_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in _contact_sites_ordered
    ]
    n_contact = len(_contact_sites_ordered)

    object_qpos_addrs = {}
    if act_scene:
        for side in ["right", "left"]:
            joint_names = [
                f"{side}_object_pos_x",
                f"{side}_object_pos_y",
                f"{side}_object_pos_z",
                f"{side}_object_rot_x",
                f"{side}_object_rot_y",
                f"{side}_object_rot_z",
            ]
            try:
                object_qpos_addrs[side] = [
                    model.jnt_qposadr[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    ]
                    for name in joint_names
                ]
            except ValueError:
                continue

    # Object DOF count
    if act_scene:
        nq_obj = 0
    elif embodiment_type == "bimanual":
        nq_obj = 14
    elif embodiment_type in ["right", "left"]:
        nq_obj = 7
    else:
        nq_obj = 0

    # Create mink configuration
    configuration = mink.Configuration(model)
    data = configuration.data

    # Build site name lists
    finger_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    wrist_sites = []
    finger_sites = []
    if embodiment_type in ["right", "bimanual"]:
        wrist_sites.append("right_palm")
        finger_sites.extend([f"right_{f}" for f in finger_names])
    if embodiment_type in ["left", "bimanual"]:
        wrist_sites.append("left_palm")
        finger_sites.extend([f"left_{f}" for f in finger_names])

    # Create IK tasks
    # Cost priority: finger_pos > wrist_pos > wrist_ori
    wrist_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=wrist_pos_cost,
            orientation_cost=wrist_ori_cost,
            lm_damping=1.0,
        )
        for s in wrist_sites
    ]

    finger_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=finger_pos_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for s in finger_sites
    ]

    posture_task = mink.PostureTask(model, cost=posture_cost)
    posture_task.set_target(configuration.q.copy())

    tasks_wrist = [posture_task, *wrist_tasks]
    tasks_all = [posture_task, *wrist_tasks, *finger_tasks]

    solver = "daqp"
    n_substeps = max(1, int(round(ref_dt / sim_dt)))

    # -- Helpers --
    def set_object_qpos(t):
        def set_object_side(side: str, qpos_obj: np.ndarray):
            qpos_addrs = object_qpos_addrs.get(side)
            if qpos_addrs is not None:
                data.qpos[qpos_addrs[:3]] = qpos_obj[:3]
                data.qpos[qpos_addrs[3:]] = Rotation.from_quat(
                    qpos_obj[3:][[1, 2, 3, 0]]
                ).as_euler("XYZ", degrees=False)
            else:
                data.qpos[-7:] = qpos_obj

        if embodiment_type == "bimanual":
            set_object_side("right", qpos_ref[t, ref_idx["right_object"]])
            set_object_side("left", qpos_ref[t, ref_idx["left_object"]])
        elif embodiment_type == "right":
            set_object_side("right", qpos_ref[t, ref_idx["right_object"]])
        elif embodiment_type == "left":
            set_object_side("left", qpos_ref[t, ref_idx["left_object"]])

    def set_wrist_targets(t):
        for wrist_task, site_name in zip(wrist_tasks, wrist_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            quat_wxyz = qpos_ref[t, ref_idx[site_name], 3:]
            wrist_task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat_wxyz, pos])))

    def set_finger_targets(t):
        for finger_task, site_name in zip(finger_tasks, finger_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            finger_task.set_target(
                mink.SE3(wxyz_xyz=np.array([1.0, 0.0, 0.0, 0.0, *pos]))
            )

    # -- Phase 1: Initialize wrist (frame 0) --
    loguru.logger.info("Phase 1: Initializing wrist position...")
    set_object_qpos(0)
    configuration.update()
    set_wrist_targets(0)
    for _ in range(wrist_init_steps):
        vel = mink.solve_ik(configuration, tasks_wrist, sim_dt, solver, damping=1e-5)
        configuration.integrate_inplace(vel, sim_dt)
        set_object_qpos(0)

    # -- Phase 2: Add finger tips (frame 0) --
    loguru.logger.info("Phase 2: Initializing finger positions...")
    configuration.update()
    set_finger_targets(0)
    for _ in range(finger_init_steps):
        vel = mink.solve_ik(configuration, tasks_all, sim_dt, solver, damping=1e-5)
        configuration.integrate_inplace(vel, sim_dt)
        set_object_qpos(0)

    # -- Main IK loop --
    loguru.logger.info(f"Running IK for {num_frames} frames...")
    qpos_list = []
    contact_pos_list = []
    images = []

    file_suffix = "_act" if act_scene else ""

    if save_video:
        import imageio

        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    run_viewer = get_viewer(show_viewer, model, data)
    rate_limiter = RateLimiter(1 / ref_dt)

    with run_viewer() as gui:
        for t in range(num_frames):
            set_wrist_targets(t)
            set_finger_targets(t)
            set_object_qpos(t)
            configuration.update()

            for _ in range(n_substeps):
                vel = mink.solve_ik(
                    configuration, tasks_all, sim_dt, solver, damping=1e-5
                )
                configuration.integrate_inplace(vel, sim_dt)
                set_object_qpos(t)

            qpos_list.append(configuration.q.copy())

            mujoco.mj_forward(model, data)
            frame_contact_pos = np.zeros((n_contact, 3), dtype=np.float32)
            for _i, _sid in enumerate(contact_site_ids):
                if _sid != -1:
                    frame_contact_pos[_i] = data.site_xpos[_sid]
            contact_pos_list.append(frame_contact_pos)

            if save_video:
                renderer.update_scene(data=data, camera="front")
                images.append(renderer.render())

            if show_viewer:
                gui.sync()
                rate_limiter.sleep()

            if t % 100 == 0:
                loguru.logger.info(f"  Frame {t}/{num_frames}")

    qpos_list = np.array(qpos_list)

    # Unwrap object Euler angles before moving average.
    # set_object_qpos uses Rotation.as_euler() which can produce ±2π jumps at
    # angle boundaries. Moving average with window=3 smears a single 2π jump
    # into three consecutive 2π/3 jumps, which np.unwrap (discont=π) can no
    # longer detect. Unwrapping first eliminates the source jump entirely.
    if act_scene:
        for dim in range(-3, 0):
            qpos_list[:, dim] = np.unwrap(qpos_list[:, dim])

    # -- Post-processing: moving average filter --
    def moving_average_filter(signal_data, window_size=5):
        return np.convolve(
            signal_data, np.ones(window_size) / window_size, mode="valid"
        )

    filtered = np.zeros(
        (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
    )
    for i in range(qpos_list.shape[1]):
        filtered[:, i] = moving_average_filter(qpos_list[:, i], average_frame_size)
    qpos_list = filtered

    contact_list = contact_ref.astype(np.float32)

    # -- Compute qvel via finite differences --
    n_filtered = qpos_list.shape[0]
    qvel_list = np.zeros((n_filtered - 1, model.nv))
    for i in range(1, n_filtered):
        mujoco.mj_differentiatePos(
            model, qvel_list[i - 1], ref_dt, qpos_list[i - 1], qpos_list[i]
        )
    qpos_list = qpos_list[1:]
    contact_list = contact_list[1:]
    contact_pos_list = np.array(contact_pos_list)[1:]
    assert qpos_list.shape[0] == qvel_list.shape[0]

    # -- Forward rollout for validation --
    mj_data_rollout = mujoco.MjData(model)
    n_rollout_substeps = max(1, int(round(ref_dt / sim_dt)))
    model.opt.timestep = ref_dt / n_rollout_substeps
    mj_data_rollout.qpos[:] = qpos_list[0]
    mj_data_rollout.qvel[:] = qvel_list[0]
    mj_data_rollout.ctrl[:] = qpos_list[0][: model.nq - nq_obj]
    for _ in range(n_rollout_substeps):
        mujoco.mj_step(model, mj_data_rollout)
    n_final = qpos_list.shape[0]
    qpos_rollout = np.zeros((n_final, model.nq))
    qpos_rollout[0] = qpos_list[0]
    for i in range(1, n_final):
        mj_data_rollout.ctrl[:] = qpos_list[i][: model.nq - nq_obj]
        noise = np.random.randn(model.nu) * 0.2
        noise[:6] *= 0.0
        noise[22:28] *= 0.0
        mj_data_rollout.ctrl[:] += noise
        for _ in range(n_rollout_substeps):
            mujoco.mj_step(model, mj_data_rollout)
        qpos_rollout[i] = mj_data_rollout.qpos.copy()

    # -- Save outputs --
    file_dir = processed_dir_robot
    if save_video:
        import imageio

        video_path = f"{file_dir}/visualization_ik{file_suffix}.mp4"
        imageio.mimsave(video_path, images, fps=int(1 / ref_dt))
        loguru.logger.info(f"Saved video to {video_path}")

    out_npz = f"{file_dir}/trajectory_kinematic{file_suffix}.npz"
    np.savez(
        out_npz,
        qpos=qpos_list,
        qvel=qvel_list,
        contact=contact_list,
        contact_pos=contact_pos_list,
        frequency=1 / ref_dt,
    )
    loguru.logger.info(f"Saved {out_npz}")

    out_npz = f"{file_dir}/trajectory_ikrollout{file_suffix}.npz"
    np.savez(out_npz, qpos=qpos_rollout)
    loguru.logger.info(f"Saved {out_npz}")


if __name__ == "__main__":
    tyro.cli(main)
