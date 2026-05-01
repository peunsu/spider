import glob
import json
import os
from contextlib import contextmanager
from os.path import join
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import pymeshlab
import tyro
from loop_rate_limiters import RateLimiter
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

# --- 누락되었던 DATACONFIGS 추가 ---
DATACONFIGS = [
    # Subject 1
    {"path": "subject_1/20231025_170231/processed_seq.pt", "start": 0, "end": 250, "obj": "G10_4", "desc": "통"},
    {"path": "subject_1/20231025_170231/processed_seq.pt", "start": 540, "end": 850, "obj": "G10_1", "desc": "커피통"},
    {"path": "subject_1/20231025_170231/processed_seq.pt", "start": 250, "end": 540, "obj": "G10_2", "desc": "로션통"},
    {"path": "subject_1/20231025_170231/processed_seq.pt", "start": 830, "end": 1075, "obj": "G10_3", "desc": "빼빼로곽"},

    # Subject 2
    {"path": "subject_2/20231022_201556/processed_seq.pt", "start": 0, "end": 380, "obj": "G05_2", "desc": "직육면체로션통"},
    {"path": "subject_2/20231022_201556/processed_seq.pt", "start": 380, "end": 882, "obj": "G05_1", "desc": "파란소스통"},
    {"path": "subject_2/20231022_203100/processed_seq.pt", "start": 0, "end": 630, "obj": "G09_4", "desc": "시럽통"},
    {"path": "subject_2/20231022_203100/processed_seq.pt", "start": 680, "end": 1280, "obj": "G09_2", "desc": "자동차"},
    {"path": "subject_2/20231023_164242/processed_seq.pt", "start": 30, "end": 510, "obj": "G19_1", "desc": "글루건"},
    {"path": "subject_2/20231023_164242/processed_seq.pt", "start": 510, "end": 960, "obj": "G19_4", "desc": "탁구채"},
    {"path": "subject_2/20231023_164242/processed_seq.pt", "start": 950, "end": 1520, "obj": "G19_2", "desc": "파란뒤집개"},
    {"path": "subject_2/20231023_164741/processed_seq.pt", "start": 0, "end": 550, "obj": "G22_3", "desc": "파란시리얼박스"},
    {"path": "subject_2/20231023_164741/processed_seq.pt", "start": 550, "end": 1050, "obj": "G22_4", "desc": "하늘색뒤집개"},
    {"path": "subject_2/20231023_164741/processed_seq.pt", "start": 1080, "end": 1580, "obj": "G22_2", "desc": "머스타드통"},

    # Subject 3
    {"path": "subject_3/20231024_154810/processed_seq.pt", "start": 0, "end": 500, "obj": "G09_4", "desc": "시럽통"},
    {"path": "subject_3/20231024_154810/processed_seq.pt", "start": 600, "end": 1000, "obj": "G09_2", "desc": "자동차"},
    {"path": "subject_3/20231024_154810/processed_seq.pt", "start": 1100, "end": 1230, "obj": "G09_1", "desc": "초콜릿곽"},

    # Subject 4
    {"path": "subject_4/20231026_162248/processed_seq.pt", "start": 0, "end": 350, "obj": "G11_2", "desc": "큰약품통"},
    {"path": "subject_4/20231026_162248/processed_seq.pt", "start": 344, "end": 774, "obj": "G11_1", "desc": "쥬스"},
    {"path": "subject_4/20231026_164958/processed_seq.pt", "start": 0, "end": 360, "obj": "G21_1", "desc": "로션통"},
    {"path": "subject_4/20231026_164958/processed_seq.pt", "start": 350, "end": 770, "obj": "G21_4", "desc": "뒤집개2"},
    {"path": "subject_4/20231026_164958/processed_seq.pt", "start": 750, "end": 1250, "obj": "G21_3", "desc": "분홍색뒤집개"},

    # Subject 6
    {"path": "subject_6/20231025_111357/processed_seq.pt", "start": 0, "end": 500, "obj": "G06_2", "desc": "큰쥬스곽"},
    {"path": "subject_6/20231025_111357/processed_seq.pt", "start": 550, "end": 1110, "obj": "G06_3", "desc": "겨자통"},
    {"path": "subject_6/20231025_111357/processed_seq.pt", "start": 1150, "end": 1810, "obj": "G06_4", "desc": "청소솔"},
    {"path": "subject_6/20231025_111357/processed_seq.pt", "start": 1850, "end": 2360, "obj": "G06_1", "desc": "책"},
    {"path": "subject_6/20231025_112332/processed_seq.pt", "start": 0, "end": 400, "obj": "G09_1", "desc": "초콜릿곽"},
    {"path": "subject_6/20231025_112332/processed_seq.pt", "start": 450, "end": 1050, "obj": "G09_3", "desc": "렌즈통"},
    {"path": "subject_6/20231025_112332/processed_seq.pt", "start": 1050, "end": 1650, "obj": "G09_2", "desc": "자동차"},
    {"path": "subject_6/20231025_112332/processed_seq.pt", "start": 1720, "end": 2255, "obj": "G09_4", "desc": "시럽통"},

    # Subject 9
    {"path": "subject_9/20231027_125019/processed_seq.pt", "start": 0, "end": 320, "obj": "G16_2", "desc": "하얀비누곽"},
    {"path": "subject_9/20231027_125019/processed_seq.pt", "start": 370, "end": 700, "obj": "G16_4", "desc": "비누곽"},
    {"path": "subject_9/20231027_125019/processed_seq.pt", "start": 730, "end": 1000, "obj": "G16_3", "desc": "쥬스곽"},
    {"path": "subject_9/20231027_125019/processed_seq.pt", "start": 1030, "end": 1415, "obj": "G16_1", "desc": "락스통"},
]

def moving_average_filter(signal, window_size=5):
    if signal.ndim == 1: signal = signal.reshape(-1, 1)
    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode="edge")
    kernel = np.ones(window_size) / window_size
    smoothed = np.array([np.convolve(padded[:, i], kernel, mode="valid") for i in range(signal.shape[1])]).T
    return smoothed.squeeze()

def ensure_quat_continuity(quats):
    """Flip quaternion signs so consecutive frames stay in the same hemisphere.

    q and -q encode the same rotation, but component-wise averaging across a
    sign boundary collapses the magnitude to ~0 and produces an arbitrary
    direction after renormalization — the root cause of instantaneous flips.
    """
    out = quats.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out

def interpolate_object_poses(extracted_poses, tracked_frames, use_smoother=True):
    trans, rots, idxs = [], [], []
    for cid in tracked_frames:
        frame_key = str(cid).zfill(6)
        if frame_key in extracted_poses:
            trans.append(np.array(extracted_poses[frame_key]["mesh_translation"]).squeeze())
            rots.append(np.array(extracted_poses[frame_key]["mesh_rotation"])[[1, 2, 3, 0]].squeeze()) # wxyz -> xyzw
            idxs.append(cid)
    trans, rots, idxs = np.array(trans), np.array(rots), np.array(idxs)
    full_idx = np.arange(idxs[0], idxs[-1] + 1)
    interp_t = interp1d(idxs, trans, axis=0, kind="linear", fill_value="extrapolate")(full_idx)
    r = Rotation.from_quat(rots)
    interp_r = Slerp(idxs, r)(full_idx).as_quat()
    # Enforce sign continuity before averaging: q and -q are the same rotation
    # but component-wise averaging across a sign flip collapses the quaternion
    # magnitude to ~0, producing an arbitrary rotation after renormalization.
    interp_r = ensure_quat_continuity(interp_r)
    if use_smoother:
        interp_t = moving_average_filter(interp_t, window_size=9)
        interp_r = moving_average_filter(interp_r, window_size=9)
        # Renormalize: averaging reduces quaternion magnitude slightly
        norms = np.linalg.norm(interp_r, axis=1, keepdims=True)
        interp_r = interp_r / np.where(norms > 1e-6, norms, 1.0)
    return {str(fid).zfill(6): {"mesh_translation": interp_t[i].tolist(), "mesh_rotation": interp_r[i].tolist()}
            for i, fid in enumerate(full_idx)}, full_idx.tolist()

def compute_mesh_centroid(mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    return np.mean(ms.current_mesh().vertex_matrix(), axis=0)

def extract_hand_data_from_kpts(keypoints, hand_id):
    """
    keypoints: (21, 3) numpy array
    th_pos: (3,) numpy array (translation)
    """
    # FINGERTIP_INDICES: [4, 8, 12, 16, 20] (Thumb, Index, Middle, Ring, Little)
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]
    fingertip_positions = keypoints[FINGERTIP_INDICES]

    # Landmark-based wrist rotation
    # keypoints[0]: Wrist, [5]: Index base, [9]: Middle base, [13]: Ring base
    z_axis = keypoints[9] - keypoints[0]
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    y_axis_aux = keypoints[5] - keypoints[13]
    y_axis_aux = y_axis_aux / (np.linalg.norm(y_axis_aux) + 1e-8)
    
    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    if hand_id == "left":
        x_axis = -x_axis
        y_axis = -y_axis
    
    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)

    th_pos = keypoints[0] # keypoints[0]이 실제 손목 위치이므로, th_pos는 그에 대한 offset으로 계산

    return th_pos, rot_mat, fingertip_positions

def process_config(cfg, dataset_dir, embodiment_type="right", show_viewer=True):
    hocap_path = Path(dataset_dir) / "raw" / "hocap"

    rel_parts = cfg["path"].split("/")
    sub_id, timestamp, obj_id = rel_parts[0], rel_parts[1], cfg["obj"]
    start_f, end_f = cfg["start"], cfg["end"]

    trajectory_dir = hocap_path / sub_id / timestamp
    mano_path = trajectory_dir / "mano_params.json"
    obj_pose_path = trajectory_dir / "object_pose.json"
    mesh_path = hocap_path / "models" / obj_id / "textured_mesh.obj"

    with open(mano_path) as f:
        mano_all = json.load(f)

    with open(obj_pose_path) as f:
        obj_all = json.load(f)

    # Object Pose 추출 및 보간
    extracted_obj = {
        str(f).zfill(6): obj_all[str(f).zfill(6)][obj_id]
        for f in range(start_f, end_f + 1)
        if str(f).zfill(6) in obj_all and obj_id in obj_all[str(f).zfill(6)]
    }

    if not extracted_obj:
        return

    interp_obj, valid_frames = interpolate_object_poses(
        extracted_obj,
        sorted([int(k) for k in extracted_obj.keys()])
    )

    N = len(valid_frames)
    q_wr = np.zeros((N, 7))
    q_fr = np.zeros((N, 5, 7))
    q_wl = np.zeros((N, 7))
    q_fl = np.zeros((N, 5, 7))
    q_or = np.zeros((N, 7))

    r_global = Rotation.from_euler("xyz", [0, 0, 0])

    for i, f_idx in enumerate(valid_frames):
        # ---------- Hand (JSON의 kpts 사용) ----------
        # mano_all["right"]["kpts"]는 (Total_Frames, 21, 3) 형태라고 가정
        kpts_frame = np.array(mano_all["right"]["kpts"][f_idx])

        hand_pos, hand_rot_mat, finger_pos = extract_hand_data_from_kpts(
            kpts_frame, "right"
        )

        # 손목 위치/회전
        q_wr[i, :3] = r_global.apply(hand_pos)
        r_wrist = Rotation.from_matrix(hand_rot_mat)
        r_wrist_final = r_global * r_wrist
        q_wr[i, 3:] = r_wrist_final.as_quat()[[3, 0, 1, 2]] # xyzw -> wxyz

        # 손가락 끝 위치
        for fj in range(5):
            q_fr[i, fj, :3] = r_global.apply(finger_pos[fj])
            q_fr[i, fj, 3:] = [1, 0, 0, 0]

        # ---------- Object ----------
        pose = interp_obj[str(f_idx).zfill(6)]
        obj_t = np.array(pose["mesh_translation"])
        quat_xyzw = np.array(pose["mesh_rotation"])
        r_obj = Rotation.from_quat(quat_xyzw)

        q_or[i, :3] = r_global.apply(obj_t)
        r_obj_final = r_global * r_obj
        q_or[i, 3:] = r_obj_final.as_quat()[[3, 0, 1, 2]]

    # -------------------------
    # Save
    # -------------------------
    task_name = f"{sub_id}-{timestamp}-{obj_id}"
    data_id = start_f

    output_dir = get_processed_data_dir(
        dataset_dir, "hocap", "mano", embodiment_type, task_name, data_id
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. mesh centroid 계산
    mesh_centroid = compute_mesh_centroid(str(mesh_path))
    loguru.logger.info(f"Mesh centroid offset: {mesh_centroid}")

    # 2. 물체 trajectory에 centroid offset 반영
    #    mesh를 centroid 기준으로 중심화하면,
    #    물체 위치도 그만큼 보정 필요
    mesh_transform = np.eye(4)
    mesh_transform[:3, 3] = mesh_centroid
    for i in range(N):
        obj_t = q_or[i, :3]
        obj_quat_wxyz = q_or[i, 3:]
        obj_quat_xyzw = obj_quat_wxyz[[1, 2, 3, 0]]
        R = Rotation.from_quat(obj_quat_xyzw).as_matrix()
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = obj_t
        
        T_corrected = T @ mesh_transform  # GigaHands와 동일한 방식
        q_or[i, :3] = T_corrected[:3, 3]
        # rotation은 변하지 않음 (순수 translation offset이므로)

    # 3. mesh를 centroid 기준으로 중심화하여 저장
    mesh_out = get_mesh_dir(dataset_dir, "hocap", obj_id)
    os.makedirs(mesh_out, exist_ok=True)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    ms.apply_filter(
        "compute_coord_by_function",
        x=f"x-({mesh_centroid[0]})",
        y=f"y-({mesh_centroid[1]})",
        z=f"z-({mesh_centroid[2]})",
    )
    ms.save_current_mesh(f"{mesh_out}/visual.obj")

    task_info = {
        "task": task_name,
        "dataset_name": "hocap",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": str(Path(mesh_out).relative_to(dataset_dir)),
        "left_object_mesh_dir": None,
        "ref_dt": 1 / 60,
    }

    with open(f"{output_dir}/../task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=q_wr,
        qpos_finger_right=q_fr,
        qpos_obj_right=q_or,
        qpos_wrist_left=q_wl,
        qpos_finger_left=q_fl,
        qpos_obj_left=np.zeros_like(q_or),
        contact=np.zeros((N, 10)),
    )

    loguru.logger.info(f"Saved raw trajectory: {output_dir}")

    if show_viewer:
        visualize_trajectory(q_wr, q_fr, q_wl, q_fl, q_or, f"{mesh_out}/visual.obj")

def visualize_trajectory(q_wr, q_fr, q_wl, q_fl, q_or, mesh_path):
    N = q_wr.shape[0]

    # 1. 모든 Mocap 데이터를 하나로 합치기 (순서가 매우 중요!)
    # XML 내부 순서: Right Wrist(1) -> Right Fingers(5) -> Left Wrist(1) -> Left Fingers(5)
    # 이후 동적으로 추가될 물체들: Right Object(1) -> Left Object(1)
    qpos_list = np.concatenate([
        q_wr[:, None],      # Right Wrist (N, 1, 7)
        q_fr,               # Right Fingers (N, 5, 7)
        q_wl[:, None],      # Left Wrist (N, 1, 7)
        q_fl,               # Left Fingers (N, 5, 7)
        q_or[:, None],      # Right Object (N, 1, 7)
        np.zeros_like(q_or[:, None]) # Left Object (N, 1, 7) - 자리 채우기용
    ], axis=1)

    # 2. MjSpec 설정 및 동적 바디 추가
    mj_spec = mujoco.MjSpec.from_file(f"{spider.ROOT}/assets/mano/empty_scene.xml")

    # Right Object Body 추가
    obj_right_handle = mj_spec.worldbody.add_body(name="right_object", mocap=True)
    
    # 참고 코드처럼 시각적 가이드를 위한 Site 추가 (반투명 빨간색 박스)
    obj_right_handle.add_site(
        name="right_object_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[1, 0, 0, 0.5],
        group=0,
    )

    # 실제 메쉬 Geom 추가
    mj_spec.add_mesh(name="right_object_mesh", file=str(mesh_path))
    obj_right_handle.add_geom(
        name="right_object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="right_object_mesh",
        pos=[0, 0, 0],
        quat=[1, 0, 0, 0],
        group=0,
        condim=1,
    )

    # Left Object Body 추가 (데이터 순서를 맞추기 위해 시각화하지 않더라도 추가 권장)
    obj_left_handle = mj_spec.worldbody.add_body(name="left_object", mocap=True)
    obj_left_handle.add_site(
        name="left_object_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[0, 1, 0, 0.2], # 연한 녹색
        group=0,
    )

    # 3. 모델 컴파일
    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)
    
    # 실제 XML에 로드된 Mocap 바디 개수에 맞춰 데이터 슬라이싱
    num_mocap = mj_model.nmocap
    qpos_list = qpos_list[:, :num_mocap, :]
    
    loguru.logger.info(f"Visualizing with {num_mocap} mocap bodies.")

    # 4. 시각화 루프
    rate_limiter = RateLimiter(60.0) # 60Hz 설정
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        cnt = 0
        while viewer.is_running():
            # Mocap 위치 및 회전 업데이트
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            
            cnt = (cnt + 1) % N
            rate_limiter.sleep()

def main(dataset_dir: str = f"{spider.ROOT}/../example_datasets"):
    dataset_dir = os.path.abspath(dataset_dir)
    loguru.logger.info("Processing trajectories using JSON keypoints...")
    
    for cfg in DATACONFIGS:
        try:
            process_config(cfg, dataset_dir, show_viewer=False)
        except Exception as e:
            loguru.logger.error(f"Failed to process {cfg['path']}: {e}")

if __name__ == "__main__":
    tyro.cli(main)