"""Convert all 6 Booster T1 pkl motion files to npz for AMP training.

Source pkl files: src/imitationlearningbooster/motions/data_pkl/{name}_booster_t1.pkl
Output npz files: src/imitationlearningbooster/motions/data/{name}_t1.npz
Each npz has: joint_pos (150,23) float32 and body kinematics (150, N_bodies, ...) float32.
"""
import os
import pickle
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path

TARGET_FPS = 50
TARGET_DURATION = 3.0
TARGET_FRAMES = int(TARGET_FPS * TARGET_DURATION)  # 150
ROOT_HEIGHT_OFFSET = 0.080
MIN_KNEE_BEND = 0.5
KNEE_INDICES = [14, 20]          # Left_Knee_Pitch, Right_Knee_Pitch
ANKLE_L_IDX, HIP_L_IDX = 15, 11  # Left_Ankle_Pitch, Left_Hip_Pitch
ANKLE_R_IDX, HIP_R_IDX = 21, 17  # Right_Ankle_Pitch, Right_Hip_Pitch

_HERE = Path(__file__).parent
_PKL_DIR = _HERE.parent.parent.parent  # Imitationlearningbooster/ root
_XML = _HERE.parent / "assets" / "booster_t1" / "T1_serial_clean.xml"
_OUT = _HERE / "data"
_OUT.mkdir(exist_ok=True)

MOTIONS = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]


def slerp_quat(q0, q1, t):
    r0 = Rotation.from_quat(q0[[1, 2, 3, 0]])  # wxyz → xyzw
    r1 = Rotation.from_quat(q1[[1, 2, 3, 0]])
    slerper = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    return slerper(t).as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz


def resample(arr, src_fps, tgt_frames):
    src_frames = len(arr)
    t_src = np.linspace(0, 1, src_frames)
    t_dst = np.linspace(0, 1, tgt_frames)
    if arr.ndim == 1:
        return np.interp(t_dst, t_src, arr).astype(np.float32)
    return np.stack([np.interp(t_dst, t_src, arr[:, i]) for i in range(arr.shape[1])], axis=1).astype(np.float32)


def rotate_z90(root_pos, root_rot_wxyz):
    """No-op: PKL data already faces +X (yaw ≈ -10°). No rotation needed."""
    return root_pos, root_rot_wxyz


def apply_foot_fix(dof_pos):
    """Keep feet level after knee clamping: ankle = -(hip_pitch + knee_pitch)."""
    dof_pos[:, ANKLE_L_IDX] = np.clip(
        -(dof_pos[:, HIP_L_IDX] + dof_pos[:, KNEE_INDICES[0]]), -0.87, 0.35
    )
    dof_pos[:, ANKLE_R_IDX] = np.clip(
        -(dof_pos[:, HIP_R_IDX] + dof_pos[:, KNEE_INDICES[1]]), -0.87, 0.35
    )
    return dof_pos


def convert_one(name: str):
    pkl_path = _PKL_DIR / f"{name}_booster_t1.pkl"
    out_path = _OUT / f"{name}_t1.npz"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    src_fps = data["fps"]
    root_pos = np.array(data["root_pos"], dtype=np.float64)   # (T, 3)
    root_rot = np.array(data["root_rot"], dtype=np.float64)   # (T, 4) XYZW
    dof_pos = np.array(data["dof_pos"], dtype=np.float64)     # (T, 23)

    # Convert root_rot from XYZW → WXYZ
    root_rot_wxyz = root_rot[:, [3, 0, 1, 2]].copy()

    # Resample to TARGET_FRAMES
    root_pos = resample(root_pos, src_fps, TARGET_FRAMES)
    root_rot_wxyz = np.stack([
        slerp_quat(root_rot_wxyz[int(i * (len(root_rot_wxyz) - 1) / (TARGET_FRAMES - 1))],
                   root_rot_wxyz[min(int(i * (len(root_rot_wxyz) - 1) / (TARGET_FRAMES - 1)) + 1, len(root_rot_wxyz) - 1)],
                   (i * (len(root_rot_wxyz) - 1) / (TARGET_FRAMES - 1)) % 1.0)
        for i in range(TARGET_FRAMES)
    ])
    dof_pos = resample(dof_pos, src_fps, TARGET_FRAMES)

    # Root height correction
    root_pos[:, 2] += ROOT_HEIGHT_OFFSET

    # Rotate -90° around Z: PKL faces +Y → NPZ faces +X.
    root_pos, root_rot_wxyz = rotate_z90(root_pos, root_rot_wxyz)

    # Clamp knee bending
    dof_pos[:, KNEE_INDICES] = np.maximum(dof_pos[:, KNEE_INDICES], MIN_KNEE_BEND)

    # NOTE: apply_foot_fix() is intentionally NOT called here.
    # The deployed model_2000.pt was trained on reference npz files (convert_booster.py)
    # which do NOT apply the foot fix. Keeping the same preprocessing ensures AMP
    # fine-tuning is consistent with the baseline policy's learned dynamics.

    # Run MuJoCo FK to get body kinematics
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data_mj = mujoco.MjData(model)
    # Extract joint names in MuJoCo XML order (skip freejoint at index 0).
    joint_names = np.array([model.joint(i).name for i in range(model.njnt)
                            if model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE])
    n_bodies = model.nbody - 1  # exclude worldbody (index 0)
    body_pos_w = np.zeros((TARGET_FRAMES, n_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((TARGET_FRAMES, n_bodies, 4), dtype=np.float32)  # wxyz
    body_lin_vel_w = np.zeros((TARGET_FRAMES, n_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((TARGET_FRAMES, n_bodies, 3), dtype=np.float32)

    for t in range(TARGET_FRAMES):
        # freejoint: qpos[0:3]=pos, qpos[3:7]=wxyz
        data_mj.qpos[:3] = root_pos[t]
        data_mj.qpos[3:7] = root_rot_wxyz[t]
        data_mj.qpos[7:] = dof_pos[t]
        if t > 0:
            dt = 1.0 / TARGET_FPS
            data_mj.qvel[:3] = (root_pos[t] - root_pos[t-1]) / dt
            data_mj.qvel[6:] = (dof_pos[t] - dof_pos[t-1]) / dt
        mujoco.mj_kinematics(model, data_mj)
        body_pos_w[t] = data_mj.xpos[1:].astype(np.float32)   # exclude worldbody at index 0
        body_quat_w[t] = data_mj.xquat[1:].astype(np.float32)  # exclude worldbody at index 0

    # Finite-diff velocities for bodies
    for t in range(1, TARGET_FRAMES):
        dt = 1.0 / TARGET_FPS
        body_lin_vel_w[t] = (body_pos_w[t] - body_pos_w[t-1]) / dt
    body_lin_vel_w[0] = body_lin_vel_w[1]

    # Shift so the lowest foot at frame 0 sits 2mm above ground.
    # Using global min would leave jump motions floating at frame 0 (foot nadir
    # occurs mid-jump, not at the start pose).
    foot_body_ids = [i - 1 for i in range(1, model.nbody)
                     if "foot" in model.body(i).name.lower()]
    foot_z_frame0 = body_pos_w[0, foot_body_ids, 2].min()
    shift = foot_z_frame0 - 0.002
    root_pos[:, 2] -= shift
    body_pos_w[:, :, 2] -= shift

    np.savez(
        out_path,
        joint_pos=dof_pos.astype(np.float32),
        joint_vel=np.gradient(dof_pos, 1.0 / TARGET_FPS, axis=0).astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=joint_names,
    )
    print(f"Saved {out_path}  joint_pos: {dof_pos.shape}")


if __name__ == "__main__":
    for name in MOTIONS:
        print(f"Converting {name}...")
        convert_one(name)
    print("Done.")
