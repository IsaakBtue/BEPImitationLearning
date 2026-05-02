"""Convert Humanoid-Goalkeeper .pt motion files to mjlab NPZ format.

NPZ keys expected by mjlab's MotionLoader (tasks/tracking/mdp/commands.py):
  joint_pos:       (N, 29)     actuated joint positions in MuJoCo order
  joint_vel:       (N, 29)     actuated joint velocities
  body_pos_w:      (N, 31, 3)  all-body world-frame positions
  body_quat_w:     (N, 31, 4)  all-body quaternions (wxyz)
  body_lin_vel_w:  (N, 31, 3)  all-body linear velocities
  body_ang_vel_w:  (N, 31, 3)  all-body angular velocities
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

# MuJoCo actuated joint order (joints 1-29 in the G1 model; joint 0 is floating_base).
# This matches csv_to_npz.py's joint_names list exactly.
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",    # 0
    "left_hip_roll_joint",     # 1
    "left_hip_yaw_joint",      # 2
    "left_knee_joint",         # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",   # 5
    "right_hip_pitch_joint",   # 6
    "right_hip_roll_joint",    # 7
    "right_hip_yaw_joint",     # 8
    "right_knee_joint",        # 9
    "right_ankle_pitch_joint", # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",         # 12
    "waist_roll_joint",        # 13 (missing in .pt — zero-filled)
    "waist_pitch_joint",       # 14 (missing in .pt — zero-filled)
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19 (missing in .pt — zero-filled)
    "left_wrist_pitch_joint",     # 20 (missing in .pt — zero-filled)
    "left_wrist_yaw_joint",       # 21 (missing in .pt — zero-filled)
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26 (missing in .pt — zero-filled)
    "right_wrist_pitch_joint",    # 27 (missing in .pt — zero-filled)
    "right_wrist_yaw_joint",      # 28 (missing in .pt — zero-filled)
]  # 29 joints total

# Mapping: .pt column index → MuJoCo actuated joint index (0-based within the 29).
# joint_id.txt defines the 21 columns in the .pt files.
PT_TO_MUJOCO_IDX: dict[int, int] = {
    0: 0,   # left_hip_pitch_joint
    1: 1,   # left_hip_roll_joint
    2: 2,   # left_hip_yaw_joint
    3: 3,   # left_knee_joint
    4: 4,   # left_ankle_pitch_joint
    5: 5,   # left_ankle_roll_joint
    6: 6,   # right_hip_pitch_joint
    7: 7,   # right_hip_roll_joint
    8: 8,   # right_hip_yaw_joint
    9: 9,   # right_knee_joint
    10: 10, # right_ankle_pitch_joint
    11: 11, # right_ankle_roll_joint
    12: 12, # waist_yaw_joint
    13: 15, # left_shoulder_pitch_joint
    14: 16, # left_shoulder_roll_joint
    15: 17, # left_shoulder_yaw_joint
    16: 18, # left_elbow_joint
    17: 22, # right_shoulder_pitch_joint
    18: 23, # right_shoulder_roll_joint
    19: 24, # right_shoulder_yaw_joint
    20: 25, # right_elbow_joint
    # Joints 13, 14, 19, 20, 21, 26, 27, 28 remain at zero
}


def _pt_joints_to_mujoco(pt_joint_arr: np.ndarray) -> np.ndarray:
    """Map (N, 21) .pt joint array to (N, 29) MuJoCo actuated joint array."""
    n = pt_joint_arr.shape[0]
    out = np.zeros((n, 29), dtype=np.float32)
    for pt_idx, mj_idx in PT_TO_MUJOCO_IDX.items():
        out[:, mj_idx] = pt_joint_arr[:, pt_idx]
    return out


def _quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Batch quaternion multiply. Inputs shape (B, 4) in wxyz order."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _angular_velocity_from_quats(quats: np.ndarray, dt: float) -> np.ndarray:
    """Estimate angular velocity from quaternion sequence (central differences).

    Args:
        quats: (N, B, 4) wxyz quaternions
        dt: time step between frames

    Returns:
        (N, B, 3) angular velocity in world frame
    """
    n, b, _ = quats.shape
    ang_vel = np.zeros((n, b, 3), dtype=np.float32)

    for t in range(1, n - 1):
        q0 = quats[t - 1]  # (B, 4) wxyz
        q1 = quats[t + 1]  # (B, 4) wxyz
        # q0_inv in wxyz = [w, -x, -y, -z]
        q0_inv = q0 * np.array([[1, -1, -1, -1]], dtype=np.float32)
        q_diff = _quat_mul_batch(q0_inv, q1)  # (B, 4)
        # axis-angle from quaternion
        vec_norm = np.linalg.norm(q_diff[:, 1:], axis=-1)  # (B,)
        angle = 2.0 * np.arctan2(vec_norm, q_diff[:, 0])   # (B,)
        axis = q_diff[:, 1:] / (vec_norm[:, None] + 1e-8)  # (B, 3)
        ang_vel[t] = axis * (angle[:, None] / (2.0 * dt))

    # Clamp boundary frames
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def convert_pt_to_npz(pt_path: Path, npz_path: Path) -> None:
    """Convert one .pt motion file to mjlab NPZ format using MuJoCo FK."""
    raw: dict = torch.load(pt_path, map_location="cpu")

    base_pos = raw["base_position"].numpy().astype(np.float32)       # (N, 3) xyz
    base_quat_xyzw = raw["base_pose"].numpy().astype(np.float32)     # (N, 4) xyzw
    joint_pos_21 = raw["joint_position"].numpy().astype(np.float32)  # (N, 21)
    joint_vel_21 = raw["joint_velocity"].numpy().astype(np.float32)  # (N, 21)
    n = base_pos.shape[0]

    joint_pos_29 = _pt_joints_to_mujoco(joint_pos_21)
    joint_vel_29 = _pt_joints_to_mujoco(joint_vel_21)

    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    data = mujoco.MjData(model)
    num_bodies = model.nbody  # 31

    body_pos_w = np.zeros((n, num_bodies, 3), dtype=np.float32)
    body_quat_wxyz = np.zeros((n, num_bodies, 4), dtype=np.float32)

    for t in tqdm(range(n), desc=f"FK {pt_path.stem}", leave=False):
        # MuJoCo free-joint qpos layout: pos(3) + quat_wxyz(4) + dofs(29)
        data.qpos[:3] = base_pos[t]
        xyzw = base_quat_xyzw[t]
        data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]  # → wxyz
        data.qpos[7:36] = joint_pos_29[t]
        mujoco.mj_kinematics(model, data)
        body_pos_w[t] = data.xpos.copy()   # (nbody, 3)
        body_quat_wxyz[t] = data.xquat.copy()  # (nbody, 4) already wxyz

    dt = 1.0 / 30.0  # source fps
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_ang_vel_w = _angular_velocity_from_quats(body_quat_wxyz, dt)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(npz_path),
        joint_pos=joint_pos_29,
        joint_vel=joint_vel_29,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_wxyz,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"  Saved {npz_path.name}  (N={n} frames, {num_bodies} bodies)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Humanoid-Goalkeeper .pt motion files to mjlab NPZ format"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing *.pt files")
    parser.add_argument("output_dir", type=Path, help="Directory for output *.npz files")
    args = parser.parse_args()

    motion_names = ["lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep"]
    for name in motion_names:
        pt_path = args.input_dir / f"{name}.pt"
        npz_path = args.output_dir / f"{name}.npz"
        if pt_path.exists():
            convert_pt_to_npz(pt_path, npz_path)
        else:
            print(f"[WARN] {pt_path} not found, skipping")


if __name__ == "__main__":
    main()
