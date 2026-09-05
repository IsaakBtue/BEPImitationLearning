"""Mirror a T1 headless NPZ motion clip left<->right.

The transform is its own inverse: mirroring a left-side clip produces a
right-side clip and vice versa. See
docs/superpowers/specs/2026-07-26-motion-mirror-tool-design.md for the full
derivation (mirror plane = world XZ, i.e. Y negates).

Usage:
    uv run sgk_mirror <input.npz> <output.npz>
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import tyro

from simple_goalkeeper.scripts.pkl_to_npz import _finite_diff_vel, _quat_ang_vel, _XML

# 21-DOF headless joint order (t1_headless.xml declaration order). For each
# slot: (mirror_slot_index, negate). Pitch joints (axis Y) are unchanged;
# Roll/Yaw joints (axis X/Z) negate. Left_X <-> Right_X slots swap; Waist
# negates in place. Derived and cross-checked in the design doc above.
_MIRROR_MAP: list[tuple[int, bool]] = [
    (4, False),   # 0  Left_Shoulder_Pitch  <-> Right_Shoulder_Pitch
    (5, True),    # 1  Left_Shoulder_Roll   <-> Right_Shoulder_Roll
    (6, False),   # 2  Left_Elbow_Pitch     <-> Right_Elbow_Pitch
    (7, True),    # 3  Left_Elbow_Yaw       <-> Right_Elbow_Yaw
    (0, False),   # 4  Right_Shoulder_Pitch <-> Left_Shoulder_Pitch
    (1, True),    # 5  Right_Shoulder_Roll  <-> Left_Shoulder_Roll
    (2, False),   # 6  Right_Elbow_Pitch    <-> Left_Elbow_Pitch
    (3, True),    # 7  Right_Elbow_Yaw      <-> Left_Elbow_Yaw
    (8, True),    # 8  Waist (self, negate)
    (15, False),  # 9  Left_Hip_Pitch       <-> Right_Hip_Pitch
    (16, True),   # 10 Left_Hip_Roll        <-> Right_Hip_Roll
    (17, True),   # 11 Left_Hip_Yaw         <-> Right_Hip_Yaw
    (18, False),  # 12 Left_Knee_Pitch      <-> Right_Knee_Pitch
    (19, False),  # 13 Left_Ankle_Pitch     <-> Right_Ankle_Pitch
    (20, True),   # 14 Left_Ankle_Roll      <-> Right_Ankle_Roll
    (9, False),   # 15 Right_Hip_Pitch      <-> Left_Hip_Pitch
    (10, True),   # 16 Right_Hip_Roll       <-> Left_Hip_Roll
    (11, True),   # 17 Right_Hip_Yaw        <-> Left_Hip_Yaw
    (12, False),  # 18 Right_Knee_Pitch     <-> Left_Knee_Pitch
    (13, False),  # 19 Right_Ankle_Pitch    <-> Left_Ankle_Pitch
    (14, True),   # 20 Right_Ankle_Roll     <-> Left_Ankle_Roll
]
assert len(_MIRROR_MAP) == 21


def _mirror_joint_array(arr: np.ndarray) -> np.ndarray:
    """Mirror (T, 21) joint_pos or joint_vel: swap L<->R slots, negate Roll/Yaw."""
    out = np.zeros_like(arr)
    for src, (dst, negate) in enumerate(_MIRROR_MAP):
        out[:, dst] = -arr[:, src] if negate else arr[:, src]
    return out


def _mirror_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Mirror (..., 4) wxyz quaternions across the world XZ plane: (w,x,y,z) -> (w,-x,y,-z)."""
    out = q.copy()
    out[..., 1] = -q[..., 1]
    out[..., 3] = -q[..., 3]
    return out


def mirror_npz(input_path: Path, output_path: Path) -> None:
    d = np.load(str(input_path))
    fps = float(d["fps"])
    joint_pos = d["joint_pos"]  # (T, 21)
    root_pos = d["body_pos_w"][:, 0, :].copy()    # (T, 3)
    root_quat = d["body_quat_w"][:, 0, :].copy()  # (T, 4) wxyz

    joint_pos_m = _mirror_joint_array(joint_pos)
    joint_vel_m = _finite_diff_vel(joint_pos_m, 1.0 / fps)

    root_pos_m = root_pos.copy()
    root_pos_m[:, 1] = -root_pos_m[:, 1]
    root_quat_m = _mirror_quat_wxyz(root_quat)
    root_quat_m /= np.linalg.norm(root_quat_m, axis=-1, keepdims=True).clip(min=1e-8)

    # Recompute full-body kinematics via FK, exactly like pkl_to_npz.py's Pass 2,
    # so every body (not just the trunk) is derived consistently -- no manual
    # per-body index swapping (e.g. left_foot_link <-> right_foot_link) needed.
    model = mujoco.MjModel.from_xml_path(str(_XML))
    mdata = mujoco.MjData(model)
    n_robot_bodies = model.nbody - 1
    T = joint_pos_m.shape[0]
    body_pos_w = np.zeros((T, n_robot_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, n_robot_bodies, 4), dtype=np.float32)

    for i in range(T):
        mdata.qpos[:3] = root_pos_m[i]
        mdata.qpos[3:7] = root_quat_m[i]
        mdata.qpos[7:] = joint_pos_m[i]
        mujoco.mj_kinematics(model, mdata)
        body_pos_w[i] = mdata.xpos[1:].copy()
        body_quat_w[i] = mdata.xquat[1:].copy()

    body_lin_vel_w = _finite_diff_vel(body_pos_w, 1.0 / fps)
    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    for b in range(n_robot_bodies):
        body_ang_vel_w[:, b, :] = _quat_ang_vel(body_quat_w[:, b, :], 1.0 / fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        fps=np.array(fps),
        joint_pos=joint_pos_m.astype(np.float32),
        joint_vel=joint_vel_m.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Mirrored {input_path.name} -> {output_path.name}  ({T} frames)")


def main(input_file: str, output_file: str) -> None:
    """Mirror a T1 headless NPZ motion clip left<->right (self-inverse transform)."""
    mirror_npz(Path(input_file), Path(output_file))


def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
