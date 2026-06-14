"""Convert Booster T1 goalkeeper PKL motions → NPZ for beyondAMP.

PKL format: {fps: int, root_pos: (T,3), root_rot: (T,4) xyzw, dof_pos: (T,23)}
NPZ output: fps, joint_pos (T,21), joint_vel (T,21), body_pos_w, body_quat_w,
            body_lin_vel_w, body_ang_vel_w  — all (T, num_bodies, 3/4)

Usage:
    uv run sgk_convert --input-dir /path/to/Motions --output-dir src/simple_goalkeeper/motions/data
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

_HERE = Path(__file__).parent
_XML = _HERE.parent / "robots" / "xmls" / "t1_headless.xml"

# Head joints (AAHead_yaw, Head_pitch) are first 2 DOFs — skip in headless output
_HEAD_JOINT_COUNT = 2

# T1 headless standing keyframe (HOME_KEYFRAME) in XML joint order.
# AMP discriminator obs uses joint_pos_rel = joint_pos - default; NPZ must match.
_T1_HEADLESS_DEFAULT_JOINT_POS = np.array([
    0.0,   # Left_Shoulder_Pitch
   -1.4,   # Left_Shoulder_Roll
    0.0,   # Left_Elbow_Pitch
   -0.4,   # Left_Elbow_Yaw
    0.0,   # Right_Shoulder_Pitch
    1.4,   # Right_Shoulder_Roll
    0.0,   # Right_Elbow_Pitch
    0.4,   # Right_Elbow_Yaw
    0.0,   # Waist
   -0.2,   # Left_Hip_Pitch
    0.0,   # Left_Hip_Roll
    0.0,   # Left_Hip_Yaw
    0.4,   # Left_Knee_Pitch
   -0.2,   # Left_Ankle_Pitch
    0.0,   # Left_Ankle_Roll
   -0.2,   # Right_Hip_Pitch
    0.0,   # Right_Hip_Roll
    0.0,   # Right_Hip_Yaw
    0.4,   # Right_Knee_Pitch
   -0.2,   # Right_Ankle_Pitch
    0.0,   # Right_Ankle_Roll
], dtype=np.float32)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert xyzw → wxyz (MuJoCo convention)."""
    return q[..., [3, 0, 1, 2]]


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply q1*q2, both in wxyz convention. Supports batched input."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _rotate_xy_minus90(pos: np.ndarray) -> np.ndarray:
    """Rotate XY positions -90° around Z: (x, y, z) → (y, -x, z).

    PKL motion data faces +Y (yaw ≈ 90°). Applying -90° around Z makes
    the robot face +X so that lateral goalkeeper saves appear as Y
    displacement in the world frame.
    """
    out = pos.copy()
    out[..., 0] =  pos[..., 1]   # x' = y
    out[..., 1] = -pos[..., 0]   # y' = -x
    return out


# -90° around Z: wxyz = [cos(-π/4), 0, 0, sin(-π/4)]
_Q_ROT_MINUS90Z = np.array([np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)], dtype=np.float64)


def _finite_diff_vel(pos: np.ndarray, dt: float) -> np.ndarray:
    """Central differences for velocity, forward/backward at endpoints."""
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def _quat_ang_vel(quats: np.ndarray, dt: float) -> np.ndarray:
    """Approximate angular velocity from quaternion sequence via log map."""
    ang_vel = np.zeros((len(quats), 3))
    for i in range(1, len(quats) - 1):
        q0 = quats[i - 1]
        q1 = quats[i + 1]
        # relative quat: q_rel = q1 * conj(q0)
        q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q0_conj
        qr = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        angle = 2.0 * np.arctan2(np.linalg.norm(qr[1:]), qr[0])
        axis_norm = np.linalg.norm(qr[1:])
        axis = qr[1:] / axis_norm if axis_norm > 1e-8 else np.zeros(3)
        ang_vel[i] = axis * angle / (2.0 * dt)
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def convert_one(
    pkl_path: Path,
    output_path: Path,
    output_fps: int = 50,
    speed_factor: float = 2.0,
) -> None:
    """Convert one PKL file.

    speed_factor compresses the time axis so the motion plays speed_factor× faster
    in simulation.  All source PKL files from this project are captured at 30 fps and
    show slow walking (feet ≤ 9 cm, trunk lateral velocity ≤ 0.6 m/s).  Compressing
    by 2× doubles all velocities and halves the motion duration, which:
      - gives the AMP discriminator a reference that demands more dynamic leg motion;
      - keeps the compressed clip (1–2 s) well within the 4 s training episode.
    Any future PKL files recorded from the same motion-capture source will be sped up
    automatically without additional flags.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    input_fps: int = data["fps"]
    root_pos = np.array(data["root_pos"], dtype=np.float32)        # (T, 3)
    root_rot_xyzw = np.array(data["root_rot"], dtype=np.float32)   # (T, 4) xyzw
    dof_pos = np.array(data["dof_pos"], dtype=np.float32)          # (T, 23)
    T_in = root_pos.shape[0]

    # Resample to output_fps, compressing the time axis by speed_factor.
    # t_out_compressed: the playback timeline at output_fps.
    # t_out_original:   the corresponding lookup positions in the original recording.
    # Velocities computed later via finite-diff on compressed positions will be
    # speed_factor× larger than the original, matching the faster playback.
    duration = (T_in - 1) / input_fps
    compressed_duration = duration / speed_factor
    t_out = np.arange(0, compressed_duration, 1.0 / output_fps)
    T_out = len(t_out)
    t_in = np.linspace(0, duration, T_in)
    t_out_orig = t_out * speed_factor   # map compressed time → original time for interp

    def resample(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((T_out, arr.shape[1]), dtype=np.float32)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(t_out_orig, t_in, arr[:, j])
        return out

    root_pos_r = resample(root_pos)
    root_rot_r = resample(root_rot_xyzw)
    root_rot_r /= np.linalg.norm(root_rot_r, axis=-1, keepdims=True).clip(min=1e-8)
    dof_pos_r = resample(dof_pos)

    root_rot_wxyz = _quat_xyzw_to_wxyz(root_rot_r)

    # ── Rotate PKL data from +Y-facing to +X-facing ──────────────────────────
    # PKL motions recorded with robot facing +Y (initial yaw ≈ 90°).
    # Apply -90° Z rotation so lateral goalkeeper saves become Y displacement.
    root_pos_r = _rotate_xy_minus90(root_pos_r)
    q_rot = _Q_ROT_MINUS90Z.reshape(1, 4).astype(np.float32)
    root_rot_wxyz = _quat_mul_wxyz(q_rot, root_rot_wxyz)

    # Snap frame-0 yaw to exactly 0° to remove any residual drift.
    w0, x0, y0, z0 = (root_rot_wxyz[0, i] for i in range(4))
    yaw0 = np.arctan2(2*(w0*z0 + x0*y0), 1 - 2*(y0*y0 + z0*z0))
    q_snap = np.array([[np.cos(-yaw0/2), 0.0, 0.0, np.sin(-yaw0/2)]], dtype=np.float32)
    root_rot_wxyz = _quat_mul_wxyz(q_snap, root_rot_wxyz)
    # Also apply the snap rotation to XY positions (rotate around Z by -yaw0).
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    xy = root_pos_r[:, :2].copy()
    root_pos_r[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    root_pos_r[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    print(f"  yaw0 before snap: {np.degrees(yaw0):+.2f}°")



    model = mujoco.MjModel.from_xml_path(str(_XML))
    mdata = mujoco.MjData(model)

    foot_names = ["left_foot_link", "right_foot_link"]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names]

    # headless XML has 21 joints (no head) → skip first 2 DOFs (AAHead_yaw, Head_pitch)
    dof_headless = dof_pos_r[:, _HEAD_JOINT_COUNT:]  # (T, 21)

    # Pass 1: find minimum foot height for z-correction
    min_foot_z = float("inf")
    for i in range(T_out):
        mdata.qpos[:3] = root_pos_r[i]
        mdata.qpos[3:7] = root_rot_wxyz[i]
        mdata.qpos[7:] = dof_headless[i]
        mujoco.mj_kinematics(model, mdata)
        for bid in foot_ids:
            min_foot_z = min(min_foot_z, float(mdata.xpos[bid, 2]))

    root_pos_r[:, 2] -= min_foot_z
    print(f"  z-correction: {-min_foot_z:+.4f} m")

    # Pass 2: FK to extract body kinematics.
    # Exclude world body (index 0) so local indices 0..nbody-2 map to Trunk..right_foot_link.
    # This matches mjlab Entity.find_bodies() which uses entity-local indices (0 = first non-world body).
    n_robot_bodies = model.nbody - 1
    body_pos_w = np.zeros((T_out, n_robot_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((T_out, n_robot_bodies, 4), dtype=np.float32)

    for i in tqdm(range(T_out), desc=pkl_path.stem, ncols=80):
        mdata.qpos[:3] = root_pos_r[i]
        mdata.qpos[3:7] = root_rot_wxyz[i]
        mdata.qpos[7:] = dof_headless[i]
        mujoco.mj_kinematics(model, mdata)
        body_pos_w[i] = mdata.xpos[1:].copy()   # skip world body (index 0)
        body_quat_w[i] = mdata.xquat[1:].copy()  # skip world body (index 0)

    body_lin_vel_w = _finite_diff_vel(body_pos_w, 1.0 / output_fps)

    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    for b in range(n_robot_bodies):
        body_ang_vel_w[:, b, :] = _quat_ang_vel(body_quat_w[:, b, :], 1.0 / output_fps)

    # Strip head joints → 21-DOF headless. Store absolute joint angles so that
    # MotionCommand._debug_vis_impl can write them directly to MuJoCo qpos and
    # the ghost robot renders in the correct pose. AMP obs uses joint_pos_abs
    # (absolute) on the robot side to match this convention.
    joint_pos = dof_pos_r[:, _HEAD_JOINT_COUNT:]
    joint_vel = _finite_diff_vel(joint_pos, 1.0 / output_fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        fps=np.array(output_fps),          # 0-d scalar — beyondAMP expects float(data["fps"])
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"  Saved → {output_path.name}  shape={joint_pos.shape}")


def main(
    input_dir: str = str(_HERE.parent / "motions" / "raw"),
    output_dir: str = str(_HERE.parent / "motions" / "data"),
    output_fps: int = 50,
    speed_factor: float = 2.0,
) -> None:
    """Convert all *.pkl in input_dir to NPZ files in output_dir.

    speed_factor: time-compression applied to every clip (default 2.0).
    Source PKL files from this project are 30 fps slow-walking recordings.
    2× compression doubles joint velocities, matching the faster movements
    needed for goalkeeper saves.  Pass --speed-factor 1.0 to disable.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    pkl_files = sorted(in_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in {in_dir}")
    print(f"Converting {len(pkl_files)} files @ {output_fps} fps  speed_factor={speed_factor}× → {out_dir}")
    for pkl in pkl_files:
        convert_one(pkl, out_dir / (pkl.stem + ".npz"), output_fps, speed_factor)
    print("Done.")


def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
