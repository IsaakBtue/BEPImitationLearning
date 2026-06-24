"""Convert lefthand_booster_t1.pkl → lefthand_t1.npz for mjlab MotionLoader.

Resamples from source fps (30 Hz) to policy fps (50 Hz) and trims to 3.0 s
(150 frames), matching the original G1 lefthand episode length exactly.

Why: the pkl is 30 fps, 123 frames = 4.1 s of motion. mjlab plays npz frames
at policy dt (0.02 s = 50 Hz), so without resampling the motion plays in
123 × 0.02 = 2.46 s — 1.37× too fast. The original G1 lefthand.pt is also
123 frames but was consumed by AMP at 30 fps (4.1 s), with a 3 s episode
covering only the first 90 source frames. We resample to 50 Hz and keep the
first 3 s → 150 frames so timing matches the original episode structure.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np

_PKL = Path("/home/isaak/BEP/ConvertData/GMRTRY/output/lefthand_booster_t1.pkl")
_XML = Path(__file__).parent.parent / "assets" / "booster_t1" / "T1_serial_clean.xml"
_OUT = Path(__file__).parent / "data" / "lefthand_t1.npz"

# Policy fps = 1 / (sim_dt × decimation) = 1 / (0.005 × 4) = 50 Hz
TARGET_FPS: int = 50
TARGET_DURATION: float = 3.0          # seconds — matches original G1 episode length
TARGET_FRAMES: int = int(TARGET_DURATION * TARGET_FPS)  # 150


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def resample_linear(arr: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Linearly resample a (T, ...) array from t_src timestamps to t_tgt."""
    flat = arr.reshape(len(t_src), -1)
    out = np.stack([np.interp(t_tgt, t_src, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape(len(t_tgt), *arr.shape[1:])


def resample_quat(q: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Linearly resample (T, 4) WXYZ quaternions, keeping consistent hemisphere, then renormalize."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    out = resample_linear(q, t_src, t_tgt)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(arr)
    vel[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    vel[0] = (arr[1] - arr[0]) / dt
    vel[-1] = (arr[-1] - arr[-2]) / dt
    return vel


def quat_ang_vel(quats: np.ndarray, dt: float) -> np.ndarray:
    """Approximate angular velocity from WXYZ quaternion sequence."""
    T = quats.shape[0]
    ang_vel = np.zeros((T, *quats.shape[1:-1], 3))
    for t in range(T):
        t0 = max(0, t - 1)
        t1 = min(T - 1, t + 1)
        dt_eff = (t1 - t0) * dt
        if dt_eff == 0:
            continue
        q0 = quats[t0]
        q1 = quats[t1]
        w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        dx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        ang_vel[t] = (2.0 / dt_eff) * np.stack([dx, dy, dz], axis=-1)
    return ang_vel


def convert():
    with open(_PKL, "rb") as f:
        data = pickle.load(f)

    src_fps: int = data["fps"]
    src_T: int = data["root_pos"].shape[0]
    src_duration: float = src_T / src_fps
    print(f"Source: {src_T} frames @ {src_fps} fps = {src_duration:.3f} s")
    print(f"Target: {TARGET_FRAMES} frames @ {TARGET_FPS} fps = {TARGET_DURATION:.3f} s")

    root_pos = data["root_pos"].astype(np.float64)        # (T, 3)
    root_rot_xyzw = data["root_rot"].astype(np.float64)   # (T, 4) XYZW
    dof_pos = data["dof_pos"].astype(np.float64)          # (T, 23)

    # ── Resample from source fps to policy fps, trim to TARGET_DURATION ──────
    t_src = np.arange(src_T) / src_fps
    # Clamp target timestamps to source range (no extrapolation).
    t_tgt = np.clip(np.arange(TARGET_FRAMES) / TARGET_FPS, 0.0, t_src[-1])

    root_pos      = resample_linear(root_pos, t_src, t_tgt)
    root_rot_xyzw = resample_quat(root_rot_xyzw, t_src, t_tgt)
    dof_pos       = resample_linear(dof_pos, t_src, t_tgt)
    T = TARGET_FRAMES
    dt = 1.0 / TARGET_FPS

    # ── Root height correction ───────────────────────────────────────────────
    # 0.080 = 0.058 (body center offset) + 0.021 (geom local z + margin) + 0.001 clearance
    root_pos[:, 2] += 0.080

    # PKL data already faces +X (yaw ≈ -10°). No rotation needed.
    root_rot_wxyz = xyzw_to_wxyz(root_rot_xyzw)  # (T, 4) WXYZ

    # ── Knee clamping ────────────────────────────────────────────────────────
    # Retargeted pkl has near-zero knee bend; robot needs ≥0.5 rad to stand stably.
    KNEE_INDICES = [14, 20]  # Left_Knee_Pitch, Right_Knee_Pitch in T1 joint order
    MIN_KNEE_BEND = 0.5
    for ki in KNEE_INDICES:
        dof_pos[:, ki] = np.maximum(dof_pos[:, ki], MIN_KNEE_BEND)

    # ── Forward kinematics ───────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(_XML))
    mj_data = mujoco.MjData(model)
    assert model.nq == 30, f"nq mismatch: {model.nq}"
    assert model.nbody == 25, f"nbody mismatch: {model.nbody}"

    n_bodies = model.nbody - 1  # exclude world body
    body_pos_w  = np.zeros((T, n_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)

    for t in range(T):
        mj_data.qpos[:3] = root_pos[t]
        mj_data.qpos[3:7] = root_rot_wxyz[t]
        mj_data.qpos[7:] = dof_pos[t]
        mujoco.mj_kinematics(model, mj_data)
        body_pos_w[t]  = mj_data.xpos[1:].astype(np.float32)
        body_quat_w[t] = mj_data.xquat[1:].astype(np.float32)

    # ── Velocities via finite differences at policy dt ───────────────────────
    joint_pos      = dof_pos.astype(np.float32)
    joint_vel      = finite_diff(joint_pos, dt)
    body_lin_vel_w = finite_diff(body_pos_w, dt)
    body_ang_vel_w = quat_ang_vel(body_quat_w.astype(np.float64), dt).astype(np.float32)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _OUT,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Saved {_OUT}")
    d = np.load(_OUT)
    for k, v in d.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    convert()
