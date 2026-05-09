"""Convert lefthand_booster_t1.pkl → lefthand_t1.npz for mjlab MotionLoader."""
from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np

_PKL = Path("/home/isaak/BEP/ConvertData/GMRTRY/output/lefthand_booster_t1.pkl")
_XML = Path(__file__).parent.parent / "assets" / "booster_t1" / "T1_serial_clean.xml"
_OUT = Path(__file__).parent / "data" / "lefthand_t1.npz"


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


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

    fps: int = data["fps"]
    dt = 1.0 / fps
    root_pos = data["root_pos"].astype(np.float64)        # (T, 3)
    root_rot_xyzw = data["root_rot"].astype(np.float64)   # (T, 4) XYZW
    dof_pos = data["dof_pos"].astype(np.float64)          # (T, 23)
    T = root_pos.shape[0]

    # Shift root z up so that foot bottoms land slightly above ground level.
    # 0.080 = 0.058 (body center offset) + 0.021 (geom local z=-0.01 + radius=0.02 + 2mm margin) + 0.001 (1mm clearance)
    root_pos[:, 2] += 0.080

    # Rotate entire motion +90° around Z to match G1/mjlab orientation.
    # Root positions: (x, y) -> (-y, x)
    xy = root_pos[:, :2].copy()
    root_pos[:, 0] = -xy[:, 1]
    root_pos[:, 1] =  xy[:, 0]

    root_rot_wxyz = xyzw_to_wxyz(root_rot_xyzw)  # (T, 4) WXYZ

    # Rotate root quaternions: q_new = q_90 * q_original (wxyz convention)
    def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return np.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], axis=-1)

    q_90 = np.array([[0.7071068, 0.0, 0.0, 0.7071068]])  # +90° around Z (wxyz)
    root_rot_wxyz = quat_mul_wxyz(q_90, root_rot_wxyz)

    # The retargeted pkl has Left_Knee_Pitch (dof 14) and Right_Knee_Pitch (dof 20) near-zero
    # (straight knees), but the robot needs ~0.5 rad bend to stand stably. Fix by clamping
    # knees to a minimum before running kinematics so body_pos_w is consistent with bent legs.
    KNEE_INDICES = [14, 20]  # Left_Knee_Pitch, Right_Knee_Pitch in T1 joint order
    MIN_KNEE_BEND = 0.5  # radians — matches stable standing pose
    for ki in KNEE_INDICES:
        dof_pos[:, ki] = np.maximum(dof_pos[:, ki], MIN_KNEE_BEND)

    model = mujoco.MjModel.from_xml_path(str(_XML))
    mj_data = mujoco.MjData(model)
    assert model.nq == 30, f"nq mismatch: {model.nq}"
    assert model.nbody == 25, f"nbody mismatch: {model.nbody}"

    n_bodies = model.nbody - 1  # exclude world body
    body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)

    for t in range(T):
        mj_data.qpos[:3] = root_pos[t]
        mj_data.qpos[3:7] = root_rot_wxyz[t]
        mj_data.qpos[7:] = dof_pos[t]
        mujoco.mj_kinematics(model, mj_data)
        body_pos_w[t] = mj_data.xpos[1:].astype(np.float32)
        body_quat_w[t] = mj_data.xquat[1:].astype(np.float32)

    joint_pos = dof_pos.astype(np.float32)
    joint_vel = finite_diff(joint_pos, dt)
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
