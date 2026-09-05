"""TEMPLATE -- adapt the constants below to the clip(s)/joints you're editing.
See SKILL.md (same directory) for the full method writeup before using this.

Combined pass: (1) damp swing-leg Hip_Pitch/Knee_Pitch toward T1 standing
neutral by some fraction, (2) solve per-frame for Ankle_Pitch/Ankle_Roll
corrections that restore exact foot flatness -- damping hip/knee moves the
foot's world orientation downstream in the kinematic chain even though the
ankle's own joint VALUES are untouched, since it's a serial chain.

Do NOT skip step (2) just because step (1) alone passes the usual sanity
checks (no NaN, no floor penetration, no velocity spike) -- those checks say
nothing about orientation. This exact mistake happened once already in this
project (2026-07-31): a hip/knee-only damping pass looked completely clean
by every existing check and still silently reintroduced the tilt the
original mocap's own leveling pass had removed. See SKILL.md's "Kinematic
Chain Rule" section.

Before running: confirm you're reading from the TRUE original clip, not an
already-corrected one (git-recover it if `motions/data/` has been
overwritten by a prior pass -- see SKILL.md step 1). Re-sourcing from an
already-corrected file stacks a nonlinear correction on top of itself.
"""
import math
import numpy as np
import mujoco
from pathlib import Path

REPO = Path("/home/isaak/BEPImitationlearning/interceptV2DualDis")
XML = REPO / "src/simple_goalkeeper/robots/xmls/t1_headless.xml"
DATA_DIR = REPO / "src/simple_goalkeeper/motions/data"
# TRUE, never-corrected source clips. If motions/data/ still holds the
# original (no prior correction pass touched it), point this at DATA_DIR
# instead. If it's already been corrected once, recover the true original
# via `git show <commit-before-your-edits>:<path> > SRC_DIR/<name>.npz`
# for each file first -- see SKILL.md step 1.
SRC_DIR = REPO / "src/simple_goalkeeper/motions/data"  # <-- ADAPT: point at true originals if DATA_DIR was already edited

# <-- ADAPT: which clips to process
FILES = [
    "example_clip_booster_t1.npz",
]

NEUTRAL_HIP = -0.3   # T1 HOME_KEYFRAME standing neutral -- adapt per joint being damped
NEUTRAL_KNEE = 0.6
DAMPING = 0.8        # fraction of excursion-from-neutral to remove -- <-- ADAPT, user-directed
CONTACT_Z = 0.032    # just above this project's grounded convention (foot z = 0.0300)
BLEND_FRAMES = 2

ANKLE_PITCH_RANGE = (-0.87, 0.35)
ANKLE_ROLL_RANGE = (-0.44, 0.44)
NEWTON_ITERS = 4
FD_EPS = 1e-3

JOINT_NAMES = [
    "Left_Shoulder_Pitch","Left_Shoulder_Roll","Left_Elbow_Pitch","Left_Elbow_Yaw",
    "Right_Shoulder_Pitch","Right_Shoulder_Roll","Right_Elbow_Pitch","Right_Elbow_Yaw",
    "Waist","Left_Hip_Pitch","Left_Hip_Roll","Left_Hip_Yaw","Left_Knee_Pitch",
    "Left_Ankle_Pitch","Left_Ankle_Roll","Right_Hip_Pitch","Right_Hip_Roll",
    "Right_Hip_Yaw","Right_Knee_Pitch","Right_Ankle_Pitch","Right_Ankle_Roll",
]
IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def _finite_diff_vel(pos, dt):
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def _quat_ang_vel(quats, dt):
    ang_vel = np.zeros((len(quats), 3))
    for i in range(1, len(quats) - 1):
        q0 = quats[i - 1]; q1 = quats[i + 1]
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


def smooth_mask(mask, width):
    kernel = np.ones(2 * width + 1) / (2 * width + 1)
    return np.clip(np.convolve(mask.astype(np.float64), kernel, mode="same"), 0.0, 1.0)


def _quat_rot(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([0.0, v[0], v[1], v[2]])

    def qmul(a, b):
        aw, ax, ay, az = a; bw, bx, by, bz = b
        return np.array([
            aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw,
        ])
    qc = np.array([w, -x, -y, -z])
    r = qmul(qmul(q_wxyz, qv), qc)
    return r[1:]


def tilt_xy(quat_wxyz):
    """Gravity vector expressed in the foot's local frame, XY components --
    (0, 0) means perfectly flat. Same metric rewards.py:feetorientation uses."""
    w, x, y, z = quat_wxyz
    q_inv = np.array([w, -x, -y, -z])
    g = _quat_rot(q_inv, np.array([0.0, 0.0, -1.0]))
    return g[:2]


def fk_foot_quat(model, mdata, root_pos, root_quat, joint_pos, foot_body_idx):
    mdata.qpos[:3] = root_pos
    mdata.qpos[3:7] = root_quat
    mdata.qpos[7:] = joint_pos
    mujoco.mj_kinematics(model, mdata)
    return mdata.xquat[foot_body_idx + 1].copy()  # +1: skip world body


def solve_flat_ankle(model, mdata, root_pos, root_quat, joint_pos, foot_body_idx,
                      pitch_j, roll_j, pitch_range, roll_range):
    """Newton-solve Ankle_Pitch/Ankle_Roll so the foot reads flat (tilt_xy = 0)."""
    jp = joint_pos.copy()
    for _ in range(NEWTON_ITERS):
        q0 = fk_foot_quat(model, mdata, root_pos, root_quat, jp, foot_body_idx)
        t0 = tilt_xy(q0)
        if np.linalg.norm(t0) < 1e-6:
            break

        jp_p = jp.copy(); jp_p[pitch_j] += FD_EPS
        t_p = tilt_xy(fk_foot_quat(model, mdata, root_pos, root_quat, jp_p, foot_body_idx))
        jp_r = jp.copy(); jp_r[roll_j] += FD_EPS
        t_r = tilt_xy(fk_foot_quat(model, mdata, root_pos, root_quat, jp_r, foot_body_idx))

        J = np.array([
            [(t_p[0] - t0[0]) / FD_EPS, (t_r[0] - t0[0]) / FD_EPS],
            [(t_p[1] - t0[1]) / FD_EPS, (t_r[1] - t0[1]) / FD_EPS],
        ])
        try:
            delta = np.linalg.solve(J, -t0)
        except np.linalg.LinAlgError:
            break
        jp[pitch_j] = np.clip(jp[pitch_j] + delta[0], *pitch_range)
        jp[roll_j] = np.clip(jp[roll_j] + delta[1], *roll_range)

    return jp[pitch_j], jp[roll_j]


def process(model, lf_idx, rf_idx, src_path: Path, dst_path: Path) -> None:
    d = np.load(src_path)
    fps = float(d["fps"])
    dt = 1.0 / fps
    joint_pos = d["joint_pos"].copy()
    body_pos_w = d["body_pos_w"]
    T = joint_pos.shape[0]

    left_swing_raw = body_pos_w[:, lf_idx, 2] > CONTACT_Z
    right_swing_raw = body_pos_w[:, rf_idx, 2] > CONTACT_Z
    left_w = smooth_mask(left_swing_raw, BLEND_FRAMES)
    right_w = smooth_mask(right_swing_raw, BLEND_FRAMES)

    def damp(col, neutral, w):
        j = IDX[col]
        old = joint_pos[:, j].copy()
        target = neutral + (1.0 - DAMPING) * (old - neutral)
        joint_pos[:, j] = old + w * (target - old)

    damp("Left_Hip_Pitch", NEUTRAL_HIP, left_w)
    damp("Left_Knee_Pitch", NEUTRAL_KNEE, left_w)
    damp("Right_Hip_Pitch", NEUTRAL_HIP, right_w)
    damp("Right_Knee_Pitch", NEUTRAL_KNEE, right_w)

    # --- NEW: restore exact foot flatness by solving Ankle_Pitch/Ankle_Roll ---
    root_pos = d["body_pos_w"][:, 0, :]
    root_quat = d["body_quat_w"][:, 0, :]
    mdata = mujoco.MjData(model)

    lp_j, lr_j = IDX["Left_Ankle_Pitch"], IDX["Left_Ankle_Roll"]
    rp_j, rr_j = IDX["Right_Ankle_Pitch"], IDX["Right_Ankle_Roll"]

    max_tilt_before = 0.0
    for i in range(T):
        q0 = fk_foot_quat(model, mdata, root_pos[i], root_quat[i], joint_pos[i], lf_idx)
        max_tilt_before = max(max_tilt_before, float(np.linalg.norm(tilt_xy(q0))))
        new_p, new_r = solve_flat_ankle(model, mdata, root_pos[i], root_quat[i], joint_pos[i],
                                         lf_idx, lp_j, lr_j, ANKLE_PITCH_RANGE, ANKLE_ROLL_RANGE)
        joint_pos[i, lp_j] = new_p
        joint_pos[i, lr_j] = new_r

        q0 = fk_foot_quat(model, mdata, root_pos[i], root_quat[i], joint_pos[i], rf_idx)
        max_tilt_before = max(max_tilt_before, float(np.linalg.norm(tilt_xy(q0))))
        new_p, new_r = solve_flat_ankle(model, mdata, root_pos[i], root_quat[i], joint_pos[i],
                                         rf_idx, rp_j, rr_j, ANKLE_PITCH_RANGE, ANKLE_ROLL_RANGE)
        joint_pos[i, rp_j] = new_p
        joint_pos[i, rr_j] = new_r

    joint_vel = _finite_diff_vel(joint_pos, dt)

    n_robot_bodies = model.nbody - 1
    new_body_pos_w = np.zeros((T, n_robot_bodies, 3), dtype=np.float32)
    new_body_quat_w = np.zeros((T, n_robot_bodies, 4), dtype=np.float32)
    max_tilt_after = 0.0
    for i in range(T):
        mdata.qpos[:3] = root_pos[i]
        mdata.qpos[3:7] = root_quat[i]
        mdata.qpos[7:] = joint_pos[i]
        mujoco.mj_kinematics(model, mdata)
        new_body_pos_w[i] = mdata.xpos[1:].copy()
        new_body_quat_w[i] = mdata.xquat[1:].copy()
        max_tilt_after = max(max_tilt_after,
                              float(np.linalg.norm(tilt_xy(new_body_quat_w[i, lf_idx]))),
                              float(np.linalg.norm(tilt_xy(new_body_quat_w[i, rf_idx]))))

    new_body_lin_vel_w = _finite_diff_vel(new_body_pos_w, dt)
    new_body_ang_vel_w = np.zeros_like(new_body_lin_vel_w)
    for b in range(n_robot_bodies):
        new_body_ang_vel_w[:, b, :] = _quat_ang_vel(new_body_quat_w[:, b, :], dt)

    contact_min = min(new_body_pos_w[:, lf_idx, 2].min(), new_body_pos_w[:, rf_idx, 2].min())
    z_shift = 0.030 - contact_min
    new_body_pos_w[:, :, 2] += z_shift

    print(f"\n=== {src_path.name} ===")
    print(f"  max foot tilt BEFORE ankle fix: {max_tilt_before:.4f}   AFTER: {max_tilt_after:.6f}")
    print(f"  re-level shift: {z_shift:+.4f} m  |  post-fix foot-z min L={new_body_pos_w[:, lf_idx, 2].min():.4f} R={new_body_pos_w[:, rf_idx, 2].min():.4f}")
    for col in ["Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Roll"]:
        j = IDX[col]
        old = d["joint_pos"][:, j]
        new = joint_pos[:, j]
        print(f"  {col:16s} mean {math.degrees(old.mean()):+6.1f} -> {math.degrees(new.mean()):+6.1f}   "
              f"min {math.degrees(old.min()):+6.1f} -> {math.degrees(new.min()):+6.1f}   "
              f"max {math.degrees(old.max()):+6.1f} -> {math.degrees(new.max()):+6.1f}")
    print(f"  nan check: joint_pos={np.isnan(joint_pos).any()} body_pos_w={np.isnan(new_body_pos_w).any()}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dst_path),
        fps=np.array(fps),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=new_body_pos_w,
        body_quat_w=new_body_quat_w,
        body_lin_vel_w=new_body_lin_vel_w,
        body_ang_vel_w=new_body_ang_vel_w,
    )
    print(f"  saved -> {dst_path}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    lf_idx = names.index("left_foot_link") - 1
    rf_idx = names.index("right_foot_link") - 1

    for fname in FILES:
        src = SRC_DIR / fname
        dst = DATA_DIR / fname
        process(model, lf_idx, rf_idx, src, dst)


if __name__ == "__main__":
    main()
