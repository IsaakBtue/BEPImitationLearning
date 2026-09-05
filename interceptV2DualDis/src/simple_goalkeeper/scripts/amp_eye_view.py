"""Render an NPZ reference clip re-anchored to whichever foot is currently
planted, instead of the way it looks in the file's own (real) root frame.

Motivation (see docs/BugFixes.md, 2026-08-01): AMP's observation for these
clips is `joint_pos`/`joint_vel` only (see `goalkeeper_multidisc_amp_cfg.py`'s
`_MULTIDISC_AMP_OBS_TERMS`) -- it never sees root orientation. A ghost
overlay that replays the file's real root quat (as `GhostMotionCommand`
correctly does -- see its own docstring, "world space, independent of the
robot") is an accurate rendering of the FILE, but it is easy to mistake for
"what AMP is being trained on," which it is not whenever a clip's root
orientation itself carries information (e.g. a deliberate torso lean) that
joint_pos alone can't express.

First attempt at this tool (superseded, do not resurrect) forced the ROOT
orientation to identity every frame. That is mathematically a valid way to
strip root information, but it silently broke this project's own
feet-always-flat invariant (every frame, both feet, tilt_xy ~ 0 -- see the
editing-amp-motion-data skill's Kinematic Chain Rule) because the recorded
joint angles were tuned assuming the REAL root, not an arbitrary level one.
The result looked like an unbalanced "leaning back" pose that had nothing to
do with the actual edit being checked, and was not a useful signal either
way -- AMP does not judge balance/lean at all (it has no orientation
reference), so a render that merely LOOKS unbalanced isn't informative
regardless of the reason.

This version anchors to the currently-PLANTED foot instead of the root.
Re-expressing every body's pose relative to any single body in the chain
(not just the root) is a rigid whole-scene transform -- it cancels the root
transform out completely (same math, different reference body), so it is
just as faithful an "AMP-eye view" as the root=identity version, but the
foot is already established-flat and grounded by construction, so the
rendering looks physically sensible: the planted foot stays visually still
and everything else (swing leg, torso, arms) moves relative to it, exactly
matching this project's real feet-flat convention. The tradeoff: since
stance alternates feet mid-clip, there is a visible jump in the render at
each foot-switch instant -- expected (same as any stance-relative gait
plot), not a bug.

Do not use the output of this module as new training data -- it is a
diagnostic rendering only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_XML = _REPO_ROOT / "src/simple_goalkeeper/robots/xmls/t1_headless.xml"

_HYSTERESIS_M = 0.015  # only switch anchor once the other foot is clearly lower, to avoid flicker
_CANONICAL_POS = np.array([0.0, 0.0, 0.030])
_CANONICAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity -> flat by construction (tilt_xy = 0)


def _finite_diff_vel(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def _quat_ang_vel(quats: np.ndarray, dt: float) -> np.ndarray:
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


def _quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([
        aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw,
    ])


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_rot(q_wxyz, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    r = _quat_mul(_quat_mul(q_wxyz, qv), _quat_conj(q_wxyz))
    return r[1:]


def make_amp_eye_view(
    src_path: Path | str,
    dst_path: Path | str,
    xml_path: Path | str = _DEFAULT_XML,
) -> None:
    """Write `dst_path` = `src_path` rigidly re-expressed, per frame, in the
    frame of whichever foot is currently planted (canonical: flat, at
    (0,0,0.03)). `joint_pos`/`joint_vel` (AMP's entire input) are untouched;
    every body's pose gets the SAME per-frame rigid transform applied, which
    exactly cancels the root transform and leaves a rendering driven purely
    by relative joint angles -- while staying visually grounded, since the
    anchor foot is already flat/planted by construction.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    lf_idx = names.index("left_foot_link") - 1
    rf_idx = names.index("right_foot_link") - 1

    d = np.load(str(src_path))
    fps = float(d["fps"])
    dt = 1.0 / fps
    body_pos_w = d["body_pos_w"]
    body_quat_w = d["body_quat_w"]
    T, n_robot_bodies, _ = body_pos_w.shape

    left_z = body_pos_w[:, lf_idx, 2]
    right_z = body_pos_w[:, rf_idx, 2]

    new_body_pos_w = np.zeros_like(body_pos_w)
    new_body_quat_w = np.zeros_like(body_quat_w)

    # Anchor to whichever foot is LOWER (closer to the ground = more likely
    # planted) rather than a fixed absolute-height threshold. A fixed
    # threshold is fragile -- some clips spend many consecutive frames with
    # neither foot below an arbitrary cutoff (e.g. both hovering in a
    # 0.030-0.040 m band during a brief double-support/near-ground phase),
    # which left the old threshold-based version frozen on whichever foot an
    # initial tie-break happened to pick, sometimes the wrong (swinging) one
    # for a long stretch. Small hysteresis avoids pure-noise flicker right at
    # a genuine crossover, where the anchor choice is inherently ambiguous.
    anchor_idx = lf_idx if left_z[0] <= right_z[0] else rf_idx
    for i in range(T):
        current_h = left_z[i] if anchor_idx == lf_idx else right_z[i]
        other_idx = rf_idx if anchor_idx == lf_idx else lf_idx
        other_h = left_z[i] if other_idx == lf_idx else right_z[i]
        if other_h < current_h - _HYSTERESIS_M:
            anchor_idx = other_idx

        anchor_pos = body_pos_w[i, anchor_idx, :]
        anchor_quat = body_quat_w[i, anchor_idx, :]

        # Solve for the rigid transform (extra_quat, extra_trans) that maps
        # anchor_pos/anchor_quat -> the canonical target:
        #   extra_quat * anchor_quat = CANONICAL_QUAT
        #   extra_trans + rot(extra_quat, anchor_pos) = CANONICAL_POS
        extra_quat = _quat_mul(_CANONICAL_QUAT, _quat_conj(anchor_quat))
        extra_trans = _CANONICAL_POS - _quat_rot(extra_quat, anchor_pos)

        for b in range(n_robot_bodies):
            new_body_quat_w[i, b] = _quat_mul(extra_quat, body_quat_w[i, b])
            new_body_pos_w[i, b] = extra_trans + _quat_rot(extra_quat, body_pos_w[i, b])

    new_body_lin_vel_w = _finite_diff_vel(new_body_pos_w, dt)
    new_body_ang_vel_w = np.zeros_like(new_body_lin_vel_w)
    for b in range(n_robot_bodies):
        new_body_ang_vel_w[:, b, :] = _quat_ang_vel(new_body_quat_w[:, b, :], dt)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dst_path),
        fps=np.array(fps),
        joint_pos=d["joint_pos"].astype(np.float32),
        joint_vel=d["joint_vel"].astype(np.float32),
        body_pos_w=new_body_pos_w.astype(np.float32),
        body_quat_w=new_body_quat_w.astype(np.float32),
        body_lin_vel_w=new_body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=new_body_ang_vel_w.astype(np.float32),
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--xml", type=Path, default=_DEFAULT_XML)
    args = parser.parse_args()
    make_amp_eye_view(args.input_file, args.output_file, xml_path=args.xml)
    print(f"[amp_eye_view] wrote {args.output_file} (re-anchored to the currently-planted foot)")


if __name__ == "__main__":
    cli()
