"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Investigates the reported "toes-up lift" exploit on the leading (ball-
assigned) foot: does its roll/pitch angular velocity spike throughout the
whole episode, or is it concentrated in a narrow window around the save
event (_ball_is_behind onset)? Real telemetry to inform a smarter gate on
`foot_ang_vel_xy` than the blunt, unconditional -3.0-on-both-feet attempt
that was reverted back to -0.25 (commit dceab30, see docs/BugFixes.md,
"2026-08-13 (same session) -- full revert...").

Does NOT import, call, or modify anything in rewards.py or
goalkeeper_env_cfg.py beyond read-only helper functions
(`_get_correct_foot_idx`, `_ball_is_behind`) that are pure state readers
with no side effects on training. Builds the exact same env/policy-loading
path as `sgk_play` (see scripts/play.py:run_play) for the multi-disc task,
then runs a plain step loop (no viewer) to collect per-step telemetry for
the assigned foot across many episodes, and writes the raw data to a CSV in
the scratchpad (not the repo) for reproducible offline analysis/plotting.

EXTENDED (follow-up): the first pass only measured the foot BODY's
world-frame angular velocity, which in a serial kinematic chain is the sum
of every upstream joint's contribution -- it can't distinguish "ankle
flexing on its own" from "whole leg swinging while the ankle stays rigid
relative to the shank." This version additionally logs, for the assigned
leg only: joint-space velocities for Hip_Pitch/Hip_Roll/Knee_Pitch/
Ankle_Pitch/Ankle_Roll (`robot.data.joint_vel`, real DOF rates, not world
body ang vel), plus the foot's ABSOLUTE static tilt off flat in degrees
(reusing `rewards.py:feetorientation`'s exact gravity-vs-foot-Z-axis math,
converted from that reward's squared-error form to a signed angle).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_foot_rotation.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_39750.pt \\
        --num-envs 128 --steps 4000 \\
        --out-csv /path/to/scratchpad/foot_rotation_probe.csv
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
    _get_actor_current_obs,
)

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/intercept_simple_goalkeeper_multidisc/"
    "2026-08-13_22-52-47_6144_footangvelrevert_2026-08-13/model_39750.pt"
)


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    num_envs: int = 128
    steps: int = 4000
    device: str | None = None
    out_csv: str = "/tmp/foot_rotation_probe.csv"
    min_ep_len: int = 10
    """Drop episode segments shorter than this many steps (boundary junk at
    the very start/end of the rollout, before the first / after the last
    reset falls inside the collection window)."""


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)
    # Imported after task registration, matching play.py's own lazy-import
    # ordering for anything that reaches into simple_goalkeeper.mdp.
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx, _ball_is_behind

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict), (
        f"Expected multi-disc dict rl_cfg for '{TASK_ID}', got {type(agent_cfg).__name__}."
    )

    env_cfg.scene.num_envs = cfg.num_envs
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    # Mirrors run_play's multi-disc branch exactly: motion_dataset=None
    # (multi-disc RSI/motion loading never reads AMPEnvWrapper.motion_dataset).
    env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

    resume_path = Path(cfg.checkpoint)
    if not resume_path.is_absolute() and not resume_path.exists():
        resume_path = Path.cwd() / resume_path
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    print(f"[INFO] Loading checkpoint: {resume_path}", file=sys.stderr)

    runner_cls = load_runner_cls(TASK_ID)
    runner = runner_cls(env, agent_cfg, log_dir=None, device=device)
    runner.load(str(resume_path), load_optimizer=False)
    act_inference = runner.get_inference_policy(device=device)

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    feet_contact = raw_env.scene["feet_contact"]
    foot_body_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]

    # Joint-space DOF rates for the 5 sagittal/frontal leg joints, both sides
    # -- order fixed so column k is the same joint for left (idx 0) and right
    # (idx 1) at that position: Hip_Pitch, Hip_Roll, Knee_Pitch, Ankle_Pitch,
    # Ankle_Roll. Matches rewards.py's own "Left_Hip_Roll, Left_Hip_Yaw,
    # Left_Hip_Pitch, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll"
    # naming convention (t1_headless.xml) -- Hip_Yaw deliberately excluded,
    # not part of this follow-up's roll/pitch question. T1's ankle has only
    # Ankle_Pitch/Ankle_Roll (no ankle yaw DOF), confirmed in t1_headless.xml.
    _JOINT_LABELS = ["hip_pitch", "hip_roll", "knee_pitch", "ankle_pitch", "ankle_roll"]
    _LEFT_JOINTS = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll"]
    _RIGHT_JOINTS = ["Right_Hip_Pitch", "Right_Hip_Roll", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"]
    left_joint_ids = robot.find_joints(_LEFT_JOINTS)[0]
    right_joint_ids = robot.find_joints(_RIGHT_JOINTS)[0]
    assert len(left_joint_ids) == 5 and len(right_joint_ids) == 5, (
        f"Expected 5 joints/side, got left={left_joint_ids}, right={right_joint_ids}"
    )

    gravity_w_unit = torch.tensor([0.0, 0.0, -1.0], device=device)

    N = cfg.num_envs
    arange_n = torch.arange(N, device=device)

    obs, _ = env.reset()

    roll_hist = np.zeros((cfg.steps, N), dtype=np.float32)
    pitch_hist = np.zeros((cfg.steps, N), dtype=np.float32)
    height_hist = np.zeros((cfg.steps, N), dtype=np.float32)
    contact_hist = np.zeros((cfg.steps, N), dtype=bool)
    behind_hist = np.zeros((cfg.steps, N), dtype=bool)
    eplen_hist = np.zeros((cfg.steps, N), dtype=np.int32)
    footidx_hist = np.zeros((cfg.steps, N), dtype=np.int8)
    joint_vel_hist = np.zeros((cfg.steps, N, 5), dtype=np.float32)  # hip_pitch,hip_roll,knee_pitch,ankle_pitch,ankle_roll
    tilt_deg_hist = np.zeros((cfg.steps, N), dtype=np.float32)

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            foot_idx = _get_correct_foot_idx(raw_env, "ball")  # (N,) 0=left,1=right
            ang_vel_w = robot.data.body_link_ang_vel_w[:, foot_body_ids, :]  # (N,2,3)
            roll = ang_vel_w[arange_n, foot_idx, 0]
            pitch = ang_vel_w[arange_n, foot_idx, 1]

            foot_pos_w = robot.data.body_link_pos_w[:, foot_body_ids, :]  # (N,2,3)
            height = foot_pos_w[arange_n, foot_idx, 2] - raw_env.scene.env_origins[:, 2]

            # Joint-space DOF rates for the assigned leg only -- select the
            # Left_* or Right_* joint column per env based on foot_idx.
            left_jvel = robot.data.joint_vel[:, left_joint_ids]    # (N, 5)
            right_jvel = robot.data.joint_vel[:, right_joint_ids]  # (N, 5)
            assigned_jvel = torch.where(
                (foot_idx == 0)[:, None], left_jvel, right_jvel
            )  # (N, 5): hip_pitch,hip_roll,knee_pitch,ankle_pitch,ankle_roll

            # Absolute static tilt off flat, assigned foot only -- reuses
            # rewards.py:feetorientation's exact gravity-vs-foot-Z-axis math
            # (gravity expressed in the foot's local frame), converted from
            # that reward's squared-XY-error form to a signed tilt angle in
            # degrees: theta = arccos(-gravity_foot_z), 0 = perfectly flat.
            feet_quat_w = robot.data.body_link_quat_w[:, foot_body_ids, :]  # (N,2,4)
            assigned_quat_w = feet_quat_w[arange_n, foot_idx]  # (N,4)
            gravity_w_exp = gravity_w_unit.expand(N, -1)
            gravity_foot = quat_apply(quat_inv(assigned_quat_w), gravity_w_exp)  # (N,3)
            cos_theta = (-gravity_foot[:, 2]).clamp(-1.0, 1.0)
            tilt_deg = torch.rad2deg(torch.acos(cos_theta))  # (N,)

            found = feet_contact.data.found  # (N, 8): [:4]=left sensors, [4:]=right
            left_contact = (found[:, :4] > 0).any(dim=-1)
            right_contact = (found[:, 4:] > 0).any(dim=-1)
            contact = torch.where(foot_idx == 0, left_contact, right_contact)

            behind = _ball_is_behind(raw_env, "ball")
            ep_len = raw_env.episode_length_buf

            roll_hist[step] = roll.detach().cpu().numpy()
            pitch_hist[step] = pitch.detach().cpu().numpy()
            height_hist[step] = height.detach().cpu().numpy()
            contact_hist[step] = contact.detach().cpu().numpy()
            behind_hist[step] = behind.detach().cpu().numpy()
            eplen_hist[step] = ep_len.detach().cpu().numpy()
            footidx_hist[step] = foot_idx.detach().cpu().numpy()
            joint_vel_hist[step] = assigned_jvel.detach().cpu().numpy()
            tilt_deg_hist[step] = tilt_deg.detach().cpu().numpy()

            if step % 200 == 0:
                print(f"[INFO] step {step}/{cfg.steps}", file=sys.stderr)

    env.close()

    # ---- post-process the raw (steps, N) arrays into per-episode records ----
    # Same reset-detection idiom play.py's AnalyticsPolicy uses
    # ("ep_buf[0] < self._prev_ep_buf[0]"), applied per-env, offline, across
    # the whole collected history instead of live.
    records = []
    ep_id = 0
    for c in range(N):
        ep_len_col = eplen_hist[:, c]
        reset_steps = (np.where(np.diff(ep_len_col) < 0)[0] + 1).tolist()
        boundaries = [0] + reset_steps + [cfg.steps]
        for k in range(len(boundaries) - 1):
            s, e = boundaries[k], boundaries[k + 1]
            if e - s < cfg.min_ep_len:
                continue
            seg_behind = behind_hist[s:e, c]
            save_idx = int(np.argmax(seg_behind)) if seg_behind.any() else None
            for t in range(s, e):
                rel_step = t - s
                rel_to_save = (t - (s + save_idx)) if save_idx is not None else np.nan
                jv = joint_vel_hist[t, c]  # (5,)
                records.append((
                    ep_id, c, rel_step, rel_to_save,
                    float(roll_hist[t, c]), float(pitch_hist[t, c]),
                    float(height_hist[t, c]),
                    bool(contact_hist[t, c]), bool(behind_hist[t, c]),
                    int(footidx_hist[t, c]),
                    float(jv[0]), float(jv[1]), float(jv[2]), float(jv[3]), float(jv[4]),
                    float(tilt_deg_hist[t, c]),
                ))
            ep_id += 1

    import csv
    columns = [
        "episode_id", "env_id", "step_since_reset", "step_since_save",
        "roll_ang_vel", "pitch_ang_vel", "foot_height",
        "foot_contact", "ball_behind", "foot_idx",
        "hip_pitch_vel", "hip_roll_vel", "knee_pitch_vel", "ankle_pitch_vel", "ankle_roll_vel",
        "foot_tilt_deg",
    ]
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(records)
    eps_with_save = {r[0] for r in records if r[3] == r[3]}  # r[3]!=r[3] iff NaN
    print(
        f"[INFO] Saved {len(records)} rows, {ep_id} episode segments "
        f"({len(eps_with_save)} with a detected save event) to {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
