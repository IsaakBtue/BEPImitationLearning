"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Investigates a user-reported asymmetry: `foot_inner_face_continuous`
apparently scores a genuinely-correct (flat, side-matched) foot orientation
the SAME as a tilted/rolled foot on near-region saves -- and the user's own
live observation is that this "same reward for a wrong orientation" problem
shows up mainly on `right_near` saves, not `left_near`.

The 2026-08-16 fix (see rewards.py) already swapped the measured axis from
the toe (local X, provably invariant to Ankle_Roll since Ankle_Roll's own
joint axis IS local X) to the lateral (local Y, which Ankle_Roll actually
rotates). That fix is symmetric in code (uses `expected_sign` identically
for both feet), so it should not itself introduce a left/right asymmetry.
This probe exists to get real evidence, not assume the fix generalizes:
replay the most recent pre-fix checkpoint (which learned under the OLD,
roll-blind reward, so any roll-tilt exploit it found is baked into its real
behavior) and compute BOTH the OLD (X-axis) and NEW (Y-axis) alignment
formulas against the actual foot orientation at every softstop-firing
moment, split by ground-truth region (`env._region_id`, 0=left_near,
2=right_near). Confirms whether the new formula genuinely scores
`right_near`'s (real, checkpoint-exhibited) rolled-foot saves lower than
`left_near`'s, or whether a further asymmetric bug remains.

Does NOT import, call, or modify rewards.py beyond read-only helper
functions (`_get_correct_foot_idx`) -- the OLD/NEW alignment formulas are
independently re-implemented here (mirroring probe_foot_rotation.py's own
"no side effects" pattern) so this script can compute both regardless of
whatever `rewards.py` currently contains.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_inner_face_axis_lr.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_13000.pt \\
        --num-envs 256 --steps 3000 \\
        --out-csv /path/to/scratchpad/inner_face_axis_probe.csv
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

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/intercept_simple_goalkeeper_multidisc/"
    "2026-08-15_23-19-54_6144_run_2026-08-15b/model_13000.pt"
)

_FOOT_TARGET_ANGLE_DEG = 50.0
_FOOT_TARGET_COS = float(np.cos(np.radians(_FOOT_TARGET_ANGLE_DEG)))
_FOOT_TARGET_SIN = float(np.sin(np.radians(_FOOT_TARGET_ANGLE_DEG)))
_FOOT_OVERSHOOT_SIGMA = 0.03  # must match rewards.py's current value


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    num_envs: int = 256
    steps: int = 3000
    device: str | None = None
    out_csv: str = "/tmp/inner_face_axis_probe.csv"


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict), (
        f"Expected multi-disc dict rl_cfg for '{TASK_ID}', got {type(agent_cfg).__name__}."
    )

    env_cfg.scene.num_envs = cfg.num_envs
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
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

    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
        _get_actor_current_obs,
    )

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    foot_body_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]

    N = cfg.num_envs
    arange_n = torch.arange(N, device=device)

    obs, _ = env.reset()
    prev_softstop = torch.zeros(N, dtype=torch.bool, device=device)

    records = []

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            softstop_flag = getattr(
                raw_env, "_softstop_flag",
                torch.zeros(N, dtype=torch.bool, device=device),
            )
            just_fired = softstop_flag & ~prev_softstop
            prev_softstop = softstop_flag.clone()

            if just_fired.any():
                foot_idx = _get_correct_foot_idx(raw_env, "ball")  # (N,) 0=left,1=right
                region_id = getattr(
                    raw_env, "_region_id",
                    torch.full((N,), -1, dtype=torch.int64, device=device),
                )

                foot_quat_w = robot.data.body_link_quat_w[:, foot_body_ids, :]  # (N,2,4)
                assigned_quat_w = foot_quat_w[arange_n, foot_idx]  # (N,4)
                expected_sign = torch.where(foot_idx == 0, 1.0, -1.0)

                forward_local = torch.tensor([1.0, 0.0, 0.0], device=device).expand(N, -1)
                forward_w = quat_apply(robot.data.root_link_quat_w, forward_local)
                lateral_local = torch.tensor([0.0, 1.0, 0.0], device=device).expand(N, -1)
                lateral_w = quat_apply(robot.data.root_link_quat_w, lateral_local)

                # ---- OLD formula: toe (local X) axis ----
                x_local = torch.tensor([1.0, 0.0, 0.0], device=device).expand(N, -1)
                foot_x_w = quat_apply(assigned_quat_w, x_local)
                old_target = torch.stack([
                    torch.full((N,), _FOOT_TARGET_COS, device=device),
                    expected_sign * _FOOT_TARGET_SIN,
                    torch.zeros(N, device=device),
                ], dim=-1)
                old_alignment = (foot_x_w * old_target).sum(dim=-1)
                old_angle_deg = torch.rad2deg(torch.acos(
                    (foot_x_w * forward_w).sum(dim=-1).clamp(-1.0, 1.0)
                ))
                old_side_sign = (foot_x_w * lateral_w).sum(dim=-1) * expected_sign
                old_overshoot_mask = (old_side_sign > 0.0) & (old_angle_deg > _FOOT_TARGET_ANGLE_DEG)
                old_overshoot_err = old_angle_deg - _FOOT_TARGET_ANGLE_DEG
                old_overshoot_reward = torch.exp(-_FOOT_OVERSHOOT_SIGMA * old_overshoot_err ** 2)
                old_reward = torch.where(old_overshoot_mask, old_overshoot_reward, old_alignment)

                # ---- NEW formula: lateral (local Y) axis ----
                y_local = torch.tensor([0.0, 1.0, 0.0], device=device).expand(N, -1)
                foot_y_w = quat_apply(assigned_quat_w, y_local)
                new_target = torch.stack([
                    -expected_sign * _FOOT_TARGET_SIN,
                    torch.full((N,), _FOOT_TARGET_COS, device=device),
                    torch.zeros(N, device=device),
                ], dim=-1)
                new_alignment = (foot_y_w * new_target).sum(dim=-1)
                new_angle_deg = torch.rad2deg(torch.acos(
                    (foot_y_w * lateral_w).sum(dim=-1).clamp(-1.0, 1.0)
                ))
                new_side_sign = -(foot_y_w * forward_w).sum(dim=-1) * expected_sign
                new_overshoot_mask = (new_side_sign > 0.0) & (new_angle_deg > _FOOT_TARGET_ANGLE_DEG)
                new_overshoot_err = new_angle_deg - _FOOT_TARGET_ANGLE_DEG
                new_overshoot_reward = torch.exp(-_FOOT_OVERSHOOT_SIGMA * new_overshoot_err ** 2)
                new_reward = torch.where(new_overshoot_mask, new_overshoot_reward, new_alignment)

                # Real roll-off-flat tilt, degrees (feetorientation's own
                # gravity-vs-foot-Z math, converted to a signed angle) --
                # independent ground truth for "was this foot actually
                # tilted/rolled at the save moment."
                gravity_w = torch.tensor([0.0, 0.0, -1.0], device=device).expand(N, -1)
                gravity_foot = quat_apply(quat_inv(assigned_quat_w), gravity_w)
                tilt_deg = torch.rad2deg(torch.acos((-gravity_foot[:, 2]).clamp(-1.0, 1.0)))

                fired_ids = just_fired.nonzero(as_tuple=True)[0]
                for i in fired_ids.tolist():
                    records.append((
                        step, i, int(region_id[i].item()), int(foot_idx[i].item()),
                        float(old_angle_deg[i].item()), bool(old_overshoot_mask[i].item()), float(old_reward[i].item()),
                        float(new_angle_deg[i].item()), bool(new_overshoot_mask[i].item()), float(new_reward[i].item()),
                        float(tilt_deg[i].item()),
                    ))

            if step % 300 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, {len(records)} saves so far", file=sys.stderr)

    env.close()

    import csv
    columns = [
        "step", "env_id", "region_id", "foot_idx",
        "old_angle_deg", "old_overshoot", "old_reward",
        "new_angle_deg", "new_overshoot", "new_reward",
        "tilt_deg",
    ]
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(records)
    print(f"[INFO] Saved {len(records)} save-moment rows to {out_path}", file=sys.stderr)

    # ---- summary, split by region (left_near vs right_near only) ----
    region_names = {0: "left_near", 1: "left_far", 2: "right_near", 3: "right_far"}
    by_region: dict[int, list[tuple]] = {}
    for r in records:
        by_region.setdefault(r[2], []).append(r)

    print("\n[SUMMARY] angle-off-target(deg) / reward, OLD vs NEW formula, by region:")
    for rid in (0, 2):  # left_near, right_near
        rows = by_region.get(rid, [])
        if not rows:
            continue
        name = region_names.get(rid, f"region_{rid}")
        old_angles = [r[4] for r in rows]
        old_rewards = [r[6] for r in rows]
        new_angles = [r[7] for r in rows]
        new_rewards = [r[9] for r in rows]
        print(f"  {name:12s} n={len(rows)}")
        print(f"    OLD: angle mean={np.mean(old_angles):.1f} p10={np.percentile(old_angles,10):.1f} "
              f"p50={np.percentile(old_angles,50):.1f} p90={np.percentile(old_angles,90):.1f}  "
              f"reward mean={np.mean(old_rewards):.3f}")
        print(f"    NEW: angle mean={np.mean(new_angles):.1f} p10={np.percentile(new_angles,10):.1f} "
              f"p50={np.percentile(new_angles,50):.1f} p90={np.percentile(new_angles,90):.1f}  "
              f"reward mean={np.mean(new_rewards):.3f}")


if __name__ == "__main__":
    main()
