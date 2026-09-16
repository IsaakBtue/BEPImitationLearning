"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Follow-up to probe_blue_landing_liftoff.py: user reports the new genuine-
lift/descent gate (rewards.py:_get_reach_target_y, 2026-09-16 fix) now
blocks landings that used to register -- "now it doesn't work when it
lands after". This probe compares the OLD gate (wide & was_airborne &
foot_in_contact & within_square) against the NEW one (same + genuinely_
lifted_and_descended) directly against the live env attributes the fix
itself writes (env._blue_peak_clearance, env._blue_min_vel_z), to see
whether real landings are being blocked, and by how much they miss the
new thresholds (_MIN_GENUINE_LIFT_HEIGHT=0.02, _MIN_DOWNWARD_VEL=0.05).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_blue_genuine_lift_gate.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
        --num-envs 512 --steps 3000 --difficulty 1.0
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

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
_MIN_GENUINE_LIFT_HEIGHT = 0.02
_MIN_DOWNWARD_VEL = 0.05
_LANDING_FORCE_THRESHOLD = 40.0


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = ""
    num_envs: int = 512
    steps: int = 3000
    difficulty: float = 1.0
    device: str | None = None


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")

    import simple_goalkeeper.tasks  # noqa: F401
    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import _get_actor_current_obs
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx, _get_reach_target_y

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict)
    env_cfg.scene.num_envs = cfg.num_envs
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

    raw_env = env.unwrapped
    N = cfg.num_envs
    arange_n = torch.arange(N, device=device)

    obs, _ = env.reset()
    raw_env._ball_difficulty = float(cfg.difficulty)

    asset_cfg = raw_env.reward_manager.get_term_cfg("blue_ball_landed").params["asset_cfg"]
    robot = raw_env.scene[asset_cfg.name]
    feet_contact = raw_env.scene["feet_contact"]

    prev_blue_landed = torch.zeros(N, dtype=torch.bool, device=device)
    prev_old_candidate = torch.zeros(N, dtype=torch.bool, device=device)

    new_landings = 0
    old_candidate_events = 0       # rising edges of the OLD gate condition
    blocked_by_new_gate = 0        # OLD gate rising edge, but genuinely_lifted_and_descended was False
    blocked_peak_clearance: list[float] = []
    blocked_min_vel_z: list[float] = []

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            # Recompute the OLD gate's ingredients live (mirrors rewards.py exactly).
            _get_reach_target_y(raw_env, "ball", asset_cfg=asset_cfg)  # ensure fresh env._blue_* this tick

            foot_idx = _get_correct_foot_idx(raw_env, "ball")
            force_per_geom = feet_contact.data.force.norm(dim=-1)
            left_force = force_per_geom[:, :4].max(dim=-1).values
            right_force = force_per_geom[:, 4:].max(dim=-1).values
            assigned_force = torch.where(foot_idx == 0, left_force, right_force)
            foot_in_contact = assigned_force > _LANDING_FORCE_THRESHOLD

            wide = raw_env._blue_wide
            was_airborne = raw_env._blue_was_airborne
            half_side = raw_env._blue_landing_half_side_current
            foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
            assigned_foot_pos = foot_pos_w[arange_n, foot_idx]
            goal_x_w = raw_env.scene.env_origins[:, 0]
            half_y = raw_env._blue_dbg_half_off + raw_env.scene.env_origins[:, 1]
            target_xy = torch.stack([goal_x_w, half_y], dim=-1)
            within_square = (
                (assigned_foot_pos[:, 0] - target_xy[:, 0]).abs() < half_side
            ) & ((assigned_foot_pos[:, 1] - target_xy[:, 1]).abs() < half_side)

            old_candidate = wide & was_airborne & foot_in_contact & within_square
            genuinely_lifted_and_descended = (
                (raw_env._blue_peak_clearance > _MIN_GENUINE_LIFT_HEIGHT)
                & (raw_env._blue_min_vel_z < -_MIN_DOWNWARD_VEL)
            )

            newly_old_candidate = old_candidate & ~prev_old_candidate
            ids = torch.where(newly_old_candidate)[0]
            if ids.numel() > 0:
                old_candidate_events += ids.numel()
                blocked = ~genuinely_lifted_and_descended[ids]
                blocked_ids = ids[blocked]
                if blocked_ids.numel() > 0:
                    blocked_by_new_gate += blocked_ids.numel()
                    for i in blocked_ids.tolist():
                        blocked_peak_clearance.append(float(raw_env._blue_peak_clearance[i].item()))
                        blocked_min_vel_z.append(float(raw_env._blue_min_vel_z[i].item()))
            prev_old_candidate = old_candidate.clone()

            blue_landed = raw_env._blue_landed_genuine
            new_landings += int((blue_landed & ~prev_blue_landed).sum().item())
            prev_blue_landed = blue_landed.clone()

            if step % 500 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, new-gate landings so far: {new_landings}", file=sys.stderr)

    env.close()

    pc = np.array(blocked_peak_clearance) if blocked_peak_clearance else np.array([0.0])
    vz = np.array(blocked_min_vel_z) if blocked_min_vel_z else np.array([0.0])

    print("\n[SUMMARY] old-gate vs new-gate comparison (live, same rollout):")
    print(f"  OLD-gate candidate rising-edge events   {old_candidate_events}")
    print(f"  of those, blocked by new genuine-lift gate  {blocked_by_new_gate}  ({blocked_by_new_gate/max(old_candidate_events,1):.1%})")
    print(f"  NEW genuine blue landings (env._blue_landed_genuine)  {new_landings}")
    print()
    print("  Blocked events' peak_clearance (m) -- threshold is 0.02:")
    print(f"    mean={pc.mean():.4f} median={np.median(pc):.4f} p90={np.percentile(pc,90):.4f} max={pc.max():.4f}")
    print(f"    fraction that DID clear the height threshold (>0.02) -- blocked by vel_z only: {(pc > _MIN_GENUINE_LIFT_HEIGHT).mean():.1%}")
    print("  Blocked events' min_vel_z (m/s) -- threshold is -0.05:")
    print(f"    mean={vz.mean():.4f} median={np.median(vz):.4f} p10={np.percentile(vz,10):.4f}")
    print(f"    fraction that DID clear the vel threshold (<-0.05) -- blocked by height only: {(vz < -_MIN_DOWNWARD_VEL).mean():.1%}")


if __name__ == "__main__":
    main()
