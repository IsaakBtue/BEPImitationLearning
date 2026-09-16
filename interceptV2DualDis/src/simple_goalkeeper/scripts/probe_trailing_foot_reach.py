"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

User report: orange/red trailing-foot approach "should go faster" -- looks
like it's not performing well. `trailing_foot_reach` (rewards.py) is the
mechanism responsible for lateral approach speed toward orange/red (sigmoid
reach x vel_sigma, same shape as footreach's own urgency term). This probe
replays a real trained checkpoint and measures that mechanism's own internal
quantities directly (not just the aggregate wandb Episode_Reward, which
can't distinguish "foot moves slowly" from "foot rarely gets assigned a wide
crossing" from "landing gate never satisfied").

Measures, restricted to steps where env._orange_wide is True and the ball
isn't behind yet:
  - vel_toward (unclamped, m/s) split into "far" (dist_to_target above the
    0.30m decel zone -- should show real approach speed) vs "near" (inside
    the decel zone, decaying toward neutral by design) -- if "far" is near
    zero, the mechanism isn't actually driving fast lateral motion.
  - vel_sigma actual multiplier applied (1.0 = no speed credit at all).
  - orange/red genuine-landing rate per wide-crossing episode, and mean
    steps-to-first-landing for orange (is it slow, or does it just never
    happen).
  - red_active activation rate (can red even turn on if orange rarely lands).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_trailing_foot_reach.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/2026-09-15_12-53-39_6144_ampfixbundle_67d22af_2026-09-15/model_5500.pt \\
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
_DECEL_ZONE = 0.30


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

    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)
    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
        _get_actor_current_obs,
    )
    from simple_goalkeeper.mdp.rewards import (
        _get_correct_foot_idx,
        _get_orange_reach_target_y,
        _get_red_reach_target_y,
        _ball_is_behind,
    )

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

    prev_orange_landed = torch.zeros(N, dtype=torch.bool, device=device)
    prev_red_landed = torch.zeros(N, dtype=torch.bool, device=device)
    wide_step_counter = torch.zeros(N, dtype=torch.int64, device=device)
    orange_land_step: list[int] = []

    far_vel_samples: list[float] = []
    near_vel_samples: list[float] = []
    far_vel_sigma_samples: list[float] = []
    near_vel_sigma_samples: list[float] = []

    wide_ep_count = 0
    ever_wide_this_ep = torch.zeros(N, dtype=torch.bool, device=device)
    orange_land_count = 0
    red_land_count = 0
    red_active_ever = torch.zeros(N, dtype=torch.bool, device=device)
    red_active_ep_count = 0

    asset_cfg = raw_env.reward_manager.get_term_cfg("trailing_foot_reach").params["asset_cfg"]
    robot_name = asset_cfg.name

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            wide = raw_env._orange_wide
            behind = _ball_is_behind(raw_env, "ball")
            active = wide & ~behind
            ever_wide_this_ep |= wide

            orange_y = _get_orange_reach_target_y(raw_env, "ball", asset_cfg=asset_cfg)
            red_y = _get_red_reach_target_y(raw_env, "ball", asset_cfg=asset_cfg)
            red_active = raw_env._red_active
            red_active_ever |= red_active
            target_y = torch.where(red_active, red_y, orange_y)

            robot = raw_env.scene[robot_name]
            foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
            foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
            foot_idx = _get_correct_foot_idx(raw_env, "ball")
            trailing_idx = 1 - foot_idx
            assigned_foot_pos = foot_pos_w[arange_n, trailing_idx]
            assigned_foot_vel_y = foot_vel_w[arange_n, trailing_idx, 1]

            goal_x_w = raw_env.scene.env_origins[:, 0]
            floor_z_w = raw_env.scene.env_origins[:, 2]
            target_point = torch.stack([goal_x_w, target_y, floor_z_w + 0.10], dim=-1)
            dist_to_target = torch.norm(assigned_foot_pos - target_point, dim=-1)

            lateral_error = target_y - assigned_foot_pos[:, 1]
            vel_toward = torch.where(lateral_error > 0, assigned_foot_vel_y, -assigned_foot_vel_y)
            vel_sigma_raw = 1.0 + 3.0 * vel_toward.clamp(0.0, 3.0)

            current_landed_genuine = torch.where(red_active, raw_env._red_landed_genuine, raw_env._orange_landed_genuine)
            approaching = wide & ~current_landed_genuine
            _decel_floor = float(getattr(raw_env, "_orange_landing_radius_current", 0.08))
            decay_frac = ((dist_to_target - _decel_floor) / (_DECEL_ZONE - _decel_floor)).clamp(0.0, 1.0)
            vel_sigma = torch.where(approaching, 1.0 + (vel_sigma_raw - 1.0) * decay_frac, vel_sigma_raw)

            mask_active = active.cpu()
            far_mask = (mask_active & (dist_to_target.cpu() > _DECEL_ZONE))
            near_mask = (mask_active & (dist_to_target.cpu() <= _DECEL_ZONE))
            if far_mask.any():
                far_vel_samples.extend(vel_toward.cpu()[far_mask].tolist())
                far_vel_sigma_samples.extend(vel_sigma.cpu()[far_mask].tolist())
            if near_mask.any():
                near_vel_samples.extend(vel_toward.cpu()[near_mask].tolist())
                near_vel_sigma_samples.extend(vel_sigma.cpu()[near_mask].tolist())

            wide_step_counter += active.long()

            orange_landed = raw_env._orange_landed_genuine
            newly_orange = orange_landed & ~prev_orange_landed
            ids = torch.where(newly_orange)[0]
            if ids.numel() > 0:
                orange_land_count += ids.numel()
                for i in ids.tolist():
                    orange_land_step.append(int(wide_step_counter[i].item()))
            prev_orange_landed = orange_landed.clone()

            red_landed = raw_env._red_landed_genuine
            red_land_count += int((red_landed & ~prev_red_landed).sum().item())
            prev_red_landed = red_landed.clone()

            done_ids = torch.where(dones.bool())[0]
            if done_ids.numel() > 0:
                for i in done_ids.tolist():
                    if bool(ever_wide_this_ep[i].item()):
                        wide_ep_count += 1
                        if bool(red_active_ever[i].item()):
                            red_active_ep_count += 1
                wide_step_counter[done_ids] = 0
                ever_wide_this_ep[done_ids] = False
                red_active_ever[done_ids] = False

            if step % 500 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, wide episodes so far: {wide_ep_count}", file=sys.stderr)

    env.close()

    far_v = np.array(far_vel_samples) if far_vel_samples else np.array([0.0])
    near_v = np.array(near_vel_samples) if near_vel_samples else np.array([0.0])
    far_vs = np.array(far_vel_sigma_samples) if far_vel_sigma_samples else np.array([1.0])
    near_vs = np.array(near_vel_sigma_samples) if near_vel_sigma_samples else np.array([1.0])
    orange_steps = np.array(orange_land_step) if orange_land_step else np.array([])

    print("\n[SUMMARY] trailing_foot_reach live diagnostic:")
    print(f"  wide-crossing episodes seen         {wide_ep_count}")
    print(f"  orange genuine landings             {orange_land_count}  ({orange_land_count / max(wide_ep_count,1):.1%} of wide episodes)")
    print(f"  red genuine landings                {red_land_count}  ({red_land_count / max(wide_ep_count,1):.1%} of wide episodes)")
    print(f"  red_active ever reached             {red_active_ep_count}  ({red_active_ep_count / max(wide_ep_count,1):.1%} of wide episodes)")
    print(f"  mean steps-to-orange-landing         {orange_steps.mean() if orange_steps.size else float('nan'):.1f}  (median {np.median(orange_steps) if orange_steps.size else float('nan'):.1f})")
    print()
    print(f"  FAR zone (dist > {_DECEL_ZONE}m, should show real approach speed):")
    print(f"    vel_toward mean/median              {far_v.mean():.3f} / {np.median(far_v):.3f} m/s")
    print(f"    vel_toward negative (moving away)   {(far_v < 0).mean():.1%}")
    print(f"    vel_toward ~0 (<0.05 m/s)            {(np.abs(far_v) < 0.05).mean():.1%}")
    print(f"    vel_sigma mean                      {far_vs.mean():.3f}  (1.0 = no speed credit)")
    print()
    print(f"  NEAR zone (dist <= {_DECEL_ZONE}m, decel-zone active by design):")
    print(f"    vel_toward mean/median              {near_v.mean():.3f} / {np.median(near_v):.3f} m/s")
    print(f"    vel_sigma mean                      {near_vs.mean():.3f}")


if __name__ == "__main__":
    main()
