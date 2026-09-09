"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Measures the assigned (leading) foot's real ground-reaction force (feet_contact
sensor, .data.force, same field penalize_sharpcontact already reads --
rewards.py:penalize_sharpcontact) at the EXACT instant a genuine blue landing
fires (env._blue_landed_genuine flipping False->True this tick), split by
region (left_far/right_far).

Motivation (user report, 2026-09-09): watching model_18750.pt with
--force-region left_far vs right_far, right_far landings looked "quite
softly" -- this probe quantifies that with real force numbers instead of a
visual impression, to calibrate a genuine-landing force threshold for
replacing _get_reach_target_y's binary feet_contact.data.found check (see
BoosterT1mjlab's own feet_landing_impact reward, which uses .data.force for
exactly this reason -- found from an earlier session's fork research).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_blue_landing_force.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
        --num-envs 512 --steps 3000 --difficulty 0.5 --domain-rand 0.5
"""
from __future__ import annotations

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
REGION_NAMES = {0: "left_near", 1: "left_far", 2: "right_near", 3: "right_far"}
FAR_REGION_IDS = (1, 3)


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = ""
    num_envs: int = 512
    steps: int = 3000
    difficulty: float | None = 0.5
    domain_rand: float | None = 0.5
    device: str | None = None


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")

    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)

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

    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
        _get_actor_current_obs,
    )

    raw_env = env.unwrapped
    N = cfg.num_envs

    if not hasattr(raw_env, "_region_id"):
        raise RuntimeError("env._region_id not found -- region-conditioned task required.")

    obs, _ = env.reset()

    if cfg.difficulty is not None:
        raw_env._ball_difficulty = float(cfg.difficulty)
        print(f"[INFO] ball_difficulty overridden to {cfg.difficulty}", file=sys.stderr)
    if cfg.domain_rand is not None:
        raw_env._domain_rand_curriculum = float(cfg.domain_rand)
        print(f"[INFO] domain_rand_curriculum overridden to {cfg.domain_rand}", file=sys.stderr)

    feet_contact = raw_env.scene["feet_contact"]

    region_forces: dict[int, list[float]] = {r: [] for r in range(4)}
    prev_landed_genuine = torch.zeros(N, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            landed_genuine = getattr(raw_env, "_blue_landed_genuine", None)
            if landed_genuine is None:
                continue
            just_landed = landed_genuine & ~prev_landed_genuine
            prev_landed_genuine = landed_genuine.clone()

            ids = torch.where(just_landed)[0]
            if ids.numel() > 0:
                foot_idx = raw_env._blue_dbg_foot_idx if hasattr(raw_env, "_blue_dbg_foot_idx") else None
                force_per_geom = feet_contact.data.force.norm(dim=-1)  # [N, 8]
                left_force = force_per_geom[:, :4].max(dim=-1).values
                right_force = force_per_geom[:, 4:].max(dim=-1).values
                for i in ids.tolist():
                    r = int(raw_env._region_id[i].item())
                    fidx = int(foot_idx[i].item()) if foot_idx is not None else 0
                    f = float(left_force[i].item() if fidx == 0 else right_force[i].item())
                    region_forces[r].append(f)

            if step % 300 == 0:
                total = sum(len(v) for v in region_forces.values())
                print(f"[INFO] step {step}/{cfg.steps}, {total} genuine landings recorded so far", file=sys.stderr)

    env.close()

    print("\n[SUMMARY] assigned-foot contact FORCE (Newtons) at the exact instant of genuine blue landing:")
    for r in range(4):
        vals = np.array(region_forces[r])
        if vals.size == 0:
            print(f"  {REGION_NAMES[r]:12s} n=0 (no genuine landings recorded)")
            continue
        pct = np.percentile(vals, [10, 25, 50, 75, 90])
        print(
            f"  {REGION_NAMES[r]:12s} n={vals.size:4d}  "
            f"mean={vals.mean():7.2f}N  p10={pct[0]:7.2f}  p25={pct[1]:7.2f}  "
            f"median={pct[2]:7.2f}  p75={pct[3]:7.2f}  p90={pct[4]:7.2f}"
        )

    far_vals = np.array(region_forces[1] + region_forces[3])
    if far_vals.size:
        print(f"\n  FAR REGIONS COMBINED: n={far_vals.size}  mean={far_vals.mean():.2f}N  "
              f"median={np.median(far_vals):.2f}N  min={far_vals.min():.2f}N  max={far_vals.max():.2f}N")


if __name__ == "__main__":
    main()
