"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Follow-up to probe_orange_landing_diagnosis.py: for every real tick (not
just each episode's single closest-approach tick) where the trailing foot
is already "positioned" (dist_to_orange < landing_radius) and "slow"
(foot_speed < landing_speed_threshold) -- i.e. exactly the two conditions a
human watching the viewer would call "it's standing on the orange ball" --
what fraction of those ticks ALSO clear the 40N force gate? Tests whether
the 40N threshold (calibrated for the LEADING/blue foot, which usually
carries full body weight) is systematically too strict for the TRAILING
foot, which may only carry partial weight while blue has already landed.

Usage: same flags as probe_orange_landing_diagnosis.py.
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

    import simple_goalkeeper.tasks  # noqa: F401

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
    obs, _ = env.reset()

    if cfg.difficulty is not None:
        raw_env._ball_difficulty = float(cfg.difficulty)
    if cfg.domain_rand is not None:
        raw_env._domain_rand_curriculum = float(cfg.domain_rand)

    positioned_slow_forces: list[float] = []
    positioned_slow_contact_true = 0
    positioned_slow_total = 0
    # Also record the actual force values regardless of the 40N cutoff, to
    # see the real distribution (not just pass/fail).

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            wide = getattr(raw_env, "_orange_wide", None)
            dist = getattr(raw_env, "_orange_dbg_dist", None)
            speed = getattr(raw_env, "_orange_dbg_speed", None)
            contact = getattr(raw_env, "_orange_dbg_contact", None)
            radius = getattr(raw_env, "_orange_landing_radius_current", None)
            speed_th = getattr(raw_env, "_orange_landing_speed_threshold_current", None)
            trailing_idx = getattr(raw_env, "_orange_dbg_foot_idx", None)
            if wide is None or dist is None or trailing_idx is None:
                continue

            feet_contact = raw_env.scene["feet_contact"]
            force_per_geom = feet_contact.data.force.norm(dim=-1)
            left_force = force_per_geom[:, :4].max(dim=-1).values
            right_force = force_per_geom[:, 4:].max(dim=-1).values
            assigned_force = torch.where(trailing_idx == 0, left_force, right_force)

            positioned_slow = wide & (dist < radius) & (speed < speed_th)
            ids = torch.where(positioned_slow)[0]
            if ids.numel() > 0:
                positioned_slow_total += ids.numel()
                positioned_slow_contact_true += int(contact[ids].sum().item())
                positioned_slow_forces.extend(assigned_force[ids].tolist())

            if step % 300 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, positioned_slow_total={positioned_slow_total}", file=sys.stderr)

    env.close()

    print(f"\n[SUMMARY] ticks where trailing foot is positioned (dist<radius) AND slow (speed<threshold): {positioned_slow_total}")
    if positioned_slow_total == 0:
        print("  none recorded.")
        return
    print(f"  of those, force > 40N (the landing gate): {positioned_slow_contact_true} "
          f"({100*positioned_slow_contact_true/positioned_slow_total:.1f}%)")
    forces = np.array(positioned_slow_forces)
    pct = np.percentile(forces, [10, 25, 50, 75, 90])
    print(f"  real force (N) distribution at those ticks: mean={forces.mean():.1f}  "
          f"p10={pct[0]:.1f}  p25={pct[1]:.1f}  median={pct[2]:.1f}  p75={pct[3]:.1f}  p90={pct[4]:.1f}")


if __name__ == "__main__":
    main()
