"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

User report (2026-09-11): "the transition from orange ball to red ball is
then broken because in the mujoco play script it always stays at
orange/yellow." A deterministic hand-forced test already proved
`_get_red_reach_target_y`'s gate logic itself is correct (red_active flips
exactly at the 25-step delay boundary, no earlier/later). This probe
measures how often the real PREREQUISITE for that gate -- blue AND orange
BOTH genuinely landing in the SAME episode -- actually happens during a
real rollout, to quantify whether "always stays orange" is explained by
this being genuinely rare rather than a code bug.

Usage: same flags as probe_orange_landing_diagnosis.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

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
    N = cfg.num_envs
    obs, _ = env.reset()

    if cfg.difficulty is not None:
        raw_env._ball_difficulty = float(cfg.difficulty)
    if cfg.domain_rand is not None:
        raw_env._domain_rand_curriculum = float(cfg.domain_rand)

    ever_wide = torch.zeros(N, dtype=torch.bool, device=device)
    ever_blue = torch.zeros(N, dtype=torch.bool, device=device)
    ever_orange = torch.zeros(N, dtype=torch.bool, device=device)
    ever_red_active = torch.zeros(N, dtype=torch.bool, device=device)
    episode_done_recorded = torch.zeros(N, dtype=torch.bool, device=device)

    finished_wide = 0
    finished_blue = 0
    finished_orange = 0
    finished_both = 0
    finished_red_active = 0

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            wide = getattr(raw_env, "_orange_wide", None)
            blue_g = getattr(raw_env, "_blue_landed_genuine", None)
            orange_g = getattr(raw_env, "_orange_landed_genuine", None)
            red_active = getattr(raw_env, "_red_active", None)
            if wide is None:
                continue

            ep_len = raw_env.episode_length_buf
            just_reset = ep_len <= 1
            newly_reset = just_reset & ~episode_done_recorded
            ids = torch.where(newly_reset)[0]
            for i in ids.tolist():
                if ever_wide[i]:
                    finished_wide += 1
                    b = bool(ever_blue[i].item())
                    o = bool(ever_orange[i].item())
                    if b:
                        finished_blue += 1
                    if o:
                        finished_orange += 1
                    if b and o:
                        finished_both += 1
                    if ever_red_active[i]:
                        finished_red_active += 1
            ever_wide[newly_reset] = False
            ever_blue[newly_reset] = False
            ever_orange[newly_reset] = False
            ever_red_active[newly_reset] = False
            episode_done_recorded = just_reset.clone()

            ever_wide |= wide
            if blue_g is not None:
                ever_blue |= blue_g
            if orange_g is not None:
                ever_orange |= orange_g
            if red_active is not None:
                ever_red_active |= red_active

            if step % 300 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, finished_wide={finished_wide} both={finished_both}", file=sys.stderr)

    env.close()

    print(f"\n[SUMMARY] finished wide episodes: {finished_wide}")
    print(f"  blue landed genuine:          {finished_blue:5d} ({100*finished_blue/max(finished_wide,1):.1f}%)")
    print(f"  orange landed genuine:        {finished_orange:5d} ({100*finished_orange/max(finished_wide,1):.1f}%)")
    print(f"  BOTH blue AND orange genuine: {finished_both:5d} ({100*finished_both/max(finished_wide,1):.1f}%)  <- red's prerequisite")
    print(f"  red_active ever fired:        {finished_red_active:5d} ({100*finished_red_active/max(finished_wide,1):.1f}%)")
    if finished_both > 0:
        print(f"\n  Of episodes where BOTH landed genuine, red_active fired in "
              f"{100*finished_red_active/finished_both:.1f}% -- if this is not ~100%, "
              f"THAT would indicate a real gate bug (it should be, modulo episode ending "
              f"before the 25-step delay elapses).")


if __name__ == "__main__":
    main()
