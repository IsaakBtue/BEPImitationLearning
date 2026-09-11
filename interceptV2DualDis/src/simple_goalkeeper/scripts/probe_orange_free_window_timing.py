"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Follow-up to probe_orange_landing_diagnosis.py. User insists orange_ball_landed
"never fires" despite watching real (genuine-looking) landings. Hypothesis:
the free-landing filter (`_ORANGE_LANDING_FREE_STEP_THRESHOLD=10`) -- tuned
against BLUE, whose target sits far from the robot's own start -- is far too
aggressive for ORANGE, whose target is by design much closer to start. A
genuinely fast, real (non-RSI-lucky) approach can complete within 10 real
steps (0.2s) for orange in a way it structurally can't for blue, so the SAME
threshold silently voids real landings for orange, not just RSI luck.

This probe records, for every RAW landing (env._orange_landed newly True,
regardless of the free classification), the exact episode_length_buf at that
tick, split into buckets:
    0-1   : essentially at/immediately after reset -- consistent with RSI
            donor pose already there (the case the filter is meant to catch)
    2-9   : early, but NOT immediate -- consistent with a genuinely fast
            real approach (the case the filter WRONGLY catches if this
            bucket is large)
    >=10  : passes the filter regardless (genuine by definition)

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

    prev_landed = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    landing_ticks: list[int] = []
    landing_was_free: list[bool] = []
    landing_dist_to_orange: list[float] = []  # how far the target was from start, for context

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            landed = getattr(raw_env, "_orange_landed", None)
            was_free = getattr(raw_env, "_orange_landed_was_free", None)
            ep_len = raw_env.episode_length_buf
            if landed is None:
                continue

            newly = landed & ~prev_landed
            prev_landed = landed.clone()
            ids = torch.where(newly)[0]
            if ids.numel() > 0:
                for i in ids.tolist():
                    landing_ticks.append(int(ep_len[i].item()))
                    landing_was_free.append(bool(was_free[i].item()) if was_free is not None else False)

            if step % 300 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, landings recorded so far: {len(landing_ticks)}", file=sys.stderr)

    env.close()

    ticks = np.array(landing_ticks)
    free = np.array(landing_was_free)
    print(f"\n[SUMMARY] total raw landings recorded: {len(ticks)}")
    if len(ticks) == 0:
        return

    b0 = (ticks <= 1)
    b1 = (ticks >= 2) & (ticks < 10)
    b2 = (ticks >= 10)
    print(f"  bucket ep_len<=1  (at/immediately-after reset, classic RSI-luck): {b0.sum()} ({100*b0.mean():.1f}%)")
    print(f"  bucket 2<=ep_len<10 (early but NOT immediate -- plausible genuine fast approach): {b1.sum()} ({100*b1.mean():.1f}%)")
    print(f"  bucket ep_len>=10 (passes the free filter regardless): {b2.sum()} ({100*b2.mean():.1f}%)")
    print(f"\n  Of ALL landings classified 'free' (was_free=True): {free.sum()} ({100*free.mean():.1f}%)")
    print(f"  -- of those free-classified landings, ep_len distribution:")
    free_ticks = ticks[free]
    if free_ticks.size:
        vals, counts = np.unique(free_ticks, return_counts=True)
        for v, c in zip(vals, counts):
            print(f"      ep_len={v:2d}: {c:4d} landings")
    print(
        "\n  Interpretation: if the 2<=ep_len<10 bucket is large, most 'free'-"
        "voided landings are NOT RSI donor poses sitting on the target at reset"
        "(that would cluster at ep_len<=1) -- they're genuine approaches that"
        "just happen to finish inside the 10-step window because orange sits"
        "close to start by design. That means the SAME free-window threshold"
        "that correctly filters blue's RSI luck is voiding orange's genuine,"
        "fast landings, since orange's target is inherently much closer."
    )


if __name__ == "__main__":
    main()
