"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Measures live region_estimator accuracy (argmax(estimate_region) vs the true
env._region_id) for one or more checkpoints -- same diagnostic class as the
2026-07-05 region_estimator collapse investigation (see interceptV2DualDis
CLAUDE.md's "Multi-disc PPO schedule" divergence row), re-run here to check
whether a similar collapse explains the 2026-09-09 double-step/blue-landing
regression between model_3000.pt and model_39750.pt of the
6144_forcelandingfix run.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_region_estimator_accuracy.py \\
        --checkpoint logs/rsl_rl/.../model_39750.pt --num-envs 512 --steps 1500
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
REGION_NAMES = {0: "left_near", 1: "left_far", 2: "right_near", 3: "right_far"}


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = ""
    num_envs: int = 512
    steps: int = 1500
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
    raw_env._ball_difficulty = float(cfg.difficulty)

    N = cfg.num_envs
    confusion = np.zeros((4, 4), dtype=np.int64)  # [true, pred]
    total = 0
    correct = 0

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            pred_region = torch.argmax(
                runner.alg.actor_critic.estimate_region, dim=-1
            )  # (N,)
            true_region = raw_env._region_id  # (N,)

            for t in range(4):
                mask_t = true_region == t
                if mask_t.any():
                    for p in range(4):
                        confusion[t, p] += int((pred_region[mask_t] == p).sum().item())

            total += N
            correct += int((pred_region == true_region).sum().item())

            obs, rew, dones, extras = env.step(actions)

            if step % 300 == 0:
                acc_so_far = correct / max(total, 1)
                print(f"[INFO] step {step}/{cfg.steps}, running accuracy {acc_so_far:.3f}", file=sys.stderr)

    env.close()

    acc = correct / max(total, 1)
    print(f"\n[SUMMARY] region_estimator accuracy for {cfg.checkpoint}:")
    print(f"  overall accuracy: {acc:.4f}  (chance = 0.25)")
    print("  confusion matrix (rows=true, cols=predicted):")
    header = "true\\pred".ljust(12) + "".join(f"{REGION_NAMES[p]:>12s}" for p in range(4))
    print("  " + header)
    for t in range(4):
        row_total = confusion[t].sum()
        row_str = "".join(
            f"{confusion[t, p]:>12d}" if row_total == 0 else f"{confusion[t, p] / row_total:>11.1%} "
            for p in range(4)
        )
        print(f"  {REGION_NAMES[t]:<12s}{row_str}   (n={row_total})")


if __name__ == "__main__":
    main()
