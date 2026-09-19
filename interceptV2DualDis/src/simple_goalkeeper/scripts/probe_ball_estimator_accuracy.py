"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Measures live ball_estimator error (estimate_ball vs the true ball_state_gt:
pos_x, pos_y, vel_x, vel_y in robot body frame) for one or more checkpoints,
in REAL units (meters / m/s per dimension) rather than the training-side
combined MSE (Loss/est_ball), which mixes position and velocity into one
number and can't say by itself whether the estimator is "half a meter off"
or just noisy.

Built 2026-09-19 after Loss/est_ball was found to climb from ~0.22 (best,
iter ~680) to a noisy 0.6-0.9 plateau that never converges even once
ball_difficulty itself goes flat (iter ~5225 onward in the
6144_forcestart_35165d5_2026-09-19 run) -- the rise while difficulty ramps
is expected (harder trajectories are harder to estimate), but persistent
oscillation after difficulty stops moving isn't explained by that alone.
Same diagnostic class as probe_region_estimator_accuracy.py -- reuse that
script's checkpoint-loading pattern.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_ball_estimator_accuracy.py \\
        --checkpoint logs/rsl_rl/.../model_9500.pt --num-envs 512 --steps 1500
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
BALL_NAME = "ball"
DIM_NAMES = ["pos_x", "pos_y", "vel_x", "vel_y"]


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
    from simple_goalkeeper.mdp.regions import ball_state_gt

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
    abs_err_sum = torch.zeros(4, device=device)
    sq_err_sum = torch.zeros(4, device=device)
    total = 0
    all_abs_err = []  # for percentiles, kept small via periodic flush
    all_ep_len = []  # ticks-since-reset, to check for a post-reset cold-start effect

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            pred_ball = runner.alg.actor_critic.estimate_ball  # (N, 4)
            true_ball = ball_state_gt(raw_env, BALL_NAME)  # (N, 4)

            err = pred_ball - true_ball
            abs_err = err.abs()
            abs_err_sum += abs_err.sum(dim=0)
            sq_err_sum += (err ** 2).sum(dim=0)
            total += N
            all_abs_err.append(abs_err.cpu().numpy())
            all_ep_len.append(raw_env.episode_length_buf.clone().cpu().numpy())

            obs, rew, dones, extras = env.step(actions)

            if step % 300 == 0:
                running_mae = (abs_err_sum / max(total, 1)).cpu().numpy()
                print(
                    f"[INFO] step {step}/{cfg.steps}, running MAE "
                    f"pos_x={running_mae[0]:.3f}m pos_y={running_mae[1]:.3f}m "
                    f"vel_x={running_mae[2]:.3f}m/s vel_y={running_mae[3]:.3f}m/s",
                    file=sys.stderr,
                )

    env.close()

    mae = (abs_err_sum / max(total, 1)).cpu().numpy()
    rmse = np.sqrt((sq_err_sum / max(total, 1)).cpu().numpy())
    all_abs_err = np.concatenate(all_abs_err, axis=0)  # (total, 4)
    all_ep_len = np.concatenate(all_ep_len, axis=0)  # (total,)

    # Post-reset cold-start check: history buffer needs `history_length`
    # (10) real ticks after a reset before it's fully current-episode data.
    print("\n  --- error vs ticks-since-reset (cold-start check) ---")
    print("  " + "".join(f"{d:>14s}" for d in DIM_NAMES) + "         n")
    bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 10_000)]
    for lo, hi in bins:
        mask = (all_ep_len > lo) & (all_ep_len <= hi)
        if mask.sum() == 0:
            continue
        bin_mae = all_abs_err[mask].mean(axis=0)
        label = f"ep_len {lo:>3d}-{hi if hi < 10_000 else 'inf':<5}"
        print(f"  {label} " + "".join(f"{v:>13.4f} " for v in bin_mae) + f" {mask.sum():>8d}")

    print(f"\n[SUMMARY] ball_estimator error for {cfg.checkpoint}:")
    print(f"  n samples: {total}")
    print("  " + "".join(f"{d:>14s}" for d in DIM_NAMES))
    print("  MAE " + "".join(f"{v:>13.4f} " for v in mae))
    print("  RMSE" + "".join(f"{v:>13.4f} " for v in rmse))
    p90 = np.percentile(all_abs_err, 90, axis=0)
    p50 = np.percentile(all_abs_err, 50, axis=0)
    print("  p50 " + "".join(f"{v:>13.4f} " for v in p50))
    print("  p90 " + "".join(f"{v:>13.4f} " for v in p90))
    print(
        "\n  (position dims in meters, velocity dims in m/s -- e.g. pos_x MAE "
        "of 0.05 means the estimator is off by 5cm on average)"
    )


if __name__ == "__main__":
    main()
