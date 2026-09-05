"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Investigates a user-reported convergence asymmetry: right-side (-Y) saves
converged correctly, left-side (+Y) saves did not converge at all under the
same checkpoint/training run.

Runs a trained checkpoint in the real multi-disc env (play=True, so region
assignment cycles randomly per episode via `randomize_region_on_reset` --
region_id is cached BEFORE each env.step() so a mid-step reset can't
mis-attribute the episode that just ended to its NEXT region) and, per
episode, records:

  - which region (left_near/left_far/right_near/right_far) it belonged to
  - the per-episode-accumulated value of every registered reward term
    (mjlab's RewardManager._step_reward, summed manually here rather than
    read from RewardManager._episode_sums, since that internal buffer is
    zeroed by the SAME env.step() call that ends the episode -- reading it
    after step() would already show the NEXT episode's zeroed state)
  - one-shot outcome latches: stopball/softstop/cleanstop fired,
    blue/orange/red genuine-landing flags, inner_face_orientation_save fired
  - which termination term fired (bad_orientation/base_height/ball_exit/
    sharpforce/time_out), read via termination_manager AFTER step() --
    confirmed elsewhere in this codebase (play.py's AnalyticsPolicy) that
    this buffer is NOT cleared by reset(), so it still reflects the episode
    that just ended.

Reports per-region: episode count, one-shot fire rates, termination-reason
distribution, and mean per-episode value for every reward term -- sorted by
the LEFT-vs-RIGHT relative gap (near vs near, far vs far) so the terms most
responsible for the asymmetry surface at the top.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_region_reward_asymmetry.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
        --num-envs 512 --steps 3000 \\
        --out-csv /path/to/scratchpad/region_reward_probe.csv
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

_LATCH_ATTRS = [
    "_sb_flag", "_softstop_flag", "_cleanstop_flag",
    "_blue_landed_genuine", "_orange_landed_genuine",
    "_red_active", "_red_landed_genuine", "_ifos_flag",
]


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = ""
    num_envs: int = 512
    steps: int = 3000
    device: str | None = None
    out_csv: str = "/tmp/region_reward_probe.csv"


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)

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
    rew_mgr = raw_env.reward_manager
    term_mgr = raw_env.termination_manager
    term_names = list(rew_mgr.active_terms)
    T = len(term_names)
    N = cfg.num_envs

    if not hasattr(raw_env, "_region_id"):
        raise RuntimeError(
            "env._region_id not found -- 'assign_static_regions' event missing "
            "from this task cfg. Only the multi-disc region-conditioned task "
            "is supported by this probe."
        )

    obs, _ = env.reset()

    ep_sum = torch.zeros(N, T, device=device)
    region_term_sum = {r: np.zeros(T) for r in range(4)}
    region_ep_count = {r: 0 for r in range(4)}
    region_latch_count = {r: {a: 0 for a in _LATCH_ATTRS} for r in range(4)}
    region_term_reason = {r: {} for r in range(4)}

    with torch.inference_mode():
        for step in range(cfg.steps):
            prev_region = raw_env._region_id.clone()

            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            ep_sum += rew_mgr._step_reward

            done_ids = torch.where(dones.bool())[0]
            if done_ids.numel() > 0:
                fired_terms = {
                    name: term_mgr.get_term(name).bool()
                    for name in term_mgr.active_terms
                }
                for i in done_ids.tolist():
                    r = int(prev_region[i].item())
                    region_term_sum[r] += ep_sum[i].cpu().numpy()
                    region_ep_count[r] += 1
                    for a in _LATCH_ATTRS:
                        t = getattr(raw_env, a, None)
                        if t is not None and bool(t[i].item()):
                            region_latch_count[r][a] += 1
                    reason = "none"
                    for name, flags in fired_terms.items():
                        if bool(flags[i].item()):
                            reason = name
                            break
                    region_term_reason[r][reason] = region_term_reason[r].get(reason, 0) + 1
                    ep_sum[i] = 0.0

            if step % 300 == 0:
                total_eps = sum(region_ep_count.values())
                print(f"[INFO] step {step}/{cfg.steps}, {total_eps} episodes so far", file=sys.stderr)

    env.close()

    import csv
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "term", "mean_per_episode"])
        for r in range(4):
            n_ep = max(region_ep_count[r], 1)
            for name, total in zip(term_names, region_term_sum[r]):
                writer.writerow([REGION_NAMES[r], name, total / n_ep])
    print(f"[INFO] Saved per-region/per-term means to {out_path}", file=sys.stderr)

    print("\n[SUMMARY] episode counts + one-shot fire rates by region:")
    for r in range(4):
        n_ep = max(region_ep_count[r], 1)
        print(f"  {REGION_NAMES[r]:12s} n_episodes={region_ep_count[r]}")
        for a in _LATCH_ATTRS:
            rate = region_latch_count[r][a] / n_ep
            print(f"    {a:24s} rate={rate:.3f}")
        reasons = sorted(region_term_reason[r].items(), key=lambda kv: -kv[1])
        reason_str = ", ".join(f"{k}={v}" for k, v in reasons)
        print(f"    terminated_by: {reason_str}")

    print("\n[SUMMARY] left_near vs right_near, left_far vs right_far -- "
          "mean per-episode reward per term, sorted by |gap| descending:")
    for lr, rr, label in ((0, 2, "near"), (1, 3, "far")):
        n_l = max(region_ep_count[lr], 1)
        n_r = max(region_ep_count[rr], 1)
        left_means = region_term_sum[lr] / n_l
        right_means = region_term_sum[rr] / n_r
        gaps = left_means - right_means
        order = np.argsort(-np.abs(gaps))
        print(f"\n  -- {label} (left_n={region_ep_count[lr]}, right_n={region_ep_count[rr]}) --")
        for idx in order[:20]:
            print(f"    {term_names[idx]:32s} left={left_means[idx]:+9.3f}  "
                  f"right={right_means[idx]:+9.3f}  gap={gaps[idx]:+9.3f}")


if __name__ == "__main__":
    main()
