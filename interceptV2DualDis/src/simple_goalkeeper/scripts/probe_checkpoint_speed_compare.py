"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Compares TWO checkpoints from the same run against the SAME env config
(play=True) to answer two user-reported questions:

  1. Did the policy "dislearn" being fast toward the green-ball (live/final)
     target between an early checkpoint and a later one? Measured via
     per-episode-summed reward terms (footreach, foot_proximity, stopball,
     softstop, cleanstop, success, sequence_promptness) and one-shot latch
     fire rates -- same accounting pattern as probe_region_reward_asymmetry.py.

  2. Is there unwanted forward (world +/-X, i.e. toward/away from where the
     ball is coming from) velocity on the assigned (leading) foot while it
     is airborne during the approach, and does that coincide with low foot
     height (a floor-clipping signature: fast horizontal motion while too
     low to clear the floor)?

Runs each checkpoint for --steps env steps with a fixed seed set immediately
before env.reset() so both runs see the same spawn RNG sequence as closely as
mjlab/mujoco-warp's parallel-env stepping allows (best-effort matching, not
bit-exact) -- makes cross-checkpoint episode-count/latch-rate deltas
attributable to the POLICY, not to a different sample of ball trajectories.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_checkpoint_speed_compare.py \\
        --checkpoint-a logs/.../model_2500.pt --label-a early \\
        --checkpoint-b logs/.../model_39750.pt --label-b final \\
        --num-envs 512 --steps 3000 \\
        --out-csv /path/to/scratchpad/checkpoint_speed_compare.csv
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

_LATCH_ATTRS = [
    "_sb_flag", "_softstop_flag", "_cleanstop_flag",
    "_blue_landed_genuine", "_orange_landed_genuine",
    "_red_active", "_red_landed_genuine", "_ifos_flag",
]

# Floor-clipping proxy thresholds -- a tick counts as "low-and-fast" if the
# assigned (leading) foot is below this height AND its world-frame X speed
# (forward/backward, i.e. toward/away from ball's -X travel direction) is
# above this speed, while airborne and approaching (ball not yet behind).
_LOW_HEIGHT_M = 0.03
_FAST_X_MPS = 0.20


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint_a: str = ""
    checkpoint_b: str = ""
    label_a: str = "A"
    label_b: str = "B"
    num_envs: int = 512
    steps: int = 3000
    seed: int = 0
    device: str | None = None
    out_csv: str = "/tmp/checkpoint_speed_compare.csv"


def _run_checkpoint(checkpoint: str, label: str, cfg: ProbeConfig, device: str) -> dict:
    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)
    from simple_goalkeeper.mdp import rewards as gk_rewards
    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import (
        _get_actor_current_obs,
    )

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict), (
        f"Expected multi-disc dict rl_cfg for '{TASK_ID}', got {type(agent_cfg).__name__}."
    )
    env_cfg.scene.num_envs = cfg.num_envs
    os.environ.setdefault("MUJOCO_GL", "egl")

    torch.manual_seed(cfg.seed)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

    resume_path = Path(checkpoint)
    if not resume_path.is_absolute() and not resume_path.exists():
        resume_path = Path.cwd() / resume_path
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    print(f"[INFO] [{label}] Loading checkpoint: {resume_path}", file=sys.stderr)

    runner_cls = load_runner_cls(TASK_ID)
    runner = runner_cls(env, agent_cfg, log_dir=None, device=device)
    runner.load(str(resume_path), load_optimizer=False)
    act_inference = runner.get_inference_policy(device=device)

    raw_env = env.unwrapped
    rew_mgr = raw_env.reward_manager
    term_mgr = raw_env.termination_manager
    term_names = list(rew_mgr.active_terms)
    T = len(term_names)
    N = cfg.num_envs

    torch.manual_seed(cfg.seed)
    obs, _ = env.reset()

    # FIX: a bare SceneEntityCfg imported from goalkeeper_env_cfg.py is NOT
    # resolved (mjlab only resolves the copy it makes for each RewardTermCfg's
    # own params dict, never the shared module-level object itself -- the
    # exact bug class documented in rewards.py's stopball docstring, FIX
    # 2026-07-23). body_ids would silently stay slice(None) (all 23 bodies),
    # so foot_pos_w[arange_n, foot_idx] with foot_idx in {0,1} would pick
    # body 0/1 (Trunk/H1), not a foot at all. Resolve by name directly.
    robot0 = raw_env.scene["robot"]
    body_ids = [robot0.body_names.index("left_foot_link"), robot0.body_names.index("right_foot_link")]

    ep_sum = torch.zeros(N, T, device=device)
    term_total = np.zeros(T)
    n_episodes = 0
    latch_count = {a: 0 for a in _LATCH_ATTRS}
    term_reason = {}

    contact_vx, contact_vy, contact_vz, contact_height = [], [], [], []
    low_and_fast_ticks = 0
    approach_ticks = 0
    # Local one-shot latch mirroring contact_yield_velocity's own pattern --
    # env._sb_deflection_now is a RAW per-tick condition (can stay True for
    # many consecutive ticks while a foot rests against the ball), not
    # itself one-shot. Sample only the first tick of each contact event.
    probe_contact_flag = torch.zeros(N, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            ep_sum += rew_mgr._step_reward

            # --- per-tick leading-foot kinematics (issue 2: forward velocity
            # at low height during approach, before the ball is behind) ---
            behind = gk_rewards._ball_is_behind(raw_env, BALL_NAME)
            foot_idx = gk_rewards._get_correct_foot_idx(raw_env, BALL_NAME)
            robot = raw_env.scene["robot"]
            foot_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
            foot_vel_w = robot.data.body_link_lin_vel_w[:, body_ids, :]
            floor_z = raw_env.scene.env_origins[:, 2]
            arange_n = torch.arange(N, device=device)
            assigned_pos = foot_pos_w[arange_n, foot_idx]
            assigned_vel = foot_vel_w[arange_n, foot_idx]
            assigned_height = (assigned_pos[:, 2] - floor_z).clamp(0.0, None)
            assigned_vx = assigned_vel[:, 0]

            approaching = ~behind
            approach_ticks += int(approaching.sum().item())
            low_and_fast = approaching & (assigned_height < _LOW_HEIGHT_M) & (assigned_vx.abs() > _FAST_X_MPS)
            low_and_fast_ticks += int(low_and_fast.sum().item())

            # Genuine contact instant, reusing the same raw flag
            # contact_yield_velocity gates on -- but that flag stays True for
            # every tick of a resting contact, so latch it locally (same
            # one-shot pattern contact_yield_velocity itself uses) to sample
            # only the FIRST tick of each contact event.
            probe_contact_flag[raw_env.episode_length_buf <= 1] = False
            deflection_now = getattr(
                raw_env, "_sb_deflection_now", torch.zeros(N, dtype=torch.bool, device=device)
            )
            contact_fired = deflection_now & ~probe_contact_flag
            probe_contact_flag |= deflection_now
            if bool(contact_fired.any().item()):
                idxs = torch.where(contact_fired)[0]
                contact_vx.extend(assigned_vel[idxs, 0].tolist())
                contact_vy.extend(assigned_vel[idxs, 1].tolist())
                contact_vz.extend(assigned_vel[idxs, 2].tolist())
                contact_height.extend(assigned_height[idxs].tolist())

            done_ids = torch.where(dones.bool())[0]
            if done_ids.numel() > 0:
                fired_terms = {
                    name: term_mgr.get_term(name).bool()
                    for name in term_mgr.active_terms
                }
                for i in done_ids.tolist():
                    term_total += ep_sum[i].cpu().numpy()
                    n_episodes += 1
                    for a in _LATCH_ATTRS:
                        t = getattr(raw_env, a, None)
                        if t is not None and bool(t[i].item()):
                            latch_count[a] += 1
                    reason = "none"
                    for name, flags in fired_terms.items():
                        if bool(flags[i].item()):
                            reason = name
                            break
                    term_reason[reason] = term_reason.get(reason, 0) + 1
                    ep_sum[i] = 0.0

            if step % 300 == 0:
                print(f"[INFO] [{label}] step {step}/{cfg.steps}, {n_episodes} episodes so far", file=sys.stderr)

    env.close()

    n_ep = max(n_episodes, 1)
    return {
        "label": label,
        "term_names": term_names,
        "term_mean": term_total / n_ep,
        "n_episodes": n_episodes,
        "latch_rate": {a: latch_count[a] / n_ep for a in _LATCH_ATTRS},
        "term_reason": term_reason,
        "contact_vx": np.array(contact_vx),
        "contact_vy": np.array(contact_vy),
        "contact_vz": np.array(contact_vz),
        "contact_height": np.array(contact_height),
        "low_and_fast_frac": low_and_fast_ticks / max(approach_ticks, 1),
        "n_contacts": len(contact_vx),
    }


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint_a or not cfg.checkpoint_b:
        raise ValueError("--checkpoint-a and --checkpoint-b are both required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    res_a = _run_checkpoint(cfg.checkpoint_a, cfg.label_a, cfg, device)
    res_b = _run_checkpoint(cfg.checkpoint_b, cfg.label_b, cfg, device)

    import csv
    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", f"mean_{cfg.label_a}", f"mean_{cfg.label_b}", "delta_b_minus_a"])
        for name, a_val, b_val in zip(res_a["term_names"], res_a["term_mean"], res_b["term_mean"]):
            writer.writerow([name, a_val, b_val, b_val - a_val])
    print(f"[INFO] Saved per-term means to {out_path}", file=sys.stderr)

    print("\n[SUMMARY] episode counts + one-shot fire rates:")
    for res in (res_a, res_b):
        print(f"  {res['label']:8s} n_episodes={res['n_episodes']}")
        for a in _LATCH_ATTRS:
            print(f"    {a:24s} rate={res['latch_rate'][a]:.3f}")
        reasons = sorted(res["term_reason"].items(), key=lambda kv: -kv[1])
        print(f"    terminated_by: " + ", ".join(f"{k}={v}" for k, v in reasons))

    print(f"\n[SUMMARY] {cfg.label_a} vs {cfg.label_b} -- mean per-episode reward per term, "
          f"sorted by |delta| descending:")
    a_means = res_a["term_mean"]
    b_means = res_b["term_mean"]
    gaps = b_means - a_means
    order = np.argsort(-np.abs(gaps))
    for idx in order[:25]:
        print(f"    {res_a['term_names'][idx]:32s} {cfg.label_a}={a_means[idx]:+9.3f}  "
              f"{cfg.label_b}={b_means[idx]:+9.3f}  delta={gaps[idx]:+9.3f}")

    print(f"\n[SUMMARY] floor-clipping proxy (height<{_LOW_HEIGHT_M}m AND |vx|>{_FAST_X_MPS}m/s "
          f"while approaching, fraction of approach ticks):")
    for res in (res_a, res_b):
        print(f"    {res['label']:8s} low_and_fast_frac={res['low_and_fast_frac']:.4f}")

    print(f"\n[SUMMARY] assigned-foot velocity at genuine contact instant (world frame):")
    for res in (res_a, res_b):
        if res["n_contacts"] == 0:
            print(f"    {res['label']:8s} n_contacts=0 (no genuine deflection observed)")
            continue
        print(f"    {res['label']:8s} n_contacts={res['n_contacts']}")
        print(f"      vx (fwd/back): mean={res['contact_vx'].mean():+.3f}  "
              f"std={res['contact_vx'].std():.3f}  "
              f"p90_abs={np.percentile(np.abs(res['contact_vx']), 90):.3f}")
        print(f"      vy (lateral):  mean={res['contact_vy'].mean():+.3f}  "
              f"std={res['contact_vy'].std():.3f}")
        print(f"      vz (vertical): mean={res['contact_vz'].mean():+.3f}  "
              f"std={res['contact_vz'].std():.3f}")
        print(f"      height:        mean={res['contact_height'].mean():.4f}  "
              f"std={res['contact_height'].std():.4f}")


if __name__ == "__main__":
    main()
