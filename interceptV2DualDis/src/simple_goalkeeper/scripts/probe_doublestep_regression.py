"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Compares two checkpoints from the SAME run (6144_forcelandingfix) to
investigate a user-reported regression: model_3000.pt appears to double-step
genuinely on far-region crossings, model_39750.pt appears to have "delearned"
it. Both replayed under the SAME fixed difficulty/domain_rand so any
difference reflects the POLICY, not curriculum drift.

Measures, per far-region (left_far/right_far) episode:
  - trailing-foot liftoff count during the approach (ball_x_local > 0.5m) --
    a genuine double-step needs >=1 real liftoff+replant of the trailing
    foot; a single continuous slide/lunge should show ~0.
  - orange/red landing force at the exact genuine-landing instant (same
    feet_contact.data.force field the 2026-09-09 force-based landing fix
    reads) -- low force there would support "sliding into the target
    instead of planting."
  - blue/orange/red genuine-landing rates and softstop (save) rate, for
    context.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_doublestep_regression.py \\
        --checkpoint-a logs/rsl_rl/.../model_3000.pt \\
        --checkpoint-b logs/rsl_rl/.../model_39750.pt \\
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
FAR_REGION_IDS = (1, 3)


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint_a: str = ""
    checkpoint_b: str = ""
    num_envs: int = 512
    steps: int = 3000
    difficulty: float = 1.0
    device: str | None = None


def _run_one(checkpoint: str, cfg: ProbeConfig, device: str) -> dict:
    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict)

    env_cfg.scene.num_envs = cfg.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

    resume_path = Path(checkpoint)
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
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    raw_env = env.unwrapped
    N = cfg.num_envs

    obs, _ = env.reset()
    raw_env._ball_difficulty = float(cfg.difficulty)

    feet_contact = raw_env.scene["feet_contact"]

    liftoff_counts = torch.zeros(N, dtype=torch.int64, device=device)
    trailing_was_airborne = torch.zeros(N, dtype=torch.bool, device=device)
    prev_blue_landed = torch.zeros(N, dtype=torch.bool, device=device)
    prev_orange_landed = torch.zeros(N, dtype=torch.bool, device=device)
    prev_red_landed = torch.zeros(N, dtype=torch.bool, device=device)

    far_liftoff_samples: list[int] = []
    blue_land_count = 0
    orange_land_count = 0
    red_land_count = 0
    orange_land_forces: list[float] = []
    softstop_count = 0
    far_ep_count = 0
    ep_sum_flag = torch.zeros(N, dtype=torch.bool, device=device)  # softstop-this-episode

    with torch.inference_mode():
        for step in range(cfg.steps):
            prev_region = raw_env._region_id.clone()

            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            # Trailing-foot liftoff counting, only meaningful pre-save on wide crossings.
            wide = getattr(raw_env, "_blue_wide", None)
            behind_fn_flag = getattr(raw_env, "_sb_flag", None)
            if wide is not None:
                foot_idx = _get_correct_foot_idx(raw_env, "ball")
                trailing_idx = 1 - foot_idx
                force_per_geom = feet_contact.data.force.norm(dim=-1)
                left_force = force_per_geom[:, :4].max(dim=-1).values
                right_force = force_per_geom[:, 4:].max(dim=-1).values
                trailing_force = torch.where(trailing_idx == 0, left_force, right_force)
                trailing_in_contact = trailing_force > 40.0
                trailing_airborne_now = wide & ~trailing_in_contact
                # count a liftoff on the rising edge (was in contact, now airborne)
                new_liftoff = wide & trailing_airborne_now & ~trailing_was_airborne
                liftoff_counts += new_liftoff.long()
                trailing_was_airborne = trailing_airborne_now | (~wide & trailing_was_airborne)

            orange_landed = getattr(raw_env, "_orange_landed_genuine", None)
            if orange_landed is not None:
                newly = orange_landed & ~prev_orange_landed
                ids = torch.where(newly)[0]
                if ids.numel() > 0:
                    fidx = _get_correct_foot_idx(raw_env, "ball")
                    trailing_idx2 = 1 - fidx
                    fpg = feet_contact.data.force.norm(dim=-1)
                    lf = fpg[:, :4].max(dim=-1).values
                    rf = fpg[:, 4:].max(dim=-1).values
                    tf = torch.where(trailing_idx2 == 0, lf, rf)
                    for i in ids.tolist():
                        orange_land_forces.append(float(tf[i].item()))
                    orange_land_count += ids.numel()
                prev_orange_landed = orange_landed.clone()

            blue_landed = getattr(raw_env, "_blue_landed_genuine", None)
            if blue_landed is not None:
                blue_land_count += int((blue_landed & ~prev_blue_landed).sum().item())
                prev_blue_landed = blue_landed.clone()

            red_landed = getattr(raw_env, "_red_landed_genuine", None)
            if red_landed is not None:
                red_land_count += int((red_landed & ~prev_red_landed).sum().item())
                prev_red_landed = red_landed.clone()

            sb_flag = getattr(raw_env, "_sb_flag", None)
            if sb_flag is not None:
                ep_sum_flag |= sb_flag

            done_ids = torch.where(dones.bool())[0]
            if done_ids.numel() > 0:
                for i in done_ids.tolist():
                    r = int(prev_region[i].item())
                    if r in FAR_REGION_IDS:
                        far_liftoff_samples.append(int(liftoff_counts[i].item()))
                        far_ep_count += 1
                        if bool(ep_sum_flag[i].item()):
                            softstop_count += 1
                liftoff_counts[done_ids] = 0
                trailing_was_airborne[done_ids] = False
                ep_sum_flag[done_ids] = False

            if step % 500 == 0:
                print(f"[INFO] {checkpoint}: step {step}/{cfg.steps}, far episodes so far: {far_ep_count}", file=sys.stderr)

    env.close()

    liftoffs = np.array(far_liftoff_samples) if far_liftoff_samples else np.array([0])
    forces = np.array(orange_land_forces) if orange_land_forces else np.array([])

    return {
        "far_episodes": far_ep_count,
        "far_softstop_rate": softstop_count / max(far_ep_count, 1),
        "mean_trailing_liftoffs": float(liftoffs.mean()),
        "pct_zero_liftoff": float((liftoffs == 0).mean()),
        "blue_landings": blue_land_count,
        "orange_landings": orange_land_count,
        "red_landings": red_land_count,
        "orange_landing_force_mean": float(forces.mean()) if forces.size else float("nan"),
        "orange_landing_force_median": float(np.median(forces)) if forces.size else float("nan"),
        "orange_landing_force_p10": float(np.percentile(forces, 10)) if forces.size else float("nan"),
    }


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint_a or not cfg.checkpoint_b:
        raise ValueError("--checkpoint-a and --checkpoint-b are required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")

    results = {}
    for label, ckpt in (("A (early)", cfg.checkpoint_a), ("B (late)", cfg.checkpoint_b)):
        results[label] = _run_one(ckpt, cfg, device)

    print("\n[SUMMARY] double-step / landing-quality comparison:")
    for label, r in results.items():
        print(f"\n  -- {label} --")
        for k, v in r.items():
            print(f"    {k:32s} {v}")


if __name__ == "__main__":
    main()
