"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Re-verifies the `env._blue_was_airborne` gate (rewards.py:_get_reach_target_y)
actually requires a REAL foot lift, not just contact-force sensor noise
crossing the 40N threshold while the foot stays essentially on the ground.
`was_airborne` only checks force (`foot_in_contact = assigned_force >
_LANDING_FORCE_THRESHOLD`) -- it has no height component at all, so this is
worth checking empirically: a foot resting near the boundary could flicker
`foot_in_contact` False for a tick from force noise alone, satisfying the
gate without ever visibly leaving the ground.

Measures, for every genuine blue landing (`env._blue_landed_genuine` flipping
False->True):
  - peak real foot height reached (Z, floor-relative, corrected for the
    known ~0.03m foot-body-link-vs-ground-contact baseline offset --
    _FOOT_CONTACT_BELOW_BODY convention, see events.py/rewards.py's own
    leading_foot_lift docstring) during the was_airborne window immediately
    before that landing.
  - how many consecutive steps was_airborne stayed True before landing
    (duration of the "airborne" window).
  - how many steps after episode reset the FIRST was_airborne flip happens
    (near-0 would suggest a spawn-settling transient, not a real step).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_blue_landing_liftoff.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
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
_FOOT_CONTACT_BELOW_BODY = 0.030  # matches events.py / leading_foot_lift's own baseline convention


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = ""
    num_envs: int = 512
    steps: int = 3000
    difficulty: float = 1.0
    device: str | None = None
    force_region: str | None = None  # "left_near"|"left_far"|"right_near"|"right_far"
    target_landings: int | None = None  # stop early once this many genuine landings observed


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")

    import simple_goalkeeper.tasks  # noqa: F401
    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import _get_actor_current_obs
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict)
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.force_region is not None:
        from simple_goalkeeper.mdp.regions import REGION_NAMES, pin_region_on_reset
        from mjlab.managers.event_manager import EventTermCfg as _EvtCfg
        region_id = REGION_NAMES.index(cfg.force_region)
        env_cfg.events["assign_static_regions"] = _EvtCfg(
            func=pin_region_on_reset, mode="reset", params={"region_id": region_id},
        )
        print(f"[INFO] Region pinned to '{cfg.force_region}'", file=sys.stderr)
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

    asset_cfg = raw_env.reward_manager.get_term_cfg("blue_ball_landed").params["asset_cfg"]
    robot = raw_env.scene[asset_cfg.name]

    prev_blue_landed = torch.zeros(N, dtype=torch.bool, device=device)
    prev_was_airborne = torch.zeros(N, dtype=torch.bool, device=device)
    peak_height_since_airborne = torch.zeros(N, device=device)
    airborne_duration = torch.zeros(N, dtype=torch.int64, device=device)
    first_airborne_step = torch.full((N,), -1, dtype=torch.int64, device=device)
    step_since_reset = torch.zeros(N, dtype=torch.int64, device=device)

    peak_heights: list[float] = []
    durations: list[int] = []
    first_airborne_steps: list[int] = []
    landings = 0

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            step_since_reset += 1

            foot_idx = _get_correct_foot_idx(raw_env, "ball")
            foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
            foot_z = foot_pos_w[arange_n, foot_idx, 2]
            floor_z = raw_env.scene.env_origins[:, 2]
            clearance = (foot_z - floor_z - _FOOT_CONTACT_BELOW_BODY).clamp(min=0.0)

            was_airborne = raw_env._blue_was_airborne
            newly_airborne = was_airborne & ~prev_was_airborne
            first_flip = newly_airborne & (first_airborne_step < 0)
            first_airborne_step = torch.where(first_flip, step_since_reset, first_airborne_step)

            peak_height_since_airborne = torch.where(
                was_airborne,
                torch.maximum(peak_height_since_airborne, clearance),
                peak_height_since_airborne,
            )
            airborne_duration = torch.where(was_airborne, airborne_duration + 1, airborne_duration)
            prev_was_airborne = was_airborne.clone()

            blue_landed = raw_env._blue_landed_genuine
            newly_landed = blue_landed & ~prev_blue_landed
            ids = torch.where(newly_landed)[0]
            if ids.numel() > 0:
                landings += ids.numel()
                for i in ids.tolist():
                    peak_heights.append(float(peak_height_since_airborne[i].item()))
                    durations.append(int(airborne_duration[i].item()))
                    if int(first_airborne_step[i].item()) >= 0:
                        first_airborne_steps.append(int(first_airborne_step[i].item()))
            prev_blue_landed = blue_landed.clone()

            done_ids = torch.where(dones.bool())[0]
            if done_ids.numel() > 0:
                peak_height_since_airborne[done_ids] = 0.0
                airborne_duration[done_ids] = 0
                first_airborne_step[done_ids] = -1
                step_since_reset[done_ids] = 0
                prev_was_airborne[done_ids] = False

            if step % 500 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, landings so far: {landings}", file=sys.stderr)
            if cfg.target_landings is not None and landings >= cfg.target_landings:
                print(f"[INFO] reached target of {cfg.target_landings} landings at step {step}, stopping early", file=sys.stderr)
                break

    env.close()

    ph = np.array(peak_heights) if peak_heights else np.array([0.0])
    du = np.array(durations) if durations else np.array([0])
    fa = np.array(first_airborne_steps) if first_airborne_steps else np.array([0])

    print("\n[SUMMARY] blue was_airborne / genuine-lift diagnostic:")
    print(f"  genuine blue landings observed        {landings}")
    print(f"  peak clearance before landing (m):    mean={ph.mean():.4f} median={np.median(ph):.4f} p10={np.percentile(ph,10):.4f} p90={np.percentile(ph,90):.4f}")
    print(f"  fraction with peak clearance < 1cm:    {(ph < 0.01).mean():.1%}  (near-0 = suspicious, likely force-sensor noise not a real lift)")
    print(f"  fraction with peak clearance < 2cm:    {(ph < 0.02).mean():.1%}")
    print(f"  was_airborne duration before landing (steps): mean={du.mean():.1f} median={np.median(du):.1f}")
    print(f"  first was_airborne flip after reset (steps):  mean={fa.mean():.1f} median={np.median(fa):.1f}  (near-0 = suspicious spawn transient)")


if __name__ == "__main__":
    main()
