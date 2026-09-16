"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

User report: "landing_ok never fires now" after several genuine-lift-gate
loosenings today (threshold 0.04->0.02, square side 0.30->0.32, dropped the
downward-velocity requirement). Hypothesis: the gate is now loose enough
that raw env._blue_landed fires very early (within the first 10 steps,
often from mere post-reset settling), tripping the SEPARATE, untouched
"free landing" exclusion (_get_reach_target_y's own
_BLUE_LANDING_FREE_STEP_THRESHOLD=10) -- so env._blue_landed IS firing, but
env._blue_landed_genuine (what landing_ok actually reads) almost always
isn't, because almost every landing gets classified "free".

Measures, left_far only: raw env._blue_landed rising-edge events vs how
many of those get excluded as "free" (episode_length_buf < 10 at the
moment they fired).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_blue_free_landing_rate.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
        --num-envs 16 --steps 1500 --difficulty 1.0
"""
from __future__ import annotations

import os
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
    num_envs: int = 16
    steps: int = 1500
    difficulty: float = 1.0
    device: str | None = None
    target_events: int = 10


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")

    import simple_goalkeeper.tasks  # noqa: F401
    from simple_goalkeeper.rsl_rl_multi.him_amp_on_policy_runner import _get_actor_current_obs
    from simple_goalkeeper.mdp.regions import REGION_NAMES, pin_region_on_reset
    from mjlab.managers.event_manager import EventTermCfg as _EvtCfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict)
    env_cfg.scene.num_envs = cfg.num_envs
    region_id = REGION_NAMES.index("left_far")
    env_cfg.events["assign_static_regions"] = _EvtCfg(
        func=pin_region_on_reset, mode="reset", params={"region_id": region_id},
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = AMPEnvWrapper(env, clip_actions=None, motion_dataset=None)

    resume_path = Path(cfg.checkpoint)
    if not resume_path.is_absolute() and not resume_path.exists():
        resume_path = Path.cwd() / resume_path
    print(f"[INFO] Loading checkpoint: {resume_path}", file=sys.stderr)
    runner_cls = load_runner_cls(TASK_ID)
    runner = runner_cls(env, agent_cfg, log_dir=None, device=device)
    runner.load(str(resume_path), load_optimizer=False)
    act_inference = runner.get_inference_policy(device=device)

    raw_env = env.unwrapped
    N = cfg.num_envs
    obs, _ = env.reset()
    raw_env._ball_difficulty = float(cfg.difficulty)

    prev_landed = torch.zeros(N, dtype=torch.bool, device=device)
    raw_events = 0
    free_events = 0
    genuine_events = 0
    landing_ep_steps: list[int] = []

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            landed = raw_env._blue_landed
            newly = landed & ~prev_landed
            ids = torch.where(newly)[0]
            if ids.numel() > 0:
                raw_events += ids.numel()
                was_free = raw_env._blue_landed_was_free[ids]
                free_events += int(was_free.sum().item())
                genuine_events += int((~was_free).sum().item())
                for i in ids.tolist():
                    landing_ep_steps.append(int(raw_env.episode_length_buf[i].item()))
            prev_landed = landed.clone()

            if step % 300 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, raw events: {raw_events}, free: {free_events}, genuine: {genuine_events}", file=sys.stderr)
            if raw_events >= cfg.target_events:
                break

    env.close()

    print("\n[SUMMARY] free-landing rate, left_far:")
    print(f"  raw env._blue_landed rising-edge events   {raw_events}")
    print(f"  classified FREE (episode_length_buf < 10) {free_events}  ({free_events/max(raw_events,1):.1%})")
    print(f"  classified GENUINE (what landing_ok reads) {genuine_events}  ({genuine_events/max(raw_events,1):.1%})")
    print(f"  episode_length_buf at moment of landing: {sorted(landing_ep_steps)}")


if __name__ == "__main__":
    main()
