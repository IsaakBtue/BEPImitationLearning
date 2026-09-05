"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

Follow-up to probe_region_reward_asymmetry.py's finding: `inner_face_orientation_save`
fires 0/789 times on left_near episodes but 730/730 on right_near episodes
(same checkpoint). This script measures the RAW quantity that reward checks
(the assigned foot's toe-axis world yaw, exactly as rewards.py:
inner_face_orientation_save/foot_inner_face_continuous compute it) at every
softstop-firing moment, split by assigned foot (left/right), to determine
whether the left foot simply never rotates (stuck near 0deg) or rotates in
the WRONG direction (toward -Y, mirroring the right foot's learned
direction instead of the correct +Y).

Does NOT import or call rewards.py -- re-implements the measurement
independently (same pattern as probe_inner_face_axis_lr.py) so it's not
possible for a shared bug to hide itself from both the reward and the probe.

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_left_right_yaw.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
        --num-envs 512 --steps 3000
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
from mjlab.utils.lab_api.math import quat_apply

TASK_ID = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
REGION_NAMES = {0: "left_near", 1: "left_far", 2: "right_near", 3: "right_far"}


@dataclass(frozen=True)
class ProbeConfig:
    checkpoint: str = ""
    num_envs: int = 512
    steps: int = 3000
    device: str | None = None


def main() -> None:
    cfg = tyro.cli(ProbeConfig)
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import simple_goalkeeper.tasks  # noqa: F401
    from simple_goalkeeper.mdp.rewards import _get_correct_foot_idx

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, dict)

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
    robot = raw_env.scene["robot"]
    foot_body_ids = robot.find_bodies(["left_foot_link", "right_foot_link"])[0]

    N = cfg.num_envs
    arange_n = torch.arange(N, device=device)

    obs, _ = env.reset()
    prev_softstop = torch.zeros(N, dtype=torch.bool, device=device)

    records = []  # (region, foot_idx, yaw_deg, target_signed_deg)

    with torch.inference_mode():
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            softstop_flag = getattr(
                raw_env, "_softstop_flag",
                torch.zeros(N, dtype=torch.bool, device=device),
            )
            just_fired = softstop_flag & ~prev_softstop
            prev_softstop = softstop_flag.clone()

            if just_fired.any():
                foot_idx = _get_correct_foot_idx(raw_env, "ball")  # (N,) 0=left,1=right
                region_id = getattr(
                    raw_env, "_region_id",
                    torch.full((N,), -1, dtype=torch.int64, device=device),
                )
                foot_quat_w = robot.data.body_link_quat_w[:, foot_body_ids, :]  # (N,2,4)
                assigned_quat_w = foot_quat_w[arange_n, foot_idx]  # (N,4)
                x_local = torch.tensor([1.0, 0.0, 0.0], device=device).expand(N, -1)
                foot_x_w = quat_apply(assigned_quat_w, x_local)
                yaw_deg = torch.rad2deg(torch.atan2(foot_x_w[:, 1], foot_x_w[:, 0]))
                expected_sign = torch.where(foot_idx == 0, 1.0, -1.0)
                target_signed_deg = expected_sign * 80.0

                fired_ids = just_fired.nonzero(as_tuple=True)[0]
                for i in fired_ids.tolist():
                    records.append((
                        int(region_id[i].item()), int(foot_idx[i].item()),
                        float(yaw_deg[i].item()), float(target_signed_deg[i].item()),
                    ))

            if step % 300 == 0:
                print(f"[INFO] step {step}/{cfg.steps}, {len(records)} saves so far", file=sys.stderr)

    env.close()

    print("\n[SUMMARY] measured toe-axis world yaw_deg at softstop-fire moment, by region:")
    by_region: dict[int, list[tuple]] = {}
    for r in records:
        by_region.setdefault(r[0], []).append(r)
    for rid in (0, 1, 2, 3):
        rows = by_region.get(rid, [])
        if not rows:
            print(f"  {REGION_NAMES.get(rid, rid):12s} n=0")
            continue
        yaws = np.array([r[2] for r in rows])
        targets = np.array([r[3] for r in rows])
        name = REGION_NAMES.get(rid, f"region_{rid}")
        print(f"  {name:12s} n={len(rows)}  target={targets[0]:+.0f}deg  "
              f"yaw: mean={yaws.mean():+7.2f} std={yaws.std():6.2f} "
              f"p10={np.percentile(yaws,10):+7.2f} p50={np.percentile(yaws,50):+7.2f} "
              f"p90={np.percentile(yaws,90):+7.2f}")


if __name__ == "__main__":
    main()
