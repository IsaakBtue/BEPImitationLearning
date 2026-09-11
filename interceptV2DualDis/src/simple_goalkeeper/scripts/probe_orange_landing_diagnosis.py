"""# DIAGNOSTIC SCRIPT -- read-only telemetry probe. No training impact.

User report (2026-09-11, run 6144_orangerestored): "orange ball landed
almost never fires but the foot does land on the orange ball." This probe
rolls out a real checkpoint across many parallel wide-crossing episodes and,
for each one, records the state of ALL FOUR gate conditions
(`_get_orange_reach_target_y`'s `candidate`/`newly_landed` logic) at the
moment the trailing foot gets closest to the orange target -- distance,
ground-contact force, foot speed, and settle count -- to identify which
condition is actually the bottleneck, instead of guessing.

Same pattern as probe_blue_landing_force.py (real env + real checkpoint +
AMPEnvWrapper, no synthetic teleporting).

Usage:
    uv run python src/simple_goalkeeper/scripts/probe_orange_landing_diagnosis.py \\
        --checkpoint logs/rsl_rl/intercept_simple_goalkeeper_multidisc/<run>/model_XXXXX.pt \\
        --num-envs 512 --steps 3000 --difficulty 0.5 --domain-rand 0.5
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

    import simple_goalkeeper.tasks  # noqa: F401  (registers TASK_ID)

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
        print(f"[INFO] ball_difficulty overridden to {cfg.difficulty}", file=sys.stderr)
    if cfg.domain_rand is not None:
        raw_env._domain_rand_curriculum = float(cfg.domain_rand)
        print(f"[INFO] domain_rand_curriculum overridden to {cfg.domain_rand}", file=sys.stderr)

    # Per-env running "closest approach to orange" record, reset whenever a
    # new episode starts (episode_length_buf <= 1) or once genuinely landed
    # (no more useful signal after that).
    best_dist = torch.full((N,), float("inf"), device=device)
    best_dist_contact = torch.zeros(N, dtype=torch.bool, device=device)
    best_dist_speed = torch.zeros(N, device=device)
    best_dist_settle = torch.zeros(N, dtype=torch.int64, device=device)
    best_dist_radius = torch.zeros(N, device=device)
    best_dist_speed_th = torch.zeros(N, device=device)
    ever_wide = torch.zeros(N, dtype=torch.bool, device=device)
    ever_landed_genuine = torch.zeros(N, dtype=torch.bool, device=device)
    ever_landed_was_free = torch.zeros(N, dtype=torch.bool, device=device)
    episode_done_recorded = torch.zeros(N, dtype=torch.bool, device=device)

    # Finished-episode stats, appended once per env each time its episode ends.
    finished_wide = 0
    finished_landed_genuine = 0
    finished_landed_was_free = 0
    closest_approach_rows = []  # (dist, contact, speed, settle, radius, speed_th)
    # Breakdown by region_id (0=left_near,1=left_far,2=right_near,3=right_far)
    # and by trailing-foot side (0=left,1=right), to catch a per-side bug
    # (e.g. force sensor mis-mapped for one side) that an aggregate rate
    # would hide.
    region_wide: dict[int, int] = {r: 0 for r in range(4)}
    region_landed: dict[int, int] = {r: 0 for r in range(4)}
    trailing_side_wide: dict[int, int] = {0: 0, 1: 0}
    trailing_side_landed: dict[int, int] = {0: 0, 1: 0}
    last_region = torch.full((N,), -1, dtype=torch.int64, device=device)
    last_trailing_idx = torch.full((N,), -1, dtype=torch.int64, device=device)

    with torch.inference_mode():
        prev_ep_len = raw_env.episode_length_buf.clone()
        for step in range(cfg.steps):
            obs_current = _get_actor_current_obs(env)
            actions = act_inference(obs_current, obs)
            obs, rew, dones, extras = env.step(actions)

            wide = getattr(raw_env, "_orange_wide", None)
            dist = getattr(raw_env, "_orange_dbg_dist", None)
            speed = getattr(raw_env, "_orange_dbg_speed", None)
            contact = getattr(raw_env, "_orange_dbg_contact", None)
            settle = getattr(raw_env, "_orange_dbg_settle", None)
            landed_genuine = getattr(raw_env, "_orange_landed_genuine", None)
            landed_was_free = getattr(raw_env, "_orange_landed_was_free", None)
            radius = getattr(raw_env, "_orange_landing_radius_current", None)
            speed_th = getattr(raw_env, "_orange_landing_speed_threshold_current", None)
            region_id = getattr(raw_env, "_region_id", None)
            trailing_idx_t = getattr(raw_env, "_orange_dbg_foot_idx", None)

            if wide is None or dist is None:
                continue

            if region_id is not None:
                last_region = torch.where(wide, region_id, last_region)
            if trailing_idx_t is not None:
                last_trailing_idx = torch.where(wide, trailing_idx_t, last_trailing_idx)

            ep_len = raw_env.episode_length_buf
            just_reset = ep_len <= 1
            # Flush finished episodes (detected via just_reset, i.e. this
            # tick started a NEW episode for that env -- the PREVIOUS
            # episode's final state is what we recorded up to last tick).
            newly_reset = just_reset & ~episode_done_recorded
            ids = torch.where(newly_reset)[0]
            for i in ids.tolist():
                if ever_wide[i]:
                    finished_wide += 1
                    r = int(last_region[i].item())
                    ti = int(last_trailing_idx[i].item())
                    if r in region_wide:
                        region_wide[r] += 1
                    if ti in trailing_side_wide:
                        trailing_side_wide[ti] += 1
                    if ever_landed_genuine[i]:
                        finished_landed_genuine += 1
                        if r in region_landed:
                            region_landed[r] += 1
                        if ti in trailing_side_landed:
                            trailing_side_landed[ti] += 1
                    if ever_landed_was_free[i]:
                        finished_landed_was_free += 1
                    if best_dist[i].item() < float("inf"):
                        closest_approach_rows.append((
                            float(best_dist[i].item()),
                            bool(best_dist_contact[i].item()),
                            float(best_dist_speed[i].item()),
                            int(best_dist_settle[i].item()),
                            float(best_dist_radius[i].item()),
                            float(best_dist_speed_th[i].item()),
                        ))
            # Reset per-episode trackers for envs that just started fresh.
            best_dist[newly_reset] = float("inf")
            ever_wide[newly_reset] = False
            ever_landed_genuine[newly_reset] = False
            ever_landed_was_free[newly_reset] = False
            last_region[newly_reset] = -1
            last_trailing_idx[newly_reset] = -1
            episode_done_recorded = just_reset.clone()

            ever_wide |= wide
            if landed_genuine is not None:
                ever_landed_genuine |= landed_genuine
            if landed_was_free is not None:
                ever_landed_was_free |= landed_was_free

            improved = wide & (dist < best_dist)
            if improved.any():
                best_dist = torch.where(improved, dist, best_dist)
                if contact is not None:
                    best_dist_contact = torch.where(improved, contact, best_dist_contact)
                if speed is not None:
                    best_dist_speed = torch.where(improved, speed, best_dist_speed)
                if settle is not None:
                    best_dist_settle = torch.where(improved, settle, best_dist_settle)
                if radius is not None:
                    best_dist_radius = torch.where(improved, torch.full_like(best_dist, float(radius)), best_dist_radius)
                if speed_th is not None:
                    best_dist_speed_th = torch.where(improved, torch.full_like(best_dist, float(speed_th)), best_dist_speed_th)

            if step % 300 == 0:
                print(
                    f"[INFO] step {step}/{cfg.steps}, finished_wide={finished_wide} "
                    f"landed_genuine={finished_landed_genuine} landed_was_free={finished_landed_was_free} "
                    f"rows={len(closest_approach_rows)}",
                    file=sys.stderr,
                )

    env.close()

    print(f"\n[SUMMARY] finished wide-crossing episodes: {finished_wide}")
    print(f"  orange_landed_genuine: {finished_landed_genuine} ({100*finished_landed_genuine/max(finished_wide,1):.1f}%)")
    print(f"  orange_landed_was_free (excluded from genuine): {finished_landed_was_free}")

    region_names = {0: "left_near", 1: "left_far", 2: "right_near", 3: "right_far"}
    print("\n[BY REGION]")
    for r in range(4):
        w = region_wide[r]
        l = region_landed[r]
        print(f"  {region_names[r]:12s} wide={w:5d}  landed_genuine={l:5d}  ({100*l/max(w,1):.1f}%)")

    side_names = {0: "left", 1: "right"}
    print("\n[BY TRAILING-FOOT SIDE]")
    for s in (0, 1):
        w = trailing_side_wide[s]
        l = trailing_side_landed[s]
        print(f"  trailing={side_names[s]:6s} wide={w:5d}  landed_genuine={l:5d}  ({100*l/max(w,1):.1f}%)")

    if not closest_approach_rows:
        print("\nNo closest-approach rows recorded -- env._orange_wide never went True. Check region distribution.")
        return

    arr = np.array(closest_approach_rows, dtype=object)
    dists = np.array([r[0] for r in closest_approach_rows])
    contacts = np.array([r[1] for r in closest_approach_rows])
    speeds = np.array([r[2] for r in closest_approach_rows])
    settles = np.array([r[3] for r in closest_approach_rows])
    radii = np.array([r[4] for r in closest_approach_rows])
    speed_ths = np.array([r[5] for r in closest_approach_rows])

    inside_radius = dists < radii
    contact_ok_given_inside = contacts[inside_radius]
    speed_ok_given_inside = (speeds < speed_ths)[inside_radius]
    settle_ok_given_inside = (settles >= 3)[inside_radius]

    print(f"\n[CLOSEST APPROACH TO ORANGE, per wide episode, n={len(dists)}]")
    print(f"  dist_to_orange:  mean={dists.mean():.3f}  median={np.median(dists):.3f}  "
          f"p90={np.percentile(dists,90):.3f}  radius(mean)={radii.mean():.3f}")
    print(f"  fraction that ever got INSIDE the landing radius: {inside_radius.mean()*100:.1f}%")
    if inside_radius.any():
        print(f"  of those inside-radius envs, at their closest tick:")
        print(f"    foot_in_contact (>40N) True: {contact_ok_given_inside.mean()*100:.1f}%")
        print(f"    foot_speed < speed_threshold True: {speed_ok_given_inside.mean()*100:.1f}%")
        print(f"    settle_count >= 3 True: {settle_ok_given_inside.mean()*100:.1f}%")
    print(f"  speed at closest tick: mean={speeds.mean():.3f} m/s, median={np.median(speeds):.3f}, "
          f"p90={np.percentile(speeds,90):.3f}, threshold(mean)={speed_ths.mean():.3f}")
    print(f"  settle_count at closest tick: mean={settles.mean():.2f}, median={np.median(settles)}, "
          f"max_observed={settles.max()}")


if __name__ == "__main__":
    main()
