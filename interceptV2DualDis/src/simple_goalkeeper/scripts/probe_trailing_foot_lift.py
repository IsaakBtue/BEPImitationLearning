"""Diagnostic-only script: teleports the robot's trailing foot to controlled
(Y offset from its current sub-target, height) positions and prints
trailing_foot_lift's actual live reward at each one, to confirm the
2026-09-11 rewrite (resting-height baseline + shrinking-target-near-orange/
red mechanism, mirroring leading_foot_lift) behaves as intended.

Not a training script. Reuses probe_leading_foot_lift.py's teleport
technique verbatim (same pitfalls apply -- fresh root pose every call, defeat
the per-tick memoization guard, reset settle/landed state before each "not
yet landed" scenario).

Run: uv run python -m simple_goalkeeper.scripts.probe_trailing_foot_lift
"""

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import simple_goalkeeper.tasks  # noqa: F401  (registers tasks)
from simple_goalkeeper.mdp.rewards import (
    trailing_foot_lift,
    _get_correct_foot_idx,
    _get_ball_crossing_y,
    _get_reach_target_y,
    _get_orange_reach_target_y,
    _get_red_reach_target_y,
)
from simple_goalkeeper.tasks.goalkeeper_env_cfg import BALL_NAME, _FEET_CFG

_FOOT_RESTING_HEIGHT = 0.03  # must match trailing_foot_lift's own constant
_FORCED_DELTA = 0.8  # env-relative crossing offset -- safely > wide_threshold (0.5)
_OUTER_ZONE_MARGIN = 0.05  # must match trailing_foot_lift's own _OUTER_ZONE_MARGIN (2026-09-11: outer zone is now current_radius + this, not a fixed absolute value)
_DECAY_STEEPNESS = 1.0


def main() -> None:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    task_id = "Mjlab-BeyondAMP-Goalkeeper-T1-MultiDisc"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    obs, _ = env.reset()
    _FEET_CFG.resolve(env.scene)

    for _ in range(5):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
        env.step(actions)

    robot = env.scene[_FEET_CFG.name]
    goal_x = env.scene.env_origins[:, 0]
    floor_z = env.scene.env_origins[:, 2]
    start_y = env.scene.env_origins[:, 1]

    env._ball_crossing_y = start_y + _FORCED_DELTA
    env._rsi_cross_y = torch.full((env.num_envs,), _FORCED_DELTA, device=device)
    # trailing_foot_lift's `active` gate reads env._orange_wide, which is only
    # ever set by _get_orange_reach_target_y reading env._blue_wide -- which
    # in turn only _get_reach_target_y (blue) sets. Must call it at least once
    # or `active` silently stays False (fallback) for the whole probe.
    _get_reach_target_y(env, BALL_NAME, asset_cfg=_FEET_CFG)
    full_y = _get_ball_crossing_y(env, BALL_NAME)
    foot_idx = _get_correct_foot_idx(env, BALL_NAME)
    trailing_idx = 1 - foot_idx
    orange_y = _get_orange_reach_target_y(env, BALL_NAME, asset_cfg=_FEET_CFG)
    red_y = _get_red_reach_target_y(env, BALL_NAME, asset_cfg=_FEET_CFG)
    print(
        f"Forced full_y (green)={full_y.item():.3f}, orange_y={orange_y.item():.3f}, "
        f"red_y={red_y.item():.3f}, leading_idx={foot_idx.item()}, trailing_idx={trailing_idx.item()}"
    )

    def teleport_trailing_foot(
        target_y: float, delta_y_from_target: float, effective_height: float,
        force_orange_landed: bool = False, force_red_landed: bool = False,
    ) -> None:
        ti = int(trailing_idx[0].item())
        foot_pos_w = robot.data.body_link_pos_w[:, _FEET_CFG.body_ids, :]
        current_foot = foot_pos_w[0, ti].clone()
        current_root_pose = robot.data.root_link_pose_w.clone()  # fresh, not a stale snapshot
        target_foot = torch.tensor(
            [
                goal_x[0].item(),
                target_y + delta_y_from_target,
                floor_z[0].item() + _FOOT_RESTING_HEIGHT + effective_height,
            ],
            device=device,
        )
        delta = target_foot - current_foot
        new_pose = current_root_pose.clone()
        new_pose[0, :3] += delta
        robot.write_root_link_pose_to_sim(new_pose, env_ids=torch.tensor([0], device=device))
        env.sim.forward()
        env._ball_crossing_y = start_y + _FORCED_DELTA
        env._rsi_cross_y = torch.full((env.num_envs,), _FORCED_DELTA, device=device)
        env.episode_length_buf += 1  # defeat the per-tick memoization guard

        env._orange_landed[:] = force_orange_landed
        env._orange_landed_was_free[:] = False
        env._orange_settle_count[:] = 0
        env._orange_was_airborne[:] = False
        env._red_landed[:] = force_red_landed
        env._red_landed_was_free[:] = False
        env._red_settle_count[:] = 0
        env._red_was_airborne[:] = False
        # blue must be genuinely landed for red_active's own gate, and the
        # 25-step delay must have elapsed, for the "post-orange" scenarios
        # below to actually exercise the red sub-target.
        env._blue_landed[:] = True
        env._blue_landed_was_free[:] = False
        if force_orange_landed:
            env._orange_landed_genuine_step[:] = env.episode_length_buf - 30

    trailing_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)  # seed env._orange_*/_red_* attributes

    print("\n=== Phase A: targeting ORANGE (not yet landed) ===")
    print(f"{'scenario':<50} {'dist':>8} {'eff_target':>11} {'reward':>9}")
    print("-" * 84)
    scenarios_orange = [
        ("far from orange (-1.0m), flat (h=0)", -1.00, 0.0),
        ("far from orange (-1.0m), lifted (h=0.10)", -1.00, 0.10),
        ("just outside decay zone (-0.25m out), h=0", -0.25, 0.0),
        ("mid decay zone (-0.10m out), flat (h=0)", -0.10, 0.0),
        ("mid decay zone (-0.10m out), h=0.10", -0.10, 0.10),
        ("AT orange (0m out), flat (h=0)", 0.00, 0.0),
        ("AT orange (0m out), h=0.10 (still lifted)", 0.00, 0.10),
        ("PAST orange / overshot (+0.20m), flat (h=0)", 0.20, 0.0),
        ("PAST orange / overshot (+0.20m), h=0.10", 0.20, 0.10),
    ]
    for label, dy, h in scenarios_orange:
        teleport_trailing_foot(orange_y.item(), dy, h)
        r = trailing_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)
        print(f"{label:<50} {dy:>8.2f} {'-':>11} {r[0].item():>9.4f}")

    print("\n=== Phase B: post-orange, targeting RED (blue+orange genuinely landed) ===")
    print(f"{'scenario':<50} {'dist':>8} {'eff_target':>11} {'reward':>9}")
    print("-" * 84)
    scenarios_red = [
        ("far from red (-1.0m), flat (h=0)", -1.00, 0.0),
        ("mid decay zone (-0.10m out), flat (h=0)", -0.10, 0.0),
        ("mid decay zone (-0.10m out), h=0.10", -0.10, 0.10),
        ("AT red (0m out), flat (h=0)", 0.00, 0.0),
        ("AT red (0m out), h=0.10 (still lifted)", 0.00, 0.10),
        ("PAST red / overshot (+0.20m), flat (h=0)", 0.20, 0.0),
    ]
    for label, dy, h in scenarios_red:
        teleport_trailing_foot(red_y.item(), dy, h, force_orange_landed=True, force_red_landed=False)
        r = trailing_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)
        print(f"{label:<50} {dy:>8.2f} {'-':>11} {r[0].item():>9.4f}")

    print("\n=== Phase C: fully done (red genuinely landed too) -- should restore to standard target ===")
    for label, dy, h in [
        ("flat (h=0), post-red-landing", 0.0, 0.0),
        ("lifted (h=0.10), post-red-landing", 0.0, 0.10),
    ]:
        teleport_trailing_foot(red_y.item(), dy, h, force_orange_landed=True, force_red_landed=True)
        r = trailing_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)
        print(f"{label:<50} {'':>8} {'0.100 (std)':>11} {r[0].item():>9.4f}  [active={bool(env._orange_wide[0] & ~env._red_landed_genuine[0])}]")

    print("\n=== Baseline sanity: flat foot raw height reads ~0.03m; after subtracting resting height -> 0 ===")
    teleport_trailing_foot(orange_y.item(), -1.0, 0.0)
    ti = int(trailing_idx[0].item())
    foot_pos_w = robot.data.body_link_pos_w[:, _FEET_CFG.body_ids, :]
    raw_z = foot_pos_w[0, ti, 2].item()
    print(f"raw world Z = {raw_z:.4f}, floor_z = {floor_z[0].item():.4f}, raw-above-floor = {raw_z - floor_z[0].item():.4f} (expect ~0.03)")


if __name__ == "__main__":
    main()
