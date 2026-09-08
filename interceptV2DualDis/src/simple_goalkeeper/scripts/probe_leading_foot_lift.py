"""Diagnostic-only script: teleports the robot's leading foot to controlled
(Y offset from blue, height) positions and prints leading_foot_lift's actual
live reward at each one, to directly confirm the shrinking-target-near-blue
mechanism (rewards.py:leading_foot_lift, 2026-09-08) behaves as intended.

Not a training script. Techniques used, in case they're needed again:
- forces a WIDE crossing by overwriting env._ball_crossing_y AND
  env._rsi_cross_y (the latter is what _get_reach_target_y's wide/narrow
  classification actually reads -- overwriting only _ball_crossing_y is not
  enough, since _rsi_cross_y takes priority when present)
- rigidly translates the whole robot (root pose only, no IK) so the leading
  foot lands at an exact target position -- unnatural pose, diagnostic only.
  IMPORTANT: read the robot's CURRENT root pose fresh on every call, not a
  pose captured once at the start -- reusing a stale snapshot silently
  breaks every teleport after the first (each one lands somewhere unrelated
  to the requested target).
- accounts for the ~0.03m foot resting-height baseline leading_foot_lift
  itself subtracts, so "h=X" here means the EFFECTIVE (post-baseline) height
  the reward function actually sees, not the raw world Z coordinate.
- bumps env.episode_length_buf (without stepping physics) to defeat
  _get_reach_target_y's per-tick memoization guard, so every teleport gets a
  freshly recomputed dist_to_blue instead of a stale one from a prior call.
- fully resets env._blue_landed/_was_free/_settle_count/_was_airborne before
  every "not yet landed" scenario -- otherwise the settle counter's
  accumulated history from a PRIOR row (e.g. several consecutive
  already-grounded-and-within-radius calls) can silently flip a landing on
  the very next row even after explicitly setting _blue_landed=False.

Run: uv run python -m simple_goalkeeper.scripts.probe_leading_foot_lift
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import simple_goalkeeper.tasks  # noqa: F401  (registers tasks)
from simple_goalkeeper.mdp.rewards import (
    leading_foot_lift,
    _get_correct_foot_idx,
    _get_ball_crossing_y,
)
from simple_goalkeeper.tasks.goalkeeper_env_cfg import BALL_NAME, _FEET_CFG

_FOOT_RESTING_HEIGHT = 0.03  # must match leading_foot_lift's own constant
_FORCED_DELTA = 0.8  # env-relative crossing offset -- safely > wide_threshold (0.5)
_OUTER_ZONE = 0.20  # must match leading_foot_lift's own _BLUE_APPROACH_OUTER_ZONE
_DECAY_STEEPNESS = 1.0  # must match leading_foot_lift's own _DECAY_STEEPNESS (2026-09-08 exponential shape, tuned down from 4.0)


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

    # Step a few times so episode_length_buf > 1 -- otherwise
    # _get_ball_crossing_y recomputes env._ball_crossing_y from scratch every
    # call (just_reset stays True), clobbering our forced crossing override.
    for _ in range(5):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
        env.step(actions)

    robot = env.scene[_FEET_CFG.name]
    goal_x = env.scene.env_origins[:, 0]
    floor_z = env.scene.env_origins[:, 2]
    start_y = env.scene.env_origins[:, 1]

    env._ball_crossing_y = start_y + _FORCED_DELTA
    env._rsi_cross_y = torch.full((env.num_envs,), _FORCED_DELTA, device=device)
    full_y = _get_ball_crossing_y(env, BALL_NAME)
    half_y = start_y + (full_y - start_y) / 2.0
    foot_idx = _get_correct_foot_idx(env, BALL_NAME)
    print(
        f"Forced full_y (green)={full_y.item():.3f}, half_y (blue)={half_y.item():.3f}, "
        f"orange_y (approx, full-0.25)={(half_y.item() - 0.25):.3f}, foot_idx={foot_idx.item()}"
    )

    def teleport_leading_foot(
        delta_y_from_blue: float, effective_height: float, force_landed_genuine: bool = False
    ) -> None:
        """Rigidly translates the whole robot so the LEADING foot ends up at
        (goal_x, half_y + delta_y_from_blue, floor_z + resting_height +
        effective_height). `effective_height` is what leading_foot_lift will
        actually measure (post baseline-subtraction)."""
        fi = int(foot_idx[0].item())
        foot_pos_w = robot.data.body_link_pos_w[:, _FEET_CFG.body_ids, :]
        current_foot = foot_pos_w[0, fi].clone()
        current_root_pose = robot.data.root_link_pose_w.clone()  # fresh, not a stale snapshot
        target_foot = torch.tensor(
            [
                goal_x[0].item(),
                half_y[0].item() + delta_y_from_blue,
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
        env.episode_length_buf += 1  # defeat the per-tick memoization guard without stepping physics

        env._blue_landed[:] = False
        env._blue_landed_was_free[:] = False
        env._blue_settle_count[:] = 0
        env._blue_was_airborne[:] = False
        if force_landed_genuine:
            env._blue_landed[:] = True
            env._blue_landed_was_free[:] = False

    leading_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)  # seed env._blue_* attributes

    scenarios = [
        ("far from blue (-1.0m), flat (h=0)", -1.00, 0.0),
        ("far from blue (-1.0m), lifted to standard (h=0.10)", -1.00, 0.10),
        ("just outside decay zone (-0.25m out), h=0", -0.25, 0.0),
        ("just outside decay zone (-0.25m out), h=0.10", -0.25, 0.10),
        ("mid decay zone (-0.10m out), flat (h=0)", -0.10, 0.0),
        ("mid decay zone (-0.10m out), h=0.02", -0.10, 0.02),
        ("mid decay zone (-0.10m out), h=0.10", -0.10, 0.10),
        ("AT blue (0m out), flat (h=0)", 0.00, 0.0),
        ("AT blue (0m out), h=0.02", 0.00, 0.02),
        ("AT blue (0m out), h=0.10 (still lifted)", 0.00, 0.10),
        ("PAST blue / overshot (+0.20m), flat (h=0)", 0.20, 0.0),
        ("PAST blue / overshot (+0.20m), h=0.02", 0.20, 0.02),
        ("PAST blue / overshot (+0.20m), h=0.10", 0.20, 0.10),
    ]

    print(f"\n{'scenario':<50} {'dist_to_blue':>13} {'eff_target':>11} {'reward':>9}")
    print("-" * 88)
    for label, dy, h in scenarios:
        teleport_leading_foot(dy, h, force_landed_genuine=False)
        r = leading_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)
        dist = env._blue_dbg_dist[0].item()
        radius = env._blue_landing_radius_current
        import math as _math
        x = max(0.0, min(1.0, (dist - radius) / (_OUTER_ZONE - radius)))
        frac = (_math.exp(_DECAY_STEEPNESS * x) - 1.0) / (_math.exp(_DECAY_STEEPNESS) - 1.0)
        eff_target = 0.0 if dy >= radius else 0.10 * frac  # overshoot forces target to 0 too
        print(
            f"{label:<50} {dist:>13.3f} {eff_target:>11.3f} {r[0].item():>9.4f}  "
            f"[wide={bool(env._blue_wide[0])} genuine={bool(env._blue_landed_genuine[0])}]"
        )

    print("\nPost-landing ('blue fired', env._blue_landed_genuine forced True):")
    print("-" * 88)
    for label, dy, h in [
        ("flat (h=0), post-landing", 0.0, 0.0),
        ("lifted (h=0.10), post-landing", 0.0, 0.10),
    ]:
        teleport_leading_foot(dy, h, force_landed_genuine=True)
        r = leading_foot_lift(env, BALL_NAME, asset_cfg=_FEET_CFG)
        print(f"{label:<50} {'':>13} {'0.100 (std)':>11} {r[0].item():>9.4f}")


if __name__ == "__main__":
    main()
