"""Goalkeeper-specific observation terms for Booster T1."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_HAND_CFG = SceneEntityCfg("robot", body_names=("left_hand_link", "right_hand_link"))


def _compute_ball_visibility(env: ManagerBasedRlEnv, ball_name: str) -> torch.Tensor:
    """Compute ball visibility mask (N,) bool: True = visible.

    Mirrors upstream compute_observations() visibility logic exactly:

    initial_vanish: ball hidden during catchstep >= startstep warmup.
        (self.catchstep < self.startstep) in original — True when warmup expired.
        Port: catchstep stored on env._catchstep (int tensor). startstep ≈ 43 (50 - randint(3,10)).
        We use a fixed startstep=43 as a conservative upper bound.

    flying: ball is in the camera field of view and approaching:
        end_target_local (ball in body frame):
            x > 0.05 AND x < 3.4   → x_local (approach axis = forwards)
            |y| < 2.0               → |y_local| < 2.0 (lateral axis = left/right)
            z < 1.8
        catchstep > 0: ball has been launched (warmup not finished)
        ball moving closer: x_local < x_last OR x_last == 0 → ball approaching

    random_vanish: ball disappears at a random step during flight.
        vanish_step sampled per-env at reset from randint(0, 30).
        ball_visible_step counts consecutive flying steps.
        random_vanish = ball_visible_step > vanish_step.

    visible = initial_vanish & flying & ~random_vanish

    Result is cached per-step so that ball_pos_b and ball_vel_b share one
    computation without re-running stateful side effects (ball_last update,
    visible_step increment). Without the cache the second caller always sees
    approaching=False (ball_last was just set to current y) → flying=False →
    visible=False, permanently zeroing ball_vel observations.
    """
    # Return cached result if already computed this step.
    if getattr(env, "_ball_vis_step", -1) == env.common_step_counter:
        return env._ball_vis_cache

    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                           # (N, 3)
    env_origins = env.scene.env_origins                              # (N, 3)

    ball_x_local = ball_pos_w[:, 0] - env_origins[:, 0]             # approach axis (X = forwards)
    ball_y_local = ball_pos_w[:, 1] - env_origins[:, 1]             # lateral axis (Y = left/right)
    ball_z_local = ball_pos_w[:, 2]                                  # absolute Z (floor at ~0)

    # initial_vanish: True once the warmup countdown has expired.
    # Upstream: (catchstep < startstep) — mask expires when catchstep drops below startstep.
    # Port: _catchstep counts down from 50; we reveal the ball when _catchstep < 43 (~startstep).
    _STARTSTEP = 43  # 50 - mid(3,10) ≈ 43
    catchstep = getattr(env, "_catchstep", None)
    if catchstep is None:
        initial_vanish = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        initial_vanish = catchstep < _STARTSTEP                      # (N,) bool

    # flying: ball in camera view, approaching the robot, launched.
    catchstep_positive = (catchstep > 0) if catchstep is not None else torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # Ball-last tracking for approach direction (mirrors upstream ball_last).
    if not hasattr(env, "_ball_obs_last_x"):
        env._ball_obs_last_x = torch.zeros(env.num_envs, device=env.device)

    approaching = (ball_x_local < env._ball_obs_last_x) | (env._ball_obs_last_x == 0.0)
    env._ball_obs_last_x = ball_x_local.clone()

    flying = (
        (ball_x_local > 0.05)  &
        (ball_x_local < 3.4)   &
        (ball_y_local.abs() < 2.0) &
        (ball_z_local < 1.8)   &
        catchstep_positive     &
        approaching
    )                                                                 # (N,) bool

    # random_vanish: ball disappears randomly after some in-flight steps.
    just_reset = env.episode_length_buf <= 1
    if not hasattr(env, "_vanish_step"):
        env._vanish_step = torch.randint(0, 30, (env.num_envs,), device=env.device)
        env._ball_visible_step = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)

    env._vanish_step[just_reset] = torch.randint(
        0, 30, (int(just_reset.sum().item()),), device=env.device
    )
    env._ball_visible_step = torch.where(flying, env._ball_visible_step + 1, torch.zeros_like(env._ball_visible_step))

    random_vanish = env._ball_visible_step > env._vanish_step        # (N,) bool

    visible = initial_vanish & flying & ~random_vanish               # (N,) bool

    env._ball_vis_cache = visible
    env._ball_vis_step = env.common_step_counter
    return visible


def ball_pos_b(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Ball position in robot base frame, zeroed during detection latency window.

    Mirrors G1's initial_vanish × random_vanish logic: ball position is hidden for
    the first 3–40 steps of each episode (0.06–0.8 s at 50 Hz), simulating camera
    detection latency and random occlusion at the moment of ball launch.
    Only ball_pos_b is zeroed; ball_vel_b is left visible (matches G1 exactly).
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    pos_b = quat_apply(quat_inv(base_quat_w), ball_pos_w - base_pos_w)

    if not hasattr(env, "_ball_vanish_steps"):
        env._ball_vanish_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        n = int(just_reset.sum())
        # G1: initial_vanish = 3–10 steps, random_vanish = 0–30 steps extra
        env._ball_vanish_steps[just_reset] = (
            torch.randint(3, 11, (n,), device=env.device)
            + torch.randint(0, 31, (n,), device=env.device)
        )
    visible = (env.episode_length_buf >= env._ball_vanish_steps).float().unsqueeze(-1)
    return pos_b * visible


def ball_vel_b(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Ball linear velocity in robot base frame. Shape (N, 3).

    Applies the same visibility mask as ball_pos_b — velocity is zeroed when the
    ball is hidden. Mirrors upstream actor_obs[:, :num_ballobs] masking which
    covers both ball position and velocity in the observation vector.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_vel_w = ball.data.root_link_lin_vel_w
    base_quat_w = robot.data.root_link_quat_w
    ball_vel_b_val = quat_apply(quat_inv(base_quat_w), ball_vel_w)

    visible = _compute_ball_visibility(env, ball_name)               # (N,) bool
    return ball_vel_b_val * visible.float().unsqueeze(-1)


def left_hand_pos_b(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _HAND_CFG
) -> torch.Tensor:
    """Left hand position in robot base frame. Shape (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    idx = asset_cfg.body_ids[0]
    hand_pos_w = robot.data.body_link_pos_w[:, idx, :]
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), hand_pos_w - base_pos_w)


def right_hand_pos_b(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _HAND_CFG
) -> torch.Tensor:
    """Right hand position in robot base frame. Shape (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    idx = asset_cfg.body_ids[1]
    hand_pos_w = robot.data.body_link_pos_w[:, idx, :]
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), hand_pos_w - base_pos_w)


def ball_intercept_pos_b(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Predicted ball intercept position (where ball crosses goal line) in base frame. Shape (N, 3).

    Mirrors G1's end_target privileged observation. Projects current ball position
    and velocity forward in time (with gravity) to estimate where the ball will
    reach the goal line (Y = env_origin_Y). Critic-only: gives the value network
    the intercept target so it can accurately credit region-appropriate behaviour.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    ball_vel_w = ball.data.root_link_lin_vel_w

    ball_y_local = ball_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ball_vy = ball_vel_w[:, 1]

    # t = -y_local / vy, positive when ball is in front and approaching (vy < 0)
    t = torch.where(ball_vy < -0.1, -ball_y_local / ball_vy, torch.zeros_like(ball_vy))
    t = torch.clamp(t, 0.0, 2.0)

    intercept_x = ball_pos_w[:, 0] + ball_vel_w[:, 0] * t
    intercept_y = env.scene.env_origins[:, 1]
    intercept_z = ball_pos_w[:, 2] + ball_vel_w[:, 2] * t - 0.5 * 9.81 * t * t

    intercept_w = torch.stack([intercept_x, intercept_y, intercept_z], dim=-1)
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), intercept_w - base_pos_w)


def reach_dist_to_intercept(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
) -> torch.Tensor:
    """Distance from nearest hand to predicted ball intercept point. Shape (N, 1).

    Mirrors G1's reach_distance privileged observation (dist column in current_obs).
    Critic-only: lets the value function see how far the hand is from the target
    intercept, so it can properly credit approach behaviour before contact.
    """
    intercept_b = ball_intercept_pos_b(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    intercept_w = base_pos_w + quat_apply(base_quat_w, intercept_b)
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    dist = torch.norm(hand_pos_w - intercept_w[:, None, :], dim=-1).min(dim=-1).values
    return dist.unsqueeze(-1)
