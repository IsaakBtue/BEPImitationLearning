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
            x > 0.05 AND x < 3.4   → port: y_local (approach axis)
            |y| < 2.0               → port: |x_local| < 2.0
            z < 1.8
        catchstep > 0: ball has been launched (warmup not finished)
        ball moving closer: x_local < x_last OR x_last == 0 → port: y_local < y_last OR y_last == 0

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

    ball_y_local = ball_pos_w[:, 1] - env_origins[:, 1]             # approach axis (Y in port)
    ball_x_local = ball_pos_w[:, 0] - env_origins[:, 0]             # lateral axis
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
    if not hasattr(env, "_ball_obs_last_y"):
        env._ball_obs_last_y = torch.zeros(env.num_envs, device=env.device)

    approaching = (ball_y_local < env._ball_obs_last_y) | (env._ball_obs_last_y == 0.0)
    env._ball_obs_last_y = ball_y_local.clone()

    flying = (
        (ball_y_local > 0.05)  &
        (ball_y_local < 3.4)   &
        (ball_x_local.abs() < 2.0) &
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
    """Ball position in robot base (Trunk) frame. Shape (N, 3).

    Applies catchstep warmup masking (Feature 7) and ball visibility masking (Feature 8).

    Mirrors upstream compute_observations() exactly:
        initial_vanish = (catchstep < startstep)
        end_target_local = quat_rotate_inverse(base_quat, ball_pos - torso_pos) * initial_vanish
        actor_obs[:, :num_ballobs] = actor_obs[:, :num_ballobs] * flying * ~random_vanish

    Ball position is zeroed when not visible — the policy must learn to act on the
    last known position during vanish windows, which was the upstream training regime.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    ball_pos_b_val = quat_apply(quat_inv(base_quat_w), ball_pos_w - base_pos_w)

    visible = _compute_ball_visibility(env, ball_name)               # (N,) bool
    return ball_pos_b_val * visible.float().unsqueeze(-1)


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
