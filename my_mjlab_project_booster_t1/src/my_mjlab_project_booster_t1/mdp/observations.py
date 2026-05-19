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
    """Ball linear velocity in robot base frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_vel_w = ball.data.root_link_lin_vel_w
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), ball_vel_w)


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
