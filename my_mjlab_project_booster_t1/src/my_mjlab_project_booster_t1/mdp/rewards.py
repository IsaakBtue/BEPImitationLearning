"""Goalkeeper task reward terms for Booster T1."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_HAND_CFG = SceneEntityCfg("robot", body_names=("left_hand_link", "right_hand_link"))
_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def _ball_is_behind(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Boolean mask (N,): ball has passed the goal line (y=0) or is moving away.
    T1 faces +Y, ball approaches from +Y, so the goal line is y=0."""
    ball: Entity = env.scene[ball_name]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    return (ball_y_local < 0.0) | (ball_y_vel < -1.0)


def eereach(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
    reach_th: float = 0.3,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Sigmoid reward for nearest hand reaching the ball."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    dist = torch.norm(hand_pos_w - ball_pos_w[:, None, :], dim=-1)
    min_dist = dist.min(dim=-1).values
    rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (min_dist - reach_th)))
    projected_grav = env.scene["robot"].data.projected_gravity_b
    upright = (1.0 - torch.clamp(torch.sum(projected_grav[:, :2] ** 2, dim=1), 0.0, 1.0))
    return rew * upright


def catch_success(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
    catch_th: float = 0.3,
) -> torch.Tensor:
    """Reward when ball is within catch threshold of the nearest hand."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    dist = torch.norm(hand_pos_w - ball_pos_w[:, None, :], dim=-1).min(dim=-1).values
    return (dist < catch_th).float()


def stopball(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
) -> torch.Tensor:
    """Reward when ball in front of goal line (y>0) has slowed or reversed.
    T1 faces +Y: ball approaches in -Y, so in_front = ball_y_local > 0."""
    ball: Entity = env.scene[ball_name]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    in_front = ball_y_local > 0.0
    deflected = ball_y_vel > -0.5
    return (in_front & deflected).float()


def stayonline(
    env: ManagerBasedRlEnv,
    line_offset: float = 0.2,
    max_offset: float = 1.2,
) -> torch.Tensor:
    """Penalty for robot retreating from the goal line (y=0).
    T1 faces +Y: lateral X movement is the dive direction and must NOT be penalized here."""
    robot: Entity = env.scene["robot"]
    y_local = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    dist = torch.clamp(y_local.abs(), line_offset, max_offset) - line_offset
    return dist


def noretreat(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalty for retreating away from the incoming ball (negative Y velocity).
    T1 faces +Y, ball comes from +Y, so retreating is moving in -Y."""
    robot: Entity = env.scene["robot"]
    y_vel = robot.data.root_link_lin_vel_w[:, 1]
    return -torch.clamp(y_vel, -1.0, 0.0)


def feetorientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Reward for keeping feet flat (gravity aligned with foot z-axis)."""
    robot: Entity = env.scene[asset_cfg.name]
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    feet_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_w_exp = gravity_w[:, None, :].expand(-1, 2, -1)
    gravity_foot = quat_apply(quat_inv(feet_quat_w), gravity_w_exp)
    err = torch.sum(gravity_foot[..., :2] ** 2, dim=-1).sum(dim=-1)
    return torch.exp(-sigma * err)


def postorientation(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Upright posture reward — active only when ball is behind or passed."""
    behind = _ball_is_behind(env, ball_name)
    grav_b = env.scene["robot"].data.projected_gravity_b
    err = torch.sum(grav_b[:, :2] ** 2, dim=1)
    return torch.exp(-3.0 * err) * behind.float()


def postangvel(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Low angular velocity reward — active when ball is behind or passed.
    All 3 axes (including yaw/Z) are penalized to prevent spinning."""
    behind = _ball_is_behind(env, ball_name)
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b
    err = torch.sum(ang_vel ** 2, dim=1)
    return torch.exp(-3.0 * err) * behind.float()


def postlinvel(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Low forward linear velocity reward — active when ball is behind or passed."""
    behind = _ball_is_behind(env, ball_name)
    lin_vel = env.scene["robot"].data.root_link_lin_vel_b
    err = lin_vel[:, 0] ** 2
    return torch.exp(-3.0 * err) * behind.float()
