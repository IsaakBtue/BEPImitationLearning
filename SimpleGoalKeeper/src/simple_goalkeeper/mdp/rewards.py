"""Goalkeeper reward terms for SimpleGoalKeeper (Phase 1 — feet only)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def _robot_x_axis_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Robot local +X unit vector in world frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    x_local = torch.zeros(env.num_envs, 3, device=env.device)
    x_local[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, x_local)


def foot_to_ball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Gaussian reward: foot midpoint XY proximity to ball. Shape (N,).

    Provides a dense signal from episode start — reward peaks when either
    foot is directly under the ball in the XY plane.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :2]  # (N, 2, 2)
    feet_mid_xy = foot_pos_w.mean(dim=1)                                  # (N, 2)
    ball_xy = ball.data.root_link_pos_w[:, :2]                           # (N, 2)

    dist = torch.norm(ball_xy - feet_mid_xy, dim=-1)
    return torch.exp(-(dist ** 2) / (std ** 2))


def ball_vx_reduction(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    max_speed: float = 8.0,
) -> torch.Tensor:
    """Reward for stopping the ball's incoming velocity along robot +X axis. Shape (N,).

    vx_local < 0 means ball coming toward robot. Reward peaks when vx_local = 0
    (ball stopped). Clamped to [0, max_speed] before normalisation so the reward
    scale is stable regardless of spawn speed.
    """
    ball: Entity = env.scene[ball_name]
    x_axis_w = _robot_x_axis_w(env)  # (N, 3)

    vx_local = (ball.data.root_link_lin_vel_w * x_axis_w).sum(dim=-1)  # (N,)
    incoming = (-vx_local).clamp(0.0, max_speed)
    return torch.exp(-(incoming ** 2) / (4.0 ** 2))


def ball_positive_vx(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    target_speed: float = 5.0,
) -> torch.Tensor:
    """Reward ball deflected back along robot +X axis. Shape (N,).

    Returns clamp(vx_local / target_speed, 0, 1) — normalised to [0, 1].
    Saturates at target_speed m/s. Only active when ball has positive local-X velocity.
    """
    ball: Entity = env.scene[ball_name]
    x_axis_w = _robot_x_axis_w(env)

    vx_local = (ball.data.root_link_lin_vel_w * x_axis_w).sum(dim=-1)
    return (vx_local / target_speed).clamp(0.0, 1.0)


def posture(
    env: "ManagerBasedRlEnv",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Gaussian reward for staying near default joint pose. Shape (N,)."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error_sq = torch.square(joint_pos - default_pos)
    return torch.exp(-torch.mean(error_sq, dim=1) / (std ** 2))


def ang_vel_xy_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Sum of squared base roll+pitch angular velocity. Shape (N,).

    Penalises excessive rolling/pitching to keep robot upright.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
