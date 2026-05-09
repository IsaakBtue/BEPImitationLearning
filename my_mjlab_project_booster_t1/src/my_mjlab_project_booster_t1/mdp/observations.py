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
    """Ball position in robot base (Trunk) frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    return quat_apply(quat_inv(base_quat_w), ball_pos_w - base_pos_w)


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
