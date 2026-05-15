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
