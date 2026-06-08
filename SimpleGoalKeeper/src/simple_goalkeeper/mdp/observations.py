"""Goalkeeper-specific observation terms for SimpleGoalKeeper (feet-only, Phase 1)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))


def _compute_ball_visibility(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Return per-env visibility mask (N,) bool. Result is cached per step.

    Three gates (all must be True for visibility):
      initial_vanish: catchstep < startstep  (warmup countdown ~1s after reset)
      flying:         ball in robot body frame: x∈(0.05,3.4), |y|<2.0, z<1.8, approaching
      ~random_vanish: ball_visible_step <= vanish_step  (random mid-flight disappearance)
    """
    if getattr(env, "_ball_vis_step", -1) == env.common_step_counter:
        return env._ball_vis_cache

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]

    # Ball position in robot body frame for flying gate.
    ball_pos_w = ball.data.root_link_pos_w           # (N, 3)
    base_pos_w = robot.data.root_link_pos_w
    base_quat_w = robot.data.root_link_quat_w
    ball_pos_b_val = quat_apply(quat_inv(base_quat_w), ball_pos_w - base_pos_w)  # (N, 3)

    x_b = ball_pos_b_val[:, 0]   # forward
    y_b = ball_pos_b_val[:, 1]   # lateral
    z_w = ball_pos_w[:, 2]       # absolute height

    # initial_vanish: True once warmup countdown expires.
    catchstep = getattr(env, "_catchstep", None)
    startstep = getattr(env, "_startstep", None)
    if catchstep is None:
        initial_vanish = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif startstep is None:
        initial_vanish = catchstep < 43
    else:
        initial_vanish = catchstep < startstep

    catchstep_positive = (catchstep > 0) if catchstep is not None else torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # Approaching: ball x_b decreasing (getting closer to robot) or first observation.
    if not hasattr(env, "_ball_obs_last_x"):
        env._ball_obs_last_x = torch.zeros(env.num_envs, device=env.device)
    approaching = (x_b < env._ball_obs_last_x) | (env._ball_obs_last_x == 0.0)
    env._ball_obs_last_x = x_b.clone()

    flying = (
        (x_b > 0.05) &
        (x_b < 3.4) &
        (y_b.abs() < 2.0) &
        (z_w < 1.8) &
        catchstep_positive &
        approaching
    )

    # random_vanish: ball disappears after a random number of consecutive flying steps.
    if not hasattr(env, "_vanish_step"):
        env._vanish_step = torch.randint(0, 30, (env.num_envs,), device=env.device)
    if not hasattr(env, "_ball_visible_step"):
        env._ball_visible_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._ball_visible_step = torch.where(
        flying,
        env._ball_visible_step + 1,
        torch.zeros_like(env._ball_visible_step),
    )
    random_vanish = env._ball_visible_step > env._vanish_step

    visible = initial_vanish & flying & ~random_vanish

    env._ball_vis_cache = visible
    env._ball_vis_step = env.common_step_counter
    return visible


def ball_pos_b(env: "ManagerBasedRlEnv", ball_name: str = "ball") -> torch.Tensor:
    """Ball position in robot body frame, zeroed when not visible. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_pos_w - robot.data.root_link_pos_w,
    )
    visible = _compute_ball_visibility(env, ball_name)
    return ball_pos_b_val * visible.float().unsqueeze(-1)


def ball_vel_b(env: "ManagerBasedRlEnv", ball_name: str = "ball") -> torch.Tensor:
    """Ball linear velocity in robot body frame, zeroed when not visible. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_vel_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_lin_vel_w,
    )
    visible = _compute_ball_visibility(env, ball_name)
    return ball_vel_b_val * visible.float().unsqueeze(-1)


def left_foot_pos_b(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Left foot position in robot body frame. Shape (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    idx = asset_cfg.body_ids[0]
    foot_pos_w = robot.data.body_link_pos_w[:, idx, :]
    return quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        foot_pos_w - robot.data.root_link_pos_w,
    )


def right_foot_pos_b(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Right foot position in robot body frame. Shape (N, 3)."""
    robot: Entity = env.scene[asset_cfg.name]
    idx = asset_cfg.body_ids[1]
    foot_pos_w = robot.data.body_link_pos_w[:, idx, :]
    return quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        foot_pos_w - robot.data.root_link_pos_w,
    )
