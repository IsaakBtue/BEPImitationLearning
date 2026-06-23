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


def ball_pos_xy_b(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    always_visible: bool = False,
) -> torch.Tensor:
    """Ball XY position in robot body frame (no Z). Shape (N, 2).

    Matches BoosterT1mjlab kick task observation space for deployment compatibility.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_pos_w - robot.data.root_link_pos_w,
    )
    if always_visible:
        return ball_pos_b_val[:, :2]
    visible = _compute_ball_visibility(env, ball_name)
    return ball_pos_b_val[:, :2] * visible.float().unsqueeze(-1)


def ball_pos_b(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    always_visible: bool = False,
) -> torch.Tensor:
    """Ball position in robot body frame, optionally gated by visibility. Shape (N, 3).

    Set always_visible=True during Phase 1 training so the policy always has the ball
    position as input. The visibility system (warmup + random vanish) is enabled for
    play/evaluation to simulate partial observability.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_pos_w - robot.data.root_link_pos_w,
    )
    if always_visible:
        return ball_pos_b_val
    visible = _compute_ball_visibility(env, ball_name)
    return ball_pos_b_val * visible.float().unsqueeze(-1)


def ball_vel_b(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    always_visible: bool = False,
) -> torch.Tensor:
    """Ball velocity in robot body frame, optionally gated by visibility. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_vel_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_lin_vel_w,
    )
    if always_visible:
        return ball_vel_b_val
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


def base_lin_vel(
    env: "ManagerBasedRlEnv",
) -> torch.Tensor:
    """Robot base linear velocity in robot body frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    return robot.data.root_link_lin_vel_b


def joint_pos_abs(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute joint positions for AMP discriminator — gated by softstop.

    After softstop fires the ball is deflected and the robot is recovering.
    Freezing AMP obs to HOME_KEYFRAME post-softstop removes the recovery
    phase from discriminator training, keeping its signal focused on the
    save-motion phase where reference data exists. This pushes d toward 0
    (higher style reward) vs letting recovery transitions drag it negative.

    Before softstop: real joint positions (save motion — reference covers this).
    After softstop:  default joint positions (neutral — excluded from training).
    """
    robot: Entity = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.clone()
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None and softstop_fired.any():
        joint_pos[softstop_fired] = robot.data.default_joint_pos[softstop_fired]
    return joint_pos


def joint_vel_abs(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute joint velocities for AMP discriminator — gated by softstop.

    Returns zero velocities after softstop fires. Mirrors joint_pos_abs gating.
    """
    robot: Entity = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.clone()
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None and softstop_fired.any():
        joint_vel[softstop_fired] = 0.0
    return joint_vel
