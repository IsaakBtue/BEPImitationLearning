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
_KNEE_BODY_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
_ARM_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
_WAIST_JOINT_CFG = SceneEntityCfg("robot", joint_names=("Waist",))
_ALL_JOINT_CFG = SceneEntityCfg("robot")


def _ball_is_behind(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Boolean mask (N,): ball has passed the goal line or is moving away.

    Ball approaches from +Y; 'behind' means ball_y < 0 (passed the robot)
    or ball moving strongly back in +Y (was deflected toward attacker).
    """
    ball: Entity = env.scene[ball_name]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    return (ball_y_local < 0.0) | (ball_y_vel > 1.0)


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
    delta_vel_threshold: float = 1.0,
) -> torch.Tensor:
    """One-time reward when ball first decelerates significantly while in front.

    Mirrors the original Humanoid-Goalkeeper logic: fires exactly once per
    episode when the ball's Y velocity increases by >2 m/s (i.e., the ball
    was approaching in -Y and decelerated or reversed). A per-env flag blocks
    subsequent firings. This prevents the continuous passive-blocking exploit
    where standing still and letting the ball roll into the body earned
    100 pts/step for the entire episode.

    Ball approaches from +Y so approaching vy < 0; deceleration/reversal
    means vy_now - vy_prev > delta_vel_threshold.
    """
    ball: Entity = env.scene[ball_name]
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]

    if not hasattr(env, "_sb_prev_vy"):
        env._sb_prev_vy = ball_y_vel.clone()
        env._sb_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Reset state for envs at the start of a new episode.
    just_reset = env.episode_length_buf <= 1
    env._sb_flag[just_reset] = False
    env._sb_prev_vy[just_reset] = ball_y_vel[just_reset].clone()

    delta_vy = ball_y_vel - env._sb_prev_vy
    in_front = ball_y_local > 0.0
    fired = (delta_vy > delta_vel_threshold) & in_front & ~env._sb_flag

    env._sb_flag |= fired
    env._sb_prev_vy.copy_(ball_y_vel)

    return fired.float()


def stayonline(
    env: ManagerBasedRlEnv,
    line_offset: float = 0.2,
    max_offset: float = 1.2,
) -> torch.Tensor:
    """Penalty for robot retreating/advancing away from the goal line.

    Ball approaches from +Y so the goal line is Y = env_origin_Y.
    The robot slides laterally in X to intercept; this penalises Y deviation.
    """
    robot: Entity = env.scene["robot"]
    y_local = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    dist = torch.clamp(y_local.abs(), line_offset, max_offset) - line_offset
    return dist


def noretreat(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalty for retreating in the robot's forward direction away from the ball.

    Uses body-frame forward (X) so the penalty stays semantically correct when
    the robot yaws during a dive. World-Y and body-X agree when the robot faces
    +Y (spawn orientation) but diverge at 30-45° yaw mid-save.
    """
    robot: Entity = env.scene["robot"]
    fwd_vel = robot.data.root_link_lin_vel_b[:, 0]
    return -torch.clamp(fwd_vel, -1.0, 0.0)


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

    Only XY axes, matching the original Humanoid-Goalkeeper (yaw/Z excluded).
    """
    behind = _ball_is_behind(env, ball_name)
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b
    err = torch.sum(ang_vel[:, :2] ** 2, dim=1)
    return torch.exp(-3.0 * err) * behind.float()


def base_ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalise base angular velocity on XY axes (roll/pitch rate).

    Ported from original _reward_ang_vel_xy. Suppresses wobbling and
    tipping without penalising intentional yaw rotation.
    """
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b
    return torch.sum(ang_vel[:, :2] ** 2, dim=1)


def postlinvel(env: ManagerBasedRlEnv, ball_name: str = "ball") -> torch.Tensor:
    """Low forward linear velocity reward — active when ball is behind or passed."""
    behind = _ball_is_behind(env, ball_name)
    lin_vel = env.scene["robot"].data.root_link_lin_vel_b
    err = lin_vel[:, 0] ** 2
    return torch.exp(-3.0 * err) * behind.float()


# ============================================================================
# Gap fixes: rewards present in upstream Humanoid-Goalkeeper but missing here
# ============================================================================

def penalize_sharpcontact(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.35,
) -> torch.Tensor:
    """Penalize trunk collapse as proxy for hard body–ground contact.

    cfrc_ext is not exposed per-body in mjlab, so we use trunk height drop as
    a proxy. Trunk Z < 0.35 m (half of normal ~0.7 m) indicates a hard fall.
    Weight: -100 (same as upstream) — fires as a binary flag.
    """
    robot: Entity = env.scene["robot"]
    trunk_z = robot.data.root_link_pos_w[:, 2]
    env_z = env.scene.env_origins[:, 2]
    return (trunk_z - env_z < height_threshold).float()


def penalize_kneeheight(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _KNEE_BODY_CFG,
    height_threshold: float = 0.12,
) -> torch.Tensor:
    """Penalize shank/knee bodies being too close to the ground (kneeling).

    Mirrors upstream _reward_penalize_kneeheight. Threshold 0.12 m catches
    kneeling without penalising intentional low crouches during a dive.
    Weight: -100 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    heights = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    env_z = env.scene.env_origins[:, 2:3]
    return (heights - env_z < height_threshold).any(dim=-1).float()


def successland(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _FEET_CFG,
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward both feet near ground after the ball passes — encourages safe landing.

    Mirrors upstream _reward_successland. Uses foot height < threshold as a
    proxy for foot–ground contact (mjlab has no per-foot contact sensor here).
    Only active when ball is behind/passed so it doesn't fire during the dive.
    Weight: 4.0 (same as upstream).
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    foot_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    env_z = env.scene.env_origins[:, 2:3]
    feet_down = (foot_z - env_z < height_threshold).all(dim=-1)
    return feet_down.float() * behind.float()


def postupperdofpos(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _ARM_JOINT_CFG,
) -> torch.Tensor:
    """Reward arm joints returning to default pose after the ball passes.

    Mirrors upstream _reward_postupperdofpos exactly: exp(-1 * sum_sq_err).
    The original code uses hardcoded sigma=-1 on the SUM (not mean), despite the
    config having target_dof_pos_sigma=-20 which is not used in the original code.
    Active only when ball is behind. Weight: 1.0 (same as upstream).
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-1.0 * err) * behind.float()


def postwaistdofpos(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _WAIST_JOINT_CFG,
) -> torch.Tensor:
    """Reward waist joint returning to default pose after the ball passes.

    Mirrors upstream _reward_postwaistdofpos exactly: exp(-3 * sum_sq_err).
    The original code uses hardcoded sigma=-3 on the SUM over the waist joints.
    Weight: 1.0 (same as upstream).
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-3.0 * err) * behind.float()


def dof_vel_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG,
    vel_limit: float = 20.0,
    soft_factor: float = 0.9,
) -> torch.Tensor:
    """Penalize joint velocities exceeding the soft velocity limit.

    Mirrors upstream _reward_dof_vel_limits. mjlab does not store per-joint
    velocity limits from the URDF, so vel_limit=20 rad/s is used as a universal
    soft cap (conservative for legs, generous for fast arm motions during a dive).
    Weight: -2.0 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    out_of_limit = (joint_vel.abs() - vel_limit * soft_factor).clamp(min=0.0)
    return out_of_limit.sum(dim=-1)


def torque_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG,
    torque_limit: float = 50.0,
    soft_factor: float = 0.95,
) -> torch.Tensor:
    """Penalize actuator torques exceeding the soft torque limit.

    Mirrors upstream _reward_torque_limits. Uses qfrc_actuator (joint-space
    actuator force). torque_limit=50 Nm is the median T1 effort limit; knees
    (60 Nm) are slightly under-penalised, arms (18 Nm) are over-penalised but
    their torques are naturally small so the penalty rarely fires there.
    Weight: -3.0 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    torques = robot.data.qfrc_actuator[:, asset_cfg.joint_ids].abs()
    out_of_limit = (torques - torque_limit * soft_factor).clamp(min=0.0)
    return out_of_limit.sum(dim=-1)


def feet_slippage(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _FEET_CFG,
    contact_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize foot XY velocity while feet are in ground contact.

    Mirrors upstream _reward_feet_slippage. Uses foot height < threshold as
    contact proxy (no foot contact sensor configured). Uses body_link_lin_vel_w
    for foot linear velocity in world frame.
    Weight: -3.0 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    foot_lin_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_xy_vel_sq = torch.sum(foot_lin_vel_w[:, :, :2] ** 2, dim=-1)  # (N, 2)

    foot_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    env_z = env.scene.env_origins[:, 2:3]
    in_contact = (foot_z - env_z < contact_height_threshold).float()

    return torch.sum(foot_xy_vel_sq * in_contact, dim=-1)


def hand_proximity_strict(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
    strict_th: float = 0.15,
) -> torch.Tensor:
    """Continuous reward when nearest hand is within strict threshold of the ball.

    Mirrors upstream _reward_success: fires every step the hand is within
    strict_th (0.15 m) of the ball, providing dense gradient for precise hand
    placement. Complements catch_success (0.5 m coarse threshold) by rewarding
    the final approach. Weight: 5.0 (same as upstream success reward).
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    dist = torch.norm(hand_pos_w - ball.data.root_link_pos_w[:, None, :], dim=-1).min(dim=-1).values
    return (dist < strict_th).float()
