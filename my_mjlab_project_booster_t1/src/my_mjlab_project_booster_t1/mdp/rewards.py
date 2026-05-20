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
    """Boolean mask (N,): ball has passed the goal line or has been deflected.

    Mirrors the original Humanoid-Goalkeeper's 'behind' condition exactly:
        behind = (ball_x < 0) | (ball_vx - initial_vx > 2.0)
    i.e. ball passed goal line OR velocity increased ≥2 m/s from its
    initial value (deflected/stopped by the robot).

    Reuses _sb_init_vy from stopball if already initialised; otherwise
    falls back to the absolute threshold so post-save rewards still gate
    correctly even if stopball hasn't run first this step.
    """
    ball: Entity = env.scene[ball_name]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    init_vy = getattr(env, "_sb_init_vy", None)
    if init_vy is not None:
        delta_vy = ball_y_vel - init_vy
        return (ball_y_local < 0.0) | (delta_vy > 2.0)
    # Fallback before stopball has run (first policy step of first episode).
    return (ball_y_local < 0.0) | (ball_y_vel > 1.0)


def eereach(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
    reach_th: float = 0.3,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Sigmoid reach reward with velocity amplification and behind-ball boost.

    Mirrors the original isaacgym _reward_eereach more faithfully:
    - vel_sigma: up to 4× multiplier when the closest hand is swinging toward the ball.
      Without this the policy learns to hover near the ball rather than commit to a strike.
    - behind boost: 2× multiplier once the ball passes, matching the original's
      vel_sigma=2.0 post-pass behaviour — keeps the robot reaching aggressively
      during the post-save recovery window.
    - upright gate: unchanged from original.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)

    # Per-hand distance and index of closest hand.
    to_ball = ball_pos_w[:, None, :] - hand_pos_w                      # (N, 2, 3)
    dist = torch.norm(to_ball, dim=-1)                                  # (N, 2)
    min_dist, closest_idx = dist.min(dim=-1)                           # (N,), (N,)

    # Base sigmoid: 1 at dist=0, 0 far away.
    rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (min_dist - reach_th)))

    # Velocity amplification: reward the closest hand actively moving toward the ball.
    hand_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    to_ball_unit = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vel_toward = (hand_vel_w * to_ball_unit).sum(dim=-1)               # (N, 2)
    closest_vel = vel_toward[torch.arange(env.num_envs, device=env.device), closest_idx]
    vel_sigma = 1.0 + 3.0 * torch.clamp(closest_vel, 0.0, 3.0)        # 1× – 4×

    # Post-pass: double the multiplier so the robot keeps committing after the ball goes by.
    behind = _ball_is_behind(env, ball_name)
    vel_sigma = torch.where(behind, vel_sigma * 2.0, vel_sigma)

    projected_grav = env.scene["robot"].data.projected_gravity_b
    upright = (1.0 - torch.clamp(torch.sum(projected_grav[:, :2] ** 2, dim=1), 0.0, 1.0))
    return rew * vel_sigma * upright


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
    delta_vel_threshold: float = 2.0,
) -> torch.Tensor:
    """One-time reward when ball decelerates ≥2 m/s from its initial Y velocity.

    Mirrors the original Humanoid-Goalkeeper _reward_stopball exactly:
    compares current ball velocity against the velocity stored at episode
    reset (not per-step delta), so the threshold is robust to air resistance.
    A per-env flag prevents re-firing after the first deceleration event.

    Ball approaches from +Y so initial vy < 0; deceleration/reversal means
    current_vy - initial_vy > delta_vel_threshold (2.0 m/s, matching G1).
    """
    ball: Entity = env.scene[ball_name]
    ball_y_vel = ball.data.root_link_lin_vel_w[:, 1]
    ball_y_local = ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]

    if not hasattr(env, "_sb_init_vy"):
        env._sb_init_vy = ball_y_vel.clone()
        env._sb_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Store the ball's initial velocity at episode reset (first step).
    just_reset = env.episode_length_buf <= 1
    env._sb_flag[just_reset] = False
    env._sb_init_vy[just_reset] = ball_y_vel[just_reset].clone()

    delta_vy = ball_y_vel - env._sb_init_vy
    in_front = ball_y_local > 0.0
    fired = (delta_vy > delta_vel_threshold) & in_front & ~env._sb_flag

    env._sb_flag |= fired

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
    soft_factor: float = 0.95,
) -> torch.Tensor:
    """Penalize actuator torques exceeding the per-joint soft torque limit.

    Uses _T1_EFFORT_MAP (sourced from T1_serial_clean.xml actuatorfrcrange)
    instead of the previous universal 50 Nm cap. Arms (18 Nm) and ankles
    (15–20 Nm) are now penalised at their actual limits rather than 2.5–3x over.
    Weight: -3.0 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    torques = robot.data.qfrc_actuator[:, asset_cfg.joint_ids].abs()

    if not hasattr(env, "_t1_effort_limits"):
        all_names = robot.joint_names
        effort_all = torch.ones(len(all_names), device=env.device) * 50.0
        for i, name in enumerate(all_names):
            if name in _T1_EFFORT_MAP:
                effort_all[i] = _T1_EFFORT_MAP[name]
        env._t1_effort_limits = effort_all

    effort_limits = env._t1_effort_limits[asset_cfg.joint_ids]
    out_of_limit = (torques - effort_limits * soft_factor).clamp(min=0.0)
    return out_of_limit.sum(dim=-1)


def feet_slippage(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _FEET_CFG,
    contact_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward feet not slipping while in ground contact.

    Mirrors upstream _reward_feet_slippage exactly:
        contactvel = sum(norm(foot_vel_3d) * in_contact, over feet)
        return exp(-10 * contactvel)
    Returns 1.0 when no contact or no slip; approaches 0 with high slip.
    Weight: +3.0 (positive reward for not slipping — same as upstream).

    Contact detection: upstream uses contact_forces > 1 N; mjlab has no
    per-foot contact sensor here so foot height < threshold is used as proxy.
    3D velocity (not just XY) matches upstream rigid_body_states[:, :, 7:10].
    """
    robot: Entity = env.scene[asset_cfg.name]
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_speed = torch.norm(foot_vel_w, dim=-1)  # (N, 2)

    foot_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    env_z = env.scene.env_origins[:, 2:3]
    in_contact = (foot_z - env_z < contact_height_threshold).float()

    contactvel = torch.sum(foot_speed * in_contact, dim=-1)
    return torch.exp(-10.0 * contactvel)


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
    stopped = getattr(env, "_sb_flag", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    multiplier = 1.0 + stopped.float()
    return (dist < strict_th).float() * multiplier


_T1_KP_MAP: dict[str, float] = {
    "Left_Shoulder_Pitch": 15.0, "Left_Shoulder_Roll": 15.0,
    "Left_Elbow_Pitch":    15.0, "Left_Elbow_Yaw":     15.0,
    "Right_Shoulder_Pitch": 15.0, "Right_Shoulder_Roll": 15.0,
    "Right_Elbow_Pitch":   15.0, "Right_Elbow_Yaw":    15.0,
    "Waist":               80.0,
    "Left_Hip_Pitch":     120.0, "Right_Hip_Pitch":    120.0,
    "Left_Hip_Roll":       80.0, "Left_Hip_Yaw":        80.0,
    "Right_Hip_Roll":      80.0, "Right_Hip_Yaw":       80.0,
    "Left_Knee_Pitch":    200.0, "Right_Knee_Pitch":   200.0,
    "Left_Ankle_Pitch":    50.0, "Right_Ankle_Pitch":   50.0,
    "Left_Ankle_Roll":     40.0, "Right_Ankle_Roll":    40.0,
}

# Per-joint effort limits sourced from KaydenKnapik/BoosterT1mjlab t1_constants.py.
# Their setup was successfully deployed on real T1 hardware. actuatorfrcrange has been
# removed from T1_serial_clean.xml so Python effort_limit is the only hard clamp,
# matching KaydenKnapik's approach (their XML has no actuatorfrcrange at all).
_T1_EFFORT_MAP: dict[str, float] = {
    "Left_Shoulder_Pitch": 36.0, "Left_Shoulder_Roll": 36.0,
    "Left_Elbow_Pitch":    36.0, "Left_Elbow_Yaw":     36.0,
    "Right_Shoulder_Pitch": 36.0, "Right_Shoulder_Roll": 36.0,
    "Right_Elbow_Pitch":   36.0, "Right_Elbow_Yaw":    36.0,
    "Waist":               40.0,
    "Left_Hip_Pitch":      55.0, "Right_Hip_Pitch":     55.0,
    "Left_Hip_Roll":       40.0, "Left_Hip_Yaw":        40.0,
    "Right_Hip_Roll":      40.0, "Right_Hip_Yaw":       40.0,
    "Left_Knee_Pitch":     65.0, "Right_Knee_Pitch":    65.0,
    "Left_Ankle_Pitch":    50.0, "Right_Ankle_Pitch":   50.0,
    "Left_Ankle_Roll":     50.0, "Right_Ankle_Roll":    50.0,
}


def torques_normalized_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG,
) -> torch.Tensor:
    """Penalize torques normalized by per-joint stiffness (kp).

    Mirrors the original _reward_torques exactly:
        return sum(square(torques / p_gains))
    Since torque = kp × (target - current), dividing by kp gives a
    dimensionless position-error proxy that is comparable across joints
    regardless of stiffness. Without normalization, high-stiffness leg joints
    (kp=200) would dominate the penalty over soft arm joints (kp=15).
    """
    robot: Entity = env.scene[asset_cfg.name]
    torques = robot.data.qfrc_actuator[:, asset_cfg.joint_ids]

    if not hasattr(env, "_t1_kp_inv"):
        all_names = robot.joint_names
        kp_all = torch.ones(len(all_names), device=env.device)
        for i, name in enumerate(all_names):
            if name in _T1_KP_MAP:
                kp_all[i] = _T1_KP_MAP[name]
        env._t1_kp_inv = 1.0 / kp_all

    kp_inv = env._t1_kp_inv[asset_cfg.joint_ids]
    return torch.sum(torch.square(torques * kp_inv), dim=-1)


def deviation_waist_joint(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _WAIST_JOINT_CFG,
) -> torch.Tensor:
    """Penalize waist joint deviation from default position at all times.

    Mirrors upstream _reward_deviation_waist_pitch_joint. G1 had a dedicated
    waist_pitch joint; T1 has a single Waist joint — same intent, same weight.
    Always-active (not gated on ball position) to keep trunk upright throughout.
    Weight: -0.001 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    delta = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(delta), dim=-1)
