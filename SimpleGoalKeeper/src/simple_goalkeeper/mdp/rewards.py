"""Goalkeeper reward terms for SimpleGoalKeeper (Phase 1 — feet only)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_FEET_CFG = SceneEntityCfg("robot", body_names=("left_foot_link", "right_foot_link"))
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_KNEE_CFG = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right"))
_ARM_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    ),
)
_WAIST_JOINT_CFG_RECOVERY = SceneEntityCfg("robot", joint_names=("Waist",))
_ALL_JOINTS_CFG = SceneEntityCfg("robot")

# Per-joint stiffness (kp) for torque normalisation. Source: ILB _T1_KP_MAP,
# which was validated on real T1 hardware via KaydenKnapik/BoosterT1mjlab.
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

# Per-joint effort limits from T1_serial_clean.xml actuatorfrcrange.
# Source: ILB _T1_EFFORT_MAP (same hardware-validated values).
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


def _ball_is_behind(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Bool mask (N,): ball has been deflected OR has crossed the goal line.

    Fires when ANY of:
      - ball_x < 0 (crossed goal line)
      - _softstop_flag is set (delta_vx > 0.2 — partial contact)
      - delta_vx > 1.0 from _sb_init_vx (full stopball deflection)

    Checking _softstop_flag (0.2 m/s threshold) means partial contacts also
    deactivate footreach and activate post-save recovery immediately, preventing
    the robot from standing still in a T-pose while tracking a ball it already
    touched.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]

    # Persistent flag: softstop fires once at delta_vx > 0.2 m/s.
    softstop_fired = getattr(env, "_softstop_flag", None)
    already_deflected = (
        softstop_fired
        if softstop_fired is not None
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )

    # Also check real-time delta_vx against stopball threshold (mirrors old behaviour).
    init_vx = getattr(env, "_sb_init_vx", None)
    if init_vx is not None:
        delta_vx = ball_x_vel - init_vx
        already_deflected = already_deflected | (delta_vx > 1.0)

    return (ball_x_local < 0.0) | already_deflected


def _robot_x_axis_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Robot local +X unit vector in world frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    x_local = torch.zeros(env.num_envs, 3, device=env.device)
    x_local[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, x_local)


def _get_ball_crossing_y(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Frozen Y coordinate where the ball will cross the goal line (x_local = 0).

    Computed once per episode at reset from the ball's initial position and velocity.
    Both footreach (phase1) and foot_proximity read this so the robot pre-positions
    toward where the ball WILL arrive, not where it currently is. Mirrors ILB's
    _ball_end_target frozen intercept pattern.
    """
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w
    ball_vel_w = ball.data.root_link_lin_vel_w
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]

    just_reset = env.episode_length_buf <= 1
    if not hasattr(env, "_ball_crossing_y"):
        env._ball_crossing_y = ball_pos_w[:, 1].clone()
    if just_reset.any():
        bvx = ball_vel_w[just_reset, 0].clamp(max=-0.1)   # approaching → negative
        bvy = ball_vel_w[just_reset, 1]
        t_cross = ball_x_local[just_reset] / (-bvx)
        env._ball_crossing_y[just_reset] = ball_pos_w[just_reset, 1] + bvy * t_cross
    return env._ball_crossing_y


def footreach(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    reach_th: float = 0.3,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Reach reward adapted from Imitationlearningbooster eereach, for feet instead of hands.

    Phase 1 (ball x_local > 1.5 m): reward lateral alignment with the FROZEN crossing Y
    (where the ball will cross the goal line), not the live ball Y. This gives a stable
    pre-positioning target even for angled shots where live ball Y != arrival Y.

    Phase 2 (ball x_local <= 1.5 m): sigmoid reach reward × lateral vel_sigma so actively
    diving/stepping toward the ball gives up to 10× the static reach reward.

    vel_sigma = 1 + 3 * clamp(vel_toward_crossing_side, 0, 3).

    Upright gate suppresses reward when falling (mirrors eereach upright gate).
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)

    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]     # (N,)
    robot_y_w = robot.data.root_link_pos_w[:, 1]                       # (N,)

    # Frozen crossing Y: where the ball will arrive at the goal line.
    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)

    # Phase 1: pre-position laterally when ball is far (> 1.5 m in front).
    lateral_error = crossing_y - robot_y_w                             # positive → target right
    asidegoal = lateral_error.clamp(-1.0, 1.0)
    asidegoal = torch.where(asidegoal.abs() < 0.3, torch.zeros_like(asidegoal), asidegoal)
    phase1_rew = 1.0 - asidegoal.abs()                                  # 1=aligned, 0=1 m off

    # Phase 2: sigmoid reach to frozen crossing point (where ball WILL arrive at goal line).
    # Mirrors ILB eereach end_target: robot must pre-position foot at intercept, not
    # chase the live ball. Using live ball caused free reward when ball rolled through center.
    goal_x_w  = env.scene.env_origins[:, 0]       # (N,)
    floor_z_w = env.scene.env_origins[:, 2]       # (N,)
    crossing_point = torch.stack(
        [goal_x_w, crossing_y, floor_z_w + 0.10], dim=-1
    )                                                                     # (N, 3)
    dist_to_crossing = torch.norm(
        foot_pos_w - crossing_point[:, None, :], dim=-1
    ).min(dim=-1).values                                                  # (N,)
    reach_rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (dist_to_crossing - reach_th)))

    # Lateral velocity toward crossing side amplifies the reach reward.
    lateral_vel_y = robot.data.root_link_lin_vel_w[:, 1]
    vel_toward = torch.where(lateral_error > 0, lateral_vel_y, -lateral_vel_y)
    vel_sigma = 1.0 + 3.0 * vel_toward.clamp(0.0, 3.0)                # 1–10×

    # Combine: phase1 when ball is far, phase2 sigmoid when close.
    phase1_mask = ball_x_local > 1.5
    taskrew = torch.where(phase1_mask, phase1_rew, reach_rew * vel_sigma)

    # Upright gate: suppress reward when robot is falling.
    projected_grav = robot.data.projected_gravity_b
    upright = 1.0 - torch.clamp(torch.sum(projected_grav[:, :2] ** 2, dim=1), 0.0, 1.0)
    behind = _ball_is_behind(env, ball_name)
    return taskrew * upright * (~behind).float()


def foot_proximity(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Dense continuous reward for foot near the frozen ball crossing point.

    Uses the frozen goal-line crossing point (env._ball_crossing_y, set by
    _get_ball_crossing_y) as the target. Rewards exp(-sigma * dist) so there is
    always a gradient pulling the foot toward the arrival point, unlike the
    one-shot stopball. Analogous to ILB's hand_proximity_strict (weight 5.0).
    Deactivates once the ball is behind (same gate as footreach).
    Weight: +5.0.
    """
    robot: Entity = env.scene[asset_cfg.name]

    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z    = env.scene.env_origins[:, 2]
    crossing_point = torch.stack(
        [goal_x_w, crossing_y, env_z + 0.10], dim=-1                   # (N, 3) — ball radius
    )

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    dist = torch.norm(foot_pos_w - crossing_point[:, None, :], dim=-1)  # (N, 2)
    min_dist = dist.min(dim=-1).values                                   # (N,)

    behind = _ball_is_behind(env, ball_name)
    return torch.exp(-sigma * min_dist) * (~behind).float()


def stopball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    delta_vel_threshold: float = 1.0,
) -> torch.Tensor:
    """One-time reward when ball X velocity increases by >= delta_vel_threshold (m/s).

    Ball approaches with negative X velocity; foot contact reverses or decelerates it.
    Fires exactly once per episode when delta_vx > threshold, providing the primary
    training signal for a successful save. Mirrors Imitationlearningbooster stopball.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    if not hasattr(env, "_sb_init_vx"):
        env._sb_init_vx = ball_x_vel.clone()
        env._sb_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._sb_flag[just_reset] = False
    env._sb_init_vx[just_reset] = ball_x_vel[just_reset].clone()

    delta_vx = ball_x_vel - env._sb_init_vx
    in_front = ball_x_local > -0.3  # allow 0.3 m past goal line: deflection accumulates gradually
    fired = (delta_vx > delta_vel_threshold) & in_front & ~env._sb_flag
    env._sb_flag |= fired
    return fired.float()


def softstop(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    velocity_threshold: float = 0.2,
) -> torch.Tensor:
    """One-time reward when ball world-X velocity exceeds velocity_threshold (m/s).

    The ball must actually reverse direction and be rolling back into the field (+X).
    This is a genuine save: foot contact deflected the ball away from the goal.
    Uses its own _softstop_flag, independent of stopball.

    Also gates _ball_is_behind: once this flag is set, footreach deactivates and
    post-save recovery rewards activate immediately.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    if not hasattr(env, "_softstop_flag"):
        env._softstop_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._softstop_flag[just_reset] = False

    in_front = ball_x_local > -0.3  # match stopball tolerance: deflection may complete just past line
    fired = (ball_x_vel > velocity_threshold) & in_front & ~env._softstop_flag
    env._softstop_flag |= fired
    return fired.float()


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
    """Reward ball deflected back along robot local +X axis. Shape (N,).

    Uses robot-local +X (not world +X) to stay consistent with local-frame ball
    spawning: a save means the ball travels back in the direction it came from,
    which is the robot's local +X.
    Returns clamp(vx_local / target_speed, 0, 1) — normalised to [0, 1].
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
    """Sum of squared base roll+pitch angular velocity. Shape (N,)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def ang_vel_z_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Squared yaw angular velocity. Shape (N,).

    A goalkeeper should face the field; spinning wastes reach radius and
    delays re-positioning. Penalised separately from roll/pitch so the weight
    can be tuned independently.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_ang_vel_b[:, 2])


def stayonline(
    env: "ManagerBasedRlEnv",
    line_offset: float = 0.2,
    max_offset: float = 1.2,
) -> torch.Tensor:
    """Penalty for robot drifting away from the goal line along global X.

    Ball approaches from local +X so the goal line is at X = env_origin_X.
    Returns the X deviation above `line_offset`, clamped to `max_offset`.
    """
    robot: Entity = env.scene["robot"]
    x_local = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    dist = torch.clamp(x_local.abs(), line_offset, max_offset) - line_offset
    return dist


def noretreat(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalty for retreating backward (negative body-frame forward velocity).

    Uses body-frame X so the penalty stays correct when the robot yaws during a dive.
    """
    robot: Entity = env.scene["robot"]
    fwd_vel = robot.data.root_link_lin_vel_b[:, 0]
    return -torch.clamp(fwd_vel, -1.0, 0.0)


def feetorientation(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Reward for keeping feet flat (gravity aligned with foot z-axis).

    Flat feet produce better ball deflections during a save.
    """
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    robot: Entity = env.scene[asset_cfg.name]
    feet_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_w_exp = gravity_w[:, None, :].expand(-1, 2, -1)
    gravity_foot = quat_apply(quat_inv(feet_quat_w), gravity_w_exp)
    err = torch.sum(gravity_foot[..., :2] ** 2, dim=-1).sum(dim=-1)
    return torch.exp(-sigma * err)


def postorientation(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Upright posture reward — always active.

    AMP only sees joint_pos/joint_vel, not root orientation, so it cannot push
    the root upright. Gating on ball-is-behind means no upright signal during
    ball approach (80% of episode), causing the policy to drift into backward lean.
    """
    grav_b = env.scene["robot"].data.projected_gravity_b
    err = torch.sum(grav_b[:, :2] ** 2, dim=1)
    return torch.exp(-3.0 * err)


def postangvel(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Low XY angular velocity reward — active when ball is behind. Mirrors ILB postangvel."""
    behind = _ball_is_behind(env, ball_name)
    ang_vel = env.scene["robot"].data.root_link_ang_vel_b
    err = torch.sum(ang_vel[:, :2] ** 2, dim=1)
    return torch.exp(-3.0 * err) * behind.float()


def postlinvel(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Low forward velocity reward — active when ball is behind. Mirrors ILB postlinvel."""
    behind = _ball_is_behind(env, ball_name)
    lin_vel = env.scene["robot"].data.root_link_lin_vel_b
    err = lin_vel[:, 0] ** 2
    return torch.exp(-3.0 * err) * behind.float()


def deviation_waist_joint(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalty for waist joint deviation from default pose (always active)."""
    robot: Entity = env.scene[asset_cfg.name]
    delta = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(delta), dim=-1)


def penalize_kneeheight(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_KNEE_CFG,
) -> torch.Tensor:
    """Penalise shank bodies dropping below min_height above floor.

    Detects kneeling/falling states that would damage real hardware.
    Returns sum of excess below threshold across both shanks.
    """
    robot: Entity = env.scene[asset_cfg.name]
    shank_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]                                # (N,)
    shank_z_local = shank_pos_w[:, :, 2] - floor_z[:, None]             # (N, 2)
    violation = torch.clamp(min_height - shank_z_local, min=0.0)        # (N, 2)
    return violation.sum(dim=-1)


def dof_vel_limits(
    env: "ManagerBasedRlEnv",
    vel_threshold: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalise joint velocities above vel_threshold (rad/s).

    10 rad/s is below all T1 actuator velocity limits.
    Returns sum of squared excess across all joints.
    """
    robot: Entity = env.scene[asset_cfg.name]
    vel = robot.data.joint_vel[:, asset_cfg.joint_ids]                  # (N, J)
    excess = torch.clamp(vel.abs() - vel_threshold, min=0.0)            # (N, J)
    return excess.pow(2).sum(dim=-1)


def torques_normalized_l2(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ALL_JOINTS_CFG,
) -> torch.Tensor:
    """Penalize torques normalized by per-joint stiffness (kp). Mirrors ILB.

    sum(square(torque / kp)) — dimensionless position-error proxy comparable
    across joints. Without normalization, high-stiffness leg joints (kp=200)
    dominate the penalty over soft arm joints (kp=15).
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


def torque_limits(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _ALL_JOINTS_CFG,
    soft_factor: float = 0.95,
) -> torch.Tensor:
    """Penalize actuator torques exceeding per-joint soft torque limit. Mirrors ILB.

    Uses _T1_EFFORT_MAP instead of a universal cap so arms (36 Nm) and
    hip-yaw joints (40 Nm) are penalised at their actual limits.
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


def postupperdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _ARM_JOINT_CFG,
) -> torch.Tensor:
    """Reward arm joints returning to default pose after ball is behind. Mirrors ILB.

    exp(-1 * sum_sq_err) × behind — bounded [0, 1], reward peaks at default pose.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-1.0 * err) * behind.float()


def postwaistdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _WAIST_JOINT_CFG_RECOVERY,
) -> torch.Tensor:
    """Reward waist joint returning to default pose after ball is behind. Mirrors ILB.

    exp(-3 * sum_sq_err) × behind — bounded [0, 1], reward peaks at default pose.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-3.0 * err) * behind.float()


def post_default_pose(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward ALL joints returning to home-keyframe default after ball is deflected.

    Covers legs, arms, and waist together — encourages the robot to stand in its
    standard goalkeeper stance (T-like pose) once the save is complete.
    exp(-sum_sq_err / std^2) × behind — bounded [0, 1].
    std=0.5 rad means reward = 0.37 when RMS joint error is 0.5 rad (≈ 28°).
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-err / (std ** 2)) * behind.float()


def penalize_sharpcontact(
    env: "ManagerBasedRlEnv",
    force_threshold: float = 1200.0,
) -> torch.Tensor:
    """Binary penalty when mean foot contact force exceeds force_threshold.

    Mirrors ILB / upstream Humanoid-Goalkeeper _reward_penalize_sharpcontact.
    Geom layout in feet_contact sensor (sorted by name):
        0: left_foot_1, 1: left_foot_2  → left foot
        2: right_foot_1, 3: right_foot_2 → right foot
    Per-foot max over geoms, then mean — matches upstream 2-body mean.
    Weight: -100.0.
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)          # [B, 8]
    left_max  = force_per_geom[:, :4].max(dim=-1).values     # left_foot1-4
    right_max = force_per_geom[:, 4:].max(dim=-1).values     # right_foot1-4
    mean_force = (left_max + right_max) / 2.0
    return (mean_force > force_threshold).float()


def penalize_self_collision(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Binary penalty when any self-collision is detected in the Trunk subtree.

    Reads from the self_collision ContactSensor. data.found shape: [B, 1].
    Weight: -50.0.
    """
    sensor: ContactSensor = env.scene["self_collision"]
    return (sensor.data.found > 0).any(dim=-1).float()


def airborne_at_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """One-time bonus when softstop fires with either foot airborne.

    softstop fires unconditionally on ball velocity; this fires on top of it
    as a quality bonus when the robot had a foot in the air at the save moment,
    rewarding the committed step/dive motion over a flat-footed shuffle.
    Weight: +15.0.
    """
    if not hasattr(env, "_aas_ss_prev"):
        env._aas_ss_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._aas_flag    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._aas_ss_prev[just_reset] = False
    env._aas_flag[just_reset]    = False

    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is None:
        return torch.zeros(env.num_envs, device=env.device)

    fc = env.scene["feet_contact"].data.found                      # (B, 8)
    any_airborne = (
        ~(fc[:, :4] > 0).any(dim=-1) |                            # left foot up
        ~(fc[:, 4:] > 0).any(dim=-1)                              # right foot up
    )

    just_fired = softstop_fired & ~env._aas_ss_prev
    env._aas_ss_prev[:] = softstop_fired

    fired = just_fired & any_airborne & ~env._aas_flag
    env._aas_flag |= fired
    return fired.float()


def inner_face_orientation_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    alignment_threshold: float = 0.4,
) -> torch.Tensor:
    """One-time bonus when softstop fires and the closest foot's inner face points at the ball.

    Replaces geom-contact detection with foot orientation checking, which is more
    robust: instead of asking "which geom was hit?", we ask "was the foot rotated
    correctly so its medial face faced the ball?".

    The inner face of each foot is its local -Y axis (left) or +Y axis (right):
      Left foot:  inner face normal in local frame = (0, -1, 0)
      Right foot: inner face normal in local frame = (0, +1, 0)
    (From XML: left inner geoms at y=-0.03/-0.01, right inner at y=+0.03/+0.01.)

    Fires once per episode when softstop transitions False→True, if at the save
    frame the foot closest to the ball has its inner-face normal aligned with the
    foot→ball direction by at least alignment_threshold (dot product, 0.4 ≈ 66°).

    No sensor needed — uses foot quaternions directly.
    Weight: +15.0.
    """
    if not hasattr(env, "_ifos_ss_prev"):
        env._ifos_ss_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ifos_flag    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._ifos_ss_prev[just_reset] = False
    env._ifos_flag[just_reset]    = False

    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is None:
        return torch.zeros(env.num_envs, device=env.device)

    just_fired = softstop_fired & ~env._ifos_ss_prev
    env._ifos_ss_prev[:] = softstop_fired

    if not just_fired.any():
        return torch.zeros(env.num_envs, device=env.device)

    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity  = env.scene[ball_name]

    foot_pos_w  = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)
    ball_pos_w  = ball.data.root_link_pos_w                               # (N, 3)

    # Find which foot is closer to the ball.
    dist = torch.norm(foot_pos_w - ball_pos_w[:, None, :], dim=-1)        # (N, 2)
    left_closer = dist[:, 0] <= dist[:, 1]                                 # (N,)

    # Inner face normal in local foot frame:
    #   left foot  → (0, -1, 0)  (medial = negative local Y)
    #   right foot → (0, +1, 0)  (medial = positive local Y)
    left_local_y  = torch.tensor([0.0, -1.0, 0.0], device=env.device).expand(env.num_envs, -1)
    right_local_y = torch.tensor([0.0,  1.0, 0.0], device=env.device).expand(env.num_envs, -1)

    # Rotate local normal into world frame using foot quaternion.
    left_inner_w  = quat_apply(foot_quat_w[:, 0, :], left_local_y)        # (N, 3)
    right_inner_w = quat_apply(foot_quat_w[:, 1, :], right_local_y)       # (N, 3)

    # Pick the closer foot's inner-face world normal.
    inner_normal_w = torch.where(
        left_closer[:, None], left_inner_w, right_inner_w
    )                                                                       # (N, 3)

    # Unit vector from chosen foot to ball.
    closer_foot_pos = torch.where(
        left_closer[:, None], foot_pos_w[:, 0, :], foot_pos_w[:, 1, :]
    )                                                                       # (N, 3)
    foot_to_ball = ball_pos_w - closer_foot_pos                            # (N, 3)
    foot_to_ball = foot_to_ball / (foot_to_ball.norm(dim=-1, keepdim=True) + 1e-6)

    # Alignment: dot product of inner face normal and foot→ball direction.
    alignment = (inner_normal_w * foot_to_ball).sum(dim=-1)                # (N,)
    oriented_correctly = alignment > alignment_threshold

    fired = just_fired & oriented_correctly & ~env._ifos_flag
    env._ifos_flag |= fired
    return fired.float()


def cleanstop(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    speed_threshold: float = 0.25,
) -> torch.Tensor:
    """One-time reward when ball nearly stops after deflection.

    Fires once per episode when softstop has already triggered AND the ball's total
    speed drops below speed_threshold (0.25 m/s). Rewards a clean foot-trap style
    save over a hard uncontrolled deflection.
    Weight: +25.0.
    """
    ball: Entity = env.scene[ball_name]
    ball_speed = ball.data.root_link_lin_vel_w.norm(dim=-1)  # (N,)

    if not hasattr(env, "_cleanstop_flag"):
        env._cleanstop_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._cleanstop_flag[just_reset] = False

    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is None:
        return torch.zeros(env.num_envs, device=env.device)

    fired = softstop_fired & (ball_speed < speed_threshold) & ~env._cleanstop_flag
    env._cleanstop_flag |= fired
    return fired.float()


def foot_clearance(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    target_height: float = 0.10,
) -> torch.Tensor:
    """Reward for lifting feet during ball approach — encourages active stepping.

    Returns max foot height above floor normalized to [0, 1], clamped at target_height (10 cm).
    Deactivates once the ball is behind to avoid rewarding post-save hopping.

    The robot sometimes shuffles without lifting feet (keeps both grounded), producing
    a slide rather than a committed step or dive. This reward creates a gradient for any
    foot-lift, making stepping/diving strictly better than shuffling.
    Weight: +2.0.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]        # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]                                     # (N,)
    foot_z_above_floor = (foot_pos_w[:, :, 2] - floor_z[:, None]).clamp(0.0, target_height)  # (N, 2)
    max_foot_height = foot_z_above_floor.max(dim=-1).values                   # (N,)
    return (max_foot_height / target_height) * (~behind).float()


def feet_slippage(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Reward feet not slipping while in ground contact.

    Mirrors upstream _reward_feet_slippage: exp(-10 * sum(foot_speed * in_contact)).
    Returns 1.0 when airborne or no slip; approaches 0 with high slip velocity.
    Geom layout in feet_contact sensor (sorted by name):
        0: left_foot_1, 1: left_foot_2 → left foot
        2: right_foot_1, 3: right_foot_2 → right foot

    Suppressed (returns 1.0) when ball is within 0.5 m of either foot: the
    contact sensor cannot distinguish ground from ball, so without this gate
    the policy learns to plant feet passively rather than sweep into the ball,
    preventing the force needed for stopball (delta_vx > 1 m/s).
    Weight: +3.0.
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]
    left_in_contact  = (found[:, :4] > 0).any(dim=-1)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)
    in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1).float()  # [B, 2]

    robot: Entity = env.scene[asset_cfg.name]
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # [B, 2, 3]
    foot_speed = torch.norm(foot_vel_w, dim=-1)                             # [B, 2]
    contactvel = torch.sum(foot_speed * in_contact, dim=-1)
    slippage_rew = torch.exp(-10.0 * contactvel)

    ball: Entity = env.scene[ball_name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # [B, 2, 3]
    ball_pos_w = ball.data.root_link_pos_w                                   # [B, 3]
    min_dist = torch.norm(foot_pos_w - ball_pos_w[:, None, :], dim=-1).min(dim=-1).values
    ball_near_foot = min_dist < 0.5
    return torch.where(ball_near_foot, torch.ones_like(slippage_rew), slippage_rew)
