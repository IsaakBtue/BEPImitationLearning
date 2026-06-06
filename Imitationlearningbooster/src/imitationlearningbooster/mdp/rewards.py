"""Goalkeeper task reward terms for Booster T1."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
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

    Reuses _sb_init_vx from stopball if already initialised; otherwise
    falls back to the absolute threshold so post-save rewards still gate
    correctly even if stopball hasn't run first this step.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    init_vx = getattr(env, "_sb_init_vx", None)
    if init_vx is not None:
        delta_vx = ball_x_vel - init_vx
        return (ball_x_local < 0.0) | (delta_vx > 1.0)
    # Fallback before stopball has run (first policy step of first episode).
    return (ball_x_local < 0.0) | (ball_x_vel > 1.0)


def eereach(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _HAND_CFG,
    reach_th: float = 0.3,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Sigmoid reach reward with velocity amplification, intercept target, and behind-ball boost.

    Mirrors the original isaacgym _reward_eereach faithfully:

    Phase 1 (ball far away, y_local > 1.5):
      - Use _ball_end_target (predicted intercept point) for lateral pre-positioning.
      - Compute aside = (end_target_x - robot_x) / 0.8, clamped ±1.
      - phase1_rew = 1 - |aside| (reward being on the correct side of the goal).

    Phase 2 (ball close, y_local ≤ 1.5):
      - Use current ball position OR end_target (if ball y_local > 0.5) for distance.
      - Sigmoid reach reward × vel_sigma multiplier.
      - Mirrors upstream: approachidx updates end_target when ball_x_local ∈ [0.1, 0.5].

    vel_sigma computation (mirrors G1 _reward_eereach jump_scale mechanism):
      - Non-jump envs (motion_type 0,1,4,5): vel_sigma = 1 + 3 × clamp(vel_toward, 0, 3)
      - Jump envs (motion_type 2,3):         vel_sigma = 1 + jump_scale × clamp(vel_toward, 0, 3)
        where jump_scale = 3 + 3 × curriculumupdate (3→9 as difficulty 0→1).
        curriculumupdate = _ball_difficulty × 2 (maps 0→1 difficulty to 0→2 curriculum stages).

    Post-pass (behind=True):
      - vel_sigma set to flat 2.0 (mirrors G1 exactly; previous port incorrectly doubled).

    Upright gate: multiply by (1 - clamp(grav_xy², 0, 1)) as in original.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    hand_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)

    # Ball distance from env origin (X axis = approach axis).
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]      # (N,)

    # Compute behind once; reused for phase1_mask guard and post-pass vel_sigma.
    behind = _ball_is_behind(env, ball_name)

    # Retrieve predicted intercept point set by _reset_ball.
    end_target = getattr(env, "_ball_end_target", None)

    # ---- Phase 1: pre-positioning when ball is far (x_local > 1.5) ----
    # Mirrors upstream:
    #   end_target_local = end_target - torso_pos
    #   asidegoal = clip(end_target_local[:, 1], -1, 1)   [original Y is lateral]
    #   asidegoal[|asidegoal| < 0.3] = 0
    #   verticalgoal = clip(torso_z - clip(end_target_z, 0.3, 1.2), 0, 1)
    #   phase1_rew = 1 - (verticalgoal + |asidegoal|) / 2
    # New system: X = approach axis, Y = lateral axis, Z = vertical (same).
    # Guard: phase1 must not fire for deflected balls still at x > 1.5 (mirrors G1 velocity check).
    phase1_mask = (ball_x_local > 1.5) & ~behind                       # (N,)

    if end_target is not None:
        root_pos_w = robot.data.root_link_pos_w                         # (N, 3)
        end_target_local = end_target - root_pos_w                      # (N, 3)

        # Lateral (Y) alignment — mirrors original end_target_local[:, 1] (Y lateral in G1).
        asidegoal = end_target_local[:, 1].clamp(-1.0, 1.0)
        asidegoal = torch.where(asidegoal.abs() < 0.3, torch.zeros_like(asidegoal), asidegoal)

        # Vertical (Z) — same in both port and original.
        torso_z = root_pos_w[:, 2]
        verticalgoal = (torso_z - end_target[:, 2].clamp(0.3, 1.2)).clamp(0.0, 1.0)

        phase1_rew = 1.0 - (verticalgoal + asidegoal.abs()) / 2.0      # (N,)
    else:
        phase1_rew = torch.zeros(env.num_envs, device=env.device)

    # ---- Phase 2: sigmoid reach reward when ball is close ----
    # Target point: use end_target when ball is between 0.5–1.5 m away (in-flight approach),
    # snap to current ball position when ≤ 0.5 m (mirrors upstream approachidx update).
    if end_target is not None:
        use_end_target = (ball_x_local > 0.5) & ~phase1_mask           # 0.5 < x ≤ 1.5
        target_pos = torch.where(
            use_end_target.unsqueeze(-1),
            end_target,
            ball_pos_w,
        )                                                                # (N, 3)
    else:
        target_pos = ball_pos_w

    to_target = target_pos[:, None, :] - hand_pos_w                    # (N, 2, 3)
    dist_to_target = torch.norm(to_target, dim=-1)                     # (N, 2)

    # --- Jump-region-aware vel_sigma (mirrors G1 _reward_eereach) ---
    # G1: upper-body world-Y lateral velocity for side-saves, world-Z for jumps.
    # Port: world-X lateral velocity (90° rotation applied), world-Z unchanged.
    difficulty = float(getattr(env, "_ball_difficulty", 0.0))
    curriculumupdate = difficulty * 2.0                                # 0→2
    jump_scale = 3.0 + 3.0 * curriculumupdate                         # 3→9 across stages

    torso_vel_w = robot.data.root_link_lin_vel_w                       # (N, 3)

    # is_left/is_jump must be computed before min_dist so we can route to the correct hand.
    try:
        motion_cmd = env.command_manager._terms.get("motion", None)
        if motion_cmd is not None and hasattr(motion_cmd, "motion_type_ids"):
            type_ids = motion_cmd.motion_type_ids                      # [N] long tensor
            is_jump  = (type_ids == 2) | (type_ids == 3)
            is_left  = (type_ids == 0) | (type_ids == 2) | (type_ids == 4)
        else:
            is_jump = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            is_left = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    except Exception:
        is_jump = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        is_left = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Route to the correct hand per motion type: left hand (idx 0) for types 0/2/4,
    # right hand (idx 1) for types 1/3/5. Mirrors G1 which selects by save region.
    min_dist = torch.where(is_left, dist_to_target[:, 0], dist_to_target[:, 1])

    # Base sigmoid: 1 at dist=0, 0 far away.
    rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (min_dist - reach_th)))

    # Side-saves: reward lateral torso X-velocity toward the target side.
    # Jumps: reward upward (Z) torso velocity for the leap.
    lateral_vel_x = torso_vel_w[:, 0]
    base_vel_sigma = torch.where(
        is_left,
        1.0 + 3.0 * torch.clamp(-lateral_vel_x, 0.0, 3.0),   # left: reward -X motion
        1.0 + 3.0 * torch.clamp( lateral_vel_x, 0.0, 3.0),   # right: reward +X motion
    )
    jump_vel_sigma = 1.0 + jump_scale * torch.clamp(torso_vel_w[:, 2], 0.0, 3.0)

    vel_sigma = torch.where(is_jump, jump_vel_sigma, base_vel_sigma)

    # Post-pass: mirrors G1 which sets vel_sigma = 2.0 (flat) when behind.
    vel_sigma = torch.where(behind, torch.full_like(vel_sigma, 2.0), vel_sigma)

    # Combine: phase1 when ball is far, phase2 when ball is close.
    taskrew = torch.where(phase1_mask, phase1_rew, rew * vel_sigma)

    projected_grav = env.scene["robot"].data.projected_gravity_b
    upright = (1.0 - torch.clamp(torch.sum(projected_grav[:, :2] ** 2, dim=1), 0.0, 1.0))
    return taskrew * upright


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
<<<<<<< Updated upstream
    """One-time reward when ball decelerates ≥1 m/s from its initial Y velocity.

    Mirrors the original Humanoid-Goalkeeper _reward_stopball logic but with a
    lower threshold (1.0 vs G1's 2.0) because MuJoCo's soft contact solver
    produces smaller velocity impulses than PhysX. At 2.0 m/s, slow-ball saves
    (ball approaching at -1 m/s, deflected to +0.5 m/s → Δvy = 1.5 m/s) never
    fire this 100-weight reward, starving the primary training signal.
=======
    """One-time reward when ball decelerates ≥2 m/s from its initial X velocity.

    Mirrors the original Humanoid-Goalkeeper _reward_stopball exactly:
    compares current ball velocity against the velocity stored at episode
    reset (not per-step delta), so the threshold is robust to air resistance.
    A per-env flag prevents re-firing after the first deceleration event.

    Ball approaches from +X so initial vx < 0; deceleration/reversal means
    current_vx - initial_vx > delta_vel_threshold (2.0 m/s, matching G1).
>>>>>>> Stashed changes
    """
    ball: Entity = env.scene[ball_name]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    if not hasattr(env, "_sb_init_vx"):
        env._sb_init_vx = ball_x_vel.clone()
        env._sb_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Store the ball's initial velocity at episode reset (first step).
    just_reset = env.episode_length_buf <= 1
    env._sb_flag[just_reset] = False
    env._sb_init_vx[just_reset] = ball_x_vel[just_reset].clone()

    delta_vx = ball_x_vel - env._sb_init_vx
    in_front = ball_x_local > 0.0
    fired = (delta_vx > delta_vel_threshold) & in_front & ~env._sb_flag

    env._sb_flag |= fired

    return fired.float()


def stayonline(
    env: ManagerBasedRlEnv,
    line_offset: float = 0.2,
    max_offset: float = 1.2,
) -> torch.Tensor:
    """Penalty for robot retreating/advancing away from the goal line.

    Ball approaches from +X so the goal line is X = env_origin_X.
    The robot slides laterally in Y to intercept; this penalises X deviation.
    """
    robot: Entity = env.scene["robot"]
    x_local = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    dist = torch.clamp(x_local.abs(), line_offset, max_offset) - line_offset
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
    force_threshold: float = 1000.0,
) -> torch.Tensor:
    """Penalize large impulsive foot contact forces.

    Mirrors upstream _reward_penalize_sharpcontact exactly:
        return (mean(norm(foot_contact_forces)) > max_contact_force) * 1.0
    where max_contact_force = 1000 N (g1_29_config.py cfg.rewards.max_contact_force).

    Upstream averages over 2 foot bodies: mean(norm(contact_forces[:, feet, :]), dim=-1).
    Port geom layout (4 geoms, sorted by name):
        index 0: left_foot_1  ─┐ left foot
        index 1: left_foot_2  ─┘
        index 2: right_foot_1 ─┐ right foot
        index 3: right_foot_2 ─┘
    Fix: per-foot max over geoms, then mean over feet — matches upstream 2-body mean.
    Weight: -100.0 (same as upstream).
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)          # [B, 4]
    left_max  = force_per_geom[:, :2].max(dim=-1).values     # max of left_foot_1, left_foot_2
    right_max = force_per_geom[:, 2:].max(dim=-1).values     # max of right_foot_1, right_foot_2
    mean_force = (left_max + right_max) / 2.0                # [B]
    return (mean_force > force_threshold).float()


def penalize_self_collision(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Binary penalty when any self-collision is detected in the Trunk subtree.

    Reads from the self_collision ContactSensor (already declared in the scene).
    The sensor monitors Trunk-subtree vs Trunk-subtree contacts; found > 0 means
    at least one matching contact was detected this step.

    data.found shape: [B, 1] (reduce="none", num_slots=1).
    Returns 1.0 on any self-collision, 0.0 otherwise.
    Weight: -50.0.
    """
    sensor: ContactSensor = env.scene["self_collision"]
    return (sensor.data.found > 0).any(dim=-1).float()


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
    air_height_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward safe landing after a jump save.

    Mirrors upstream _reward_successland exactly:
        jump = root_z > 1.0
        has_in_air |= jump
        has_contact = foot_0_contact & foot_1_contact
        one_feet_contact = exactly one foot down & has_in_air
        successful_landings = has_contact & has_in_air
        air_reward = has_in_air.float()
        landing_reward = successful_landings.float() * 5.0
        one_feet_punish = one_feet_contact.float() * -1.0
        jump_ids = end_regions == 2 | 3   ← upstream gates on jump-region envs only
        return (air_reward + landing_reward + one_feet_punish) * jump_ids

    Port adaptation:
    - No region partitioning: all envs are treated as jump-region envs, BUT
      we require _has_in_air so the reward only fires after actual jumps (never
      fires for ground-level saves). This is functionally equivalent to the
      upstream jump_ids gate.
    - Contact detection: foot height < height_threshold (mjlab proxy for contact sensor).
    - _has_in_air reset on episode_length_buf <= 1.

    Weight: 4.0 (same as upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    # Initialise per-env _has_in_air buffer.
    if not hasattr(env, "_has_in_air"):
        env._has_in_air = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Reset at start of each episode.
    just_reset = env.episode_length_buf <= 1
    env._has_in_air[just_reset] = False

    # Track whether the robot has actually left the ground this episode.
    root_z = robot.data.root_link_pos_w[:, 2]
    env_z  = env.scene.env_origins[:, 2]
    env._has_in_air |= (root_z - env_z > air_height_threshold)

    # Foot contact via height proxy.
    foot_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]    # [B, 2]
    env_z_2d = env.scene.env_origins[:, 2:3]                          # [B, 1]
    foot_down = (foot_z - env_z_2d < height_threshold)                # [B, 2]

    has_contact       = foot_down[:, 0] & foot_down[:, 1]
    one_feet_contact  = (foot_down[:, 0] ^ foot_down[:, 1]) & env._has_in_air

    successful_landings = has_contact & env._has_in_air

    landing_reward  = successful_landings.float() * 5.0
    one_feet_punish = one_feet_contact.float() * -1.0

    # air_reward intentionally omitted: T1 has no jump-region partitioning.
    # G1 gates this on jump_ids (end_regions == 2|3); without that gate,
    # emitting +1/step for any stumble corrupts the reward signal.
    return landing_reward + one_feet_punish


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
) -> torch.Tensor:
    """Reward feet not slipping while in ground contact.

    Mirrors upstream _reward_feet_slippage exactly:
        contactvel = sum(norm(foot_vel_3d) * in_contact, over feet)
        return exp(-10 * contactvel)
    Returns 1.0 when no contact or no slip; approaches 0 with high slip.
    Weight: +3.0 (positive reward for not slipping — same as upstream).

    Contact detection: physics-based via feet_contact sensor (found > 0).
    Replaces the previous foot-height proxy which returned exp(0) = 1.0
    whenever feet were in the air (e.g., during a dive), inflating WandB
    curves and masking true slippage behaviour.

    Geom layout from feet_contact sensor (sorted by name):
        index 0: left_foot_1  ─┐ left foot
        index 1: left_foot_2  ─┘
        index 2: right_foot_1 ─┐ right foot
        index 3: right_foot_2 ─┘

    Velocity from body_link_lin_vel_w matches upstream rigid_body_states[:, feet, 7:10].
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 4]

    left_in_contact  = (found[:, 0] > 0) | (found[:, 1] > 0)  # [B]
    right_in_contact = (found[:, 2] > 0) | (found[:, 3] > 0)  # [B]
    in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1).float()  # [B, 2]

    robot: Entity = env.scene[asset_cfg.name]
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # [B, 2, 3]
    foot_speed = torch.norm(foot_vel_w, dim=-1)                             # [B, 2]

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
