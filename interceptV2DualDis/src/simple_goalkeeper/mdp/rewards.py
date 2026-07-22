"""Goalkeeper reward terms for SimpleGoalKeeper (Phase 1 — feet only)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from simple_goalkeeper.robots.t1_constants import (
    ANKLE_ACTUATOR,
    ARM_ACTUATOR,
    HIP_PITCH_ACTUATOR,
    KNEE_ACTUATOR,
    WAIST_HIP_ROLL_YAW_ACTUATOR,
)

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
# FIX 2026-07-22: added alongside postupperdofpos/postwaistdofpos -- see
# postlegdofpos's docstring below for why G1 has no equivalent reward to
# port for these joints, and why that's a task-structure fact, not a gap
# in the existing G1-parity port.
_LEG_JOINT_CFG_RECOVERY = SceneEntityCfg(
    "robot",
    joint_names=(
        "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Hip_Pitch", "Left_Knee_Pitch",
        "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Hip_Pitch", "Right_Knee_Pitch",
        "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
)
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

# Per-joint velocity limits (rad/s), from real T1 motor specs already defined
# in t1_constants.py (ElectricActuator.velocity_limit, rated RPM converted to
# rad/s -- t1_constants._rpm()). The T1 MJCF (xmls/t1_headless.xml) defines no
# <actuator>/<general> block and no per-joint velocity attribute at all --
# joints only carry a position `range` -- so there is no XML-sourced velocity
# limit to read; MuJoCo does not require or enforce one on its own. These
# values are the correct, confidently-sourced alternative: real hardware motor
# ratings already used elsewhere in this codebase (ARM/WAIST_HIP_ROLL_YAW/
# HIP_PITCH/KNEE/ANKLE actuators feed action_scale and kp/kd), just never
# wired into a joint-velocity limit before now. Source of item 7's 2026-07-20
# reward audit fix -- see docs/BugFixes.md.
_T1_VEL_LIMIT_MAP: dict[str, float] = {
    "Left_Shoulder_Pitch": ARM_ACTUATOR.velocity_limit, "Left_Shoulder_Roll": ARM_ACTUATOR.velocity_limit,
    "Left_Elbow_Pitch":    ARM_ACTUATOR.velocity_limit, "Left_Elbow_Yaw":     ARM_ACTUATOR.velocity_limit,
    "Right_Shoulder_Pitch": ARM_ACTUATOR.velocity_limit, "Right_Shoulder_Roll": ARM_ACTUATOR.velocity_limit,
    "Right_Elbow_Pitch":   ARM_ACTUATOR.velocity_limit, "Right_Elbow_Yaw":    ARM_ACTUATOR.velocity_limit,
    "Waist":               WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit,
    "Left_Hip_Pitch":      HIP_PITCH_ACTUATOR.velocity_limit, "Right_Hip_Pitch": HIP_PITCH_ACTUATOR.velocity_limit,
    "Left_Hip_Roll":       WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit, "Left_Hip_Yaw": WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit,
    "Right_Hip_Roll":      WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit, "Right_Hip_Yaw": WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit,
    "Left_Knee_Pitch":     KNEE_ACTUATOR.velocity_limit, "Right_Knee_Pitch": KNEE_ACTUATOR.velocity_limit,
    "Left_Ankle_Pitch":    ANKLE_ACTUATOR.velocity_limit, "Right_Ankle_Pitch": ANKLE_ACTUATOR.velocity_limit,
    "Left_Ankle_Roll":     ANKLE_ACTUATOR.velocity_limit, "Right_Ankle_Roll":  ANKLE_ACTUATOR.velocity_limit,
}


def _get_correct_foot_idx(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Determine which foot (left=0, right=1) should contact the ball.

    Ball crossing Y > env origin Y → left foot (0) should reach.
    Ball crossing Y <= env origin Y → right foot (1) should reach.

    Returns: (N,) tensor of dtype int64, values 0 or 1.
    """
    crossing_y = _get_ball_crossing_y(env, ball_name)
    is_left_ball = crossing_y > env.scene.env_origins[:, 1]
    foot_idx = torch.where(is_left_ball,
                           torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                           torch.ones(env.num_envs, dtype=torch.long, device=env.device))
    return foot_idx


def _ball_is_behind(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Bool mask (N,): ball has been deflected OR has crossed the goal line.

    Fires when ANY of:
      - ball_x < 0 (crossed goal line)
      - _softstop_flag is set (ball reversed to positive X velocity)
      - _sb_flag is set (delta_vx > 1.0 — hard contact, even without full reversal)

    Both flags needed: softstop alone misses hard partial deflections; stopball alone
    misses soft full reversals. Together they prevent footreach from running on
    after any meaningful contact, stopping the farm-footreach-post-stopball exploit.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    softstop_fired = getattr(env, "_softstop_flag", None)
    stopball_fired = getattr(env, "_sb_flag", None)
    already_deflected = (
        (softstop_fired if softstop_fired is not None
         else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        | (stopball_fired if stopball_fired is not None
           else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    )

    return (ball_x_local < 0.0) | already_deflected


def _robot_x_axis_w(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Robot local +X unit vector in world frame. Shape (N, 3)."""
    robot: Entity = env.scene["robot"]
    x_local = torch.zeros(env.num_envs, 3, device=env.device)
    x_local[:, 0] = 1.0
    return quat_apply(robot.data.root_link_quat_w, x_local)


def _get_ball_crossing_y(env: "ManagerBasedRlEnv", ball_name: str) -> torch.Tensor:
    """Frozen Y coordinate where the ball will cross the goal line (x_local = 0).

    Uses _rsi_cross_y set analytically by reset_ball_rolling at spawn time (no buffer
    lag). Converts from env-relative to world frame by adding env_origin_y.
    Falls back to velocity-based estimate only when _rsi_cross_y is unavailable.
    """
    just_reset = env.episode_length_buf <= 1
    if not hasattr(env, "_ball_crossing_y"):
        env._ball_crossing_y = env.scene.env_origins[:, 1].clone()

    if just_reset.any():
        rsi_cross_y = getattr(env, "_rsi_cross_y", None)
        if rsi_cross_y is not None:
            # Analytically computed at spawn: no buffer lag, exact match with RSI pool.
            env._ball_crossing_y[just_reset] = (
                env.scene.env_origins[just_reset, 1] + rsi_cross_y[just_reset]
            )
        else:
            ball: Entity = env.scene[ball_name]
            ball_pos_w = ball.data.root_link_pos_w
            ball_vel_w = ball.data.root_link_lin_vel_w
            ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]
            bvx = ball_vel_w[just_reset, 0].clamp(max=-0.1)
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

    green-ball-baseline (2026-07-10): the two-stage blue-waypoint mechanism
    (SimpleGoalKeeper port) has been removed on this branch to isolate and
    verify the core AMP+PPO+save-reward mechanism without it -- see
    docs/BugFixes.md. The reach target is always the direct crossing point,
    matching G1's own eereach (no intermediate waypoint concept upstream).
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)

    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]     # (N,)
    robot_y_w = robot.data.root_link_pos_w[:, 1]                       # (N,)

    # Frozen crossing Y (where the ball will cross the goal line) — used both as
    # the reach target and for foot-side selection below.
    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)
    reach_target_y = crossing_y                                         # (N,)

    # Phase 1: pre-position laterally when ball is far (> 1.5 m in front).
    lateral_error = reach_target_y - robot_y_w                          # positive → target right
    asidegoal = lateral_error.clamp(-1.0, 1.0)
    asidegoal = torch.where(asidegoal.abs() < 0.3, torch.zeros_like(asidegoal), asidegoal)
    phase1_rew = 1.0 - asidegoal.abs()                                  # 1=aligned, 0=1 m off

    # Phase 2 crossing point: frozen when far, live ball when close (mirrors G1
    # end_target update). G1 post_physics_step lines 203-206: end_target switches
    # to ball_states[:, :3] when balllocal < 0.5 — robot converges on the actual
    # ball in the final 0.5 m. The X coordinate stays at goal_x_w (stayonline
    # constraint keeps the foot at the goal line).
    goal_x_w  = env.scene.env_origins[:, 0]       # (N,)
    floor_z_w = env.scene.env_origins[:, 2]       # (N,)
    live_y = ball_pos_w[:, 1]
    live_z = ball_pos_w[:, 2]
    ball_close = ball_x_local < 0.5                                     # (N,) bool — mirrors G1 balllocal < 0.5

    # Switch from frozen crossing point to live ball Y/Z when ball is within 0.5 m.
    target_y = torch.where(ball_close, live_y, reach_target_y)
    target_z = torch.where(ball_close, live_z, floor_z_w + 0.10)

    crossing_point = torch.stack(
        [goal_x_w, target_y, target_z], dim=-1
    )                                                                     # (N, 3)
    # Side-specific foot: left foot (idx 0) for +Y crossing, right (idx 1) for -Y.
    is_left_ball = crossing_y > env.scene.env_origins[:, 1]
    foot_idx = torch.where(is_left_ball,
                           torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                           torch.ones(env.num_envs, dtype=torch.long, device=env.device))
    foot_pos_active = foot_pos_w[torch.arange(env.num_envs, device=env.device), foot_idx]
    dist_to_crossing = torch.norm(foot_pos_active - crossing_point, dim=-1)  # (N,)
    reach_rew = 1.0 - 1.0 / (1.0 + torch.exp(-sigma * (dist_to_crossing - reach_th)))

    # Lateral velocity toward crossing side amplifies the reach reward.
    lateral_vel_y = robot.data.root_link_lin_vel_w[:, 1]
    vel_toward = torch.where(lateral_error > 0, lateral_vel_y, -lateral_vel_y)

    # FIX 2026-07-15 (REVERTED 2026-07-18): this used to apply a region-
    # conditional multiplier here, escalating for far regions with
    # far_scale = 3.0 + 1.5*curriculumupdate, modeled on G1's jump-region
    # jump_scale = 3.0 + 3.0*curriculumupdate (legged_robot.py:1379-1388,
    # end_regions 2/3). Two independent subagent audits (2026-07-18, no
    # shared context, unanimous) found this was the wrong G1 region: jump's
    # escalating scale keys off VERTICAL velocity and is paired with two
    # jump-region-specific safety rewards (_reward_airfeetorientation,
    # _reward_successland, both gated on root_states[:,2]>1.0 -- a literal
    # dive) that manage foot orientation while airborne and landing quality.
    # None of that exists for a foot-only, always-grounded task -- which is
    # plausibly why the escalation caused the overshoot-and-fall behavior
    # this fix's slope-halving (07-15) only partially addressed.
    #
    # G1's STEP region (ranges_4/ranges_5, end_regions 4/5) is the actual
    # mechanical match: grounded, lateral velocity, and uses the exact SAME
    # FLAT scale (1 + 3.0*v) as every non-jump region -- no curriculumupdate
    # term at all (legged_robot.py:1382,1384-1385). The "far crossings need
    # more incentive than near" premise this multiplier was built on has no
    # support in the correct G1 analog; step's own difficulty ramp comes
    # entirely from its width curriculum (see far_travel_curriculum,
    # events.py), not from an escalating reward multiplier. Reverted to the
    # flat 3.0 for every region -- see docs/BugFixes.md 2026-07-18.
    vel_scale = torch.full_like(vel_toward, 3.0)
    vel_sigma = 1.0 + vel_scale * vel_toward.clamp(0.0, 3.0)  # flat 1-10x for every region, matching G1's hand/step formula

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
    """Dense continuous reward for foot near the ball crossing point.

    Uses the frozen goal-line crossing point (env._ball_crossing_y, set by
    _get_ball_crossing_y) as the target when the ball is far (>= 0.5 m from goal line).
    When the ball is within 0.5 m, switches to the live ball Y/Z position (mirrors G1
    end_target update in post_physics_step lines 203-206). Rewards exp(-sigma * dist)
    so there is always a gradient pulling the foot toward the arrival point, unlike the
    one-shot stopball. Deactivates once the ball is behind (same gate as footreach).
    Weight: +5.0.

    green-ball-baseline (2026-07-10): two-stage blue-waypoint mechanism removed,
    see footreach's docstring.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]     # (N,)

    # Frozen crossing Y — used both as the reach target and for foot-side selection.
    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)
    reach_target_y = crossing_y                                         # (N,)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z    = env.scene.env_origins[:, 2]

    # Live ball switch: when ball is within 0.5 m of goal line, track live position.
    # Mirrors G1 post_physics_step: end_target = ball_states[:, :3] when balllocal < 0.5.
    ball_close = ball_x_local < 0.5
    live_y = ball_pos_w[:, 1]
    live_z = ball_pos_w[:, 2]
    target_y = torch.where(ball_close, live_y, reach_target_y)
    target_z = torch.where(ball_close, live_z, env_z + 0.10)

    crossing_point = torch.stack(
        [goal_x_w, target_y, target_z], dim=-1                         # (N, 3)
    )

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
    is_left_ball = crossing_y > env.scene.env_origins[:, 1]
    foot_idx = torch.where(is_left_ball,
                           torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                           torch.ones(env.num_envs, dtype=torch.long, device=env.device))
    foot_pos_active = foot_pos_w[torch.arange(env.num_envs, device=env.device), foot_idx]
    min_dist = torch.norm(foot_pos_active - crossing_point, dim=-1)     # (N,)

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

    green-ball-baseline (2026-07-10): the two-stage blue-waypoint landing gate
    has been removed on this branch -- fires unconditionally on any qualifying
    deflection, no longer requires a prior blue-midpoint landing. See
    docs/BugFixes.md.

    FIX 2026-07-14: was missing any check on WHICH foot deflected the ball --
    a deflection off the wrong (non-assigned) foot fired full reward exactly
    like a correct-foot save. Now requires the task-assigned foot
    (_get_correct_foot_idx) to actually be in contact, matching softstop's
    own correct-foot gate (which already existed but only fed downstream
    quality bonuses, never gated softstop itself either -- see that fix).

    FIX 2026-07-15: that check used "feet_contact", a GROUND-contact sensor
    (secondary=None -- fires whenever a foot touches anything, mostly the
    ground) -- not ball-specific. Tried switching to a dedicated
    "ball_contact" foot-vs-ball ContactSensorCfg instead, but that made
    stopball/softstop stop firing almost entirely in practice (MuJoCo
    contact detection for a small rolling ball vs. foot geoms is
    apparently too narrow a window to reliably catch at the exact step
    the velocity threshold trips). REVERTED to "feet_contact".

    FIX 2026-07-15 (second pass): instead, requires stopball's own raw
    deflection condition to also be true the SAME step softstop's condition
    fires (env._sb_deflection_now, set below, read by softstop) -- a
    genuine single-contact event should trip both stopball's delta-vx
    check and softstop's absolute-reversal check at essentially the same
    physical instant; a coincidental "wrong foot happens to be standing
    near the ball as it settles" scenario should not.
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

    foot_idx = _get_correct_foot_idx(env, ball_name)
    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
    left_in_contact = (found[:, :4] > 0).any(dim=-1)   # (B,)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)  # (B,)
    foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)
    correct_foot_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]

    # Raw condition (not gated by the one-shot ~env._sb_flag latch) -- this
    # is "is a deflection happening right now", read by softstop below.
    # Registered before "softstop" in goalkeeper_env_cfg.py's rewards dict,
    # so this is fresh (this step, not stale) by the time softstop runs.
    deflection_now = (delta_vx > delta_vel_threshold) & in_front & correct_foot_contact
    env._sb_deflection_now = deflection_now

    fired = deflection_now & ~env._sb_flag
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

    Also gates _ball_is_behind and tracks which foot was in contact when softstop fires
    (used by single_foot_save, inner_face_orientation_save, cleanstop, airborne_at_save).

    green-ball-baseline (2026-07-10): the two-stage blue-waypoint landing gate
    has been removed on this branch, see stopball's docstring.

    FIX 2026-07-14: correct-foot contact was computed AFTER `fired`, only to
    record env._softstop_correct_foot for downstream quality bonuses -- it
    never gated `fired` itself, so a deflection off the wrong (non-assigned)
    foot earned this reward's full weight (up to 262.5, the single largest
    term in the whole reward table) exactly like a correct-foot save. Now
    computed unconditionally and required in `fired`. env._softstop_correct_foot
    is consequently always True whenever _softstop_flag is True -- downstream
    gates on it become no-ops, which is fine (correct-foot is now guaranteed
    at this point rather than merely tracked).

    FIX 2026-07-15: that check used "feet_contact", a GROUND-contact sensor
    (secondary=None) -- not ball-specific, so a foot merely standing on the
    ground satisfied it regardless of which foot actually deflected the
    ball. Tried a dedicated "ball_contact" foot-vs-ball sensor instead, but
    that made stopball/softstop stop firing almost entirely in practice.
    REVERTED to "feet_contact".

    FIX 2026-07-15 (second pass): instead, now additionally requires
    env._sb_deflection_now (stopball's own raw deflection condition, set
    THIS SAME step -- stopball is registered before softstop in
    goalkeeper_env_cfg.py's rewards dict) to also be true. A genuine
    single-contact event should trip both stopball's delta-vx check and
    softstop's absolute-reversal check at essentially the same physical
    instant; requiring both prevents a wrong-foot-standing-nearby
    coincidence from firing this on its own. See stopball's docstring.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_vel = ball.data.root_link_lin_vel_w[:, 0]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    if not hasattr(env, "_softstop_flag"):
        env._softstop_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._softstop_correct_foot = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._softstop_flag[just_reset] = False
    env._softstop_correct_foot[just_reset] = False

    in_front = ball_x_local > -0.3

    foot_idx = _get_correct_foot_idx(env, ball_name)
    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
    left_in_contact = (found[:, :4] > 0).any(dim=-1)   # (B,)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)  # (B,)
    foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)
    correct_foot_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]

    same_step_as_stopball = getattr(
        env, "_sb_deflection_now", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )

    fired = (
        (ball_x_vel > velocity_threshold) & in_front & correct_foot_contact
        & same_step_as_stopball & ~env._softstop_flag
    )
    env._softstop_correct_foot[fired] = correct_foot_contact[fired]

    env._softstop_flag |= fired
    return fired.float()


def success(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    strict_th: float = 0.15,
) -> torch.Tensor:
    """Continuing, doubled-after-save, close-to-target reward. Ports G1 _reward_success.

    G1 (legged_robot.py:1402-1403): `(success_flag + 1.0) * (dist < strict_th)`.
    success_flag is set once inside _reward_stopball, on the exact same
    delta-vx deflection event that fires stopball itself (legged_robot.py:
    1411-1413: `success_ids = (stop_flag==0) & changevel; success_flag[success_ids]
    = 1.0`), and persists for the rest of the episode (only cleared on reset,
    legged_robot.py:721). `dist` is a continuously-updated hand-to-end_target
    distance recomputed every physics step for every env regardless of phase
    (post_physics_step lines 203-223) -- crucially NOT gated by "ball behind":
    the reward keeps firing at 2x (once success_flag is set) for as long as the
    hand stays within strict_th of the target, including after the save.
    strict_th=0.15 (g1_29_config.py:342), tighter than eereach's own reach_th=0.2
    -- success requires closer proximity than the reach reward's own threshold.

    SGK port:
      - success_flag -> env._sb_flag. stopball() (this module) sets _sb_flag on
        the identical triggering event (its own delta_vx > threshold deflection,
        `fired = deflection_now & ~env._sb_flag; env._sb_flag |= fired`) and
        clears it only at reset -- the same one-shot-then-persistent semantics
        as G1's success_flag, on the same underlying event (a qualifying
        deflection). stopball is registered before this term in
        goalkeeper_env_cfg.py's rewards dict, so _sb_flag is fresh (this step)
        by the time this runs.
      - dist -> distance from the task-assigned foot (_get_correct_foot_idx) to
        the same frozen-far/live-close crossing_point used by footreach and
        foot_proximity (see those docstrings: frozen crossing_y when the ball
        is >= 0.5 m from the goal line, live ball Y/Z when closer -- mirrors
        G1's own end_target update, post_physics_step lines 203-206). Not
        gated by _ball_is_behind, matching G1 exactly: the foot is rewarded for
        staying planted at the save spot through the rest of the episode, not
        only during the approach.

    Weight/curriculum: G1 uses 5 -> 12.5 (success_init=5.0, g1_29_config.py:300,
    scaled by the same weight = base*(1+0.5*curriculumupdate) formula as every
    other curriculum-scaled reward here, max at curriculumupdate=3). Ported
    verbatim as `success_curriculum` in goalkeeper_env_cfg.py. Not folded into
    the item-8 stopball/softstop/quality-bonus peak-magnitude cap: G1 itself
    keeps `success` as its own separate-scale term, outside `_reward_stopball`'s
    own weight ceiling -- this port preserves that same separation.
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                                # (N, 3)
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]        # (N,)
    crossing_y = _get_ball_crossing_y(env, ball_name)                     # (N,)

    goal_x_w = env.scene.env_origins[:, 0]
    env_z    = env.scene.env_origins[:, 2]
    ball_close = ball_x_local < 0.5
    target_y = torch.where(ball_close, ball_pos_w[:, 1], crossing_y)
    target_z = torch.where(ball_close, ball_pos_w[:, 2], env_z + 0.10)
    crossing_point = torch.stack([goal_x_w, target_y, target_z], dim=-1)  # (N, 3)

    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]     # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                       # (N,)
    arange = torch.arange(env.num_envs, device=env.device)
    foot_pos_active = foot_pos_w[arange, foot_idx]
    dist = torch.norm(foot_pos_active - crossing_point, dim=-1)            # (N,)

    success_flag = getattr(
        env, "_sb_flag", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    return (success_flag.float() + 1.0) * (dist < strict_th).float()


def single_foot_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    window: int = 3,
) -> torch.Tensor:
    """One-time bonus when stopball and softstop both fire within `window` steps AND correct foot made contact.

    Same-step coincidence was too strict: MuJoCo soft contacts accumulate over
    several steps, so stopball (delta_vx > threshold) and softstop (ball_x_vel > threshold)
    can fire 1-3 steps apart on a clean single-foot save. A two-touch exploit still
    fires them many steps apart (first foot slows, second foot taps back).

    Tracks the episode step when each flag first rose, fires when both are set,
    within window, and the correct foot was in contact at softstop moment.
    Two int32 tensors, negligible GPU cost.
    """
    sb_flag = getattr(env, "_sb_flag", None)
    ss_flag = getattr(env, "_softstop_flag", None)

    if sb_flag is None or ss_flag is None:
        return torch.zeros(env.num_envs, device=env.device)

    _UNSET = -999
    if not hasattr(env, "_sfs_flag"):
        env._sfs_flag    = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._sfs_sb_step = torch.full((env.num_envs,), _UNSET, dtype=torch.int32, device=env.device)
        env._sfs_ss_step = torch.full((env.num_envs,), _UNSET, dtype=torch.int32, device=env.device)
        env._sfs_sb_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._sfs_ss_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._sfs_flag[just_reset]    = False
    env._sfs_sb_step[just_reset] = _UNSET
    env._sfs_ss_step[just_reset] = _UNSET
    env._sfs_sb_prev[just_reset] = False
    env._sfs_ss_prev[just_reset] = False

    step = env.episode_length_buf.to(torch.int32)

    sb_just_fired = sb_flag & ~env._sfs_sb_prev
    ss_just_fired = ss_flag & ~env._sfs_ss_prev
    env._sfs_sb_step[sb_just_fired] = step[sb_just_fired]
    env._sfs_ss_step[ss_just_fired] = step[ss_just_fired]
    env._sfs_sb_prev[:] = sb_flag
    env._sfs_ss_prev[:] = ss_flag

    both_recorded = (env._sfs_sb_step >= 0) & (env._sfs_ss_step >= 0)
    within_window = (env._sfs_sb_step - env._sfs_ss_step).abs() <= window
    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = both_recorded & within_window & correct_foot & ~env._sfs_flag
    env._sfs_flag |= fired
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


def foot_ang_vel_xy(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Sum of squared foot roll+pitch angular velocity across both feet.

    Penalises active foot rotation in XY (heel-first landings, ankle twist during
    dives). Orthogonal to feetorientation: that measures static tilt; this measures
    how fast the foot is rotating into or out of a tilt.
    """
    robot: Entity = env.scene[asset_cfg.name]
    foot_ang_vel_w = robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]  # [B, 2, 3]
    return torch.sum(foot_ang_vel_w[:, :, :2] ** 2, dim=-1).sum(dim=-1)


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

    FIX 2026-07-22: unregistered from goalkeeper_env_cfg.py (along with the
    shank_height termination) -- user found shank height false-positives on
    legitimate deep lunges (research via play.py's new per-episode min-height
    + terminated_by reporting, docs/BugFixes.md same date). Replaced by
    penalize_baseheight + the retuned base_height termination below.
    Function kept (not deleted) in case shank-based gating is revisited.
    """
    robot: Entity = env.scene[asset_cfg.name]
    shank_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]                                # (N,)
    shank_z_local = shank_pos_w[:, :, 2] - floor_z[:, None]             # (N, 2)
    violation = torch.clamp(min_height - shank_z_local, min=0.0)        # (N, 2)
    return violation.sum(dim=-1)


def penalize_baseheight(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.59,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Penalise root (Trunk) height dropping below min_height above floor.

    FIX 2026-07-22: replaces penalize_kneeheight as the graded warning half
    of this project's fall-prevention pair (paired with the base_height
    termination, goalkeeper_env_cfg.py, both user-tuned from real play
    session data via the min_base/min_Lsh/min_Rsh + terminated_by reporting
    added to play.py this same date). Shank height fired on legitimate deep
    lunges (a real athletic motion for a foot-only save) because a lunge
    inherently brings one knee close to the floor even while balanced;
    root/torso height stays much higher during an intentional lunge and
    only drops when the robot is actually falling, matching the user's own
    read of the play-session data. 0.59 / termination's 0.57 mirrors the
    existing kneeheight(0.295)/shank_height(0.27) pair's ~2.5cm graded-
    warning-before-hard-cutoff gap.

    Mirrors penalize_kneeheight's shape exactly (clamp(min_height - z, 0)),
    just on the single root body instead of two shank bodies.
    """
    robot: Entity = env.scene[asset_cfg.name]
    base_z_w = robot.data.root_link_pos_w[:, 2]           # (N,)
    floor_z = env.scene.env_origins[:, 2]                  # (N,)
    base_z_local = base_z_w - floor_z                       # (N,)
    return torch.clamp(min_height - base_z_local, min=0.0)  # (N,)


def dof_vel_limits(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    soft_factor: float = 0.9,
) -> torch.Tensor:
    """Penalise joint velocities above per-joint T1 motor velocity limits.

    Mirrors G1 _reward_dof_vel_limits exactly (legged_robot.py:1550-1553):
        sum(clip(|dof_vel| - dof_vel_limits * soft_dof_vel_limit, min=0))
    with soft_dof_vel_limit=0.9 (g1_29_config.py:349) -- both the per-joint
    sourcing and the 0.9 safety margin, and the LINEAR (not squared) excess.

    FIX 2026-07-20 (reward audit item 7): previously a single flat 10 rad/s
    cap for every joint regardless of actual per-joint limit, with a squared
    excess kernel -- both diverged from G1. The flat 10 rad/s cap silently
    never fired for T1's slower joints (e.g. arms at ~9.3 rad/s, waist/hip-
    roll/hip-yaw at ~7.3 rad/s -- see _T1_VEL_LIMIT_MAP): a joint already
    past ITS real limit could sit under the flat 10 and be judged compliant
    by this reward. Real per-joint limits sourced from t1_constants.py motor
    specs (ElectricActuator.velocity_limit); the T1 MJCF itself defines no
    joint velocity limit at all (only a position `range`), so there is no
    XML value to use instead -- see docs/BugFixes.md for the full check.
    """
    robot: Entity = env.scene[asset_cfg.name]
    vel = robot.data.joint_vel[:, asset_cfg.joint_ids]                  # (N, J)

    if not hasattr(env, "_t1_vel_limits"):
        all_names = robot.joint_names
        # Fallback for any unmapped joint (should not occur -- all 21 headless
        # T1 DOF are covered by _T1_VEL_LIMIT_MAP): use the slowest-rated
        # group (waist/hip-roll/hip-yaw) as a conservative default.
        vel_all = torch.full(
            (len(all_names),), WAIST_HIP_ROLL_YAW_ACTUATOR.velocity_limit, device=env.device
        )
        for i, name in enumerate(all_names):
            if name in _T1_VEL_LIMIT_MAP:
                vel_all[i] = _T1_VEL_LIMIT_MAP[name]
        env._t1_vel_limits = vel_all

    vel_limits = env._t1_vel_limits[asset_cfg.joint_ids] * soft_factor   # (J,)
    excess = torch.clamp(vel.abs() - vel_limits, min=0.0)                # (N, J)
    return excess.sum(dim=-1)


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


def postlegdofpos(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _LEG_JOINT_CFG_RECOVERY,
) -> torch.Tensor:
    """Reward leg joints returning to default (standing) pose after ball is behind.

    FIX 2026-07-22: user reported the policy settles into a "weird pose" after
    a save rather than a sensible standing pose. G1's post-save pose recovery
    (postupperdofpos/postwaistdofpos, both above) only covers elbow+wrist
    ("upper_body_joint_indices", legged_robot.py:1315) and waist
    (legged_robot.py:1306) -- G1's own leg_joint_indices (all 12 hip/knee/
    ankle joints, legged_robot.py:1276-1284) have NO post-save reward at
    all. That's not a gap in this port -- it's correct for G1's own task:
    G1 catches with its HANDS, so the catching limb (arms, via elbow+wrist)
    needs an explicit pull back to neutral after use, while the legs are
    mostly just standing/braced throughout and don't drift far from default
    on their own.

    This project's task structurally inverts which limb does the catching:
    SGK saves with its FEET, so the legs are the limb that gets thrown into
    an extreme, off-default configuration during a dive/step/reach -- and
    until this fix, nothing pulled them back afterward, matching the
    reported "weird pose" symptom exactly (only arms/waist had a return-to-
    default incentive; legs, wherever the save left them, had none). This
    mirrors postupperdofpos's exact shape (exp(-1*sum_sq_err), same gentle
    exponent, same _ball_is_behind gate) applied to the role-equivalent
    limb group for this task rather than G1's literal joint list --
    consistent with the top-level CLAUDE.md instruction to check whether a
    G1 decision "was designed for hands and needs adaptation for feet."

    exp(-1 * sum_sq_err) x behind -- bounded [0, 1], reward peaks at default pose.
    """
    behind = _ball_is_behind(env, ball_name)
    robot: Entity = env.scene[asset_cfg.name]
    delta = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    err = torch.sum(torch.square(delta), dim=-1)
    return torch.exp(-1.0 * err) * behind.float()



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


def penalize_wrong_foot_ball_contact(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """Binary penalty when the ball touches the WRONG (non-assigned) foot.

    FIX 2026-07-22: user reported (from reviewing checkpoints around
    iteration ~10000 of green_gradpen10_2026-07-21) that the policy learned
    to catch/stop the ball with its non-leading foot -- i.e. whichever foot
    happens to be nearby, not the task-assigned one (_get_correct_foot_idx,
    based on which side the ball is crossing on) -- and still farm cleanstop/
    softstop/single_foot_save because their existing "correct foot" gate
    uses "feet_contact", a GROUND-contact sensor (see softstop's FIX
    2026-07-15 docstring): the assigned foot merely standing on the ground
    at the same instant satisfies that gate even when a DIFFERENT foot is
    the one actually touching the ball. That gate is intentionally left as
    is (a prior attempt to tighten it via a ball-specific sensor made those
    rewards stop firing almost entirely -- same docstring), so this adds an
    independent, direct penalty instead: uses "ball_contact", the dedicated
    foot-vs-ball ContactSensorCfg (primary=foot geoms, secondary=ball_geom,
    goalkeeper_env_cfg.py) that only fires on genuine foot-ball contact, and
    fires whenever the foot NOT matching _get_correct_foot_idx is in that
    contact set -- independent of whatever the existing correct-foot gates
    believe. Not gated by _ball_is_behind: any wrong-foot ball touch, before
    or after a save, is the behavior being discouraged.

    Geom layout in ball_contact sensor (sorted by name, matches feet_contact):
        0-3: left_foot1-4 -> left foot, 4-7: right_foot1-4 -> right foot.
    Weight: -30.0.
    """
    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) 0=left, 1=right
    sensor: ContactSensor = env.scene["ball_contact"]
    found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
    left_touch = (found[:, :4] > 0).any(dim=-1)   # (B,)
    right_touch = (found[:, 4:] > 0).any(dim=-1)  # (B,)
    foot_touch = torch.stack([left_touch, right_touch], dim=-1)  # (B, 2)
    wrong_foot_idx = 1 - foot_idx
    wrong_foot_touch = foot_touch[torch.arange(env.num_envs, device=env.device), wrong_foot_idx]
    return wrong_foot_touch.float()


def airborne_at_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
) -> torch.Tensor:
    """One-time bonus when softstop fires with correct foot airborne.

    softstop fires unconditionally on ball velocity; this fires on top of it
    as a quality bonus when the robot had the correct foot in the air at the save moment,
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
    left_in_contact = (fc[:, :4] > 0).any(dim=-1)
    right_in_contact = (fc[:, 4:] > 0).any(dim=-1)
    foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)

    foot_idx = _get_correct_foot_idx(env, ball_name)
    correct_foot_in_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]
    correct_foot_airborne = ~correct_foot_in_contact

    just_fired = softstop_fired & ~env._aas_ss_prev
    env._aas_ss_prev[:] = softstop_fired

    correct_foot_contact = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = just_fired & correct_foot_airborne & correct_foot_contact & ~env._aas_flag
    env._aas_flag |= fired
    return fired.float()


def inner_face_orientation_save(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    alignment_threshold: float = 0.7,
) -> torch.Tensor:
    """One-time bonus when softstop fires and the assigned foot is turned sideways
    (block posture), rotated toward the side it's actually saving on.

    Checks that the foot's long axis (toe-heel, local X) is parallel to world Y at the save
    moment — meaning the foot is rotated 90° so its broad inner face is presented to the ball
    rather than the toe or heel.

    quat_apply(foot_quat, [1,0,0]) · [0,1,0] * expected_sign > threshold
    threshold=0.7 ≈ cos(45°): foot long axis within 45° of world Y, in the correct direction.

    FIX 2026-07-10: was `.abs()` on the alignment ("either +Y or -Y direction is a
    valid sideways save"). That reasoning was wrong -- a left-side save needs the
    foot rotated toward +Y, a right-side save needs -Y; they are mirror images, not
    interchangeable. abs() rewarded either direction equally, so the policy had no
    pressure to learn the correct, side-matched rotation -- it could satisfy this
    reward on every save by consistently rotating one way, even when that's
    backwards for half the saves. Confirmed live: a trained checkpoint's right foot
    (right-side saves) learned a genuine ~38 degree rotation toward -Y, while the
    left foot (left-side saves) stayed near-neutral (~3 degrees) -- it never
    discovered the mirror-image +Y rotation, because nothing demanded it
    specifically. See docs/BugFixes.md.

    FIX 2026-07-10 (assigned foot, not geometrically-closest foot): was
    `left_closer = dist[:,0] <= dist[:,1]` -- the GEOMETRICALLY closest foot to the
    ball's live position at the save moment, not the FIXED, task-assigned foot
    (_get_correct_foot_idx, based on which side the ball crossed on). If the
    assigned foot overshoots past the ball, the stationary lagging (wrong) foot can
    become geometrically closer at that instant, so this orientation check would
    silently evaluate the WRONG foot's quaternion -- while `correct_foot` below
    (from _softstop_correct_foot, itself built from _get_correct_foot_idx) still
    correctly requires the ASSIGNED foot to have made contact. That mismatch let
    the assigned foot's genuinely-correct orientation go unrewarded (checking the
    lagging foot instead, usually not correctly oriented -- false negative) or, in
    the reverse case, let a coincidentally-oriented lagging foot fire the reward
    despite the real save foot doing nothing to earn it (false positive). Now uses
    _get_correct_foot_idx directly, consistent with `correct_foot`'s own gate.
    expected_sign is +1 for the left foot (should rotate toward +Y), -1 for the
    right foot (should rotate toward -Y).

    This is orthogonal to feetorientation (which constrains roll/pitch — foot flatness).
    Together they enforce: flat foot AND turned sideways, in the correct direction = correct
    block posture.
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

    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)

    # FIX 2026-07-10: was the geometrically-closest foot to the ball's live
    # position (dist[:,0] <= dist[:,1]) -- see docstring. Now the fixed,
    # task-assigned foot, matching correct_foot's own gate below.
    foot_idx = _get_correct_foot_idx(env, ball_name)                       # (N,) 0=left, 1=right
    is_left = foot_idx == 0

    # Foot long axis (toe direction) in local frame = (1, 0, 0).
    # Rotate into world frame for each foot.
    foot_long_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    left_long_w  = quat_apply(foot_quat_w[:, 0, :], foot_long_local)      # (N, 3)
    right_long_w = quat_apply(foot_quat_w[:, 1, :], foot_long_local)      # (N, 3)

    # Pick the assigned foot's long axis.
    foot_long_w = torch.where(is_left[:, None], left_long_w, right_long_w)  # (N, 3)

    # "Lengthy side parallel to Y, in the correct direction" = long axis aligned
    # with +Y for the left foot, -Y for the right foot.
    world_y = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, -1)
    y_alignment_signed = (foot_long_w * world_y).sum(dim=-1)               # (N,)
    expected_sign = torch.where(is_left, 1.0, -1.0)                        # (N,)
    oriented_correctly = (y_alignment_signed * expected_sign) > alignment_threshold

    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = just_fired & oriented_correctly & correct_foot & ~env._ifos_flag
    env._ifos_flag |= fired
    return fired.float()


def foot_inner_face_continuous(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Continuous reward for rotating the assigned foot's inner face toward the ball,
    in the correct (side-matched) direction.

    Active every step while the ball is live (same ~behind gate as footreach).
    Uses the same assigned foot as footreach (_get_correct_foot_idx: left=0 for +Y
    balls, right=1 for -Y balls).

    Metric: foot_long_axis_w · robot_y_w * expected_sign — negative/zero when foot
    points forward or the wrong way, up to 1 when fully turned sideways in the
    correct direction (+Y for the left foot, -Y for the right foot). Uses robot
    local Y (not world Y) so a dive yaw doesn't degrade the signal.

    FIX 2026-07-10: was `.abs()` — same bug as inner_face_orientation_save (see
    that function's docstring for the full analysis and live evidence). This is
    the DENSE, per-step version of the same reward, so it was actually the
    dominant source of the wrong-direction-tolerant gradient (far more total
    signal than the one-shot bonus) -- fixing this one specifically should matter
    most. Deliberately left unclamped (can go negative for a wrong-direction
    rotation, not just zero) -- a mild, informative penalty gives clearer gradient
    toward the correct mirror-image direction than merely withholding reward,
    which would look identical to "never rotated at all" from the policy's
    perspective.
    """
    robot: Entity = env.scene[asset_cfg.name]

    foot_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)

    foot_idx = _get_correct_foot_idx(env, ball_name)  # (N,) — 0=left, 1=right

    foot_long_local = torch.zeros(env.num_envs, 3, device=env.device)
    foot_long_local[:, 0] = 1.0                                              # local X = toe dir
    left_long_w  = quat_apply(foot_quat_w[:, 0, :], foot_long_local)        # (N, 3)
    right_long_w = quat_apply(foot_quat_w[:, 1, :], foot_long_local)        # (N, 3)
    foot_long_w  = torch.where((foot_idx == 0)[:, None], left_long_w, right_long_w)  # (N, 3)

    robot_y_local = torch.zeros(env.num_envs, 3, device=env.device)
    robot_y_local[:, 1] = 1.0
    robot_y_w = quat_apply(robot.data.root_link_quat_w, robot_y_local)      # (N, 3)

    expected_sign = torch.where(foot_idx == 0, 1.0, -1.0)                   # (N,) left=+Y, right=-Y
    alignment = (foot_long_w * robot_y_w).sum(dim=-1) * expected_sign       # (N,) in [-1, 1]

    behind = _ball_is_behind(env, ball_name)
    return alignment * (~behind).float()


def cleanstop(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    speed_threshold: float = 0.25,
) -> torch.Tensor:
    """One-time reward when ball nearly stops after deflection AND correct foot made contact.

    Fires once per episode when softstop has already triggered AND the ball's total
    speed drops below speed_threshold (0.25 m/s) AND the correct foot was in contact
    at softstop moment. Rewards a clean foot-trap style save over a hard uncontrolled deflection.
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

    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = softstop_fired & (ball_speed < speed_threshold) & correct_foot & ~env._cleanstop_flag
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

    Suppressed only for the correct foot when the ball is within 0.5 m of it: the
    contact sensor cannot distinguish ground from ball, so without this gate
    the policy learns to plant feet passively rather than sweep into the ball,
    preventing the force needed for stopball (delta_vx > 1 m/s).
    The trailing foot is NOT gated — it must remain non-sliding throughout.

    Sliding is measured as the worst-case horizontal speed across two contact
    points per foot: the center-bottom and the toe tip.  Rigid-body kinematics:
        v_contact = v_body + ω × r_contact
    Center-bottom: r = (0, 0, -0.03) in foot-local (capsule z=-0.01 + radius 0.02).
    Toe-tip: r = (0.105, 0.01, -0.03) in foot-local (foot4 forward tip + radius).
    When the trailing foot tilts toe-down, the center-bottom sample underestimates
    the actual toe sliding velocity; the toe-tip sample catches it.
    Weight: +3.0.
    """
    _CONTACT_Z_LOCAL = 0.03  # capsule z (-0.01) + radius (0.02)
    # foot4 extends to x=+0.105, y=+0.01 in foot-local frame
    _TOE_X_LOCAL = 0.105
    _TOE_Y_LOCAL = 0.01

    sensor: ContactSensor = env.scene["feet_contact"]
    found = sensor.data.found  # [B, 8]
    left_in_contact  = (found[:, :4] > 0).any(dim=-1)
    right_in_contact = (found[:, 4:] > 0).any(dim=-1)
    in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1).float()  # [B, 2]

    robot: Entity = env.scene[asset_cfg.name]
    foot_vel_w   = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]   # [B, 2, 3]
    foot_omega_w = robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]   # [B, 2, 3]
    foot_quat_w  = robot.data.body_link_quat_w[:, asset_cfg.body_ids, :]      # [B, 2, 4]

    def _contact_speed(r_local_offset: torch.Tensor) -> torch.Tensor:
        """Horizontal speed at a contact point given foot-local offset [B,2,3]."""
        r_world = quat_apply(
            foot_quat_w.reshape(-1, 4),
            r_local_offset.reshape(-1, 3),
        ).reshape(env.num_envs, 2, 3)
        v = foot_vel_w + torch.cross(foot_omega_w, r_world, dim=-1)
        return torch.norm(v[:, :, :2], dim=-1)  # [B, 2]

    # Center-bottom contact point
    r_center = torch.zeros_like(foot_vel_w)
    r_center[:, :, 2] = -_CONTACT_Z_LOCAL
    speed_center = _contact_speed(r_center)

    # Toe-tip contact point (left foot4 / right foot3 end, worst case for tilted foot).
    # Left foot (idx 0): foot4 tip at y=+0.01; right foot (idx 1): foot3 tip at y=-0.01.
    r_toe = torch.zeros_like(foot_vel_w)
    r_toe[:, :, 0] = _TOE_X_LOCAL
    r_toe[:, 0, 1] = _TOE_Y_LOCAL    # left foot
    r_toe[:, 1, 1] = -_TOE_Y_LOCAL   # right foot (mirrored)
    r_toe[:, :, 2] = -_CONTACT_Z_LOCAL
    speed_toe = _contact_speed(r_toe)

    # Take the worst-case (highest) sliding speed across the two sample points
    foot_speed = torch.maximum(speed_center, speed_toe)                      # [B, 2]

    ball: Entity = env.scene[ball_name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]      # [B, 2, 3]
    ball_pos_w = ball.data.root_link_pos_w                                   # [B, 3]

    # Suppress slippage penalty for the correct foot only when ball is near it.
    # The trailing foot keeps its full penalty so it cannot drag freely.
    foot_dist = torch.norm(foot_pos_w - ball_pos_w[:, None, :], dim=-1)    # [B, 2]
    foot_idx = _get_correct_foot_idx(env, ball_name)                        # (N,)
    arange = torch.arange(env.num_envs, device=env.device)
    correct_foot_near_ball = foot_dist[arange, foot_idx] < 0.5             # (N,)
    suppress = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    suppress[arange, foot_idx] = correct_foot_near_ball

    contactvel_per_foot = foot_speed * in_contact                           # [B, 2]
    contactvel_per_foot = contactvel_per_foot.masked_fill(suppress, 0.0)
    contactvel = contactvel_per_foot.sum(dim=-1)
    # FIX 2026-07-20 (reward audit item 6): was -50.0, 5x steeper than G1's
    # exp(-10*contactvel) (legged_robot.py:1473) -- the docstring above already
    # claimed to mirror G1's -10 kernel, so this was a straightforward
    # coefficient bug (doc/code mismatch), not a documented divergence. No
    # comment or commit anywhere justified -50 -- it has been -50.0 since this
    # function's first commit (c2b8b69), i.e. a porting error, not later drift.
    # Reverted to G1's literal -10.
    return torch.exp(-10.0 * contactvel)
