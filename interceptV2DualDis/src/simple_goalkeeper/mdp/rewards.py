"""Goalkeeper reward terms for SimpleGoalKeeper (Phase 1 — feet only)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .regions import REGION_NAMES as _REGION_NAMES

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Region-conditioned task only (env._region_id set by regions.assign_static_regions):
# which REGION_NAMES entries are a "_far" region. See _get_reach_target_y.
_REGION_IS_FAR = torch.tensor([name.endswith("_far") for name in _REGION_NAMES], dtype=torch.bool)

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


def _get_reach_target_y(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    wide_threshold: float = 0.5,
    landing_radius: float = 0.08,
    landing_speed_threshold: float = 0.5,
) -> torch.Tensor:
    """Two-stage reach target for wide crossings (ported from SimpleGoalKeeper
    2026-07-03/07-04/07-05, see SimpleGoalKeeper/CLAUDE.md divergence table).

    For crossings where |crossing_y - start_y| > wide_threshold, the target is
    the MIDPOINT between the robot's stance and the true crossing point until
    the assigned foot (_get_correct_foot_idx) physically lands there (airborne
    then in ground contact via the feet_contact sensor, within landing_radius
    of the midpoint), then switches to the full crossing point. Narrow
    crossings always target the full point. There is no elapsed-time fallback
    -- a crossing that never lands stays targeting the midpoint for the whole
    episode (hard gate, per SGK's 2026-07-04 finding that a soft/timeout gate
    let the policy skip the waypoint and still collect full credit).

    FIX 2026-07-07 (superseded below): landing_radius tightened 0.05 -> 0.02.
    User-reported symptom: blue_ball_landed fired even when the assigned
    foot's actual ground contact was near the GREEN target, with the policy
    still doing one continuous big step/leap rather than a genuine paced
    double-step. Blue and green are collinear (same X, only Y differs) and
    always >= wide_threshold/2 apart for a wide crossing, so a foot planting
    AT green cannot itself be within 0.05m of blue -- the actual mechanism is
    a foot sweeping through (not stopping at) the blue Y-coordinate mid-swing
    during one continuous stride toward green, catching a momentary/glancing
    ground-contact reading (feet_contact sensor noise, a low-clearance
    shuffle step, foot-scuff during swing phase) while passing near blue's
    exact coordinate, before continuing on to plant at green.

    FIX 2026-07-07 (radical tightening of stopball/softstop, then this):
    after gating stopball/softstop on genuine (non-RSI) landings too, the
    user reported the play script showing the robot sweep straight past blue
    to the intercept point, correctly earning zero footreach/stopball/
    softstop for it -- the gate was finally airtight, but a fresh policy
    still wasn't discovering the genuine two-stage plant. Root cause: a
    *pure position* landing check has no way to distinguish a deliberate
    stop from a fast pass-through, and footreach's own vel_sigma term
    (up to 10x, rewards speed toward whichever point is currently the reach
    target -- see footreach()) actively rewards NOT decelerating while
    approaching blue. At landing_radius=0.02 this combination made a
    genuine, deliberate plant almost geometrically impossible to land
    exactly (2 cm, single 0.02s physics step) while doing nothing to
    disincentivize sweeping through fast -- shrinking the radius alone
    fights the symptom (accidental grazes) without fixing the cause
    (nothing requires the foot to actually stop). Added a velocity gate
    (landing_speed_threshold) instead: the assigned foot's horizontal speed
    must be below this threshold at the moment of contact, in addition to
    being within landing_radius. This directly rules out fast sweeps
    regardless of radius, so the radius itself was loosened back up
    (0.02 -> 0.08) to make a genuine, deliberate plant actually achievable
    -- precision now comes from requiring the foot to be slow, not from
    requiring pixel-perfect position. Matches the follow-up flagged in the
    prior fix entry ("a velocity-based check ... to distinguish a genuine
    plant from a foot merely passing through at speed"). See docs/BugFixes.md.

    Ported because AMP's 2-frame transition discriminator cannot judge global
    trajectory shape or step count -- a smooth fast leap to the far target
    satisfies it as well as a genuine multi-step reference clip, so nothing in
    the existing reward pushed toward a paced multi-step approach. See
    SimpleGoalKeeper's own investigation (footreach/vel_sigma rewards raw
    speed toward a single fixed target, not gait).

    Root XY is pinned to env.scene.env_origins at reset in this project too
    (reset_base event), so "robot start Y" can be read live with no new cache.
    """
    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y

    rel = getattr(env, "_rsi_cross_y", None)
    lateral = rel if rel is not None else (full_y - start_y)
    wide = lateral.abs() > wide_threshold

    # FIX 2026-07-07: for the region-conditioned task, a region's own far/near
    # label (env._region_id) is authoritative over the threshold check above.
    # _rsi_cross_y is the ball's position AT the goal line (x=0); _REGION_Y_END_RANGE
    # is defined on y_end, the aim point 0.3m BEHIND the goal line. Since
    # _rsi_cross_y = y_start + (y_end-y_start)*f with f = x_start/(x_start+0.3) < 1,
    # the crossing point is always smaller in magnitude than y_end -- silently
    # misclassifying ~17-20% of legitimately-far episodes (those with |y_end| near
    # the region's own 0.5m inner edge) as narrow. No G1 equivalent (regions are a
    # new mechanism, see regions.py); see docs/BugFixes.md.
    region_id = getattr(env, "_region_id", None)
    if region_id is not None:
        wide = wide | _REGION_IS_FAR.to(env.device)[region_id]

    # Cached so footreach/foot_proximity/stopball/softstop can gate on
    # "wide AND not yet landed" without recomputing the crossing geometry.
    env._blue_wide = wide

    half_y = start_y + (full_y - start_y) / 2.0

    n = env.num_envs
    if not hasattr(env, "_blue_was_airborne"):
        env._blue_was_airborne = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_landed = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_airborne_at_reset = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._blue_settle_count = torch.zeros(n, dtype=torch.int64, device=env.device)
    just_reset = env.episode_length_buf <= 1
    env._blue_was_airborne[just_reset] = False
    env._blue_landed[just_reset] = False
    env._blue_airborne_at_reset[just_reset] = False
    env._blue_settle_count[just_reset] = 0

    try:
        robot: Entity = env.scene[asset_cfg.name]
        feet_contact: ContactSensor = env.scene["feet_contact"]
    except KeyError:
        robot = None
        feet_contact = None

    if robot is not None and feet_contact is not None:
        foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]   # (N, 2, 3)
        foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
        foot_idx = _get_correct_foot_idx(env, ball_name)                    # (N,)
        arange_n = torch.arange(n, device=env.device)
        assigned_foot_pos = foot_pos_w[arange_n, foot_idx]                  # (N, 3)
        assigned_foot_vel = foot_vel_w[arange_n, foot_idx]                  # (N, 3)

        found = feet_contact.data.found                                    # (N, 8)
        left_in_contact = (found[:, :4] > 0).any(dim=-1)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)
        foot_in_contact = torch.where(foot_idx == 0, left_in_contact, right_in_contact)  # (N,)

        currently_airborne = ~foot_in_contact
        first_time_airborne = currently_airborne & ~env._blue_was_airborne
        near_reset = env.episode_length_buf <= 2
        env._blue_airborne_at_reset |= first_time_airborne & near_reset
        env._blue_was_airborne |= currently_airborne

        goal_x_w = env.scene.env_origins[:, 0]
        target_point_xy = torch.stack([goal_x_w, half_y], dim=-1)  # (N, 2)
        # Horizontal-only distance -- foot_in_contact already guarantees ground
        # height whenever this fires, so a Z term would be redundant at best,
        # actively unsatisfiable at worst (see SGK's 2026-07-05 fix).
        dist_to_blue = torch.norm(assigned_foot_pos[:, :2] - target_point_xy, dim=-1)
        # FIX 2026-07-07: horizontal speed gate -- a foot merely sweeping
        # through blue's coordinate at speed (mid-swing toward green) must
        # not count as landing there. Only a foot that has actually
        # decelerated (genuine plant) satisfies this alongside the position
        # check. See docstring above.
        foot_speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

        # FIX 2026-07-08: settle window, not an instantaneous check. A live
        # diagnostic (docs/BugFixes.md, 2026-07-07) showed the pure
        # instantaneous version of this check (speed < threshold on the exact
        # same step as the position check) drove genuine landings to 0% --
        # a real footstrike still carries residual swing velocity for a few
        # steps after first ground contact, decaying toward zero as weight
        # transfers and friction/contact damping take hold, not instantly.
        # Require the foot to stay in contact AND within landing_radius for
        # _BLUE_SETTLE_STEPS consecutive steps (ruling out a fast sweep-
        # through, which can't hold position that long) before checking
        # velocity, giving a genuine plant time to actually decelerate.
        candidate = wide & env._blue_was_airborne & foot_in_contact & (dist_to_blue < landing_radius)
        env._blue_settle_count = torch.where(
            candidate, env._blue_settle_count + 1, torch.zeros_like(env._blue_settle_count)
        )
        _BLUE_SETTLE_STEPS = 3
        newly_landed = (
            (env._blue_settle_count >= _BLUE_SETTLE_STEPS)
            & (foot_speed < landing_speed_threshold)
            & ~env._blue_landed
        )
        env._blue_landed |= newly_landed

    phase1_active = wide & ~env._blue_landed
    return torch.where(phase1_active, half_y, full_y)


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

    # True frozen crossing Y (final arrival point) — used only for foot-side selection
    # below, which must not flip mid-episode based on the two-stage reach target.
    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)
    # Reach target: full crossing point, or the two-stage midpoint-then-full schedule
    # on wide crossings (ported from SimpleGoalKeeper). See _get_reach_target_y.
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,)

    # Phase 1: pre-position laterally when ball is far (> 1.5 m in front).
    lateral_error = reach_target_y - robot_y_w                          # positive → target right
    asidegoal = lateral_error.clamp(-1.0, 1.0)
    asidegoal = torch.where(asidegoal.abs() < 0.3, torch.zeros_like(asidegoal), asidegoal)
    phase1_rew = 1.0 - asidegoal.abs()                                  # 1=aligned, 0=1 m off

    # Phase 2 crossing point: frozen (two-stage) when far, live ball when close (mirrors
    # G1 end_target update). G1 post_physics_step lines 203-206: end_target switches to
    # ball_states[:, :3] when balllocal < 0.5 — robot converges on the actual ball in the
    # final 0.5 m. The X coordinate stays at goal_x_w (stayonline constraint keeps the
    # foot at the goal line).
    goal_x_w  = env.scene.env_origins[:, 0]       # (N,)
    floor_z_w = env.scene.env_origins[:, 2]       # (N,)
    live_y = ball_pos_w[:, 1]
    live_z = ball_pos_w[:, 2]
    # (N,) bool — mirrors G1 balllocal < 0.5, but on wide crossings this switch is
    # ALSO gated on having genuinely landed at the blue midpoint: otherwise the
    # policy can skip the landing gate entirely and still converge on the live
    # ball once it closes to 0.5 m. See _get_reach_target_y.
    ball_close = (ball_x_local < 0.5) & (~env._blue_wide | env._blue_landed)

    # Switch from frozen (two-stage) target to live ball Y/Z when ball is within 0.5 m.
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
    vel_sigma = 1.0 + 3.0 * vel_toward.clamp(0.0, 3.0)                # 1–10× (matches G1 eereach)

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
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w                              # (N, 3)
    ball_x_local = ball_pos_w[:, 0] - env.scene.env_origins[:, 0]     # (N,)

    # True frozen crossing Y (final arrival point) — used only for foot-side
    # selection below, which must not flip mid-episode with the two-stage target.
    crossing_y = _get_ball_crossing_y(env, ball_name)                  # (N,)
    # Reach target: full crossing point, or the two-stage midpoint-then-full
    # schedule on wide crossings (ported from SimpleGoalKeeper). See _get_reach_target_y.
    reach_target_y = _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # (N,)
    goal_x_w = env.scene.env_origins[:, 0]
    env_z    = env.scene.env_origins[:, 2]

    # Live ball switch: when ball is within 0.5 m of goal line, track live position.
    # Mirrors G1 post_physics_step: end_target = ball_states[:, :3] when balllocal < 0.5.
    # On wide crossings this is ALSO gated on having genuinely landed at the blue
    # midpoint (mirrors footreach) — see _get_reach_target_y.
    ball_close = (ball_x_local < 0.5) & (~env._blue_wide | env._blue_landed)
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


def blue_ball_landed(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """One-shot bonus when the assigned foot lands at the blue (midpoint) target.

    Ported from SimpleGoalKeeper. Fires the first step the two-stage gate
    (env._blue_landed) becomes true -- the assigned foot was airborne at some
    point after reset, then came into ground contact within landing_radius of
    the midpoint target. Narrow crossings never fire this (env._blue_wide is
    always false for them). See _get_reach_target_y.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_landed is fresh this step

    if not hasattr(env, "_blue_landed_bonus_flag"):
        env._blue_landed_bonus_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    just_reset = env.episode_length_buf <= 1
    env._blue_landed_bonus_flag[just_reset] = False

    fired = env._blue_landed & ~env._blue_landed_bonus_flag
    env._blue_landed_bonus_flag |= fired
    return fired.float()


def blue_overshoot_penalty(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    landing_radius: float = 0.08,
    max_overshoot: float = 0.5,
) -> torch.Tensor:
    """FEAT 2026-07-08: penalize the assigned foot for advancing past the blue
    midpoint, toward green, on a wide crossing that hasn't been genuinely
    landed yet -- makes "walk straight past blue" actively costly instead of
    merely unrewarded.

    Rationale: before this term, a wide episode where the policy ignores the
    blue waypoint entirely earned exactly the same reward (zero, from the
    landing-gated stopball/softstop/blue_ball_landed) as one where it tried
    and failed. With no cost differential, PPO has no gradient pushing it
    away from ignoring wide episodes altogether -- and since narrow episodes
    (~half the region-sampled distribution) already pay out fully with
    minimal movement, "specialize in narrow, ignore wide" is a stable local
    optimum. This term breaks that: overshooting past blue while unlanded is
    now worse than stopping short of it or reaching it, on every step, not
    just a missed one-time bonus.

    Deliberately does NOT touch footreach's vel_sigma (which already
    amplifies reward for velocity toward whichever point is the CURRENT
    reach target -- reach_target_y flips from blue to green exactly on
    env._blue_landed, see _get_reach_target_y -- so it already only rewards
    speed toward blue pre-landing, never toward green early). The combination
    is intentional: vel_sigma still rewards approaching blue fast, while this
    term makes carrying that speed past blue without stopping costly --
    together they should shape the same accelerate-then-decelerate-to-a-plant
    profile actually observed in the LeftDoubleStep/RightDoubleStep AMP demo
    clips (see docs/BugFixes.md "do i need a better dataset" investigation),
    rather than requiring a separate deceleration reward term.

    landing_radius matches _get_reach_target_y's own arrival radius (0.08) as
    a deadband -- being within it counts as "at blue", not overshooting.
    max_overshoot bounds the per-step penalty for episodes where the policy
    is still far past blue (e.g. early in training, mid-flight toward green)
    so a single step can't dominate the return.

    Zero on narrow crossings (env._blue_wide always false) and once genuinely
    landed (env._blue_landed true) -- see _get_reach_target_y for both.
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_wide/_blue_landed fresh

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    half_y = start_y + (full_y - start_y) / 2.0
    direction = torch.sign(full_y - start_y)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                   # (N,)
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_y = foot_pos_w[arange_n, foot_idx, 1]                # (N,)

    signed_progress = direction * (assigned_foot_y - half_y)
    overshoot = torch.clamp(signed_progress - landing_radius, min=0.0, max=max_overshoot)

    phase1_active = env._blue_wide & ~env._blue_landed
    return overshoot * phase1_active.float()


def blue_stick_landing(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
    dist_sigma: float = 15.0,
    speed_sigma: float = 3.0,
) -> torch.Tensor:
    """FEAT 2026-07-08: dense reward for the assigned foot being simultaneously
    CLOSE to and SLOW near the blue midpoint on a wide, unlanded crossing --
    exp(-dist_sigma * dist_to_blue) * exp(-speed_sigma * foot_speed), peaking
    exactly at "close AND stopped" (a genuine plant), zero elsewhere.

    Rationale: two escalation-triggered health checks (iteration 2000 and
    3750 of run amp_g1_parity_2026-07-08b, docs/BugFixes.md) measured genuine
    blue landings at exactly 0.0% each time, over 2600+ wide episodes per
    check, despite the settle-window landing-check fix and full AMP-parity
    pass both already being in effect. blue_overshoot_penalty's per-episode
    value stayed flat and strongly negative across the whole run (~-0.65 to
    -0.86, iterations 0-5500) instead of shrinking -- the policy wasn't
    learning to avoid it. Diagnosis: nothing in the reward stack gives dense
    credit for the EXACT joint condition (close + slow) the settle-window
    check requires -- footreach/foot_proximity reward proximity to blue, and
    vel_sigma rewards speed toward blue, but nothing rewards decelerating
    once close. The only payoff for actually achieving "close and slow" was
    the sparse, conjunctive settle-window event itself (contact + radius for
    3 consecutive steps + speed threshold) -- a policy has to discover that
    combination by chance before any gradient reinforces it, and evidently
    hasn't in 5500+ iterations. This term gives a smooth reward basin that
    peaks at the target joint condition, so ordinary gradient ascent can find
    it incrementally (get closer -> more reward; get slower while close ->
    more reward) instead of requiring the exact conjunction to be stumbled
    into blind. Mirrors this project's own existing `cleanstop` (rewards low
    BALL speed near a target after softstop) -- same "reward stillness near a
    point" pattern, applied to the FOOT approaching blue instead of the ball
    after a save.

    dist_sigma=15 makes the distance factor decay to ~0.5 by ~5cm, ~0.1 by
    ~15cm -- tight enough to require actually being near blue, not just
    anywhere on the approach path. speed_sigma=3 makes the speed factor decay
    to ~0.5 by ~0.23 m/s, ~0.1 by ~0.77 m/s -- rewards genuine deceleration
    without demanding literal zero velocity. Neither tuned against real
    footstrike dynamics; flagged for revisit if this doesn't move the
    genuine-landing rate.

    Zero on narrow crossings and once genuinely landed -- mirrors
    blue_overshoot_penalty's phase1_active gate exactly (_get_reach_target_y).
    """
    _get_reach_target_y(env, ball_name, asset_cfg=asset_cfg)  # ensure _blue_wide/_blue_landed fresh

    full_y = _get_ball_crossing_y(env, ball_name)                 # (N,) world Y
    start_y = env.scene.env_origins[:, 1]                         # (N,) world Y
    half_y = start_y + (full_y - start_y) / 2.0
    goal_x_w = env.scene.env_origins[:, 0]
    target_xy = torch.stack([goal_x_w, half_y], dim=-1)           # (N, 2)

    robot: Entity = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_vel_w = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    foot_idx = _get_correct_foot_idx(env, ball_name)                   # (N,)
    arange_n = torch.arange(env.num_envs, device=env.device)
    assigned_foot_pos = foot_pos_w[arange_n, foot_idx]                 # (N, 3)
    assigned_foot_vel = foot_vel_w[arange_n, foot_idx]                 # (N, 3)

    dist = torch.norm(assigned_foot_pos[:, :2] - target_xy, dim=-1)
    speed = torch.norm(assigned_foot_vel[:, :2], dim=-1)

    phase1_active = env._blue_wide & ~env._blue_landed
    return torch.exp(-dist_sigma * dist) * torch.exp(-speed_sigma * speed) * phase1_active.float()


def stopball(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    delta_vel_threshold: float = 1.0,
) -> torch.Tensor:
    """One-time reward when ball X velocity increases by >= delta_vel_threshold (m/s).

    Ball approaches with negative X velocity; foot contact reverses or decelerates it.
    Fires exactly once per episode when delta_vx > threshold, providing the primary
    training signal for a successful save. Mirrors Imitationlearningbooster stopball.

    Landing gate (ported from SimpleGoalKeeper 2026-07-05): on wide crossings
    (env._blue_wide), this can only fire once the assigned foot has genuinely
    landed at the blue midpoint (env._blue_landed) -- otherwise the policy can
    skip the two-stage waypoint entirely with one continuous reach and still
    collect the full save reward. Narrow crossings are unaffected.

    FIX 2026-07-07 (radical tightening): landing_ok used to accept ANY
    _blue_landed, including RSI-assisted "free" landings where the reset
    donor pose already had the assigned foot airborne near the blue midpoint
    at episode start (env._blue_airborne_at_reset). A live diagnostic against
    model_4500.pt of this run measured 30.8% of wide episodes satisfying the
    old gate, but only 1.4% (of all wide episodes) were genuine -- 29.4% were
    pure RSI credit, meaning the gate was being bypassed for free most of the
    time and never forcing the policy to learn the two-stage step. Now
    requires env._blue_landed & ~env._blue_airborne_at_reset (same definition
    as metrics.blue_landed_genuine) so only a landing the policy actually
    caused this episode unlocks stopball/softstop on wide crossings. See
    docs/BugFixes.md.
    """
    _get_reach_target_y(env, ball_name)  # ensure _blue_wide/_blue_landed are fresh this step

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
    landing_ok = ~env._blue_wide | (env._blue_landed & ~env._blue_airborne_at_reset)
    fired = (delta_vx > delta_vel_threshold) & in_front & landing_ok & ~env._sb_flag
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

    Landing gate (ported from SimpleGoalKeeper 2026-07-05): same wide-crossing
    landing gate as stopball -- see that function's docstring, including the
    2026-07-07 tightening to genuine-only landings.
    """
    _get_reach_target_y(env, ball_name)  # ensure _blue_wide/_blue_landed are fresh this step

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
    landing_ok = ~env._blue_wide | (env._blue_landed & ~env._blue_airborne_at_reset)
    fired = (ball_x_vel > velocity_threshold) & in_front & landing_ok & ~env._softstop_flag

    # Track correct foot contact at softstop moment.
    if fired.any():
        foot_idx = _get_correct_foot_idx(env, ball_name)
        sensor: ContactSensor = env.scene["feet_contact"]
        found = sensor.data.found  # [B, 8]: 0-3=left, 4-7=right
        left_in_contact = (found[:, :4] > 0).any(dim=-1)   # (B,)
        right_in_contact = (found[:, 4:] > 0).any(dim=-1)  # (B,)
        foot_in_contact = torch.stack([left_in_contact, right_in_contact], dim=-1)  # (B, 2)

        correct_foot_contact = foot_in_contact[torch.arange(env.num_envs, device=env.device), foot_idx]
        env._softstop_correct_foot[fired] = correct_foot_contact[fired]

    env._softstop_flag |= fired
    return fired.float()


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
    """One-time bonus when softstop fires and the closest foot is turned sideways (block posture).

    Checks that the foot's long axis (toe-heel, local X) is parallel to world Y at the save
    moment — meaning the foot is rotated 90° so its broad inner face is presented to the ball
    rather than the toe or heel.

    |quat_apply(foot_quat, [1,0,0]) · [0,1,0]| > threshold
    threshold=0.7 ≈ cos(45°): foot long axis within 45° of world Y.

    This is orthogonal to feetorientation (which constrains roll/pitch — foot flatness).
    Together they enforce: flat foot AND turned sideways = correct block posture.
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

    # Foot long axis (toe direction) in local frame = (1, 0, 0).
    # Rotate into world frame for each foot.
    foot_long_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    left_long_w  = quat_apply(foot_quat_w[:, 0, :], foot_long_local)      # (N, 3)
    right_long_w = quat_apply(foot_quat_w[:, 1, :], foot_long_local)      # (N, 3)

    # Pick the closer foot's long axis.
    foot_long_w = torch.where(left_closer[:, None], left_long_w, right_long_w)  # (N, 3)

    # "Lengthy side parallel to Y" = long axis aligned with world Y.
    # abs() because either +Y or -Y direction is a valid sideways save.
    world_y = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(env.num_envs, -1)
    y_alignment = (foot_long_w * world_y).sum(dim=-1).abs()                # (N,)
    oriented_correctly = y_alignment > alignment_threshold

    correct_foot = getattr(env, "_softstop_correct_foot", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    fired = just_fired & oriented_correctly & correct_foot & ~env._ifos_flag
    env._ifos_flag |= fired
    return fired.float()


def foot_inner_face_continuous(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_FEET_CFG,
) -> torch.Tensor:
    """Continuous reward for rotating the assigned foot's inner face toward the ball.

    Active every step while the ball is live (same ~behind gate as footreach).
    Uses the same assigned foot as footreach (_get_correct_foot_idx: left=0 for +Y
    balls, right=1 for -Y balls).

    Metric: |foot_long_axis_w · robot_y_w| — 0 when foot points forward, 1 when
    foot is fully turned sideways (inner face presented in the robot's lateral direction).
    Uses robot local Y (not world Y) so a dive yaw doesn't degrade the signal.
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

    alignment = (foot_long_w * robot_y_w).sum(dim=-1).abs()                 # (N,) in [0, 1]

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
    return torch.exp(-50.0 * contactvel)
