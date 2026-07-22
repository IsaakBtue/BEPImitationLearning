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


# G1 flying-mask front edge: ball visible only while x_body > 0.05 m
# (Humanoid-Goalkeeper legged_robot.py:401, end_target_local[:,0] > 0.05).
_BEHIND_TORSO_X = 0.05


def ball_pos_xy_b(
    env: "ManagerBasedRlEnv",
    ball_name: str = "ball",
    always_visible: bool = False,
    hide_behind_torso: bool = False,
    hide_after_steps: int | None = None,
    noise_scale: float = 0.0,
) -> torch.Tensor:
    """Ball XY position in robot body frame (no Z). Shape (N, 2).

    Matches BoosterT1mjlab kick task observation space for deployment compatibility.

    Post-save release gate v2 — a G1-faithful subset of the flying mask
    (legged_robot.py:401), ported from SimpleGoalKeeper (ce69f36) after its
    v1 latched-flag gate destabilized a training run:

    - hide_behind_torso: zero once the ball is behind the torso front edge
      (x_body < 0.05, G1's end_target_local[:,0] > 0.05 visibility bound).
    - hide_after_steps: zero once the episode step exceeds the window
      (G1's catchstep > 0, sized to the flight-time distribution).
    - noise_scale: uniform noise applied BEFORE the mask, matching G1's
      noise-then-mask ordering (legged_robot.py:425-426) — a hidden ball is
      exact zeros, never noise around zero. Replaces manager noise on this
      term (mjlab applies manager noise after the term returns, which turns
      the hidden ball into a phantom ball near the robot's feet).

    Neither hide condition can fire while the ball is still approaching, so
    visibility during the approach and save is preserved by construction.
    G1's warmup blackout, random vanish, and approach/cone checks are
    intentionally NOT ported (user constraint: no pre-save blindness).
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    ball_pos_b_val = quat_apply(
        quat_inv(robot.data.root_link_quat_w),
        ball.data.root_link_pos_w - robot.data.root_link_pos_w,
    )
    visible = torch.ones(env.num_envs, dtype=torch.bool, device=ball_pos_b_val.device)
    if not always_visible:
        visible &= _compute_ball_visibility(env, ball_name)
    if hide_behind_torso:
        visible &= ball_pos_b_val[:, 0] >= _BEHIND_TORSO_X
    if hide_after_steps is not None:
        visible &= env.episode_length_buf <= hide_after_steps
    out = ball_pos_b_val[:, :2]
    if noise_scale > 0.0:
        out = out + (2.0 * torch.rand_like(out) - 1.0) * noise_scale
    return out * visible.float().unsqueeze(-1)


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

    FIX 2026-07-15: now respects asset_cfg.joint_ids (was always the full
    21-DOF vector regardless of asset_cfg) so callers can restrict this to a
    joint subset -- e.g. excluding arms from the AMP discriminator entirely,
    since arms have no task-grounding reward and AMP-only imitation of them
    was suspected of producing a "fake gradient" (weird, ungrounded arm
    motion). See goalkeeper_multidisc_amp_cfg.py's "amp" group override.
    """
    robot: Entity = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids].clone()
    default_joint_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None and softstop_fired.any():
        joint_pos[softstop_fired] = default_joint_pos[softstop_fired]
    return joint_pos


def joint_vel_abs(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute joint velocities for AMP discriminator — gated by softstop.

    Returns zero velocities after softstop fires. Mirrors joint_pos_abs gating.

    FIX 2026-07-15: now respects asset_cfg.joint_ids -- see joint_pos_abs.
    """
    robot: Entity = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids].clone()
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None and softstop_fired.any():
        joint_vel[softstop_fired] = 0.0
    return joint_vel


_ARM_JOINT_NAMES: tuple[str, ...] = (
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
)


def joint_pos_abs_arms_masked_by_region(
    env: "ManagerBasedRlEnv",
    far_region_ids: tuple[int, ...] = (1, 3),
) -> torch.Tensor:
    """Like joint_pos_abs (full 21-DOF, softstop-gated), but additionally
    freezes arm joints to their default pose for envs whose env._region_id
    is in far_region_ids -- keeping the tensor a uniform 21-dim shape (the
    multi-disc architecture routes ONE shared amp_obs tensor to whichever
    per-region discriminator matches each env, so all regions must produce
    the same shape) while making arm columns uninformative (constant, so
    the discriminator can't learn anything from them) specifically for far
    regions. Near regions keep real, live arm motion in their AMP input.

    FIX 2026-07-16: replaces a prior attempt (2026-07-15) that excluded arms
    from ALL 4 regions' AMP input uniformly (via joint slicing, not
    masking) -- that made arm swinging WORSE, not better, per user report
    (with zero AMP constraint of any kind on arms, they had even less
    incentive to move sensibly). This targets only the far/double-step
    regions the original "weird arm swinging" complaints were about,
    leaving near regions unaffected.
    """
    robot: Entity = env.scene["robot"]
    joint_pos = robot.data.joint_pos.clone()
    default_joint_pos = robot.data.default_joint_pos
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None and softstop_fired.any():
        joint_pos[softstop_fired] = default_joint_pos[softstop_fired]

    if not hasattr(env, "_arm_joint_col_mask"):
        arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
        col_mask = torch.zeros(joint_pos.shape[-1], dtype=torch.bool, device=env.device)
        col_mask[torch.tensor(arm_ids, dtype=torch.long, device=env.device)] = True
        env._arm_joint_col_mask = col_mask

    region_id = getattr(env, "_region_id", None)
    if region_id is not None:
        is_far = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for r in far_region_ids:
            is_far |= region_id == r
        full_mask = is_far.unsqueeze(-1) & env._arm_joint_col_mask.unsqueeze(0)
        joint_pos = torch.where(full_mask, default_joint_pos, joint_pos)

    return joint_pos


def joint_vel_abs_arms_masked_by_region(
    env: "ManagerBasedRlEnv",
    far_region_ids: tuple[int, ...] = (1, 3),
) -> torch.Tensor:
    """Like joint_vel_abs (full 21-DOF, softstop-gated), but additionally
    zeroes arm joint velocity columns for envs whose env._region_id is in
    far_region_ids -- same rationale and column-masking mechanism as
    joint_pos_abs_arms_masked_by_region (uniform 21-dim shape across all
    4 regions' shared amp_obs tensor; arm columns made uninformative only
    for far/double-step regions). Zero is the natural "frozen" velocity
    (unlike joint_pos, there's no "default" velocity to substitute).

    FIX 2026-07-22: added alongside joint_pos to give the AMP discriminator
    joint-velocity information -- see docs/BugFixes.md, "AMP fix: joint_vel
    added to multi-disc AMP observation". The motion-dataset side already
    supported this generically (MotionDatasetCfg.freeze_joint_names zeroes
    whichever amp_obs_terms are configured, including joint_vel) with no
    code changes needed there -- only this policy-side term was missing.
    """
    robot: Entity = env.scene["robot"]
    joint_vel = robot.data.joint_vel.clone()
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None and softstop_fired.any():
        joint_vel[softstop_fired] = 0.0

    if not hasattr(env, "_arm_joint_col_mask"):
        arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES, preserve_order=True)
        col_mask = torch.zeros(joint_vel.shape[-1], dtype=torch.bool, device=env.device)
        col_mask[torch.tensor(arm_ids, dtype=torch.long, device=env.device)] = True
        env._arm_joint_col_mask = col_mask

    region_id = getattr(env, "_region_id", None)
    if region_id is not None:
        is_far = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for r in far_region_ids:
            is_far |= region_id == r
        full_mask = is_far.unsqueeze(-1) & env._arm_joint_col_mask.unsqueeze(0)
        joint_vel = torch.where(full_mask, torch.zeros_like(joint_vel), joint_vel)

    return joint_vel
