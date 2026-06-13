"""Event functions and curriculum helpers for SimpleGoalKeeper."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


# Easy (difficulty=0.0) spawn ranges: short distance, centred, slow, low.
# Hard (difficulty=1.0) ranges are passed as params to reset_ball_local_frame.
_EASY_DIST    = (1.5, 2.5)
_EASY_LATERAL = (-0.15, 0.15)
_EASY_Z_START = (0.1, 0.35)
_EASY_Z_END   = (0.05, 0.2)
_EASY_SPEED   = (2.0, 3.5)


def _lerp_range(
    easy: tuple[float, float],
    hard: tuple[float, float],
    d: float,
) -> tuple[float, float]:
    """Linearly interpolate a (lo, hi) range between easy and hard endpoints."""
    return (
        easy[0] + d * (hard[0] - easy[0]),
        easy[1] + d * (hard[1] - easy[1]),
    )


def _yaw_only_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Return a quaternion containing ONLY the yaw component of q_wxyz.

    Isolating yaw ensures quat_apply only rotates XY — pitch/roll on the robot
    body does not tilt the ball spawn position or velocity.
    """
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    out = torch.zeros_like(q_wxyz)
    out[:, 0] = torch.cos(half)   # w
    out[:, 3] = torch.sin(half)   # z
    return out


def reset_ball_local_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (2.0, 4.0),
    lateral_range: tuple[float, float] = (-0.5, 0.5),
    spawn_height_range: tuple[float, float] = (0.2, 0.7),
    arrive_height_range: tuple[float, float] = (0.1, 0.4),
    speed_range: tuple[float, float] = (3.0, 7.0),
) -> None:
    """Spawn ball in front of robot, always approaching from +X direction.

    Ball spawns in front of the robot (local +X) and moves toward it (local -X).
    Ball position and direction are world-orientation-independent (robot-local frame,
    using yaw-only quaternion for XY rotation). Heights are floor-absolute so the
    ball arc is unaffected by robot trunk height.

    When env._ball_difficulty is set (by ball_difficulty_curriculum), ranges are
    linearly interpolated from easy (difficulty=0.0) to the configured hard params
    (difficulty=1.0). Without the curriculum, difficulty defaults to 1.0 so the
    full configured ranges are used from the start.

    Physics: vz is computed so the ball arrives at `arrive_height` when it reaches
    the robot, following a parabolic arc matching Imitationlearningbooster's _shoot_ball.

    Args:
        dist_range:          horizontal distance from robot to spawn (m)
        lateral_range:       lateral offset in robot +Y (m); ±0.5 covers both posts
        spawn_height_range:  ball height above floor at spawn (m)
        arrive_height_range: target ball height above floor at robot position (m)
        speed_range:         horizontal approach speed (m/s)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    # Curriculum difficulty lerps from easy ranges (d=0) to configured params (d=1).
    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    dist_r    = _lerp_range(_EASY_DIST,    dist_range,          d)
    lateral_r = _lerp_range(_EASY_LATERAL, lateral_range,       d)
    z_start_r = _lerp_range(_EASY_Z_START, spawn_height_range,  d)
    z_end_r   = _lerp_range(_EASY_Z_END,   arrive_height_range, d)
    speed_r   = _lerp_range(_EASY_SPEED,   speed_range,         d)

    g = 9.81
    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene["robot"]

    robot_pos_w  = robot.data.root_link_pos_w[env_ids]      # (n, 3)
    robot_quat_w = robot.data.root_link_quat_w[env_ids]     # (n, 4) wxyz
    yaw_q        = _yaw_only_quat(robot_quat_w)             # pure-yaw quat

    floor_z = env.scene.env_origins[env_ids, 2]             # (n,) floor per env

    dist    = sample_uniform(*dist_r,    (n,), env.device)
    lateral = sample_uniform(*lateral_r, (n,), env.device)
    z_start = floor_z + sample_uniform(*z_start_r, (n,), env.device)
    z_end   = floor_z + sample_uniform(*z_end_r,   (n,), env.device)
    speed_h = sample_uniform(*speed_r,  (n,), env.device)

    # Ball spawn: dist forward (+X) + lateral in robot yaw frame.
    # Ball approaches robot from +X direction (in front).
    local_xy = torch.stack([dist, lateral, torch.zeros_like(dist)], dim=-1)
    world_xy = quat_apply(yaw_q, local_xy)
    ball_pos = torch.empty((n, 3), device=env.device)
    ball_pos[:, 0] = robot_pos_w[:, 0] + world_xy[:, 0]
    ball_pos[:, 1] = robot_pos_w[:, 1] + world_xy[:, 1]
    ball_pos[:, 2] = z_start

    # Horizontal velocity: toward robot along -local_X.
    local_vel_h = torch.stack(
        [-speed_h, torch.zeros_like(speed_h), torch.zeros_like(speed_h)], dim=-1
    )
    world_vel_h = quat_apply(yaw_q, local_vel_h)

    # Gravity-compensating vz: z_end = z_start + vz*t - 0.5*g*t^2
    t_flight = dist / speed_h
    vz = ((z_end - z_start) + 0.5 * g * t_flight ** 2) / t_flight

    ball_vel = torch.empty((n, 3), device=env.device)
    ball_vel[:, 0] = world_vel_h[:, 0]
    ball_vel[:, 1] = world_vel_h[:, 1]
    ball_vel[:, 2] = vz

    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_link_velocity_to_sim(
        torch.cat([ball_vel, torch.zeros((n, 3), device=env.device)], dim=-1),
        env_ids=env_ids,
    )

    _init_visibility_state(env, env_ids)


def _init_visibility_state(env: "ManagerBasedRlEnv", env_ids: torch.Tensor) -> None:
    """Reset per-env visibility counters at episode start."""
    n = env.num_envs

    if not hasattr(env, "_catchstep"):
        env._catchstep = torch.zeros(n, dtype=torch.long, device=env.device)
    env._catchstep[env_ids] = 50

    if not hasattr(env, "_startstep"):
        env._startstep = torch.zeros(n, dtype=torch.long, device=env.device)
    env._startstep[env_ids] = 50 - torch.randint(3, 11, (len(env_ids),), device=env.device)

    if not hasattr(env, "_vanish_step"):
        env._vanish_step = torch.zeros(n, dtype=torch.long, device=env.device)
    env._vanish_step[env_ids] = torch.randint(0, 30, (len(env_ids),), device=env.device)

    if not hasattr(env, "_ball_visible_step"):
        env._ball_visible_step = torch.zeros(n, dtype=torch.long, device=env.device)
    env._ball_visible_step[env_ids] = 0

    if not hasattr(env, "_ball_obs_last_x"):
        env._ball_obs_last_x = torch.zeros(n, device=env.device)
    env._ball_obs_last_x[env_ids] = 0.0


def tick_catchstep(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
) -> None:
    """Decrement catchstep warmup counter each step (interval event, interval=(0,0))."""
    if hasattr(env, "_catchstep"):
        env._catchstep = (env._catchstep - 1).clamp(min=0)


class ball_difficulty_curriculum:
    """Curriculum that ramps ball shot difficulty from easy (centred, slow) to full range.

    Mirrors Imitationlearningbooster's ball_difficulty_curriculum.
    Sets env._ball_difficulty (0.0 → 1.0); reset_ball_local_frame reads this
    to lerp spawn ranges from easy to the configured hard values.

    Stages (matching upstream iteration thresholds with num_steps_per_env=24):
        step=0:            difficulty=0.0  (easy centre shots)
        step=stage1_step:  difficulty=0.5  (moderate range)
        step=stage2_step:  difficulty=1.0  (full configured range)
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        self._stages = cfg.params["stages"]
        if not hasattr(env, "_ball_difficulty"):
            env._ball_difficulty = 0.0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict:
        current = 0.0
        for stage in self._stages:
            if env.common_step_counter >= stage["step"]:
                current = stage["difficulty"]
        env._ball_difficulty = current
        return {"ball_difficulty": torch.tensor(current)}


def ball_exit_termination(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    behind_threshold: float = -0.5,
) -> torch.Tensor:
    """Terminate when ball has clearly passed the goal line.

    Previously also terminated on deflection (sb_flag & ball_x_vel > 0.5), which
    ended episodes immediately after any body contact — before feet could reach the
    ball. Now only the goal-line crossing terminates, giving feet time to contact
    the ball even after a torso deflection.
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return ball_x_local < behind_threshold
