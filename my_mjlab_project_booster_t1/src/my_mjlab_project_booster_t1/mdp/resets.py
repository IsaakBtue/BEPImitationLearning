"""Reset functions and curriculum helpers for Booster T1 goalkeeper."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


def _shoot_ball(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str) -> None:
    """Shared ball-launch logic: spawn at +Y, aim toward goal line (Y≈0)."""
    ball: Entity = env.scene[ball_name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    g = 9.81
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    # Ball comes from +Y direction (rotated 90° vs original G1 +X setup).
    y_start = sample_uniform(3.0, 5.0, (n,), device=env.device)
    x_end = sample_uniform(-1.2, 1.2, (n,), device=env.device)
    z_end = sample_uniform(0.1, 1.6, (n,), device=env.device)
    x_start = sample_uniform(-1.8, 1.8, (n,), device=env.device)
    z_start = sample_uniform(0.3, 1.8, (n,), device=env.device)

    t_flight = sample_uniform(0.5, 1.0, (n,), device=env.device)

    dx = x_end - x_start
    dy = -y_start - 0.3        # target Y ≈ -0.3 (just behind goal line)
    dz = z_end - z_start

    vx = dx / t_flight
    vy = dy / t_flight
    vz = (dz + 0.5 * g * t_flight**2) / t_flight

    ball_pos_w = torch.stack([
        origins[:, 0] + x_start,
        origins[:, 1] + y_start,
        origins[:, 2] + z_start,
    ], dim=1)
    ball_quat_w = torch.zeros((n, 4), device=env.device)
    ball_quat_w[:, 0] = 1.0
    ball_pose = torch.cat([ball_pos_w, ball_quat_w], dim=-1)

    ball_vel = torch.stack([vx, vy, vz], dim=1)
    ball_ang_vel = torch.zeros((n, 3), device=env.device)
    ball_velocity = torch.cat([ball_vel, ball_ang_vel], dim=-1)

    ball.write_root_link_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(ball_velocity, env_ids=env_ids)


class ball_difficulty_curriculum:
    """Curriculum that ramps up ball shot difficulty over training.

    Mirrors the upstream assign_ball_states curriculum in Humanoid-Goalkeeper:
        command_ranges[:, 0] = clip(command_ranges[:, 0] - 0.3 * curriculumupdate, bound)
    The upstream expands the shot range over time; we do the same via a staged
    difficulty float (0.0 → 1.0) stored on env._ball_difficulty.

    Stages (matching upstream's stage1/stage2 thresholds):
        step=0:           difficulty=0.0 (easy centre shots)
        step=stage1_step: difficulty=0.5 (moderate range)
        step=stage2_step: difficulty=1.0 (full range)

    _reset_ball in commands.py reads env._ball_difficulty to interpolate ranges.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: ManagerBasedRlEnv) -> None:
        self._stages = cfg.params["stages"]
        if not hasattr(env, "_ball_difficulty"):
            env._ball_difficulty = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict:
        current_difficulty = 0.0
        for stage in self._stages:
            if env.common_step_counter >= stage["step"]:
                current_difficulty = stage["difficulty"]
        env._ball_difficulty = current_difficulty
        return {"ball_difficulty": torch.tensor(current_difficulty)}


def reset_ball_training(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str = "ball") -> None:
    """Reset ball with random trajectory for training.

    Ball spawns at positive Y (3–5 m in front of robot) and is aimed toward the
    goal line (Y≈0) at a random lateral X and height Z target. Without this,
    the default reset_scene_to_default places the ball at the env origin with zero
    velocity, so stopball/eereach never receive meaningful gradients.
    """
    _shoot_ball(env, env_ids, ball_name)


def reset_ball_autonomous(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str = "ball") -> None:
    """Reset ball with random trajectory for autonomous play."""
    _shoot_ball(env, env_ids, ball_name)


def sharpforce_termination(
    env: ManagerBasedRlEnv,
    max_contact_force: float = 1500.0,
) -> torch.Tensor:
    """Terminate when mean foot contact force exceeds threshold.

    Mirrors upstream Humanoid-Goalkeeper sharpforce_buf termination:
        terminate = mean(norm(contact_forces[:, feet, :])) > 1.5 * max_contact_force
    where max_contact_force = 1000 N, giving a termination threshold of 1500 N.

    Upstream averages over 2 foot bodies (contact_feet_indices has 2 entries).
    Port geom layout (4 geoms, sorted by name):
        index 0: left_foot_1  ─┐ left foot
        index 1: left_foot_2  ─┘
        index 2: right_foot_1 ─┐ right foot
        index 3: right_foot_2 ─┘
    Fix: per-foot max over geoms, then mean over feet — matches upstream 2-body mean.

    Returns [B] bool tensor: True → terminate this environment.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)          # [B, 4]
    left_max  = force_per_geom[:, :2].max(dim=-1).values     # max of left_foot_1, left_foot_2
    right_max = force_per_geom[:, 2:].max(dim=-1).values     # max of right_foot_1, right_foot_2
    mean_force = (left_max + right_max) / 2.0                # [B]
    return mean_force > max_contact_force
