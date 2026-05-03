"""Reset functions for autonomous goalkeeper (no motion tracking required)."""
from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import sample_uniform


def reset_ball_autonomous(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str = "ball") -> None:
    """Reset ball with random trajectory independent of motion type.

    Ball spawns 3-5m away with random y/z position and computed velocity
    to reach a random end target. This is suitable for autonomous play
    where the policy learns to react to any incoming ball.

    Args:
        env: The environment instance
        env_ids: Tensor of environment indices to reset
        ball_name: Name of the ball entity in the scene
    """
    ball: Entity = env.scene[ball_name]

    # Handle startup mode (env_ids=None means all envs)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    g = 9.81
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]  # (n, 3)

    # Ball start: x=3-5m ahead of goal, y/z broadly randomized
    x_start = sample_uniform(3.0, 5.0, (n,), device=env.device)

    # Random end target (fully autonomous, not tied to motion types)
    y_end = sample_uniform(-1.2, 1.2, (n,), device=env.device)
    z_end = sample_uniform(0.1, 1.6, (n,), device=env.device)
    y_start = sample_uniform(-1.8, 1.8, (n,), device=env.device)
    z_start = sample_uniform(0.3, 1.8, (n,), device=env.device)

    t_flight = sample_uniform(0.5, 1.0, (n,), device=env.device)

    # Compute velocity to reach end target from start in t_flight seconds
    dx = -x_start - 0.3  # end x ≈ -0.3 (slightly behind goal line)
    dy = y_end - y_start
    dz = z_end - z_start

    vx = dx / t_flight
    vy = dy / t_flight
    vz = (dz + 0.5 * g * t_flight**2) / t_flight

    # Positions in world frame
    ball_pos_w = torch.stack([
        origins[:, 0] + x_start,
        origins[:, 1] + y_start,
        origins[:, 2] + z_start,
    ], dim=1)
    ball_quat_w = torch.zeros((n, 4), device=env.device)
    ball_quat_w[:, 0] = 1.0  # identity wxyz
    ball_pose = torch.cat([ball_pos_w, ball_quat_w], dim=-1)  # (n, 7)

    ball_vel = torch.stack([vx, vy, vz], dim=1)  # (n, 3)
    ball_ang_vel = torch.zeros((n, 3), device=env.device)
    ball_velocity = torch.cat([ball_vel, ball_ang_vel], dim=-1)  # (n, 6)

    ball.write_root_link_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(ball_velocity, env_ids=env_ids)
