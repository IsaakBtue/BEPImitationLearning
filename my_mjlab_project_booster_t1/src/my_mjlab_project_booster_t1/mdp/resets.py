"""Reset functions for Booster T1 goalkeeper."""
from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import sample_uniform


def reset_ball_autonomous(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str = "ball") -> None:
    """Reset ball with random trajectory for autonomous play."""
    ball: Entity = env.scene[ball_name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    g = 9.81
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    # Ball spawn rotated 90° clockwise: comes from +Y direction instead of +X
    y_start = sample_uniform(3.0, 5.0, (n,), device=env.device)
    x_end = sample_uniform(-1.2, -0.2, (n,), device=env.device)
    z_end = sample_uniform(0.1, 1.6, (n,), device=env.device)
    x_start = sample_uniform(-1.8, 1.8, (n,), device=env.device)
    z_start = sample_uniform(0.3, 1.8, (n,), device=env.device)

    t_flight = sample_uniform(0.5, 1.0, (n,), device=env.device)

    dx = x_end - x_start
    dy = -y_start - 0.3
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
