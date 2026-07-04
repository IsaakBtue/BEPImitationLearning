"""Static 4-way region partitioning and region-conditioned ball spawn.

Region assignment is a permanent, one-time split of the parallel env batch
into 4 contiguous blocks — mirrors Humanoid-Goalkeeper's `end_regions`
mechanism (legged_gym/legged_gym/envs/base/legged_robot.py:916-924), which
splits `num_envs` into 6 fixed blocks at startup and never reassigns them.
Here it's 4: left_near, left_far, right_near, right_far.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .events import reset_ball_rolling

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

REGION_NAMES: tuple[str, ...] = ("left_near", "left_far", "right_near", "right_far")

# Per-region ball-spawn y_start_range / y_end_range. Side sign matches the
# existing convention: positive Y crossing = left, negative Y crossing =
# right (see rewards.py:_get_correct_foot_idx). |cross_y| < 0.5 = near,
# >= 0.5 = far, matching the design spec's threshold.
_REGION_Y_START_RANGE: dict[int, tuple[float, float]] = {
    0: (0.0, 0.3),     # left_near
    1: (0.0, 0.3),     # left_far
    2: (-0.3, 0.0),    # right_near
    3: (-0.3, 0.0),    # right_far
}
_REGION_Y_END_RANGE: dict[int, tuple[float, float]] = {
    0: (0.15, 0.5),    # left_near: crosses on the left, under 0.5 m
    1: (0.5, 0.9),     # left_far: crosses on the left, at/above 0.5 m
    2: (-0.5, -0.15),  # right_near
    3: (-0.9, -0.5),   # right_far
}


def assign_static_regions(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: split env.num_envs into 4 fixed contiguous region blocks.

    Sets env._region_id (int64, shape (num_envs,), values 0-3). Called once
    with mode="startup" — env_ids is ignored (region assignment always
    covers the full batch and is never reassigned on reset).
    """
    n = env.num_envs
    quarter = n // 4
    remainder = n - quarter * 4
    counts = [quarter, quarter, quarter, quarter + remainder]
    region_id = torch.cat([
        torch.full((counts[r],), r, dtype=torch.int64, device=env.device)
        for r in range(4)
    ])
    env._region_id = region_id


def reset_ball_rolling_by_region(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (1.5, 3.5),
    t_flight_range: tuple[float, float] = (0.7, 1.1),
    spawn_z: float = 0.12,
) -> None:
    """Region-conditioned ball spawn: calls reset_ball_rolling once per region
    subset of env_ids, using that region's y_start_range/y_end_range so the
    spawned ball actually produces that region's category of shot.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    region_id = env._region_id[env_ids]
    for r in range(4):
        mask = region_id == r
        if not mask.any():
            continue
        reset_ball_rolling(
            env,
            env_ids[mask],
            ball_name,
            dist_range=dist_range,
            y_start_range=_REGION_Y_START_RANGE[r],
            y_end_range=_REGION_Y_END_RANGE[r],
            t_flight_range=t_flight_range,
            spawn_z=spawn_z,
        )


def region_id_gt(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Ground-truth region id for the region_estimator's cross-entropy target.
    Shape (N, 1), float32 (cross-entropy target is cast to long by the caller).

    The observation manager probes each term's output shape during env
    construction (ObservationManager.__init__ -> _prepare_terms), which runs
    *before* the ``assign_static_regions`` startup event populates
    ``env._region_id``. Return a correctly-shaped zeros tensor in that window;
    every real ``observation_manager.compute()`` happens after startup events,
    so training/eval always sees the true region ids.
    """
    region_id = getattr(env, "_region_id", None)
    if region_id is None:
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    return region_id.float().unsqueeze(-1)


def ball_state_gt(env: "ManagerBasedRlEnv", ball_name: str = "ball") -> torch.Tensor:
    """Ground-truth ball state for the ball_estimator's MSE target.

    Shape (N, 4): (pos_x, pos_y, vel_x, vel_y) in robot body frame. Same
    frame convention as observations.py:ball_pos_b/ball_vel_b (always
    visible here — this is privileged critic-only info, not gated).
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[ball_name]
    quat_i = quat_inv(robot.data.root_link_quat_w)
    pos_b = quat_apply(quat_i, ball.data.root_link_pos_w - robot.data.root_link_pos_w)
    vel_b = quat_apply(quat_i, ball.data.root_link_lin_vel_w)
    return torch.cat([pos_b[:, :2], vel_b[:, :2]], dim=-1)
