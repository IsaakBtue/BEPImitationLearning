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
    """Shared ball-launch logic for autonomous play.

    Uses the same +X approach axis as _reset_ball in commands.py so that ball
    observations and the visibility check (ball_x_local > 0.05) work correctly.
    Ball spawns at x_start ∈ [3.0, 4.5] m and aims toward X ≈ -0.3 (goal line),
    with random lateral Y and height Z targets covering the full save range.
    """
    ball: Entity = env.scene[ball_name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    g = 9.81
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    # Ball approaches from +X (same axis as training _reset_ball).
    x_start = sample_uniform(3.0, 4.5, (n,), device=env.device)
    y_start = sample_uniform(-0.8, 0.8, (n,), device=env.device)
    z_start = sample_uniform(0.5, 1.4, (n,), device=env.device)
    # End targets cover the full bilateral range so both hands are exercised in play.
    y_end = sample_uniform(-0.65, 0.65, (n,), device=env.device)
    z_end = sample_uniform(0.2, 1.4, (n,), device=env.device)

    t_flight = sample_uniform(0.5, 1.0, (n,), device=env.device)

    dx = -x_start - 0.3       # target X ≈ -0.3 (just behind goal line)
    dy = y_end - y_start
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


class task_reward_curriculum:
    """Scale eereach / hand_proximity_strict / stopball weights with _curriculumupdate.

    Mirrors G1 legged_robot.py lines 359-364 (called inside compute_reward() every step):
        reward_scales["eereach"]  = eereach_init  * (1 + 0.5 * curriculumupdate)
        reward_scales["success"]  = success_init  * (1 + 0.5 * curriculumupdate)
        reward_scales["stopball"] = stop_init     * (1 + 0.5 * curriculumupdate)

    curriculumupdate ∈ {0, 1, 2, 3}; at max the weights reach 2.5× their init value.
    G1 config inits: eereach=10, success=5, stopball=100 → peak 25 / 12.5 / 250.
    Our inits:       eereach=10, hand_proximity_strict=5, stopball=100 → same peak.

    Must be listed AFTER adaptive_curriculum_update in cfg.curriculum so it reads
    the _curriculumupdate value that was just set by that term.
    """

    _REWARD_NAMES = ("eereach", "hand_proximity_strict", "stopball")

    def __init__(self, cfg: "CurriculumTermCfg", env: ManagerBasedRlEnv) -> None:
        self._init_weights: dict[str, float] = {}
        for name in self._REWARD_NAMES:
            try:
                term_cfg = env.reward_manager.get_term_cfg(name)
                self._init_weights[name] = term_cfg.weight
            except (ValueError, AttributeError):
                pass

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict:
        cu = getattr(env, "_curriculumupdate", 0.0)
        scale = 1.0 + 0.5 * cu  # 1.0 → 2.5 as curriculumupdate goes 0 → 3
        for name, init_w in self._init_weights.items():
            try:
                env.reward_manager.get_term_cfg(name).weight = init_w * scale
            except (ValueError, AttributeError):
                pass
        return {"task_reward_scale": torch.tensor(scale)}


class adaptive_curriculum_update:
    """Episode-length-based adaptive curriculum driver matching G1's mechanism exactly.

    G1 legged_robot.py line 329 (inside reset_idx, guarded by 500-step gate):
        self.curriculumupdate = int(mean(episode_length_buf[env_ids].float()) / 50.)

    This fires at most once per 500 sim steps and uses the mean episode length of
    currently-resetting envs. At max episode length (150 steps at 3 s / 0.02 s dt),
    curriculumupdate saturates at int(150/50) = 3.

    Sets on env:
        _curriculumupdate: float in {0.0, 1.0, 2.0, 3.0} — mirrors G1 curriculumupdate
        _ball_difficulty:  float in [0.0, 1.0] = _curriculumupdate / 3.0
            → consumed by _reset_ball in commands.py to interpolate ball shot ranges
            → consumed by eereach reward function for jump_scale
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: ManagerBasedRlEnv) -> None:
        if not hasattr(env, "_curriculumupdate"):
            env._curriculumupdate = 0.0
        if not hasattr(env, "_ball_difficulty"):
            env._ball_difficulty = 0.0
        if not hasattr(env, "_last_curriculum_step"):
            env._last_curriculum_step = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        min_gate: int = 500,
    ) -> dict:
        if (env.common_step_counter - env._last_curriculum_step) > min_gate:
            # Use only truly-terminated (fallen) envs, not timeouts.
            # T1's stable standing keyframe means all episodes time out at step 150 even
            # with a random policy → raw mean_ep_len = 150 on the very first check →
            # ball_difficulty jumps to 1.0 before the policy has learned anything.
            # Using reset_terminated (actual falls) means a passive robot (no falls)
            # keeps curriculum at 0; advancement only happens as the policy takes AMP
            # motions that occasionally topple the robot, then recovers.
            terminated = env.reset_terminated[env_ids]  # True=fell, False=timeout
            if terminated.any():
                mean_ep_len = env.episode_length_buf[env_ids[terminated]].float().mean().item()
            else:
                # All envs timed out — passive or perfect robot; hold current difficulty.
                env._last_curriculum_step = env.common_step_counter
                return {
                    "curriculumupdate": torch.tensor(env._curriculumupdate),
                    "ball_difficulty": torch.tensor(env._ball_difficulty),
                }

            env._curriculumupdate = float(int(mean_ep_len / 50.0))
            new_difficulty = min(1.0, env._curriculumupdate / 3.0)
            # Monotonic — mirrors G1's effective behaviour (command_ranges only expand,
            # never shrink: legged_robot.py L333-336 accumulate via torch.clip each call).
            # Once the robot has genuinely earned a difficulty level it is not pulled back.
            env._ball_difficulty = max(getattr(env, "_ball_difficulty", 0.0), new_difficulty)
            env._last_curriculum_step = env.common_step_counter
        return {
            "curriculumupdate": torch.tensor(env._curriculumupdate),
            "ball_difficulty": torch.tensor(env._ball_difficulty),
        }


def reset_ball_training(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str = "ball") -> None:
    """Reset ball with random trajectory for training.

    Ball spawns at positive Y (3–5 m in front of robot) and is aimed toward the
    goal line (Y≈0) at a random lateral X and height Z target. Without this,
    the default reset_scene_to_default places the ball at the env origin with zero
    velocity, so stopball/eereach never receive meaningful gradients.
    """
    _shoot_ball(env, env_ids, ball_name)


def reset_ball_autonomous(env: ManagerBasedRlEnv, env_ids: torch.Tensor, ball_name: str = "ball") -> None:
    """Reset ball with random trajectory for autonomous play (symmetric, no motion-type routing)."""
    _shoot_ball(env, env_ids, ball_name)


def reset_ball_per_motion(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball_name: str = "ball",
    difficulty_override: float = -1.0,
) -> None:
    """Reset ball using the motion-type-specific target zone.

    Reads motion_type_ids from the 'motion' command manager term and routes
    to the correct _BALL_END_RANGES entry — lefthand/jump/step motions target
    +Y (left hand), righthand/jump/step target -Y (right hand).

    Used in play mode so that --motion-file lefthand_t1.npz makes the ball
    always fly to the +Y (green axis) side, matching training behaviour exactly.
    Falls back to symmetric spawn if no motion command is registered.

    Args:
        difficulty_override: If >= 0, use this difficulty instead of env._ball_difficulty.
            Pass 1.0 in play mode to use the full training range.
    """
    from imitationlearningbooster.mdp.commands import _BALL_END_RANGES, _BALL_END_RANGES_EASY

    ball: Entity = env.scene[ball_name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    g = 9.81
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]

    # Try to read motion_type_ids from command manager.
    motion_types = None
    try:
        cmd = env.command_manager._terms.get("motion", None)
        if cmd is not None and hasattr(cmd, "motion_type_ids"):
            motion_types = cmd.motion_type_ids[env_ids]
    except Exception:
        pass

    x_start = sample_uniform(3.0, 4.5, (n,), device=env.device)
    y_start = sample_uniform(-0.8, 0.8, (n,), device=env.device)
    z_start = sample_uniform(0.5, 1.4, (n,), device=env.device)

    if motion_types is not None:
        if difficulty_override >= 0.0:
            difficulty = max(0.0, min(1.0, difficulty_override))
        else:
            difficulty = float(getattr(env, "_ball_difficulty", 0.0))
        difficulty = max(0.0, min(1.0, difficulty))
        end_ranges_full = torch.tensor(_BALL_END_RANGES, device=env.device)
        end_ranges_easy = torch.tensor(_BALL_END_RANGES_EASY, device=env.device)
        end_ranges = end_ranges_easy + difficulty * (end_ranges_full - end_ranges_easy)
        per_env = end_ranges[motion_types]                                    # [n, 4]
        y_end = sample_uniform(per_env[:, 0], per_env[:, 1], (n,), device=env.device)
        z_end = sample_uniform(per_env[:, 2], per_env[:, 3], (n,), device=env.device)
    else:
        # Fallback: full bilateral range (no motion type info available).
        y_end = sample_uniform(-0.65, 0.65, (n,), device=env.device)
        z_end = sample_uniform(0.20, 1.40, (n,), device=env.device)

    t_flight = sample_uniform(0.5, 1.0, (n,), device=env.device)

    dx = -x_start - 0.3
    dy = y_end - y_start
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
