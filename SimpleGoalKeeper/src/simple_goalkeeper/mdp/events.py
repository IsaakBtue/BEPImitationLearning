"""Event functions and curriculum helpers for SimpleGoalKeeper."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

_MOTIONS_DIR = Path(__file__).parents[1] / "motions" / "data"
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


# Easy (difficulty=0.0) spawn ranges: short distance, centred, slow, low.
# Hard (difficulty=1.0) ranges are passed as params to reset_ball_local_frame.
_EASY_DIST    = (2.5, 3.5)
_EASY_Y_START = (-0.05, 0.05)  # lateral spawn: very centred on easy (hard = ±1.0 m via params)
_EASY_Y_END   = (-0.05, 0.05)  # goal target Y: dead centre on easy
_EASY_Z_START = (0.1, 0.25)
_EASY_Z_END   = (0.05, 0.15)
_EASY_SPEED   = (1.0, 1.5)     # slow on easy → longer t_flight, more reaction time


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


# Z offset baked into NPZ files (2026-06-30): body_pos_w Z was shifted +0.030 m
# in all NPZ files so the foot capsule contact surface sits at floor level.
# No runtime correction needed here anymore.

# Lateral crossing-Y thresholds for 4-tier distance-conditioned RSI.
_SINGLE_THRESH = 0.30   # |cross_y| < 0.30          → single  (standing reset)
_DOUBLE_THRESH = 0.50   # 0.30 ≤ |cross_y| < 0.50   → double  (MediumStep, SafeMedium)
_TRIPLE_THRESH = 0.70   # 0.50 ≤ |cross_y| < 0.70   → triple  (FarStep, SafeFar)
                        # |cross_y| ≥ 0.70           → wide    (DoubleStep, TripleStep)

# Exact filename stem (lowercased) → (side, pool) mapping.
# Name-specific so adding new files never silently mis-classifies.
_STEM_TO_POOL: dict[str, tuple[str, str]] = {
    # double  (0.35–0.65 m)
    "leftsafemedium1_booster_t1":     ("left",  "double"),
    "rightsafemedium1_booster_t1":    ("right", "double"),
    # triple  (0.65–0.85 m)
    "leftsafefar1_booster_t1":        ("left",  "triple"),
    "rightsafefar1_booster_t1":       ("right", "triple"),
    # wide    (≥ 0.85 m)
    "leftdoublestep_own_booster_t1":  ("left",  "wide"),
    "lefttriplestep_own_booster_t1":  ("left",  "wide"),
    "rightdoublestep_own_booster_t1": ("right", "wide"),
    "righttriplestep_own_booster_t1": ("right", "wide"),
    # single-range files (< 0.35 m) → standing pose, not RSI pools; listed so
    # the init loop doesn't warn about unknown files.
    "leftstep_own_booster_t1":        None,
    "leftsafe1_booster_t1":           None,
    "leftsafefront1_booster_t1":      None,
    "rightstep_own_booster_t1":       None,
    "rightsafe1_booster_t1":          None,
    "rightsafefront1_booster_t1":     None,
}


def _load_pool(files: list, dev: str) -> dict[str, torch.Tensor]:
    """Concatenate NPZ frames from a list of files into one pool dict."""
    arrays: dict[str, list] = {k: [] for k in
                                ("joint_pos", "joint_vel", "root_pos",
                                 "root_quat", "root_lin_vel", "root_ang_vel")}
    for f in files:
        data = np.load(str(f))
        arrays["joint_pos"].append(data["joint_pos"])
        arrays["joint_vel"].append(data["joint_vel"])
        arrays["root_pos"].append(data["body_pos_w"][:, 0, :])
        arrays["root_quat"].append(data["body_quat_w"][:, 0, :])
        arrays["root_lin_vel"].append(data["body_lin_vel_w"][:, 0, :])
        arrays["root_ang_vel"].append(data["body_ang_vel_w"][:, 0, :])
    return {k: torch.from_numpy(np.vstack(v)).float().to(dev) for k, v in arrays.items()}


class MotionResetManager:
    """Distance-conditioned RSI: selects motion frames based on ball's predicted
    goal-line crossing position.  4 distance tiers × 2 sides = 8 pools.

    reset_ball fires BEFORE reset_from_motion_data (see goalkeeper_env_cfg.py),
    so the new ball velocity is already set when reset() is called.
    """

    _instance: "MotionResetManager | None" = None

    def __init__(self) -> None:
        # Combined pool (fallback / AMP reference compatibility)
        self.frames: dict[str, torch.Tensor] = {}
        # Per-type pools keyed by (side, steps): side ∈ {left, right}, steps ∈ {single, double, triple, wide}
        self.pools: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> "MotionResetManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, env: "ManagerBasedRlEnv") -> None:
        """Load NPZ files into 8 per-type pools + one combined fallback pool."""
        if "joint_pos" in self.frames:
            return

        motion_files = sorted(_MOTIONS_DIR.glob("*.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No NPZ motion files in {_MOTIONS_DIR}")

        dev = env.device

        bucket: dict[tuple[str, str], list] = {
            ("left",  "double"): [], ("left",  "triple"): [], ("left",  "wide"): [],
            ("right", "double"): [], ("right", "triple"): [], ("right", "wide"): [],
        }
        all_files = []
        for f in motion_files:
            stem = f.stem.lower()
            if stem not in _STEM_TO_POOL:
                print(f"[MotionResetManager] WARNING: {f.name} not in _STEM_TO_POOL — added to combined pool only")
                all_files.append(f)
                continue
            key = _STEM_TO_POOL[stem]
            all_files.append(f)
            if key is not None:   # None = single-range, goes to combined pool only
                bucket[key].append(f)

        for key, files in bucket.items():
            if files:
                self.pools[key] = _load_pool(files, dev)
                n = self.pools[key]["joint_pos"].shape[0]
                print(f"[MotionResetManager] {key}: {n} frames from {len(files)} file(s)")
            else:
                print(f"[MotionResetManager] WARNING: no files for pool {key}")

        self.frames = _load_pool(all_files, dev)
        print(f"[MotionResetManager] Combined pool: {self.frames['joint_pos'].shape[0]} frames")

    def _write_rsi_state(
        self,
        env: "ManagerBasedRlEnv",
        ids: torch.Tensor,
        pool: dict[str, torch.Tensor],
        robot: "Entity",
    ) -> None:
        """Write root pose + velocity + joint state from a random frame in pool."""
        n = len(ids)
        frame_ids = torch.randint(0, pool["joint_pos"].shape[0], (n,), device=env.device)

        root_pos  = pool["root_pos"][frame_ids]
        root_quat = pool["root_quat"][frame_ids]
        positions = env.scene.env_origins[ids].clone()
        positions[:, 2] = root_pos[:, 2]  # Z offset baked into NPZ (2026-06-30)
        robot.write_root_link_pose_to_sim(
            torch.cat([positions, root_quat], dim=-1), env_ids=ids
        )

        robot.write_root_link_velocity_to_sim(
            torch.cat([pool["root_lin_vel"][frame_ids],
                       pool["root_ang_vel"][frame_ids]], dim=-1),
            env_ids=ids,
        )

        joint_pos = pool["joint_pos"][frame_ids].clone()
        joint_vel = pool["joint_vel"][frame_ids].clone()
        limits = robot.data.soft_joint_pos_limits[ids]
        joint_pos.clamp_(limits[..., 0], limits[..., 1])
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    def reset(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        rsi_fraction: float = 0.8,
    ) -> None:
        """80% distance-conditioned RSI + 20% HOME_KEYFRAME standing.

        reset_ball fires first, so ball pos/vel reflect the NEW episode trajectory.
        We predict the ball's goal-line crossing Y and pick:
          left/right — from sign of crossing_y
          tier       — from |crossing_y| vs thresholds:
            < 0.35 m  → standing pose (no RSI)
            0.35–0.65 → double  (MediumStep, SafeMedium)
            0.65–0.85 → triple  (FarStep, SafeFar)
            ≥ 0.85    → wide    (DoubleStep, TripleStep)

        20% standing resets prevent the AMP dive→RSI transition artefact and
        ensure the policy learns to save from a standing start (deployment scenario).
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        if len(env_ids) == 0:
            return

        robot: Entity = env.scene[asset_cfg.name]
        n = len(env_ids)

        use_rsi     = torch.rand(n, device=env.device) < rsi_fraction
        rsi_local   = use_rsi.nonzero(as_tuple=True)[0]
        stand_local = (~use_rsi).nonzero(as_tuple=True)[0]

        # Collect extra env_ids that are redirected from RSI → standing.
        extra_stand_ids: list[torch.Tensor] = []

        # ── 80 %: distance-conditioned RSI ────────────────────────────────
        if len(rsi_local) > 0:
            ids_rsi = env_ids[rsi_local]

            # Read the crossing Y stored by reset_ball_rolling (fires before this event).
            # Using the stored value avoids the entity data-buffer lag: ball.data.*
            # is only refreshed after the next physics step, so reading it here would
            # give the PREVIOUS episode's ball state.
            cross_y = getattr(env, "_rsi_cross_y", None)
            if cross_y is None:
                # Fallback for the very first reset before reset_ball has ever fired.
                self._write_rsi_state(env, ids_rsi, self.frames, robot)
            else:
                cy     = cross_y[ids_rsi]
                is_left = cy > 0
                abs_cy  = cy.abs()

                is_single = abs_cy < _SINGLE_THRESH
                is_double = (abs_cy >= _SINGLE_THRESH) & (abs_cy < _DOUBLE_THRESH)
                is_triple = (abs_cy >= _DOUBLE_THRESH) & (abs_cy < _TRIPLE_THRESH)
                is_wide   = abs_cy >= _TRIPLE_THRESH

                # Single-range: ball arrives near centre — start standing so the
                # policy reacts freely rather than committing to a lateral step.
                single_local = is_single.nonzero(as_tuple=True)[0]
                if len(single_local) > 0:
                    extra_stand_ids.append(ids_rsi[single_local])

                for (side, steps), mask in [
                    (("left",  "double"), is_left  & is_double),
                    (("left",  "triple"), is_left  & is_triple),
                    (("left",  "wide"),   is_left  & is_wide),
                    (("right", "double"), ~is_left & is_double),
                    (("right", "triple"), ~is_left & is_triple),
                    (("right", "wide"),   ~is_left & is_wide),
                ]:
                    local_ids = mask.nonzero(as_tuple=True)[0]
                    if len(local_ids) == 0:
                        continue
                    pool = self.pools.get((side, steps), self.frames)
                    self._write_rsi_state(env, ids_rsi[local_ids], pool, robot)

        # ── HOME_KEYFRAME standing pose (20% random + all single-range) ───
        ids_stand_parts: list[torch.Tensor] = []
        if len(stand_local) > 0:
            ids_stand_parts.append(env_ids[stand_local])
        ids_stand_parts.extend(extra_stand_ids)
        if ids_stand_parts:
            ids_stand = torch.cat(ids_stand_parts)
            ns = len(ids_stand)
            home_pos = robot.data.default_joint_pos[ids_stand]
            home_vel = torch.zeros(ns, home_pos.shape[-1], device=env.device)
            robot.write_joint_state_to_sim(home_pos, home_vel, env_ids=ids_stand)
            # Root state already correct from reset_base.


def init_motion_loader(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: load all NPZ motion files into MotionResetManager."""
    MotionResetManager.get().init(env)


def reset_from_motion_data(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
    """Reset event: 50% RSI from NPZ motion frame, 50% HOME_KEYFRAME standing pose."""
    mgr = MotionResetManager.get()
    mgr.init(env)
    mgr.reset(env, env_ids, asset_cfg, rsi_fraction=0.5)


def reset_ball_local_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (2.0, 4.0),
    y_start_range: tuple[float, float] = (-0.8, 0.8),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    spawn_height_range: tuple[float, float] = (0.2, 0.7),
    arrive_height_range: tuple[float, float] = (0.1, 0.4),
    speed_range: tuple[float, float] = (3.0, 7.0),
) -> None:
    """Spawn ball aimed at the goal area from a variable lateral position.

    Ball spawns at (dist, y_start) in the robot's local frame and is aimed toward
    (-0.3, y_end) — 0.3 m behind the goal line so the ball retains forward momentum
    at interception (matches ILB). The velocity vector has both X and Y components so
    the ball crosses the goal at a realistic angle.

    Spawning in the robot's local frame means the policy is orientation-invariant:
    the ball always arrives from the robot's front regardless of world yaw. The yaw
    penalty (ang_vel_z) penalises spinning while keeping the coordinate system clean.

    y_start_range ±0.8 m matches ILB (narrowed from ±1.5 m to avoid >50° approach
    angles that produce no gradient signal).

    Args:
        dist_range:          forward (local +X) distance from robot to spawn (m)
        y_start_range:       lateral spawn offset in robot +Y (m)
        y_end_range:         goal target lateral position in robot +Y (m)
        spawn_height_range:  ball height above floor at spawn (m)
        arrive_height_range: target ball height above floor at goal line (m)
        speed_range:         total horizontal approach speed magnitude (m/s)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    # Curriculum difficulty lerps from easy ranges (d=0) to configured params (d=1).
    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    dist_r    = _lerp_range(_EASY_DIST,    dist_range,          d)
    y_start_r = _lerp_range(_EASY_Y_START, y_start_range,       d)
    y_end_r   = _lerp_range(_EASY_Y_END,   y_end_range,         d)
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

    x_start = sample_uniform(*dist_r,    (n,), env.device)   # forward distance
    y_start = sample_uniform(*y_start_r, (n,), env.device)   # lateral spawn
    y_end   = sample_uniform(*y_end_r,   (n,), env.device)   # goal target Y
    z_start = floor_z + sample_uniform(*z_start_r, (n,), env.device)
    z_end   = floor_z + sample_uniform(*z_end_r,   (n,), env.device)
    speed_h = sample_uniform(*speed_r,   (n,), env.device)   # total horizontal speed

    # Ball spawn position in local frame: forward x_start, lateral y_start.
    local_spawn = torch.stack([x_start, y_start, torch.zeros_like(x_start)], dim=-1)
    world_spawn_xy = quat_apply(yaw_q, local_spawn)
    ball_pos = torch.empty((n, 3), device=env.device)
    ball_pos[:, 0] = robot_pos_w[:, 0] + world_spawn_xy[:, 0]
    ball_pos[:, 1] = robot_pos_w[:, 1] + world_spawn_xy[:, 1]
    ball_pos[:, 2] = z_start

    # Velocity direction: from spawn (x_start, y_start) toward (-0.3, y_end).
    # Aiming 0.3 m behind the goal line (matches ILB) so the ball still has forward
    # momentum when it reaches the robot — avoids deceleration to zero right at x=0.
    dx_local = -(x_start + 0.3)     # negative: ball moves toward robot then past it
    dy_local = y_end - y_start       # lateral component; nonzero for angled shots
    horiz_dist = torch.sqrt(dx_local ** 2 + dy_local ** 2)
    t_flight = horiz_dist / speed_h

    vx_local = dx_local / t_flight
    vy_local = dy_local / t_flight

    local_vel_h = torch.stack([vx_local, vy_local, torch.zeros_like(vx_local)], dim=-1)
    world_vel_h = quat_apply(yaw_q, local_vel_h)

    # Gravity-compensating vz: z_end = z_start + vz*t - 0.5*g*t^2
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


def reset_ball_global_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    x_start_range: tuple[float, float] = (2.0, 3.5),
    y_start_range: tuple[float, float] = (-0.5, 0.5),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    z_start_range: tuple[float, float] = (0.10, 0.20),
    z_end_range: tuple[float, float] = (0.05, 0.10),
    t_flight_range: tuple[float, float] = (0.35, 0.60),
) -> None:
    """Spawn ball aimed at goal from global +X direction (mirrors ILB _reset_ball).

    Ball spawns at (env_origin + x_start, y_start, z_start) in world frame and
    travels toward (env_origin - 0.3, y_end, z_end). Uses t_flight directly (not
    speed) so vz stays bounded regardless of distance — short t_flight keeps the
    arc flat at foot/ankle level.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    _EASY_T = (0.30, 0.45)
    x_start_r  = _lerp_range((2.0, 2.5),   x_start_range,  d)
    y_start_r  = _lerp_range((-0.2, 0.2),  y_start_range,  d)
    y_end_r    = _lerp_range((-0.1, 0.1),  y_end_range,    d)
    z_start_r  = _lerp_range((0.05, 0.10), z_start_range,  d)
    z_end_r    = _lerp_range((0.05, 0.08), z_end_range,    d)
    t_range    = _lerp_range(_EASY_T,      t_flight_range, d)

    g = 9.81
    ball: Entity = env.scene[ball_name]
    origins = env.scene.env_origins[env_ids]

    x_start  = sample_uniform(*x_start_r,  (n,), env.device)
    y_start  = sample_uniform(*y_start_r,  (n,), env.device)
    y_end    = sample_uniform(*y_end_r,    (n,), env.device)
    z_start  = origins[:, 2] + sample_uniform(*z_start_r, (n,), env.device)
    z_end    = origins[:, 2] + sample_uniform(*z_end_r,   (n,), env.device)
    t_flight = sample_uniform(*t_range,    (n,), env.device)

    ball_pos = torch.stack([
        origins[:, 0] + x_start,
        origins[:, 1] + y_start,
        z_start,
    ], dim=-1)
    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0

    dx = -(x_start + 0.3)
    dy = y_end - y_start
    dz = z_end - z_start

    vx = dx / t_flight
    vy = dy / t_flight
    vz = (dz + 0.5 * g * t_flight ** 2) / t_flight

    ball_vel = torch.stack([vx, vy, vz], dim=-1)

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
        env._ball_visible_step = torch.zeros(n, device=env.device)
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


class reward_curriculum_ep_len:
    """G1-style episode-length-driven reward weight curriculum.

    Mirrors G1 legged_robot.py:325-364 exactly:
        every 500 env steps:
            curriculumupdate = int(mean_episode_length / ep_len_divisor)  # integer 0-3
        then per reward:
            weight = base_weight * (1 + 0.5 * curriculumupdate)

    All instances share env._curriculumupdate so the episode-length sampling
    is identical across all ramped rewards (same as G1's single class variable).

    G1 extremes: ep_len=150 (full episode) → curriculumupdate=3 → weight = 2.5 × base.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._reward_name     = p["reward_name"]
        self._base_weight     = p["base_weight"]
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        self._last_update     = -(self._update_interval)  # fire immediately on first call
        self._term_cfg        = env.reward_manager.get_term_cfg(self._reward_name)
        if not hasattr(env, "_curriculumupdate"):
            env._curriculumupdate = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        reward_name: str,
        base_weight: float,
        **kwargs,
    ) -> dict:
        # Update shared curriculumupdate once per window (first term to run wins).
        if env.common_step_counter - self._last_update >= self._update_interval:
            self._last_update = env.common_step_counter
            if len(env_ids) > 0:
                mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
            else:
                mean_ep_len = 0.0
            cu = int(mean_ep_len / self._ep_len_divisor)
            # Monotonic: once a level is reached it never drops. Prevents oscillation
            # near the boundary (e.g. ep_len flickering across 144 = 48×3).
            env._curriculumupdate = max(env._curriculumupdate, cu)

        # G1 formula — weight = base * (1 + 0.5 * cu), cu capped at 3 naturally by ep_len.
        new_weight = self._base_weight * (1.0 + 0.5 * env._curriculumupdate)
        self._term_cfg.weight = new_weight
        return {"weight": torch.tensor(float(new_weight))}


class correct_foot_save_curriculum:
    """Step-up curriculum for correct-foot-save quality bonuses.

    Reads the shared env._curriculumupdate (written by reward_curriculum_ep_len
    instances) and doubles the reward weight the moment cu reaches a threshold.

    Formula: weight = base_weight * multiplier
        where multiplier = 1.0 when cu < activate_at_cu
                         = 2.0 when cu >= activate_at_cu

    Intended for one-time quality bonuses that only make sense once the policy
    has already learned to save reliably (cu >= 3 = ep_len ≈ 144 steps):
        single_foot_save, inner_face_orientation_save, cleanstop, airborne_at_save.

    One curriculum entry per reward (same pattern as reward_curriculum_ep_len).
    Shares env._curriculumupdate — no extra episode-length sampling.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._reward_name    = p["reward_name"]
        self._base_weight    = p["base_weight"]
        self._activate_at_cu = p.get("activate_at_cu", 3)
        self._term_cfg       = env.reward_manager.get_term_cfg(self._reward_name)
        if not hasattr(env, "_curriculumupdate"):
            env._curriculumupdate = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        reward_name: str,
        base_weight: float,
        **kwargs,
    ) -> dict:
        multiplier = 2.0 if env._curriculumupdate >= self._activate_at_cu else 1.0
        new_weight = self._base_weight * multiplier
        self._term_cfg.weight = new_weight
        return {"weight": torch.tensor(float(new_weight))}


class ball_difficulty_curriculum:
    """Adaptive difficulty curriculum — direct port of Humanoid-Goalkeeper (G1) approach.

    G1 legged_robot.py:325-336: every `update_interval` per-env steps compute
        curriculumupdate = int(mean_episode_length / ep_len_divisor)
    and expand command ranges by `step_size × curriculumupdate`.
    Longer completed episodes → robot has mastered current difficulty → advance faster.
    Short episodes (robot falling, ball escaping) → advance slowly or not at all.

    SGK maps G1's raw range expansion onto the 0→1 `_ball_difficulty` scalar:
        difficulty += step_size × curriculumupdate   (clamped to [0, 1])

    Default params mirror G1 exactly:
        ep_len_divisor = 50   (same as G1)
        update_interval = 500 (same as G1, in per-env steps)
        step_size = 0.01      (difficulty units per curriculumupdate per check;
                               at ep_len=144→curriculumupdate=2, reaches 1.0
                               in ~50 checks ≈ 1000 iters with 24 steps/iter)
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._step_size      = p.get("step_size",      0.01)
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        self._last_update     = -(self._update_interval)
        if not hasattr(env, "_ball_difficulty"):
            env._ball_difficulty = 0.0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        # Only update every `update_interval` per-env steps (mirrors G1's 500-step cadence).
        if env.common_step_counter - self._last_update < self._update_interval:
            return {"ball_difficulty": torch.tensor(env._ball_difficulty)}

        self._last_update = env.common_step_counter

        # G1: curriculumupdate = int(mean_episode_length / 50)
        # Uses lengths of just-completed episodes (env_ids are resetting envs).
        if len(env_ids) > 0:
            mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
        else:
            mean_ep_len = 0.0
        curriculumupdate = int(mean_ep_len / self._ep_len_divisor)

        # Direct curriculum mapping (matches G1: difficulty = f(cu) only, not accumulated).
        # At cu=3 (ep_len≈144), difficulty = 1.0. Monotonic: never goes backward even
        # if ep_len temporarily drops (e.g., during policy collapse or harder balls).
        new_difficulty = min(1.0, curriculumupdate / 3.0)
        env._ball_difficulty = max(env._ball_difficulty, new_difficulty)
        return {"ball_difficulty": torch.tensor(env._ball_difficulty)}


def sharpforce_termination(
    env: "ManagerBasedRlEnv",
    max_contact_force: float = 1500.0,
) -> torch.Tensor:
    """Terminate when mean foot contact force exceeds threshold.

    Mirrors upstream Humanoid-Goalkeeper sharpforce_buf termination.
    Requires feet_contact ContactSensor in the scene.
    Geom layout (sorted by name): left_foot_1, left_foot_2, right_foot_1, right_foot_2.
    Returns [B] bool tensor: True → terminate.
    """
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)       # [B, 8]
    left_max  = force_per_geom[:, :4].max(dim=-1).values  # left_foot1-4
    right_max = force_per_geom[:, 4:].max(dim=-1).values  # right_foot1-4
    mean_force = (left_max + right_max) / 2.0
    return mean_force > max_contact_force


def reset_ball_rolling(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (1.5, 2.5),
    y_start_range: tuple[float, float] = (-0.5, 0.5),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    t_flight_range: tuple[float, float] = (0.7, 1.1),
    spawn_z: float = 0.10,
) -> None:
    """Spawn ball at ground level in world (global) frame — rolling ground pass.

    Spawns at (env_origin_x + x_start, env_origin_y + y_start) in world frame so
    the ball always approaches along world -X regardless of robot yaw. This keeps
    all world-frame reward terms (stopball uses world-X velocity, stayonline uses
    world-X position, ball_exit_termination uses world-X) perfectly aligned with
    the ball trajectory.

    vz=0: ball drops to ground in <0.05 s then rolls at foot/ankle height.
    Curriculum lerps from easy to hard via env._ball_difficulty.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    _EASY_DIST_R      = (1.5, 2.0)
    _EASY_Y_ROLL      = (-0.05, 0.05)
    _EASY_T_FLIGHT_R  = (0.9, 1.3)   # easy: long flight → slow balls; hard: 0.7–1.1 s

    dist_r      = _lerp_range(_EASY_DIST_R,     dist_range,     d)
    y_start_r   = _lerp_range(_EASY_Y_ROLL,     y_start_range,  d)
    t_flight_r  = _lerp_range(_EASY_T_FLIGHT_R, t_flight_range, d)

    ball: Entity = env.scene[ball_name]
    origins = env.scene.env_origins[env_ids]   # world goal-line origin per env
    floor_z = origins[:, 2]

    x_start  = sample_uniform(*dist_r,     (n,), env.device)
    y_start  = sample_uniform(*y_start_r,  (n,), env.device)
    t_flight = sample_uniform(*t_flight_r, (n,), env.device)

    # Two-sided dead zone for y_end (mirrors G1's ±0.2 m minimum offset).
    # At d=0: ball always arrives 0.15–0.35 m left or right of center — never
    # through the robot's legs regardless of RSI stance width.
    # At d=1: magnitude range opens to [0.0, max_half] covering the full range.
    # The dead zone shrinks linearly so there is no curriculum cliff at any d.
    _Y_INNER = 0.15   # minimum offset from center at d=0
    _Y_OUTER = 0.35   # maximum offset from center at d=0
    max_half = max(abs(y_end_range[0]), abs(y_end_range[1]))
    inner = max(_Y_INNER * (1.0 - d), 0.1)
    outer = _Y_OUTER + (max_half - _Y_OUTER) * d
    mag   = sample_uniform(inner, outer, (n,), env.device)
    side  = torch.where(torch.rand(n, device=env.device) > 0.5,
                        torch.ones(n, device=env.device),
                        -torch.ones(n, device=env.device))
    y_end = side * mag

    # Ball spawns in world frame: x_start metres in front of goal line, y_start to the side.
    ball_pos = torch.stack([
        origins[:, 0] + x_start,
        origins[:, 1] + y_start,
        floor_z + spawn_z,
    ], dim=-1)

    # World-frame velocity: aimed from (x_start, y_start) → goal-line target (-0.3, y_end).
    # Speed is derived from geometry and flight time (mirrors G1's assign_ball_states).
    dx = -(x_start + 0.3)          # world -X: from spawn toward 0.3 m behind goal line
    dy = y_end - y_start            # world Y: lateral sweep to target

    ball_vel = torch.stack([
        dx / t_flight,              # world vx (negative = approaching)
        dy / t_flight,              # world vy
        torch.zeros_like(x_start),  # vz = 0: drops instantly, rolls at foot level
    ], dim=-1)

    ball_quat = torch.zeros((n, 4), device=env.device)
    ball_quat[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, ball_quat], dim=-1), env_ids=env_ids
    )
    # Pure-rolling angular velocity: no-slip condition for ball rolling in XY plane.
    # Without this the ball starts sliding (zero ω, nonzero v) and loses 2/7*v0 to
    # sliding friction before reaching rolling — at max spawn 3.5 m/s that is exactly
    # 1.0 m/s, enough to fire stopball from floor friction alone (false positive).
    # ωy = vx/r, ωx = -vy/r eliminates the sliding phase entirely.
    _BALL_RADIUS = 0.11
    ball_ang_vel = torch.stack([
        -ball_vel[:, 1] / _BALL_RADIUS,
         ball_vel[:, 0] / _BALL_RADIUS,
        torch.zeros(n, device=env.device),
    ], dim=-1)
    ball.write_root_link_velocity_to_sim(
        torch.cat([ball_vel, ball_ang_vel], dim=-1),
        env_ids=env_ids,
    )

    # Store predicted goal-line crossing Y (relative to env origin) for RSI pool selection.
    # Computes where ball crosses goal line, accounting for diagonal trajectory.
    # Ball travels from (x_start, y_start) to (-0.3, y_end) along a straight line.
    # At x = -0.3 (goal line), Y = y_start + (y_end - y_start) * (distance along X) / (total distance).
    # Total horizontal distance = sqrt((x_start + 0.3)^2 + (y_end - y_start)^2)
    # Distance along X at goal line = x_start + 0.3
    # So cross_y = y_start + (y_end - y_start) * (x_start + 0.3) / horiz_dist
    # Computed here while x_start/y_start/y_end are in scope — avoids the entity data
    # buffer lag that would corrupt a post-write read of ball.data.root_link_lin_vel_w.
    if not hasattr(env, "_rsi_cross_y"):
        env._rsi_cross_y = torch.zeros(env.num_envs, device=env.device)
    horiz_dist = torch.sqrt((x_start + 0.3) ** 2 + (y_end - y_start) ** 2)
    env._rsi_cross_y[env_ids] = y_start + (y_end - y_start) * (x_start + 0.3) / horiz_dist

    _init_visibility_state(env, env_ids)


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
