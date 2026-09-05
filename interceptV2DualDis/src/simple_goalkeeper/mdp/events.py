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
_SINGLE_THRESH = 0.20   # |cross_y| < 0.20          → single  (standing reset)
_DOUBLE_THRESH = 0.40   # 0.20 ≤ |cross_y| < 0.40   → double  (MediumStep, SafeMedium)
_TRIPLE_THRESH = 0.60   # 0.40 ≤ |cross_y| < 0.60   → triple  (FarStep, SafeFar)
                        # |cross_y| ≥ 0.60           → wide    (DoubleStep, TripleStep)

# Exact filename stem (lowercased) → (side, pool) mapping.
# Name-specific so adding new files never silently mis-classifies.
_STEM_TO_POOL: dict[str, tuple[str, str]] = {
    # double  (0.20–0.40 m, see _DOUBLE_THRESH above)
    "leftsafemedium1_booster_t1":     ("left",  "double"),
    "rightsafemedium1_booster_t1":    ("right", "double"),
    # triple  (0.40–0.60 m, see _TRIPLE_THRESH above)
    "leftsafefar1_booster_t1":        ("left",  "triple"),
    "rightsafefar1_booster_t1":       ("right", "triple"),
    # wide    (≥ 0.60 m, see _TRIPLE_THRESH above)
    "leftdoublestep_own_booster_t1":  ("left",  "wide"),
    "lefttriplestep_own_booster_t1":  ("left",  "wide"),
    "rightdoublestep_own_booster_t1": ("right", "wide"),
    "righttriplestep_own_booster_t1": ("right", "wide"),
    # single-range files (< 0.20 m, see _SINGLE_THRESH above) → standing pose,
    # not RSI pools; listed so the init loop doesn't warn about unknown files.
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
    """Loads NPZ motion frames into per-tier pools and the combined pool.

    `reset()` (registered as reset_from_motion_data) is a literal port of
    Humanoid-Goalkeeper G1's continue_keep mechanism (legged_gym/legged_gym/
    envs/base/legged_robot.py:657-682, _reset_dofs) — see docs/superpowers/
    plans/2026-07-01-live-env-rsi.md for the full comparison. It does not use
    the tier pools built here at all; those exist only for the sgk_play_rsi
    diagnostic script (scripts/play_rsi_doublestep.py), which locks playback
    to a specific motion pool for visual inspection.
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
        """Literal port of Humanoid-Goalkeeper G1's continue_keep mechanism.

        Source: Humanoid-Goalkeeper/legged_gym/legged_gym/envs/base/legged_robot.py
        _reset_dofs, lines 657-682, with G1's ACTIVE config values from
        legged_gym/legged_gym/envs/g1/g1_29_config.py:279-282
        (continue_keep=True, randomize_initial_joint_pos=True,
        initial_joint_pos_scale=[0.5, 1.5], initial_joint_pos_offset=[-0.1, 0.1]);
        both files are read-only upstream references, not modified:

            dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
            dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
            if self.cfg.domain_rand.continue_keep and torch.rand(1).item() > 0.2:
                self.dof_pos[env_ids] = self.dof_pos[torch.randint(0, self.num_envs, (len(env_ids),), device=...)]
            else:
                init_dos_pos = self.standpos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=...)
                init_dos_pos += torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=...)
                self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
            self.dof_vel[env_ids] = 0.

        One coin flip per reset() call decides ALL of env_ids together — this
        is G1's own behavior (`torch.rand(1)`, not one flip per env). On the
        rsi_fraction branch, dof_pos is copied from `torch.randint(0, num_envs, ...)`:
        ANY currently-running env, with no side/tier matching, no exclusion of
        the current reset batch, no minimum-age check, and — matching G1
        exactly — NO clamping: G1's `torch.clip` only appears in the OTHER
        branch, never on the continue_keep copy. This deliberately replaces
        the side/tier-conditioned pool system this project used previously —
        see docs/superpowers/plans/2026-07-01-live-env-rsi.md for the full
        comparison, including a fidelity audit that caught the clamp being
        on the wrong branch and this else-branch's randomization being
        missing entirely in an earlier version of this port.

        dof_vel is always zeroed, matching G1's unconditional
        `self.dof_vel[env_ids] = 0.` (not just on the continue_keep branch).
        Root pose/velocity are untouched here in both branches — reset_base
        handles them, matching G1's separate _reset_root_states (note: G1's
        _reset_root_states also randomizes root velocity ±0.3 unconditionally
        on every reset, which this project's reset_base does not — a known,
        separate divergence outside reset_from_motion_data's scope).

        FIX 2026-07-20: the else-branch clip below previously used
        `robot.data.joint_pos_limits` (mjlab's HARD limit field, straight from
        `mj_model.jnt_range`). That does NOT match G1: the pseudocode quoted
        above clips to `self.dof_pos_limits`, but G1's `_process_dof_props`
        (legged_robot.py:545-563) OVERWRITES `self.dof_pos_limits` in place to
        `mean +/- 0.5*range*soft_dof_pos_limit` (`cfg.rewards.soft_dof_pos_limit
        = 0.9`, g1_29_config.py:348) before `_reset_dofs` ever runs -- the hard
        limits survive separately as `self.hard_dof_pos_limits`, which
        `_reset_dofs` never touches. So G1's reset clip is SOFT (90% of URDF
        range), not hard, despite the variable name. mjlab keeps the same two
        fields split out explicitly: `joint_pos_limits` (hard, from
        `mj_model.jnt_range`) vs `soft_joint_pos_limits` (mean +/-
        0.5*range*soft_joint_pos_limit_factor, computed in
        `entity.py:_initialize_data`). `t1_constants.py:112` already sets
        `soft_joint_pos_limit_factor=0.9`, the same 0.9 G1 uses, so
        `robot.data.soft_joint_pos_limits` is the exact equivalent of G1's
        `self.dof_pos_limits` at reset time. `_write_rsi_state` above already
        used the correct soft field; only this branch had the hard/soft mixup.
        See docs/BugFixes.md.
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        if len(env_ids) == 0:
            return

        robot: Entity = env.scene[asset_cfg.name]
        n = len(env_ids)

        if torch.rand(1, device=env.device).item() > (1.0 - rsi_fraction):
            donor_idx = torch.randint(0, env.num_envs, (n,), device=env.device)
            joint_pos = robot.data.joint_pos[donor_idx].clone()
        else:
            default_pos = robot.data.default_joint_pos[env_ids]
            num_dof = default_pos.shape[-1]
            scale = sample_uniform(0.5, 1.5, (n, num_dof), env.device)
            offset = sample_uniform(-0.1, 0.1, (n, num_dof), env.device)
            joint_pos = default_pos * scale + offset
            # FIX 2026-07-20: was robot.data.joint_pos_limits (HARD limits) --
            # G1's self.dof_pos_limits is overwritten to SOFT limits before
            # _reset_dofs runs. See this method's docstring.
            limits = robot.data.soft_joint_pos_limits[env_ids.long()]
            joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Root state already correct from reset_base.


def init_motion_loader(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: load all NPZ motion files into MotionResetManager."""
    MotionResetManager.get().init(env)


def reset_from_motion_data(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
    """Reset event: continue_keep-style donor copy vs default pose.

    FIX 2026-07-01: this wrapper hardcoded rsi_fraction=0.5 (a silent 50/50
    split), diverging from both G1's actual 80/20 (`torch.rand(1).item() > 0.2`,
    legged_robot.py:669) and this project's own reset()/CLAUDE.md-documented
    80/20 intent. Caught by an independent fidelity audit — see
    docs/superpowers/plans/2026-07-01-live-env-rsi.md.
    """
    mgr = MotionResetManager.get()
    mgr.init(env)
    mgr.reset(env, env_ids, asset_cfg, rsi_fraction=0.8)


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


_CURRICULUM_EMA_ALPHA = 0.3

# G1 ranges_4 (step, left side) width curriculum: init=[0.2,1.2], max=[0.0,1.8].
# Shared by far_travel_curriculum (both edges) and reset_ball_rolling's near-
# region default branch (outer edge only -- near's inner is a deliberate
# leg-clearance dead-zone minimum, not a partition artifact, so it doesn't
# move -- see reset_ball_rolling's one-sided branch, 2026-07-18 entry).
_G1_STEP_INNER_FRAC = 0.2 / 1.8
_G1_STEP_OUTER_FRAC = 1.2 / 1.8


def _update_smoothed_ep_len(env: "ManagerBasedRlEnv", mean_ep_len: float) -> float:
    """Exponentially smooth the raw per-window mean_ep_len reading before any
    curriculum term computes cu from it. Shared across reward_curriculum_ep_len
    and ball_difficulty_curriculum (both use the same ep_len_divisor=50) so
    they stay synchronized rather than oscillating independently.

    FIX 2026-07-10: the 2026-07-09 bidirectional-curriculum fix (removing the
    monotonic ratchet) correctly stopped the "stuck forever at a level it
    can't handle" failure, but a live run (rsi_practice_curriculum_2026-07-10)
    showed the opposite overcorrection: with no damping, a single noisy
    500-step window's mean_ep_len is enough to flip cu, and ball_difficulty
    was observed oscillating 0.667<->0.333 repeatedly for the entire run
    (never settling at either level for more than 1-2 updates), with
    mean_episode_length bouncing noisily (78-113) and blue_overshoot_penalty
    flat/oscillating around -0.4 to -0.5 the whole run instead of improving --
    consistent with a policy that never gets a stable enough training
    distribution to consolidate a behavior before the ground shifts again.
    The original (removed) monotonic design's own docstring flagged this
    exact risk ("prevents oscillation near the boundary") but fixed it the
    wrong way (freezing forever) instead of damping the input signal.
    OpenAI's ADR (arXiv 1910.07113) avoids this the right way: difficulty
    boundaries move by small steps based on a rolling performance buffer, not
    an instant snap to the latest noisy reading. This EMA (alpha=0.3, i.e.
    30% weight to the newest window, 70% to history) is the minimal version
    of that idea -- a genuine, sustained shift still moves cu within a few
    updates, but a single noisy window can no longer flip it outright.
    See docs/BugFixes.md.

    FIX 2026-07-20: this function is the single point every curriculum term
    calls into, but it used to actually re-apply the EMA update on every call
    -- and up to 5 curriculum term instances (ball_difficulty_curriculum,
    3x reward_curriculum_ep_len [softstop/footreach/stopball], plus
    far_travel_curriculum on the multi-disc task) share the same
    update_interval=500 and the same env_ids each window, so all of them
    become eligible in the SAME CurriculumManager.compute() call and each
    independently called this function with the identical mean_ep_len
    reading. Applying the EMA update k times in a row with the same input m
    gives effective weight 1-(1-alpha)^k on the newest reading instead of the
    intended alpha=0.3 -- k=4 (single-disc task): 76%; k=5 (multi-disc task,
    +far_travel): 83%. That defeats the whole point of this function (damping
    single-window noise). Fixed by guarding the update itself with a shared
    env._last_ema_update_step: the first curriculum term to call this
    function in a given common_step_counter actually applies the EMA step;
    every other term calling in during that same step just reads back the
    already-updated cached value. Chosen over gating in each term class
    because this function is already the single shared choke point every
    caller goes through -- no changes needed to the term classes themselves.
    See docs/BugFixes.md.
    """
    current_step = env.common_step_counter
    if not hasattr(env, "_smoothed_ep_len"):
        env._smoothed_ep_len = mean_ep_len
        env._last_ema_update_step = current_step
    elif getattr(env, "_last_ema_update_step", None) != current_step:
        env._smoothed_ep_len = (
            _CURRICULUM_EMA_ALPHA * mean_ep_len + (1.0 - _CURRICULUM_EMA_ALPHA) * env._smoothed_ep_len
        )
        env._last_ema_update_step = current_step
    # else: already updated this window by an earlier curriculum term in the
    # same compute() call -- no-op, just return the cached value below.
    return env._smoothed_ep_len


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

    FIX 2026-07-10: cu is now computed from an EMA-smoothed mean_ep_len
    (_update_smoothed_ep_len), not the raw per-window reading -- see that
    function's docstring for why.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._reward_name     = p["reward_name"]
        self._base_weight     = p["base_weight"]
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        # FIX 2026-07-20: was -(update_interval), which makes
        # `common_step_counter - self._last_update >= update_interval` true
        # immediately at common_step_counter=0 -- fires on the very first
        # eligible call (potentially iteration 1) using whatever noisy/near-
        # zero episode-length reading exists that early. G1 requires
        # `common_step_counter > 500` before its first curriculum update
        # (legged_robot.py:325, with `last_step_counter=0` at init,
        # legged_robot.py:881) -- i.e. a full window must elapse first.
        # Initializing to 0 (matching G1's last_step_counter=0) reproduces
        # that: first eligible fire is common_step_counter >= update_interval.
        # See docs/BugFixes.md.
        self._last_update     = 0
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
            smoothed_ep_len = _update_smoothed_ep_len(env, mean_ep_len)
            cu = int(smoothed_ep_len / self._ep_len_divisor)
            # FIX 2026-07-09: was `max(env._curriculumupdate, cu)` (monotonic
            # ratchet, never drops). G1 (legged_robot.py:329) has no such floor
            # -- curriculumupdate is a fresh recomputation every update, so
            # reward/difficulty scales ease back down if performance regresses.
            # The ratchet meant a policy that hit a hard patch after advancing
            # (e.g. a difficulty jump it couldn't yet handle) had no path back
            # to easier practice -- confirmed live: ball_difficulty hit 1.0,
            # shank_height terminations and episode length regressed hard, and
            # never recovered over 2500+ further iterations because nothing
            # could ease back off. See docs/BugFixes.md.
            env._curriculumupdate = cu

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
    and expand command ranges by `step_size × curriculumupdate`:
        command_ranges[i] = clip(command_ranges[i] ± 0.3*curriculumupdate, bound_lo, bound_hi)
    This is an ACCUMULATOR, not a fresh recompute -- each update nudges the
    range outward by a small bounded step. Since curriculumupdate = int(...)
    is never negative, this can only grow or hold flat; a bad window just
    means zero growth that cycle, not retreat. Contrast with G1's REWARD
    weights (eereach/success/stopball, compute_reward() lines 359-364),
    which ARE freshly, directly recomputed every step from curriculumupdate
    and genuinely go up and down -- G1 uses two different patterns for two
    different things.

    FIX 2026-07-10: the 2026-07-09 fix ("remove the monotonic ratchet")
    correctly matched G1's reward-weight pattern for reward_curriculum_ep_len,
    but WRONGLY applied that same "fresh, fully bidirectional recompute"
    pattern here too (`difficulty = min(1.0, cu/3.0)` every update) --
    this is G1's reward-weight pattern, not its difficulty pattern. A live
    run (rsi_practice_curriculum_2026-07-10) showed ball_difficulty
    oscillating 0.667<->0.333 repeatedly the entire run, never settling --
    exactly the failure mode G1's accumulator can't have by construction
    (it structurally cannot move backward). Reverted to a step-size-based
    accumulator matching G1's actual command_ranges mechanism: difficulty
    only ever increases (or holds flat), clipped to 1.0, never resets to a
    fresh absolute value. See docs/BugFixes.md.

    Default params mirror G1 exactly:
        ep_len_divisor = 50   (same as G1)
        update_interval = 500 (same as G1, in per-env steps)
        step_size = 0.01      (difficulty units per curriculumupdate per check;
                               at cu=2 sustained, reaches 1.0 in ~25 checks;
                               original project value, unused since the
                               2026-07-09 fix, now restored to actual use)
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._step_size      = p.get("step_size",      0.01)
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        # FIX 2026-07-20: was -(update_interval) -- see reward_curriculum_ep_len's
        # __init__ comment for the full explanation. 0 matches G1's
        # last_step_counter=0 init, requiring a full window before first fire.
        self._last_update     = 0
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
        # FIX 2026-07-10: EMA-smoothed via the same shared _update_smoothed_ep_len
        # as reward_curriculum_ep_len, so both curricula stay synchronized and
        # neither oscillates independently on single-window noise. See that
        # function's docstring.
        if len(env_ids) > 0:
            mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
        else:
            mean_ep_len = 0.0
        smoothed_ep_len = _update_smoothed_ep_len(env, mean_ep_len)
        curriculumupdate = int(smoothed_ep_len / self._ep_len_divisor)

        # FIX 2026-07-10: accumulator, matching G1's ACTUAL command_ranges
        # mechanism (legged_robot.py:329-335: range = clip(range ± 0.3*cu,
        # bound_lo, bound_hi) -- an incremental nudge each update, never a
        # fresh absolute recompute). curriculumupdate is never negative, so
        # this can only grow or hold flat -- structurally cannot oscillate,
        # unlike the 2026-07-09 version's `difficulty = min(1.0, cu/3.0)`
        # direct recompute (that formula matches G1's REWARD weights, not
        # its difficulty mechanism -- see class docstring). No ratchet
        # needed here: G1's own mechanism has no floor/ceiling logic beyond
        # the clip, because monotonic-non-decreasing is its natural,
        # structural behavior, not a bolted-on guard.
        env._ball_difficulty = min(1.0, env._ball_difficulty + self._step_size * curriculumupdate)
        return {"ball_difficulty": torch.tensor(env._ball_difficulty)}


class far_travel_curriculum:
    """Curriculum over required LATERAL TRAVEL DISTANCE for far-region crossings,
    decoupled from ball_difficulty, mirroring G1's STEP region (ranges_4/
    ranges_5) -- not its jump region.

    Context (docs/superpowers/specs/2026-07-18-doublestep-research-and-plan.md,
    recommendation B): reset_ball_rolling's one-sided-range branch (used by
    reset_ball_rolling_by_region for left_far/right_far) previously lerped its
    outer y_end bound directly on env._ball_difficulty -- the SAME signal that
    also governs spawn distance, ball speed, and reaction time. Since
    ball_difficulty saturates to 1.0 within the first few thousand iterations
    (observed), the far region's full 1.3m lateral travel requirement was live
    from essentially the start of training, stacked on top of max ball speed
    and minimum reaction time -- with no separate ramp for the one dimension
    that actually requires a genuine multi-step gait.

    FIX 2026-07-18 (same day, two independent subagent audits, no shared
    context, unanimous): the first two versions of this class (single-edge
    growth with a fixed 0.5m inner bound; then a 0.65 initial-fraction seed)
    were both justified by citing G1's JUMP region (ranges_2/ranges_3) --
    "the only regions requiring extra reach." That's the wrong region. Jump's
    reward (_reward_eereach, legged_robot.py:1379-1388) keys off VERTICAL
    velocity with an ESCALATING scale (jump_scale=3.0+3.0*cu) and is paired
    with dedicated airborne/landing rewards (_reward_airfeetorientation,
    _reward_successland, both gated on root_states[:,2]>1.0 -- a literal
    dive). None of that exists for a foot-only, always-grounded task. G1's
    STEP region (ranges_4/ranges_5) is the actual mechanical match: reward
    keys off LATERAL velocity with a FLAT scale (1+3.0*v, same formula as the
    hand region, no curriculumupdate term at all), height is permanently
    frozen ([0.1,0.3] both init and max -- no height axis, exactly like this
    project), and it never touches the jump-gated airborne/landing rewards.
    Jump's frozen-inner-bound pattern (which the first two versions of this
    class ported) turned out to be a COINCIDENCE specific to jump -- its
    width floor (0) happens to already equal its init value (0) -- not a
    general "freeze the inner edge" design. Step's actual width curriculum
    moves BOTH edges: g1_29_config.py ranges_4: init [0.2,1.2] -> max
    [0.0,1.8] (inner shrinks toward the floor, outer grows toward the
    ceiling), identical in shape to the hand region.

    This class now moves both env._far_inner and env._far_outer, mirroring
    G1's actual per-edge accumulator (legged_robot.py:333-334:
    command_ranges[:,0] -= 0.3*cu / command_ranges[:,1] += 0.3*cu, same step
    for both edges) instead of a single growing fraction. Both are seeded
    using STEP's own init-fraction-of-span, applied to this task's own
    floor/ceiling (lo=0.5, the near/far boundary -- FIX 2026-08-01, reverted
    from 0.65 back to 0.5, user request, kept in sync with regions.py's
    _REGION_Y_END_RANGE and rewards.py's wide_threshold -- a structural
    constraint G1 doesn't have, since G1 separates regions by height, not
    width, so its width curricula are free to shrink toward a true
    geometric floor of 0; ours can't go below 0.5 without re-colliding
    with the near region):
        inner_frac = 0.2/1.8 = 0.1111  (G1 step ranges_4 width[0] / maxw[1])
        outer_frac = 1.2/1.8 = 0.6667  (G1 step ranges_4 width[1] / maxw[1])
        far_inner starts at lo + inner_frac*(hi-lo) ~= 0.567, shrinks to lo (0.5)
        far_outer starts at lo + outer_frac*(hi-lo) ~= 0.900, grows to hi (1.1)
    Both driven by the same shared EMA-smoothed episode-length signal (cu) and
    the same step_size per update, matching G1's identical 0.3*cu for both
    edges.

    FIX 2026-07-18 (same day, caught ~300 iterations into the run): because
    far_inner/far_outer share the exact same cu signal as ball_difficulty_
    curriculum (same _update_smoothed_ep_len, same ep_len_divisor=50, same
    update_interval=500), the two curricula's relative speed is set entirely
    by (distance to cover) / (step per cu-unit) -- NOT by step_size alone.
    The previous default (0.004, carried over unchanged from the pre-step-
    region single-edge design) was sized for a far_travel_frac that had to
    travel the FULL [0,1] range; after this seeded far_inner/far_outer 65-67%
    of the way to their bounds already, the remaining distance is much
    smaller (0.267m for outer, 0.089m for inner, vs ball_difficulty's full
    1.0), so the SAME step_size emptied it far faster: outer needed only 83
    cu-units to fully saturate (vs ball_difficulty's 100), inner only 28 --
    confirmed live (iteration ~296: far_inner already 47% of the way to its
    floor while ball_difficulty was only 12% of the way to 1.0) -- the exact
    opposite of the intended lag. New default 0.0013 targets far_outer (the
    binding constraint, larger gap) needing ~250 cu-units (~2.5x
    ball_difficulty's 100, matching this class's original slower-than-
    ball_difficulty intent); far_inner needs ~83 cu-units under the same
    step (close to, not wildly ahead of, ball_difficulty's own pace).

    reset_ball_rolling reads env._far_inner/env._far_outer directly (via
    use_far_travel_curriculum=True, wired only for far regions in
    reset_ball_rolling_by_region), replacing the generic lo/hi/d-based lerp
    for that branch entirely. footreach's far_scale escalation (rewards.py),
    also ported from jump's jump_scale, is a separate fix -- see BugFixes.md.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        p = cfg.params
        self._step_size       = p.get("step_size",       0.0013)
        self._update_interval = p.get("update_interval", 500)
        self._ep_len_divisor  = p.get("ep_len_divisor",   50)
        self._lo              = p.get("lo", 0.5)  # FIX 2026-08-01: reverted to 0.5 (was 0.65 since 2026-07-23)
        self._hi              = p.get("hi", 1.0)  # FIX 2026-08-06: 1.1 -> 1.0 (was 1.3 pre-2026-08-01)
        # FIX 2026-07-20: was -(update_interval) -- see reward_curriculum_ep_len's
        # __init__ comment for the full explanation. 0 matches G1's
        # last_step_counter=0 init, requiring a full window before first fire.
        self._last_update     = 0
        span = self._hi - self._lo
        if not hasattr(env, "_far_inner"):
            env._far_inner = self._lo + _G1_STEP_INNER_FRAC * span
        if not hasattr(env, "_far_outer"):
            env._far_outer = self._lo + _G1_STEP_OUTER_FRAC * span

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        if env.common_step_counter - self._last_update < self._update_interval:
            return {"far_inner": torch.tensor(env._far_inner), "far_outer": torch.tensor(env._far_outer)}

        self._last_update = env.common_step_counter

        if len(env_ids) > 0:
            mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
        else:
            mean_ep_len = 0.0
        # Reuses the SAME shared EMA state as ball_difficulty_curriculum /
        # reward_curriculum_ep_len (_update_smoothed_ep_len is idempotent per
        # call within a window and all three curricula read the same smoothed
        # signal) so this doesn't introduce a second, independently-noisy
        # episode-length estimate.
        smoothed_ep_len = _update_smoothed_ep_len(env, mean_ep_len)
        curriculumupdate = int(smoothed_ep_len / self._ep_len_divisor)

        span = self._hi - self._lo
        step = self._step_size * span * curriculumupdate
        env._far_inner = max(self._lo, env._far_inner - step)
        env._far_outer = min(self._hi, env._far_outer + step)
        return {"far_inner": torch.tensor(env._far_inner), "far_outer": torch.tensor(env._far_outer)}


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


def shank_height_termination(
    env: "ManagerBasedRlEnv",
    min_height: float = 0.24,
    asset_cfg: "SceneEntityCfg" = SceneEntityCfg("robot", body_names=("Shank_Left", "Shank_Right")),
) -> torch.Tensor:
    """Terminate when either knee (shank body origin) drops below min_height above floor.

    Catches deep single-step lunges that the kneeheight penalty alone cannot prevent.
    Fires when the worst-case shank across left/right is below threshold.
    """
    robot: Entity = env.scene[asset_cfg.name]
    shank_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    floor_z = env.scene.env_origins[:, 2]
    shank_z = shank_pos_w[:, :, 2] - floor_z[:, None]                  # (N, 2)
    return shank_z.min(dim=-1).values < min_height


def reset_ball_rolling(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (1.5, 2.5),
    y_start_range: tuple[float, float] = (-0.5, 0.5),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    t_flight_range: tuple[float, float] = (0.7, 1.1),
    spawn_z: float = 0.10,
    y_end_outer_frac: float | None = None,
    use_far_travel_curriculum: bool = False,
) -> None:
    """Spawn ball at ground level in world (global) frame — rolling ground pass.

    Spawns at (env_origin_x + x_start, env_origin_y + y_start) in world frame so
    the ball always approaches along world -X regardless of robot yaw. This keeps
    all world-frame reward terms (stopball uses world-X velocity, stayonline uses
    world-X position, ball_exit_termination uses world-X) perfectly aligned with
    the ball trajectory.

    vz=0: ball drops to ground in <0.05 s then rolls at foot/ankle height.
    Only the target y_end window (below) is curriculum-scaled via
    env._ball_difficulty / far_travel_curriculum, matching G1's actual
    mechanism -- see the FIX 2026-07-20 note below.

    use_far_travel_curriculum: 2026-07-18 (research-doc recommendation B; G1
    step-region redesign same day). When True, the ONE-SIDED range branch's
    inner AND outer y_end bounds are read directly from
    env._far_inner/env._far_outer (far_travel_curriculum, decoupled from
    env._ball_difficulty) instead of the generic lo/lo+(hi-lo)*d lerp -- see
    that class's docstring. Wired True only for far regions by
    regions.reset_ball_rolling_by_region; near-region y_end and the two-sided
    branch keep using env._ball_difficulty directly.

    FIX 2026-07-20 (curriculum/RSI audit item 14, re-derived from scratch):
    dist_range, y_start_range, and t_flight_range used to ALSO be lerped from
    an "easy" sub-range up to the configured range via env._ball_difficulty,
    the same signal driving y_end below. G1's actual mechanism
    (legged_robot.py:771-815 assign_ball_states) does NOT do this: spawn
    distance is a hardcoded constant (`2.0*rand()+3.0`, line 779, no config
    field, no curriculum), spawn Y is sampled from `self.init_ranges`
    (line 780), which is assigned once at init from config maxw/maxh bounds
    (lines 960-963) and never reassigned by the curriculum update
    (lines 325-336 only ever writes `self.command_ranges`), and t_flight is
    another hardcoded constant (`0.4+0.6*rand()`, line 803, play override at
    line 806 also hardcoded). Only the END target -- `ball_end_local`'s Y
    (line 786, `self.command_ranges[:,0:2]`) and Z (line 787,
    `self.command_ranges[:,2:4]`) -- reads the curriculum-updated
    `command_ranges` accumulator (lines 333-336). So G1 curriculum-scales the
    catch WINDOW only; spawn distance, spawn lateral offset, and reaction
    time are sampled from their full fixed range from iteration 0. The
    dist_r/y_start_r/t_flight_r lerps here were an SGK-only addition with no
    G1 equivalent -- removed; these three are now always sampled from the
    full configured range, matching G1. The y_end curriculum below is
    UNCHANGED: it is the one dimension G1's own command_ranges mechanism
    really does scale, confirmed by the line citations above. See
    docs/BugFixes.md for the full reasoning table (dead code note:
    reset_ball_local_frame/reset_ball_global_frame in this module still lerp
    all of dist/y_start/z_start/z_end/speed via the same pattern, but neither
    is wired into any task cfg -- verified unreferenced outside their own
    definitions -- so they were left as-is, out of scope for this fix).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    ball: Entity = env.scene[ball_name]
    origins = env.scene.env_origins[env_ids]   # world goal-line origin per env
    floor_z = origins[:, 2]

    # FIX 2026-07-20: dist_range/y_start_range/t_flight_range are sampled
    # directly, NOT lerped by d -- G1 never curriculum-scales spawn distance,
    # spawn lateral offset, or reaction time (see docstring above). `d` is
    # still used below for the target y_end window, which G1 DOES scale.
    x_start  = sample_uniform(*dist_range,     (n,), env.device)
    y_start  = sample_uniform(*y_start_range,  (n,), env.device)
    t_flight = sample_uniform(*t_flight_range, (n,), env.device)

    if y_end_range[0] * y_end_range[1] > 0:
        # One-sided range (region-conditioned calls via reset_ball_rolling_by_region,
        # e.g. left_near=(0.15,0.5), right_far=(-0.9,-0.5)): the sign is the whole
        # point of the region label (left vs right) -- it must never be randomized.
        # BUG (fixed 2026-07-06): this branch used to fall through to the two-sided
        # dead-zone logic below, which (a) re-randomizes side with a 50/50 coin flip
        # regardless of y_end_range's actual sign, and (b) derives its magnitude
        # bounds from a fixed generic constant (_Y_OUTER=0.35) and max_half, ignoring
        # the region's own inner bound entirely -- so e.g. a "far" region
        # (0.5,0.9) sampled magnitudes as low as 0.1, spilling deep into "near"
        # territory. Verified empirically: ~50% wrong sign, and "far" regions landed
        # in their own intended [0.5,0.9] band only ~25% of the time. This silently
        # decorrelated the region_estimator's ground-truth label (env._region_id)
        # from the ball's actual observable trajectory, and fed the actor a
        # region_arg conditioning signal that didn't match the real shot ~50-75%
        # of the time -- the direct cause of the region_estimator's persistent
        # left/right confusion and the failure to converge on far-side
        # double/triple-step motions. Fix: derive lo/hi from this range's own
        # bounds and lerp the sampled magnitude within [lo,hi] only, sign fixed.
        lo = min(abs(y_end_range[0]), abs(y_end_range[1]))
        hi = max(abs(y_end_range[0]), abs(y_end_range[1]))
        sign_val = 1.0 if y_end_range[1] > 0 else -1.0
        # FIX 2026-08-15 (user request, near-region-only skewed sampling):
        # True only in the near-region `else` branch below, False for the
        # `y_end_outer_frac` override and `use_far_travel_curriculum` (far)
        # branches -- both left as plain uniform, per the user's explicit
        # "for the near region" scoping.
        is_near_region_branch = False
        if y_end_outer_frac is not None:
            inner = lo + (hi - lo) * max(0.0, min(1.0, y_end_outer_frac))
            outer = hi
        elif use_far_travel_curriculum:
            # FIX 2026-07-18 (G1 step-region redesign): both edges move,
            # driven by far_travel_curriculum's env._far_inner/_far_outer.
            # Defaults to the full [lo,hi] span if the curriculum hasn't run
            # yet (play/eval with no CurriculumManager), matching how
            # env._ball_difficulty itself defaults to full difficulty (1.0)
            # in that same situation.
            inner = float(getattr(env, "_far_inner", lo))
            outer = float(getattr(env, "_far_outer", hi))
        else:
            # FIX 2026-07-18: outer used to start exactly at lo (outer=lo+
            # (hi-lo)*0 when d=0), collapsing near regions to a single
            # degenerate point at the start of training -- same bug class as
            # far_travel_curriculum's original bug, same finding: no G1
            # region ever starts at a degenerate point. Only near hits this
            # branch in practice (far uses use_far_travel_curriculum; the
            # plain single-disc task always uses the two-sided branch
            # above). Seeds outer at G1 step's own outer-fraction-of-span
            # instead of lo, then lerps toward hi as ball_difficulty (no
            # separate decoupled signal needed here -- near doesn't require
            # a genuine multi-step skill, so there's no "races ball_difficulty"
            # concern the way there was for far) advances from 0 to 1.
            # inner stays fixed at lo -- FIX 2026-08-15 (user request):
            # regions.py's _REGION_Y_END_RANGE near-region lo lowered
            # 0.15 -> 1e-4 (effectively 0; see that file's own comment for
            # why not a literal 0.0), so `inner` now allows the target to
            # land almost exactly at center.
            inner = lo
            outer_seed = lo + _G1_STEP_OUTER_FRAC * (hi - lo)
            outer = outer_seed + (hi - outer_seed) * d
            is_near_region_branch = True
        if is_near_region_branch:
            # FIX 2026-08-15 (user request): "i want it a little rare to be
            # in the center because there is nothing much to learn from
            # those." Was a plain uniform sample over [inner, outer] --
            # replaced with a power-transform skew (u~Uniform(0,1),
            # mag=inner+(outer-inner)*u**0.5) that biases density toward the
            # outer/harder end. p=0.5 chosen via AskUserQuestion (offered
            # 0.6/0.35/custom -- user picked a milder custom 0.5, "a little
            # rare," not the aggressive 0.35 option). Only applied to the
            # near-region branch (`is_near_region_branch`, set just above)
            # -- far regions (use_far_travel_curriculum) and the one-sided-
            # outer-frac override (y_end_outer_frac) are unaffected, per
            # explicit "for the near region" scoping.
            u = sample_uniform(0.0, 1.0, (n,), env.device)
            mag = inner + (outer - inner) * u.pow(0.5)
        else:
            mag = sample_uniform(inner, outer, (n,), env.device)
        y_end = sign_val * mag
    else:
        # Two-sided dead zone for y_end (mirrors G1's ±0.2 m minimum offset).
        # At d=0: ball always arrives 0.15–0.35 m left or right of center — never
        # through the robot's legs regardless of RSI stance width.
        # At d=1: magnitude range opens to [0.0, max_half] covering the full range.
        # The dead zone shrinks linearly so there is no curriculum cliff at any d.
        _Y_INNER = 0.15   # minimum offset from center at d=0
        _Y_OUTER = 0.35   # maximum offset from center at d=0
        max_half = max(abs(y_end_range[0]), abs(y_end_range[1]))
        if y_end_outer_frac is not None:
            # Testing/play override: ignore the difficulty curriculum entirely for
            # this dimension and pin to the outer band of the true (max-difficulty)
            # range -- e.g. y_end_outer_frac=0.8 samples only the top 20% of
            # max_half, regardless of env._ball_difficulty's current value.
            inner = max_half * max(0.0, min(1.0, y_end_outer_frac))
            outer = max_half
        else:
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
    # Ball travels from (x_start, y_start) to (-0.3, y_end) along a straight line, linearly
    # parametrized by fraction f in [0,1]: X(f) = x_start - f*(x_start+0.3). The goal line is
    # at X=0 (NOT at the aim point X=-0.3), so solving X(f)=0 gives f = x_start/(x_start+0.3).
    # cross_y = y_start + (y_end - y_start) * f.
    # FIX 2026-07-07: previously used f = (x_start+0.3)/horiz_dist (horiz_dist = the Euclidean
    # spawn-to-aim-point distance) -- that's cos(trajectory angle), not the goal-line crossing
    # fraction, and it silently overestimated f for every diagonal shot (e.g. x_start=2.0,
    # dy=0.5: buggy f=0.977 vs correct f=0.870). Ported from the same bug in SimpleGoalKeeper's
    # reset_ball_rolling (fixed there same day) -- see docs/BugFixes.md.
    if not hasattr(env, "_rsi_cross_y"):
        env._rsi_cross_y = torch.zeros(env.num_envs, device=env.device)
    env._rsi_cross_y[env_ids] = y_start + (y_end - y_start) * (x_start / (x_start + 0.3))

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


# Matches t1_constants.py's own `_foot_regex` -- the 8 foot collision geoms
# (left/right_foot1..4_collision) that FULL_COLLISION gives priority=1,
# solref=(0.01, 1.0). Since foot priority beats the ball's own solref for
# ANY foot-ball contact (confirmed live, 2026-08-23: varying the ball's own
# solref changed foot-ball bounce restitution 0.0% -- identical 0.1289 at
# every value tested), THIS is the geom that has to be randomized to vary
# save-contact bounciness at all, not the ball. See rewards on restitution
# investigation, 2026-08-23.
_FOOT_COLLISION_CFG = SceneEntityCfg(
    "robot", geom_names=(r"^(left|right)_foot\d+_collision$",)
)


def randomize_foot_ball_restitution(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    dampratio_range: tuple[float, float] = (0.35, 1.0),
    asset_cfg: SceneEntityCfg = _FOOT_COLLISION_CFG,
) -> None:
    """Randomize save-contact bounciness per env, matching G1's own
    `randomize_restitution` (`g1_29_config.py`: `restitution_range=[0.0,1.0]`,
    resampled every reset).

    G1 randomizes the BALL's restitution directly (Isaac Gym/PhysX resolves
    contact restitution as a genuine per-shape material property). MuJoCo has
    no literal restitution scalar -- bounciness is emergent from a
    (timeconst, dampratio) solref pair -- AND this project's foot geoms
    already carry `priority=1` (`FULL_COLLISION`, `t1_constants.py`), which
    makes MuJoCo use the foot's own solref *entirely* for foot-ball contact,
    ignoring the ball's. So randomizing the ball (the literal G1-equivalent
    target) would be a no-op for the actual save; this randomizes the winning
    geom (the foot) instead to get the same real effect G1 gets.

    `dampratio_range=(0.35, 1.0)` was calibrated live against the real
    `mujoco_warp` engine (production's own timestep=0.005,
    integrator=implicitfast, iterations=10, ls_iterations=20 --
    `velocity_env_cfg.py`, no override in this project), holding
    `timeconst=0.01` fixed (production's own value) and sweeping only
    dampratio -- the two-parameter version was non-monotonic. Measured
    first-bounce restitution across the calibrated range:

        dampratio=1.00 (current production) -> restitution 0.129
        dampratio=0.70                      -> restitution 0.231
        dampratio=0.50                      -> restitution 0.424
        dampratio=0.35                      -> restitution 0.834
        dampratio=0.30                      -> restitution 1.124 (unphysical -- excluded)

    So dampratio=1.0 is the LEAST bouncy end (matches current production
    exactly, restitution~0.13) and dampratio=0.35 is the MOST bouncy end of
    the verified-stable range (restitution~0.83) -- sampling uniform over
    dampratio does not give a uniform restitution distribution (the mapping
    is nonlinear), a known simplification, not yet refit to be
    restitution-uniform. `timeconst` is deliberately NOT randomized -- values
    below ~0.005 (this project's timestep) were confirmed live to explode
    (NaN) on the real mujoco_warp engine.

    All 8 foot geoms (both feet, all 4 capsules each) get the SAME per-env
    value -- a physical foot doesn't have per-capsule independent
    bounciness. Only the dampratio axis (index 1) is touched; `timeconst`
    (index 0) and every other FULL_COLLISION field (condim, friction,
    priority itself) are untouched.

    FIX 2026-08-23: does NOT use mjlab's own `dr._core._randomize_model_field`
    (what every other DR function in this file's sibling `dr/geom.py` uses)
    -- confirmed live it computes a DIFFERENT, off-by-one geom index set
    than `asset_cfg.geom_ids`/`robot.find_geoms()` for this scene (wrote to
    `left_foot2/3/4_collision` + one unrelated extra geom, never
    `left_foot1_collision`, same one-off pattern on the right side) --
    likely a genuine indexing bug in that private helper for this asset,
    not something to route around by guessing at a fix. Writes directly to
    `env.sim.model.geom_solref` using `asset_cfg.geom_ids`, independently
    verified twice (matches a fresh `robot.find_geoms()` call exactly) to
    be correct for this scene.

    Not yet validated against a live training run.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    geom_ids = torch.as_tensor(asset_cfg.geom_ids, device=env.device, dtype=torch.long)

    lo, hi = dampratio_range
    sampled = sample_uniform(lo, hi, (len(env_ids),), env.device)  # (n_envs,)

    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    env.sim.model.geom_solref[env_grid, geom_grid, 1] = sampled.unsqueeze(-1).expand(-1, len(geom_ids))

    # Stash the sampled per-env value for the P-panel plot (play.py) and any
    # reward/diagnostic code that wants it.
    if not hasattr(env, "_foot_restitution_dampratio"):
        env._foot_restitution_dampratio = torch.ones(env.num_envs, device=env.device)
    env._foot_restitution_dampratio[env_ids] = sampled
