"""Event functions and curriculum helpers for SimpleGoalKeeperHim (HIM-PPO, 2-disc)."""
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

# Left-group NPZ files (disc 0)
_LEFT_FILES = [
    "LeftDoubleStep_own_booster_t1.npz",
    "LeftQuickStep_own_booster_t1.npz",
    "LeftStep_own_booster_t1.npz",
    "LeftTripleStep_own_booster_t1.npz",
]
# Right-group NPZ files (disc 1)
_RIGHT_FILES = [
    "RightDoubleStep_own_booster_t1.npz",
    "Rightstep_own_booster_t1.npz",
    "RightTripleStep_own_booster_t1.npz",
]

# Easy (difficulty=0.0) spawn ranges: short distance, centred, slow, low.
# Hard (difficulty=1.0) ranges are passed as params to reset_ball_local_frame.
_EASY_DIST    = (2.0, 3.0)
_EASY_Y_START = (-0.05, 0.05)
_EASY_Y_END_ABS = 0.05   # absolute upper bound for easy; sign is set per disc
_EASY_Z_START = (0.1, 0.25)
_EASY_Z_END   = (0.05, 0.15)
_EASY_SPEED   = (2.0, 3.0)


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
    """Return a quaternion containing ONLY the yaw component of q_wxyz."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    out = torch.zeros_like(q_wxyz)
    out[:, 0] = torch.cos(half)   # w
    out[:, 3] = torch.sin(half)   # z
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Motion type assignment (static partition: left / right)
# ─────────────────────────────────────────────────────────────────────────────

def assign_motion_types(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: assign each env permanently to disc 0 (left) or disc 1 (right).

    N >= 2: first half → 0 (left disc), second half → 1 (right disc). Balanced split.
    N == 1: random assignment so play/debug mode sees both disc sides over multiple runs.
    The runner reads env._motion_type_ids for AMP reward routing.
    """
    N = env.num_envs
    ids = torch.zeros(N, dtype=torch.long, device=env.device)
    if N >= 2:
        ids[N // 2:] = 1
    else:
        # Single env (play mode): N//2==0 would assign all to right; randomise instead.
        ids[0] = int(torch.randint(0, 2, (1,)).item())
    env._motion_type_ids = ids


# ─────────────────────────────────────────────────────────────────────────────
# Motion reset manager — per-group RSI
# ─────────────────────────────────────────────────────────────────────────────

class MotionResetManager:
    """Manages motion frame data for full RSI, split by left/right group.

    Full RSI writes root Z + quaternion + linear/angular velocity + joint state
    from a random motion frame so the initial physics state is self-consistent.
    Root XY is kept from env_origins so the goalkeeper always starts at the goal line.
    """

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self.frames: list[dict[str, torch.Tensor]] = [{}, {}]  # [left_group, right_group]

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_group(self, file_list: list[str], device: str) -> dict[str, torch.Tensor]:
        joint_pos_list, joint_vel_list = [], []
        root_pos_list, root_quat_list = [], []
        root_lin_vel_list, root_ang_vel_list = [], []
        for fname in file_list:
            fpath = _MOTIONS_DIR / fname
            data = np.load(str(fpath))
            joint_pos_list.append(data["joint_pos"])
            joint_vel_list.append(data["joint_vel"])
            root_pos_list.append(data["body_pos_w"][:, 0, :])
            root_quat_list.append(data["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(data["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(data["body_ang_vel_w"][:, 0, :])

        def stack(lst):
            return torch.from_numpy(np.vstack(lst)).float().to(device)

        return {
            "joint_pos":    stack(joint_pos_list),
            "joint_vel":    stack(joint_vel_list),
            "root_pos":     stack(root_pos_list),
            "root_quat":    stack(root_quat_list),
            "root_lin_vel": stack(root_lin_vel_list),
            "root_ang_vel": stack(root_ang_vel_list),
        }

    def init(self, env: "ManagerBasedRlEnv") -> None:
        """Load NPZ files for each group (left, right)."""
        if self.frames[0]:
            return
        dev = env.device
        self.frames[0] = self._load_group(_LEFT_FILES, dev)
        self.frames[1] = self._load_group(_RIGHT_FILES, dev)
        print(
            f"[MotionResetManager] left={self.frames[0]['joint_pos'].shape[0]} frames, "
            f"right={self.frames[1]['joint_pos'].shape[0]} frames"
        )

    def reset(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ) -> None:
        """RSI: 80% copy live-env state, 20% NPZ frame — mirrors G1 continue_keep=True.

        80% of resets sample joint+root state from a random currently-running environment.
        This keeps the RSI distribution on-manifold: it matches wherever the policy
        currently operates, so reset states are always physically plausible.

        20% of resets use a pre-recorded NPZ stepping frame.  This ensures stepping-motion
        diversity even at the start of training when all live envs are near-standing.

        Root XY is always taken from env_origins (goal line fixed).
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        if len(env_ids) == 0:
            return

        robot: Entity = env.scene[asset_cfg.name]
        n = len(env_ids)

        motion_ids = getattr(env, "_motion_type_ids", None)
        if motion_ids is None:
            group_ids = torch.zeros(n, dtype=torch.long, device=env.device)
        else:
            group_ids = motion_ids[env_ids]

        joint_pos_out = torch.zeros(n, self.frames[0]["joint_pos"].shape[1], device=env.device)
        joint_vel_out = torch.zeros_like(joint_pos_out)
        # Root XY always from env_origins (goal line fixed).  NPZ path overwrites Z only
        # (root_pos[:, 2]) so the recording's world-space XY is never applied.
        positions_out = env.scene.env_origins[env_ids].clone()
        # Identity quaternion (w=1, x=y=z=0); NPZ path overwrites with frame quat.
        quat_out      = torch.zeros(n, 4, device=env.device)
        quat_out[:, 0] = 1.0
        lin_vel_out   = torch.zeros(n, 3, device=env.device)
        ang_vel_out   = torch.zeros(n, 3, device=env.device)

        # Split: 80% live-env DOF copy, 20% NPZ frames.
        use_live = torch.rand(n, device=env.device) < 0.8

        # --- 80%: G1-faithful DOF-position-only copy ---
        # Mirrors G1 _reset_dofs (legged_robot.py:669-680):
        #   dof_pos[env_ids] = dof_pos[random_src]   ← joint pose diversity
        #   dof_vel[env_ids] = 0.                     ← velocities always zero
        #   root state: always home position (env_origins) + small yaw jitter
        #
        # CRITICAL: only copy from envs that are currently standing (root Z > 0.5 m above floor).
        # With a random policy at training start all live envs crash in <5 steps.
        # Copying joint positions from a crashed robot into a standing-height root creates
        # a physically impossible state (limbs underground) → sharpforce on step 1 →
        # episodes stay at length 1–3 forever.  Fall back to the robot's default joint
        # positions (HOME_KEYFRAME: bent knees, arms neutral) for crashed source envs.
        live_mask = use_live.nonzero(as_tuple=False).flatten()
        if len(live_mask) > 0:
            src_ids = torch.randint(0, env.num_envs, (len(live_mask),), device=env.device)
            rdata = robot.data

            # Source env standing check: root height > 0.5 m above its floor tile.
            floor_z_src = env.scene.env_origins[src_ids, 2]
            src_upright = (rdata.root_link_pos_w[src_ids, 2] - floor_z_src) > 0.5  # (n_live,)

            # default_joint_pos = HOME_KEYFRAME initial state (bent knees, arms neutral).
            default_jp = rdata.default_joint_pos   # (N, dof) or (1, dof)
            if default_jp.shape[0] == env.num_envs:
                default_jp_src = default_jp[src_ids]
            else:
                default_jp_src = default_jp[:1].expand(len(live_mask), -1)

            joint_pos_out[live_mask] = torch.where(
                src_upright.unsqueeze(-1),
                rdata.joint_pos[src_ids],   # standing env → live copy
                default_jp_src,             # crashed env → HOME_KEYFRAME
            )
            # joint_vel_out stays zero (G1: dof_vel[env_ids] = 0. after live copy)
            # Add small yaw jitter so the robot doesn't always face exactly +X
            yaw = (torch.rand(len(live_mask), device=env.device) - 0.5) * 0.2  # ±0.1 rad
            half = yaw * 0.5
            quat_out[live_mask, 0] = torch.cos(half)   # w
            quat_out[live_mask, 3] = torch.sin(half)   # z
            # positions_out stays at env_origins; lin/ang_vel stays zero

        # --- 20%: sample from NPZ motion frames (per-group) ---
        npz_mask = (~use_live).nonzero(as_tuple=False).flatten()
        if len(npz_mask) > 0:
            npz_group_ids = group_ids[npz_mask]
            for g in range(2):
                gm = (npz_group_ids == g).nonzero(as_tuple=False).flatten()
                if len(gm) == 0:
                    continue
                frames = self.frames[g]
                total_frames = frames["joint_pos"].shape[0]
                frame_ids = torch.randint(0, total_frames, (len(gm),), device=env.device)
                idx = npz_mask[gm]
                root_pos  = frames["root_pos"][frame_ids]
                root_quat = frames["root_quat"][frame_ids]
                positions_out[idx, 2] = root_pos[:, 2]
                quat_out[idx]         = root_quat
                lin_vel_out[idx]      = frames["root_lin_vel"][frame_ids]
                ang_vel_out[idx]      = frames["root_ang_vel"][frame_ids]
                joint_pos_out[idx]    = frames["joint_pos"][frame_ids]
                joint_vel_out[idx]    = frames["joint_vel"][frame_ids]

        robot.write_root_link_pose_to_sim(
            torch.cat([positions_out, quat_out], dim=-1), env_ids=env_ids
        )
        robot.write_root_link_velocity_to_sim(
            torch.cat([lin_vel_out, ang_vel_out], dim=-1), env_ids=env_ids
        )
        robot.write_joint_state_to_sim(joint_pos_out, joint_vel_out, env_ids=env_ids)


def init_motion_loader(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Startup event: load all NPZ motion files into MotionResetManager."""
    MotionResetManager.get().init(env)


def reset_from_motion_data(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
    """Reset event: full RSI from a random frame in the env's motion group."""
    mgr = MotionResetManager.get()
    mgr.init(env)
    mgr.reset(env, env_ids, asset_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Ball spawn with left/right discriminator routing
# ─────────────────────────────────────────────────────────────────────────────

def reset_ball_local_frame(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (2.0, 4.0),
    y_start_range: tuple[float, float] = (-0.8, 0.8),
    y_end_max: float = 1.5,
    spawn_height_range: tuple[float, float] = (0.05, 0.35),
    arrive_height_range: tuple[float, float] = (0.05, 0.25),
    speed_range: tuple[float, float] = (3.0, 7.0),
) -> None:
    """Spawn ball in robot local frame, routing y_end sign by discriminator group.

    Left envs (disc 0): y_end ∈ [0, y_end_max]   (positive Y = left in robot frame)
    Right envs (disc 1): y_end ∈ [-y_end_max, 0]  (negative Y = right in robot frame)

    The y_end_max hard value lerps from _EASY_Y_END_ABS (easy) to y_end_max (hard)
    with the curriculum difficulty. y_start_range and other dimensions are
    curriculum-lerped the same as SGK's reset_ball_local_frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    dist_r    = _lerp_range(_EASY_DIST,    dist_range,          d)
    y_start_r = _lerp_range((-0.05, 0.05), y_start_range,       d)
    z_start_r = _lerp_range(_EASY_Z_START, spawn_height_range,  d)
    z_end_r   = _lerp_range(_EASY_Z_END,   arrive_height_range, d)
    speed_r   = _lerp_range(_EASY_SPEED,   speed_range,         d)
    y_end_hard = _EASY_Y_END_ABS + d * (y_end_max - _EASY_Y_END_ABS)

    g = 9.81
    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene["robot"]

    robot_pos_w  = robot.data.root_link_pos_w[env_ids]
    robot_quat_w = robot.data.root_link_quat_w[env_ids]
    yaw_q        = _yaw_only_quat(robot_quat_w)

    floor_z = env.scene.env_origins[env_ids, 2]

    x_start = sample_uniform(*dist_r,    (n,), env.device)
    y_start = sample_uniform(*y_start_r, (n,), env.device)
    z_start = floor_z + sample_uniform(*z_start_r, (n,), env.device)
    z_end   = floor_z + sample_uniform(*z_end_r,   (n,), env.device)
    speed_h = sample_uniform(*speed_r,   (n,), env.device)

    # Route y_end sign by discriminator group.
    motion_ids = getattr(env, "_motion_type_ids", None)
    if motion_ids is not None:
        group = motion_ids[env_ids]                               # (n,) 0=left,1=right
        # Left envs: y_end in [0, y_end_hard]; Right envs: y_end in [-y_end_hard, 0].
        y_end_u = torch.rand(n, device=env.device) * y_end_hard  # uniform in [0, hard]
        y_end = torch.where(group == 0, y_end_u, -y_end_u)
    else:
        y_end = sample_uniform(-y_end_hard, y_end_hard, (n,), env.device)

    local_spawn = torch.stack([x_start, y_start, torch.zeros_like(x_start)], dim=-1)
    world_spawn_xy = quat_apply(yaw_q, local_spawn)
    ball_pos = torch.empty((n, 3), device=env.device)
    ball_pos[:, 0] = robot_pos_w[:, 0] + world_spawn_xy[:, 0]
    ball_pos[:, 1] = robot_pos_w[:, 1] + world_spawn_xy[:, 1]
    ball_pos[:, 2] = z_start

    dx_local = -(x_start + 0.3)
    dy_local = y_end - y_start
    horiz_dist = torch.sqrt(dx_local ** 2 + dy_local ** 2)
    t_flight = horiz_dist / speed_h

    vx_local = dx_local / t_flight
    vy_local = dy_local / t_flight

    local_vel_h = torch.stack([vx_local, vy_local, torch.zeros_like(vx_local)], dim=-1)
    world_vel_h = quat_apply(yaw_q, local_vel_h)

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


def reset_ball_rolling(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    ball_name: str,
    dist_range: tuple[float, float] = (1.5, 2.5),
    y_start_range: tuple[float, float] = (-0.5, 0.5),
    y_end_range: tuple[float, float] = (-0.5, 0.5),
    speed_range: tuple[float, float] = (2.0, 3.5),
    spawn_z: float = 0.12,
) -> None:
    """Spawn ball at ground level with only horizontal velocity (ground-rolling pass).

    Routes y_end sign by discriminator group (same as reset_ball_local_frame).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)

    d = float(getattr(env, "_ball_difficulty", 1.0))
    d = max(0.0, min(1.0, d))

    _EASY_DIST_R  = (1.5, 2.0)
    _EASY_Y_ROLL  = (-0.05, 0.05)
    _EASY_SPEED_R = (1.5, 2.5)
    _y_end_max_hard = abs(y_end_range[1]) if y_end_range[1] != 0 else 0.5

    dist_r    = _lerp_range(_EASY_DIST_R,   dist_range,    d)
    y_start_r = _lerp_range(_EASY_Y_ROLL,   y_start_range, d)
    speed_r   = _lerp_range(_EASY_SPEED_R,  speed_range,   d)
    y_end_hard = _EASY_Y_END_ABS + d * (_y_end_max_hard - _EASY_Y_END_ABS)

    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene["robot"]

    robot_pos_w  = robot.data.root_link_pos_w[env_ids]
    robot_quat_w = robot.data.root_link_quat_w[env_ids]
    yaw_q        = _yaw_only_quat(robot_quat_w)

    floor_z = env.scene.env_origins[env_ids, 2]

    x_start = sample_uniform(*dist_r,    (n,), env.device)
    y_start = sample_uniform(*y_start_r, (n,), env.device)
    speed_h = sample_uniform(*speed_r,   (n,), env.device)

    # Route y_end sign by discriminator group.
    motion_ids = getattr(env, "_motion_type_ids", None)
    if motion_ids is not None:
        group = motion_ids[env_ids]
        y_end_u = torch.rand(n, device=env.device) * y_end_hard
        y_end = torch.where(group == 0, y_end_u, -y_end_u)
    else:
        y_end = sample_uniform(-y_end_hard, y_end_hard, (n,), env.device)

    local_spawn = torch.stack([x_start, y_start, torch.zeros_like(x_start)], dim=-1)
    world_spawn_xy = quat_apply(yaw_q, local_spawn)

    ball_pos = torch.empty((n, 3), device=env.device)
    ball_pos[:, 0] = robot_pos_w[:, 0] + world_spawn_xy[:, 0]
    ball_pos[:, 1] = robot_pos_w[:, 1] + world_spawn_xy[:, 1]
    ball_pos[:, 2] = floor_z + spawn_z

    dx_local = -(x_start + 0.3)
    dy_local = y_end - y_start
    horiz_dist = torch.sqrt(dx_local ** 2 + dy_local ** 2)
    t_flight = horiz_dist / speed_h

    vx_local = dx_local / t_flight
    vy_local = dy_local / t_flight

    local_vel_h = torch.stack([vx_local, vy_local, torch.zeros_like(vx_local)], dim=-1)
    world_vel_h = quat_apply(yaw_q, local_vel_h)

    ball_vel = torch.zeros((n, 3), device=env.device)
    ball_vel[:, 0] = world_vel_h[:, 0]
    ball_vel[:, 1] = world_vel_h[:, 1]

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

    # Reset post-save timeout counter so each new episode starts fresh.
    if not hasattr(env, "_post_save_steps"):
        env._post_save_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    env._post_save_steps[env_ids] = 0

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


class ball_difficulty_curriculum:
    """Episode-length-adaptive ball difficulty — mirrors G1's curriculum driver.

    Advances difficulty when the policy starts surviving longer, not on a fixed
    step schedule.  This prevents premature difficulty increases when the policy
    is still unstable.

    G1 logic (legged_robot.py:325):
        curriculumupdate = int(mean(episode_length_buf[env_ids]) / 50)
        command_ranges expand by 0.3 * curriculumupdate each update

    Our adaptation:
        curriculumupdate 0 → difficulty 0.0  (mean ep_len < 50 steps = < 1 s)
        curriculumupdate 1 → difficulty 0.5  (mean ep_len 50-100 steps = 1-2 s)
        curriculumupdate 2 → difficulty 1.0  (mean ep_len > 100 steps = > 2 s)

    Updates at most once every min_gate sim steps to avoid oscillation.
    """

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        if not hasattr(env, "_ball_difficulty"):
            env._ball_difficulty = 0.0
        self._min_gate: int = cfg.params.get("min_gate", 500)
        self._last_update: int = 0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        min_gate: int = 500,
    ) -> dict:
        # Throttle: don't update more often than min_gate sim steps.
        if (env.common_step_counter - self._last_update) < self._min_gate:
            return {}
        if env_ids is None or len(env_ids) == 0:
            return {}
        self._last_update = env.common_step_counter

        # Mean episode length of just-terminated envs (mirrors G1 reset_idx).
        mean_ep_len = env.episode_length_buf[env_ids].float().mean().item()
        curriculumupdate = min(int(mean_ep_len / 50.0), 2)   # 0, 1, or 2
        new_difficulty = min(curriculumupdate / 2.0, 1.0)     # 0.0, 0.5, or 1.0

        env._ball_difficulty = new_difficulty
        return {"ball_difficulty": torch.tensor(new_difficulty)}


def sharpforce_termination(
    env: "ManagerBasedRlEnv",
    max_contact_force: float = 1500.0,
) -> torch.Tensor:
    """Terminate when mean foot contact force exceeds threshold."""
    sensor: ContactSensor = env.scene["feet_contact"]
    force_per_geom = sensor.data.force.norm(dim=-1)
    left_max  = force_per_geom[:, :4].max(dim=-1).values
    right_max = force_per_geom[:, 4:].max(dim=-1).values
    mean_force = (left_max + right_max) / 2.0
    return mean_force > max_contact_force


def ball_exit_termination(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    behind_threshold: float = -0.5,
) -> torch.Tensor:
    """Terminate when ball has clearly passed the goal line."""
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return ball_x_local < behind_threshold


def post_save_timeout_termination(
    env: "ManagerBasedRlEnv",
    ball_name: str,
    timeout_steps: int = 80,
) -> torch.Tensor:
    """Terminate N steps after the ball has been deflected or has crossed the goal line.

    Prevents episodes where the robot saves the ball, then slowly topples over
    for several seconds without triggering bad_orientation (57°) or base_height
    (0.4 m) — both thresholds too lenient for a gradual fall after a dive.

    Counter is reset to 0 each episode via _init_visibility_state.
    Does NOT fire while ball is still in front (counter stays at 0).
    """
    ball: Entity = env.scene[ball_name]
    ball_x_local = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    # Mirror _ball_is_behind from rewards.py: deflected OR crossed goal line.
    softstop_fired = getattr(env, "_softstop_flag", None)
    if softstop_fired is not None:
        ball_behind = softstop_fired | (ball_x_local < 0.0)
    else:
        ball_behind = ball_x_local < 0.0

    if not hasattr(env, "_post_save_steps"):
        env._post_save_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Increment while ball is behind; hold at 0 while ball is in front.
    env._post_save_steps = torch.where(
        ball_behind,
        env._post_save_steps + 1,
        torch.zeros_like(env._post_save_steps),
    )

    return env._post_save_steps >= timeout_steps
